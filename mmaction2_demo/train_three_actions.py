"""Fine-tune three independent action heads from explicitly reviewed windows."""

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import time

os.environ['MPLBACKEND'] = 'Agg'

LABELS = ('strike', 'grapple', 'fall')
SPLITS = ('train', 'development', 'test')
PARENT_SHA = 'ddb09180284c4093189c53b7decbfcc5b51fd2d9b7700536d104524bceb5c344'
TEST_ONLY_HASHES = {
    'be235fef9bda633f5f63ccbc28c1dbb6c0138728c5b1696d86f7a6e0c66f4516',
    '7c57683db153b4a05b362763e69f9ebc86c3cc9833c88d789a33bccc3b5cb2bf',
    'a8893c380048907d7be5902a97c620667bdc310f0d50719950d2b964324f7324',
    'd2f2d7a8fb72fb59b2c8326b0d26d7bca504a1a3ff11dd5ba16bcfb837d1b6d4',
}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.partial.json')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def audit(manifest, root, verify_files=True):
    if manifest.get('schema_version') != 1 or manifest.get('labels') != list(LABELS):
        raise ValueError('Expected explicit strike/grapple/fall labels; binary fight labels cannot be converted')
    root = Path(root).resolve()
    rows, identities, files = [], {}, set()
    counts = {split: {label: {'positive_groups': set(), 'negative_groups': set()} for label in LABELS} for split in SPLITS}
    issues = []
    for source in manifest.get('sources', []):
        split, group = source['split'], source['group']
        if split not in SPLITS or not group:
            raise ValueError('Invalid source split/group')
        if group in {'airtlab/violent/37', 'airtlab/violent/87', 'gmdcsa24/subject_4'} and split != 'test':
            raise ValueError('Previously tested source must remain test-only: ' + group)
        sha = source['sha256']
        if sha in TEST_ONLY_HASHES and split != 'test':
            raise ValueError('Previously tested video must remain test-only')
        if len(sha) != 64 or any(character not in '0123456789abcdef' for character in sha):
            raise ValueError('Invalid source checksum')
        for key in ('group:' + group, 'sha256:' + sha, 'identity:' + source.get('identity_group', group)):
            if key in identities and identities[key] != split:
                raise ValueError('Source/group leakage across splits')
            identities[key] = split
        path = (root / source['video']).resolve()
        if not path.is_relative_to(root) or path == root:
            raise ValueError('Video must stay inside the public dataset root')
        if path in files:
            raise ValueError('Duplicate video entry; combine its reviewed windows')
        files.add(path)
        if verify_files and (not path.is_file() or digest(path) != sha):
            raise ValueError('Missing video or checksum mismatch: ' + str(path))
        if not source.get('reviewer') or not source.get('public_source_url', '').startswith('https://'):
            raise ValueError('Review and public-source provenance required')
        anchors = set()
        for window in source.get('windows', []):
            anchor, targets = window['anchor_seconds'], window['targets']
            if not isinstance(anchor, (float, int)) or not math.isfinite(anchor) or anchor < 0 or anchor in anchors:
                raise ValueError('Invalid or duplicate window anchor')
            anchors.add(anchor)
            if set(targets) != set(LABELS) or any(value is not None and (type(value) is not int or value not in (0, 1)) for value in targets.values()):
                raise ValueError('Each target must be explicit 0/1 or null (unknown), not a binary fight alias')
            if all(value is None for value in targets.values()):
                continue
            for label, value in targets.items():
                if value is not None:
                    counts[split][label]['positive_groups' if value else 'negative_groups'].add(group)
            rows.append({'path': str(path), 'sha256': sha, 'video': source['video'], 'group': group,
                         'split': split, 'anchor': float(anchor), 'targets': [targets[label] for label in LABELS]})
    summary = {split: {label: {key: len(groups) for key, groups in values.items()} for label, values in part.items()} for split, part in counts.items()}
    for split, part in summary.items():
        minimum = 3 if split == 'train' else 2
        for label, values in part.items():
            for kind, count in values.items():
                if count < minimum:
                    issues.append(f'{split}/{label}/{kind}: {count}, need >= {minimum}')
    return rows, {'status': 'ready' if not issues else 'needs_labels', 'labels': list(LABELS),
                  'windows': len(rows), 'coverage': summary, 'issues': issues,
                  'note': 'Minimum coverage is only a pilot gate, not proof of dataset quality.'}


def feature_rows(rows, root, weights, device, deadline):
    import numpy as np
    from action_engine import ActionEngine
    from train_action_transition_v2 import frames_for_video, sampled_clip

    engine = ActionEngine(weights=weights, device=device, person_recovery=True)
    loaded = []
    for number, row in enumerate(rows):
        if time.monotonic() >= deadline:
            raise TimeoutError('Feature budget exhausted; completed cache files are retained')
        key = hashlib.sha256(json.dumps([row['sha256'], row['anchor'], digest(Path(__file__).with_name('action_engine.py')),
                                        digest(Path(__file__).with_name('person_recovery.py'))]).encode()).hexdigest()
        target = root / (key + '.npz')
        if not target.exists():
            frames, fps = frames_for_video(row['path'])
            if row['anchor'] >= len(frames) / fps:
                raise ValueError('Reviewed anchor outside decoded source: ' + row['video'])
            bag, _, info = engine.extract_features(sampled_clip(frames, fps, row['anchor']))
            temporary = target.with_suffix('.partial.npz')
            np.savez_compressed(temporary, bag=bag, people=info['people'])
            temporary.replace(target)
        with np.load(target, allow_pickle=False) as saved:
            bag = saved['bag'].copy()
        if bag.ndim != 2 or bag.shape[1] != 4096 or not np.isfinite(bag).all():
            raise ValueError('Invalid feature cache')
        loaded.append({**row, 'bag': bag})
        if (number + 1) % 10 == 0:
            print('FEATURES', number + 1, '/', len(rows), flush=True)
    return loaded


def train(rows, weights, output, deadline, epochs, fingerprint, device):
    import numpy as np
    import torch
    from sklearn.metrics import average_precision_score, precision_recall_fscore_support
    from action_engine import InteractionHead, SOURCE_SHA, SPEC

    parent_path = weights / 'interaction_fall_wrestle.pt'
    if digest(parent_path) != PARENT_SHA:
        raise ValueError('Wrong warm-start parent')
    state = torch.load(parent_path, map_location='cpu', weights_only=True)
    torch.manual_seed(42)
    heads = torch.nn.ModuleList([InteractionHead() for _ in LABELS]).to(device)
    for head in heads:
        head.load_state_dict(state, strict=True)
    optimizer = torch.optim.AdamW(heads.parameters(), lr=1e-4, weight_decay=1e-3)
    partitions = {split: [row for row in rows if row['split'] == split] for split in SPLITS}
    training = [row for row in partitions['train'] if len(row['bag'])]
    if not training:
        raise ValueError('No train windows with detected people')

    def logits(row):
        bag = torch.as_tensor(row['bag'], device=device)
        return torch.stack([head(bag).max() for head in heads])

    @torch.inference_mode()
    def predict(items):
        heads.eval()
        return np.asarray([logits(row).sigmoid().cpu().tolist() if len(row['bag']) else [0.] * 3 for row in items])

    def metrics(items, values, thresholds):
        result = {}
        for index, label in enumerate(LABELS):
            known = [(row, value[index]) for row, value in zip(items, values) if row['targets'][index] is not None]
            truth = [row['targets'][index] for row, _ in known]
            scores = [score for _, score in known]
            if set(truth) != {0, 1}:
                raise ValueError('Need positive and negative known labels: ' + label)
            precision, recall, f1, _ = precision_recall_fscore_support(truth, np.asarray(scores) >= thresholds[index], average='binary', zero_division=0)
            result[label] = {'windows': len(known), 'ap': float(average_precision_score(truth, scores)),
                             'precision': float(precision), 'recall': float(recall), 'f1': float(f1),
                             'no_person': sum(not len(row['bag']) for row, _ in known)}
        return result

    history, best, best_value, stale = [], None, -1., 0
    targets = np.asarray([[np.nan if value is None else value for value in row['targets']] for row in training])
    positives = np.nansum(targets, axis=0)
    negatives = np.nansum(1 - targets, axis=0)
    if (positives == 0).any() or (negatives == 0).any():
        raise ValueError('Detection failures removed a train class; do not train an unsupported head')
    positive_weight = torch.tensor(negatives / positives, dtype=torch.float32, device=device).clamp(.25, 8)
    for epoch in range(epochs):
        if time.monotonic() >= deadline - 10:
            break
        heads.train()
        losses = []
        for indices in torch.randperm(len(training)).split(8):
            if time.monotonic() >= deadline - 10:
                break
            batch = [training[index] for index in indices.tolist()]
            values = torch.stack([logits(row) for row in batch])
            truth = values.new_tensor([[0 if value is None else value for value in row['targets']] for row in batch])
            mask = values.new_tensor([[value is not None for value in row['targets']] for row in batch])
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(values, truth, pos_weight=positive_weight, reduction='none') * mask).sum() / mask.sum().clamp_min(1)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(heads.parameters(), 2.)
            optimizer.step(); losses.append(float(loss.detach()))
        if not losses:
            break
        development = predict(partitions['development'])
        measured = metrics(partitions['development'], development, [.5] * 3)
        quality = float(np.mean([item['ap'] for item in measured.values()]))
        history.append({'epoch': epoch + 1, 'loss': float(np.mean(losses)), 'development_macro_ap': quality})
        print('EPOCH', json.dumps(history[-1]), flush=True)
        write(output / 'history.json', history)
        if quality > best_value + 1e-4:
            best_value, best, stale = quality, copy.deepcopy(heads.state_dict()), 0
        else:
            stale += 1
        if stale >= 12:
            break
    if best is None:
        raise TimeoutError('No validated epoch completed')
    heads.load_state_dict(best)
    development = predict(partitions['development'])
    thresholds = []
    for index in range(3):
        known = [(row['targets'][index], score[index]) for row, score in zip(partitions['development'], development) if row['targets'][index] is not None]
        truth, values = np.asarray(known).T
        candidates = np.unique(np.r_[.5, values])
        def rank(threshold):
            precision, recall, f1, _ = precision_recall_fscore_support(truth, values >= threshold, average='binary', zero_division=0)
            return f1, precision, threshold
        thresholds.append(float(max(candidates, key=rank)))
    temporary = output / 'three_actions.partial.pt'
    torch.save({key: value.cpu() for key, value in heads.state_dict().items()}, temporary)
    temporary.replace(output / 'three_actions.pt')
    metadata = {'schema_version': 1, 'labels': list(LABELS), 'output': 'independent_sigmoid',
                'thresholds': dict(zip(LABELS, thresholds)), 'parent_checkpoint_sha256': PARENT_SHA,
                'checkpoint_sha256': digest(output / 'three_actions.pt'), 'source_sha256': SOURCE_SHA,
                'spec': SPEC, 'approved_for_demo': False, 'fingerprint': fingerprint}
    write(output / 'three_actions.json', metadata)
    report = {'status': 'completed', 'fine_tuning': 'three warm-started interaction heads; frozen AVA/detector',
              'fingerprint': fingerprint, 'approved_for_demo': False,
              'development': metrics(partitions['development'], development, thresholds),
              'test': metrics(partitions['test'], predict(partitions['test']), thresholds),
              'limitations': ['Window-level weak spatial supervision; no actor identity ground truth.',
                              'Independent thresholds selected on development only; camera acceptance not measured.']}
    write(output / 'report.json', report)
    import zipfile
    with zipfile.ZipFile(output / 'three_actions_pilot.zip', 'w', zipfile.ZIP_DEFLATED) as package:
        for name in ('three_actions.pt', 'three_actions.json', 'report.json', 'protocol.json'):
            package.write(output / name, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'root', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--weights', type=Path)
    parser.add_argument('--stage', choices=('audit', 'train'), default='audit')
    parser.add_argument('--minutes', type=int, default=30)
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if not 1 <= args.minutes <= 120 or not 1 <= args.epochs <= 300:
        parser.error('Bounded GPU budget required')
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    rows, result = audit(manifest, args.root)
    write(args.output / 'label_audit.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if args.stage == 'audit':
        return
    if result['status'] != 'ready':
        raise ValueError('Training blocked: missing reviewed per-action labels. No GPU model loaded.')
    if args.weights is None:
        parser.error('--weights required for training')
    protocol = {'manifest_sha256': digest(args.manifest), 'trainer_sha256': digest(Path(__file__)),
                'engine_sha256': digest(Path(__file__).with_name('action_engine.py')),
                'recovery_sha256': digest(Path(__file__).with_name('person_recovery.py')),
                'parent_checkpoint_sha256': PARENT_SHA, 'seed': 42}
    marker = args.output / 'protocol.json'
    if marker.exists() and json.loads(marker.read_text()) != protocol:
        raise ValueError('Run changed; use a new output folder')
    write(marker, protocol)
    deadline = time.monotonic() + args.minutes * 60
    cache = args.output / 'feature_cache'; cache.mkdir(exist_ok=True)
    rows = feature_rows(rows, cache, args.weights, args.device, deadline)
    train(rows, args.weights, args.output, deadline, args.epochs, protocol, args.device)


if __name__ == '__main__':
    main()
