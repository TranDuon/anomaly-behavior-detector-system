"""Bounded public-data comparison: frozen and last-stage ST-GCN adaptation."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

os.environ['MPLBACKEND'] = 'Agg'

import cv2
import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

from pose_specialists import LABELS, SPEC, SOURCE_SHA, SOURCE_URL, PoseSpecialists, candidate_bags, track_window
from train_three_actions import audit, digest, write


def extract(source, target, model, deadline):
    capture = cv2.VideoCapture(str(source))
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not capture.isOpened() or not 1 <= fps <= 120:
        raise ValueError('Invalid source FPS')
    detections, frames, times = [], [], []
    frame_index, next_time = 0, 0.
    image_shape = None

    def flush():
        if not frames:
            return
        predictions = model.predict(frames, imgsz=640, conf=.3, device='cuda:0', verbose=False)
        for result in predictions:
            records = []
            if result.keypoints is not None:
                for box, points in zip(result.boxes.xyxy.cpu().numpy(), result.keypoints.data.cpu().numpy()):
                    if points.shape == (17, 3) and np.isfinite(points).all() and (points[:, 2] >= .3).sum() >= 6:
                        records.append({'box': box.tolist(), 'points': points.tolist()})
            detections.append(records[:6])
        frames.clear()

    try:
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError('Pose extraction budget reached')
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = frame_index / fps
            frame_index += 1
            if timestamp + 1e-6 < next_time:
                continue
            if frame.shape[1] > 960:
                frame = cv2.resize(frame, (960, round(frame.shape[0] * 960 / frame.shape[1])))
            image_shape = frame.shape[:2]
            frames.append(frame)
            times.append(timestamp)
            while next_time <= timestamp + 1e-6:
                next_time += 1 / SPEC['fps']
            if len(frames) == 8:
                flush()
        flush()
    finally:
        capture.release()
    write(target, {'detections': detections, 'timestamps': times, 'image_shape': image_shape})


def prepare(rows, cache, pose_weights, deadline):
    from ultralytics import YOLO
    estimator = YOLO(str(pose_weights))
    loaded, sources = [], {}
    for row in rows:
        if row['sha256'] not in sources:
            path = cache / (row['sha256'] + '_' + digest(pose_weights)[:12] + '.json')
            if not path.exists():
                extract(row['path'], path, estimator, deadline)
            sources[row['sha256']] = json.loads(path.read_text())
            print('POSE SOURCES', len(sources), flush=True)
        source = sources[row['sha256']]
        timestamps = np.asarray(source['timestamps'])
        desired = row['anchor'] + (np.arange(SPEC['frames']) - 15) / SPEC['fps']
        indices = np.abs(timestamps[:, None] - desired[None, :]).argmin(axis=0)
        detections = [source['detections'][index] for index in indices]
        tracks, boxes, coverage = track_window(detections, source['image_shape'])
        bags, pairs, _ = candidate_bags(tracks)
        loaded.append({**row, 'bags': bags, 'pairs': pairs, 'coverage': coverage,
                       'targets': [row['targets'][1], row['targets'][2]], 'people': len(boxes)})
    return loaded


def measure(rows, values, thresholds):
    result = {}
    for index, label in enumerate(LABELS):
        known = [(row, value[index]) for row, value in zip(rows, values) if row['targets'][index] is not None]
        truth = [row['targets'][index] for row, _ in known]
        scores = np.asarray([value for _, value in known])
        if set(truth) != {0, 1}:
            raise ValueError('Missing positive or negative partition: ' + label)
        precision, recall, f1, _ = precision_recall_fscore_support(truth, scores >= thresholds[index], average='binary', zero_division=0)
        result[label] = {'windows': len(known), 'precision': float(precision), 'recall': float(recall),
                         'f1': float(f1), 'ap': float(average_precision_score(truth, scores)),
                         'no_pose': sum(not len(row['bags']) for row, _ in known)}
    return result


def experiment(rows, weights, output, deadline, unfreeze, epochs):
    torch.manual_seed(42)
    model = PoseSpecialists(weights).cuda()
    model.configure_training(unfreeze)
    backbone = [parameter for parameter in model.backbone.parameters() if parameter.requires_grad]
    heads = [parameter for name, parameter in model.named_parameters() if not name.startswith('backbone.')]
    groups = [{'params': heads, 'lr': 3e-4}]
    if backbone:
        groups.append({'params': backbone, 'lr': 1e-5})
    optimizer = torch.optim.AdamW(groups, weight_decay=.01)
    partitions = {split: [row for row in rows if row['split'] == split] for split in ('train', 'development', 'test')}
    training = [row for row in partitions['train'] if len(row['bags'])]
    for index, label in enumerate(LABELS):
        supported = [row['targets'][index] for row in training if (index != 0 or row['pairs'].any())]
        if not {0, 1}.issubset(set(supported)):
            raise ValueError('Pose tracking removed a training class: ' + label)

    def logits(row):
        return model(torch.as_tensor(row['bags'], device='cuda'),
                     torch.as_tensor(row['pairs'], device='cuda')).max(dim=0).values

    @torch.inference_mode()
    def predict(items):
        model.eval()
        return np.asarray([logits(row).sigmoid().cpu().tolist() if len(row['bags']) else [0., 0.] for row in items])

    history, best, best_ap, stale = [], None, -1., 0
    for epoch in range(epochs):
        if time.monotonic() >= deadline - 30:
            break
        model.configure_training(unfreeze)
        losses = []
        for number in torch.randperm(len(training)).tolist():
            if time.monotonic() >= deadline - 30:
                break
            row = training[number]
            values = logits(row)
            mask = values.new_tensor([value is not None and (index != 0 or row['pairs'].any())
                                      for index, value in enumerate(row['targets'])])
            if not mask.any():
                continue
            targets = values.new_tensor([value or 0 for value in row['targets']])
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(values, targets, reduction='none') * mask).sum() / mask.sum()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        if not losses:
            break
        metrics = measure(partitions['development'], predict(partitions['development']), [.5, .5])
        quality = np.mean([value['ap'] for value in metrics.values()])
        history.append({'epoch': epoch + 1, 'loss': float(np.mean(losses)), 'development': metrics})
        write(output / 'history.json', history)
        print(output.name, 'EPOCH', epoch + 1, 'dev_macro_AP', round(float(quality), 4), flush=True)
        if quality > best_ap + .001:
            best, best_ap, stale = copy.deepcopy(model.state_dict()), quality, 0
        else:
            stale += 1
        if stale >= 10:
            break
    if best is None:
        raise TimeoutError('No validated epoch completed')
    model.load_state_dict(best)
    development = predict(partitions['development'])
    thresholds = []
    for index in range(2):
        def rank(threshold):
            metrics = measure(partitions['development'], development, [threshold if slot == index else .5 for slot in range(2)])[LABELS[index]]
            return metrics['precision'] >= .8, metrics['f1'], metrics['precision'], threshold
        candidates = np.unique(np.r_[.5, development[:, index]])
        candidates = candidates[(candidates > 0) & (candidates < 1)]
        thresholds.append(float(max(candidates, key=rank)))
    torch.save({key: value.cpu() for key, value in best.items()}, output / 'pose_specialists.pt')
    report = {'status': 'completed', 'labels': LABELS, 'spec': SPEC, 'unfreeze_last_two_stages': unfreeze,
              'source_sha256': digest(weights), 'checkpoint_sha256': digest(output / 'pose_specialists.pt'),
              'thresholds': dict(zip(LABELS, thresholds)), 'approved_for_demo': False,
              'development': measure(partitions['development'], development, thresholds),
              'test': measure(partitions['test'], predict(partitions['test']), thresholds),
              'limitations': ['Small repeated AI-assisted review set, not independent acceptance.',
                              'YOLO pose differs from original NTU HRNet pose.',
                              'Weak person/pair attribution; automatic promotion is prohibited.']}
    write(output / 'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'root', 'output', 'pose-weights'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--minutes', type=int, default=30)
    parser.add_argument('--epochs', type=int, default=80)
    args = parser.parse_args()
    if not 1 <= args.minutes <= 45 or not 1 <= args.epochs <= 120:
        parser.error('Invalid bounded budget')
    started = time.monotonic()
    deadline = started + args.minutes * 60
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    rows, audit_result = audit(manifest, args.root)
    write(args.output / 'audit.json', audit_result)
    if audit_result['status'] != 'ready':
        raise ValueError('Reviewed label audit not ready')
    protocol = {'spec': SPEC, 'manifest_sha256': digest(args.manifest), 'pose_sha256': digest(args.pose_weights),
                'trainer_sha256': digest(__file__), 'model_sha256': digest(Path(__file__).with_name('pose_specialists.py'))}
    write(args.output / 'protocol.json', protocol)
    cache = args.output.parent / ('pose_cache_' + hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:12])
    cache.mkdir(exist_ok=True)
    weights = cache / 'stgcn_ntu60.pth'
    if not weights.exists():
        temporary = weights.with_suffix('.partial')
        urllib.request.urlretrieve(SOURCE_URL, temporary)
        if digest(temporary) != SOURCE_SHA:
            raise ValueError('Official pretrained checkpoint checksum mismatch')
        temporary.replace(weights)
    if digest(weights) != SOURCE_SHA:
        raise ValueError('Cached pretrained checkpoint checksum mismatch')
    rows = prepare(rows, cache, args.pose_weights, deadline - 120)
    write(args.output / 'tracking_quality.json', [{key: row[key] for key in ('video', 'anchor', 'split', 'people', 'coverage')} for row in rows])
    reports = {}
    for name, unfreeze in [('frozen', False), ('last_two_stages', True)]:
        target = args.output / name
        target.mkdir()
        remaining = deadline - time.monotonic()
        stage_deadline = time.monotonic() + remaining / (2 if name == 'frozen' else 1)
        reports[name] = experiment(rows, weights, target, stage_deadline, unfreeze, args.epochs)
    selected = max(reports, key=lambda name: sum(value['ap'] for value in reports[name]['development'].values()))
    write(args.output / 'comparison.json', {'status': 'completed', 'selected_on': 'development_macro_ap',
          'selected': selected, 'reports': reports, 'elapsed_seconds': time.monotonic() - started,
          'automatic_install': False, 'private_videos_uploaded': False})
    print('COMPLETED', args.output / 'comparison.json', flush=True)


if __name__ == '__main__':
    main()
