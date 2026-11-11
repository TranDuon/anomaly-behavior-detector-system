"""Score the reviewed public corpus before any further training or deployment claim."""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.request

import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

from action_engine import ActionEngine, ROOT, SOURCE_SHA, SPEC, digest
from three_actions_engine import load_three_actions
from train_action_transition_v2 import frames_for_video, sampled_clip


LABELS = ('strike', 'grapple', 'fall')


def measurements(rows, label, threshold, field='scores'):
    selected = [row for row in rows if row['targets'][label] is not None]
    truth = [row['targets'][label] for row in selected]
    values = [row[field][label] for row in selected]
    precision, recall, f1, _ = precision_recall_fscore_support(truth, np.asarray(values) >= threshold,
                                                            average='binary', zero_division=0)
    return {'windows': len(selected), 'precision': float(precision), 'recall': float(recall),
            'f1': float(f1), 'ap': float(average_precision_score(truth, values)),
            'no_person': sum(row['people'] == 0 for row in selected)}


def choose_threshold(rows, label):
    values = [row['scores'][label] for row in rows if row['targets'][label] is not None]
    candidates = np.unique(np.clip([.4, *values], 1e-6, 1 - 1e-6))
    def rank(threshold):
        result = measurements(rows, label, threshold)
        return result['precision'] >= .8, result['f1'], result['precision'], threshold
    return float(max(candidates, key=rank))


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / 'results/three_actions_review/reviewed_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    rows_path = output / 'public_scores.json'
    identity = {'manifest_sha256': digest(manifest_path), 'engine_sha256': digest(ROOT / 'action_engine.py'),
                'grapple_checkpoint_sha256': digest(ROOT / 'weights/three_actions.pt')}
    marker = output / 'extraction.json'
    if marker.exists() and json.loads(marker.read_text()) != identity:
        raise ValueError('Evaluation input changed; use another output folder')
    marker.write_text(json.dumps(identity), encoding='utf-8')
    rows = json.loads(rows_path.read_text()) if rows_path.exists() else []
    completed = {(row['video'], row['anchor']) for row in rows}
    engine = ActionEngine(person_recovery=True)
    _, heads = load_three_actions(ROOT / 'weights', 'cuda:0')
    torch.set_num_threads(4)
    with tempfile.TemporaryDirectory(dir=output) as temporary:
        for source in manifest['sources']:
            if all((source['video'], window['anchor_seconds']) in completed for window in source['windows']):
                continue
            url = source['download_url']
            if not url.startswith('https://raw.githubusercontent.com/'):
                raise ValueError('Unexpected public video host')
            target = Path(temporary) / Path(source['video']).name
            with urllib.request.urlopen(url, timeout=90) as response:
                payload = response.read()
            if len(payload) != source['size'] or hashlib.sha256(payload).hexdigest() != source['sha256']:
                raise ValueError('Wrong public video checksum')
            target.write_bytes(payload)
            frames, fps = frames_for_video(target)
            for window in source['windows']:
                anchor = window['anchor_seconds']
                if (source['video'], anchor) in completed:
                    continue
                clip = sampled_clip(frames, fps, anchor)
                with torch.inference_mode():
                    bag, _, info = engine.extract_features(clip)
                    old = {label: float(head(torch.from_numpy(bag).cuda()).sigmoid().max()) if len(bag) else 0.
                           for label, head in zip(LABELS, heads)}
                scores = {'strike': info['base_score'], 'grapple': old['grapple'],
                          'fall': max((person['ava_fall_score'] for person in info['actions']), default=0.)}
                rows.append({'video': source['video'], 'group': source['group'], 'split': source['split'],
                             'anchor': anchor, 'targets': window['targets'], 'people': info['people'],
                             'scores': scores, 'previous_three_heads': old})
                rows_path.write_text(json.dumps(rows, indent=2), encoding='utf-8')
            target.unlink()
            print('PUBLIC SOURCES', len({row['video'] for row in rows}), '/', len(manifest['sources']), flush=True)
    development = [row for row in rows if row['split'] == 'development']
    test = [row for row in rows if row['split'] == 'test']
    thresholds = {label: choose_threshold(development, label) for label in LABELS}
    previous = json.loads((ROOT / 'weights/three_actions.json').read_text())['thresholds']
    report = {'status': 'completed', 'scope': 'Repeated public AI-reviewed windows; not independent human gold or camera acceptance',
              'threshold_selection': 'Development only: prefer precision >= .8, then maximum F1',
              'thresholds': thresholds,
              'development': {label: measurements(development, label, thresholds[label]) for label in LABELS},
              'test': {label: measurements(test, label, thresholds[label]) for label in LABELS},
              'previous_three_heads_test': {label: measurements(test, label, previous[label], 'previous_three_heads') for label in LABELS}}
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    running = json.loads((ROOT / 'weights/interaction_fixed_running.json').read_text())
    calibration = {**identity, 'source_sha256': SOURCE_SHA, 'spec': SPEC,
                   'running_checkpoint_sha256': digest(ROOT / 'weights/interaction_fixed_running.pt'),
                   'thresholds': {**thresholds, 'fast_running': running['running_threshold']},
                   'labels': [*LABELS, 'fast_running'], 'approved_for_demo': False,
                   'scope': report['scope'], 'test': report['test']}
    (ROOT / 'weights/separate_actions_calibration.json').write_text(json.dumps(calibration, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
