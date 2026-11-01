import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

from action_engine import ROOT, SOURCE_SHA, SPEC, digest
from train_interaction import annotation_at, event_metrics, metrics, validate


def run():
    manifest = ROOT.parent / 'benchmarks/mmaction_interaction_public_v1.json'
    corpus = json.loads(manifest.read_text(encoding='utf-8'))
    validate(corpus)
    rows, values = [], []
    with zipfile.ZipFile(ROOT / 'downloads/public_feature_cache.zip') as cache:
        for source in corpus['sources']:
            anchor = 32 / 30
            while anchor + 32 / 30 < source['duration']:
                key = hashlib.sha256((source['video'] + f'/{anchor:.6f}').encode()).hexdigest()[:24]
                with cache.open(key + '.npz') as stream, np.load(stream, allow_pickle=False) as entry:
                    score, people = float(entry['base']), int(entry['people'])
                label, action = annotation_at(source, anchor)
                rows.append(dict(video=source['video'], dataset=source['dataset'], group=source['group'], split=source['split'], label=label, action=action, people=people, timestamp=anchor + 32 / 30))
                values.append(score)
                anchor += SPEC['decision_stride_seconds']

    def subset(split):
        selected = [(row, score) for row, score in zip(rows, values) if row['split'] == split]
        return [row for row, _ in selected], [score for _, score in selected]

    calibration, calibration_scores = subset('calibration')
    thresholds = np.unique(np.clip(np.concatenate(([0.4], np.quantile(calibration_scores, np.linspace(0, 1, 48)))), 1e-6, 1 - 1e-6))
    choices = []
    for threshold in thresholds:
        for confirm in (0.4, 0.8, 1.2):
            events = event_metrics(calibration, calibration_scores, corpus['sources'], float(threshold), confirm)
            windows = metrics(calibration, calibration_scores, float(threshold))
            rank = (min(events['precision'], events['recall']), windows['f1'], -events['false_alerts_per_normal_minute'], float(threshold), confirm)
            choices.append((rank, float(threshold), confirm))
    _, threshold, confirm = max(choices)
    report = {'scope': 'Repeated-public pilot calibration only; no private camera data or training; test not used for selection', 'threshold': threshold, 'confirm_seconds': confirm, 'cache_sha256': digest(ROOT / 'downloads/public_feature_cache.zip'), 'splits': {}}
    for split in ('calibration', 'test'):
        selected, scores = subset(split)
        report['splits'][split] = {}
        for name, limit, seconds in (('default', 0.4, 0.4), ('calibrated', threshold, confirm)):
            report['splits'][split][name] = {'windows': metrics(selected, scores, limit), 'events': event_metrics(selected, scores, corpus['sources'], limit, seconds)}
    config = dict(source_sha256=SOURCE_SHA, engine_sha256=hashlib.sha256((ROOT / 'action_engine.py').read_text(encoding='utf-8').encode('utf-8')).hexdigest(), manifest_sha256=digest(manifest), spec=SPEC, threshold=threshold, confirm_seconds=confirm, approved_for_demo=False, scope=report['scope'])
    (ROOT / 'results/original_calibration_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (ROOT / 'config').mkdir(exist_ok=True)
    (ROOT / 'config/original_calibration.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    run()
