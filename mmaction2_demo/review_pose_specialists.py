"""Verify a completed pose trial and select each label on development only."""

import argparse
import io
import json
import math
from pathlib import Path
import zipfile

import torch

from pose_specialists import LABELS, SOURCE_SHA, SPEC, PoseSpecialists
from train_three_actions import digest, write


def select_per_label(reports):
    if not reports:
        raise ValueError('Missing experiment reports')
    for report in reports.values():
        if report.get('status') != 'completed':
            raise ValueError('Incomplete experiment')
        for label in LABELS:
            for metric in ('ap', 'f1', 'precision', 'recall'):
                value = report['development'][label][metric]
                if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError('Invalid development metric')
    return {label: max(reports, key=lambda name: (reports[name]['development'][label]['ap'],
                        reports[name]['development'][label]['f1'])) for label in LABELS}


def review(archive, output):
    with zipfile.ZipFile(archive) as package:
        members = [item for item in package.infolist() if not item.is_dir()]
        allowed = {'audit.json', 'comparison.json', 'protocol.json', 'tracking_quality.json'}
        allowed.update(f'{stage}/{name}' for stage in ('frozen', 'last_two_stages')
                       for name in ('history.json', 'report.json', 'pose_specialists.pt'))
        if {item.filename for item in members} != allowed or sum(item.file_size for item in members) > 100 * 1024**2:
            raise ValueError('Unexpected trial bundle members or size')
        comparison = json.loads(package.read('comparison.json'))
        protocol = json.loads(package.read('protocol.json'))
        root = Path(__file__).parent
        if (protocol['spec'] != SPEC or protocol['trainer_sha256'] != digest(root/'train_pose_specialists.py')
                or protocol['model_sha256'] != digest(root/'pose_specialists.py')):
            raise ValueError('Trial source/preprocessing mismatch; review original source bundle')
        reports, history = {}, {}
        for stage in ('frozen', 'last_two_stages'):
            report = json.loads(package.read(stage + '/report.json'))
            content = package.read(stage + '/pose_specialists.pt')
            import hashlib
            if (report != comparison['reports'][stage] or report['source_sha256'] != SOURCE_SHA
                    or report['spec'] != SPEC or list(report['labels']) != list(LABELS)
                    or hashlib.sha256(content).hexdigest() != report['checkpoint_sha256']):
                raise ValueError('Trial checkpoint/report mismatch')
            state = torch.load(io.BytesIO(content), map_location='cpu', weights_only=True)
            model = PoseSpecialists()
            model.load_state_dict(state, strict=True)
            if any(not torch.isfinite(value).all() for value in state.values()):
                raise ValueError('Non-finite checkpoint')
            reports[stage] = report
            history[stage] = json.loads(package.read(stage + '/history.json'))
        selected = select_per_label(reports)
        summary = {'status': 'verified_not_installed', 'bundle_sha256': digest(archive),
                   'selected_per_label': selected, 'selection_basis': 'development AP, then F1; test never used',
                   'reports': reports,
                   'best_epochs': {stage: max(items, key=lambda item: sum(value['ap'] for value in item['development'].values()))['epoch']
                                   for stage, items in history.items()},
                   'automatic_install': False,
                   'limitations': ['Repeated 37-video AI-assisted review, not independent acceptance.',
                                   'Pair attribution and fall recall remain insufficient.',
                                   'A shared macro-AP winner must not replace both independent branches.']}
        output.mkdir(parents=True, exist_ok=False)
        for item in members:
            target = output / item.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(package.read(item))
        write(output / 'review.json', summary)
        return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('output', type=Path)
    arguments = parser.parse_args()
    print(json.dumps(review(arguments.archive, arguments.output), indent=2))
