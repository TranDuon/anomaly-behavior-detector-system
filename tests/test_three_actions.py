import copy
import hashlib
from pathlib import Path
import time

import pytest

from mmaction2_demo.train_three_actions import LABELS, audit


def manifest():
    sources = []
    for split, count in [('train', 3), ('development', 2), ('test', 2)]:
        for number in range(count):
            for label in LABELS:
                group = f'{split}/{number}/{label}'
                sources.append({'video': group + '.mp4', 'group': group,
                                'sha256': hashlib.sha256(group.encode()).hexdigest(),
                                'split': split, 'reviewer': 'test-reviewer', 'public_source_url': 'https://example.org/dataset',
                                'windows': [{'anchor_seconds': 1.2, 'targets': {name: int(name == label) for name in LABELS}},
                                            {'anchor_seconds': 4.0, 'targets': dict.fromkeys(LABELS, 0)}]})
    return {'schema_version': 1, 'labels': list(LABELS), 'sources': sources}


def test_requires_separate_positive_and_negative_groups_for_all_labels(tmp_path):
    rows, report = audit(manifest(), tmp_path, verify_files=False)
    assert report['status'] == 'ready'
    assert len(rows) == 42
    assert audit({'schema_version': 1, 'labels': list(LABELS), 'sources': []}, tmp_path)[1]['status'] == 'needs_labels'


def test_unknown_is_not_implicitly_negative_and_actions_can_overlap(tmp_path):
    document = manifest()
    document['sources'][0]['windows'][0]['targets'] = {'strike': None, 'grapple': 1, 'fall': 1}
    rows, _ = audit(document, tmp_path, verify_files=False)
    assert rows[0]['targets'] == [None, 1, 1]


def test_refuses_binary_alias_and_source_leakage(tmp_path):
    document = manifest()
    document['labels'] = ['fight_wrestle_fall']
    with pytest.raises(ValueError):
        audit(document, tmp_path, verify_files=False)
    document = manifest()
    document['sources'][-1]['group'] = document['sources'][0]['group']
    with pytest.raises(ValueError, match='leakage'):
        audit(document, tmp_path, verify_files=False)


def test_refuses_changed_checksums_path_escape_and_test_reuse(tmp_path):
    original = manifest()
    for changed in [{'video': '../private.mp4'}, {'group': 'airtlab/violent/87'}, {'sha256': 'bad'}]:
        document = copy.deepcopy(original)
        document['sources'][0].update(changed)
        with pytest.raises(ValueError):
            audit(document, tmp_path, verify_files=False)


def test_refuses_missing_or_unreviewed_files(tmp_path):
    with pytest.raises(ValueError, match='Missing video'):
        audit(manifest(), tmp_path)
    document = manifest()
    document['sources'][0]['reviewer'] = ''
    with pytest.raises(ValueError, match='provenance'):
        audit(document, tmp_path, verify_files=False)


def test_refuses_duplicate_anchors_and_implicit_labels(tmp_path):
    for value in [True, 'fight', -1, 2]:
        document = manifest()
        document['sources'][0]['windows'][0]['targets']['strike'] = value
        with pytest.raises(ValueError):
            audit(document, tmp_path, verify_files=False)
    document = manifest()
    document['sources'][0]['windows'].append(document['sources'][0]['windows'][0])
    with pytest.raises(ValueError):
        audit(document, tmp_path, verify_files=False)


def test_three_heads_train_and_save_separate_outputs(tmp_path, monkeypatch):
    import numpy as np
    import torch
    from mmaction2_demo.train_three_actions import train

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'mmaction2_demo'))
    torch.set_num_threads(2)
    rows, _ = audit(manifest(), tmp_path, verify_files=False)
    for row in rows:
        row['bag'] = np.zeros((1, 4096), np.float32)
        row['bag'][0, :3] = row['targets']
    weights = Path(__file__).resolve().parents[1] / 'mmaction2_demo/weights'
    fingerprint = {'test': 'synthetic only'}
    (tmp_path/'protocol.json').write_text(__import__('json').dumps(fingerprint))
    train(rows, weights, tmp_path, time.monotonic() + 40, 2, fingerprint, 'cpu')
    state = torch.load(tmp_path/'three_actions.pt', weights_only=True)
    assert {key.split('.')[0] for key in state} == {'0', '1', '2'}
    assert (tmp_path/'three_actions_pilot.zip').is_file()
