from pathlib import Path
import json

import pytest

from fight.mmaction_runtime import MMActionRuntime
from fight.runtime import EventGate


def runtime():
    model = MMActionRuntime(three_actions=True, allow_pilot=True)
    model.action_gates = {name: EventGate(.8, .56, confirm_seconds=.4, release_seconds=.25)
                          for name in ('strike', 'grapple', 'fall')}
    return model


def prediction(strike=.01, grapple=.01, fall=.01, people=2):
    return {'score': max(strike, grapple, fall), 'people': people, 'regions': [],
            'action_scores': dict(zip(('strike', 'grapple', 'fall'), (strike, grapple, fall))),
            'action_regions': {'strike': [[1, 1, 2, 2]], 'grapple': [[3, 3, 4, 4]], 'fall': [[5, 5, 6, 6]]}}


def test_grapple_and_fall_can_overlap_without_strike():
    model = runtime()
    assert model.update_decision(prediction(grapple=.9, fall=.9), 0) == ('verifying', None)
    output = prediction(grapple=.9, fall=.9)
    assert model.update_decision(output, .54) == ('fighting', 'started')
    assert output['active_behaviors'] == ['grapple', 'fall']
    assert output['regions'] == [[3, 3, 4, 4], [5, 5, 6, 6]]


def test_single_spike_and_lost_person_do_not_stick():
    model = runtime()
    model.update_decision(prediction(strike=.9), 0)
    assert model.update_decision(prediction(), .54) == ('normal', None)
    model.update_decision(prediction(fall=.9), 1.08)
    model.update_decision(prediction(fall=.9), 1.62)
    assert model.update_decision(prediction(people=0), 2.16) == ('insufficient', 'ended')
    assert all(not gate.active for gate in model.action_gates.values())
    model.reset()
    assert model.result['behavior_scope'] == 'three_actions'
    assert model.result['active_behaviors'] == []


def test_separate_loader_is_tensor_only_and_rejects_tampering(tmp_path, monkeypatch):
    import torch
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'mmaction2_demo'))
    from action_engine import InteractionHead, SOURCE_SHA, SPEC, digest
    from three_actions_engine import LABELS, PARENT_SHA, load_three_actions

    checkpoint = tmp_path / 'three_actions.pt'
    torch.save(torch.nn.ModuleList([InteractionHead() for _ in LABELS]).state_dict(), checkpoint)
    metadata = {'labels': list(LABELS), 'output': 'independent_sigmoid',
                'source_sha256': SOURCE_SHA, 'spec': SPEC, 'parent_checkpoint_sha256': PARENT_SHA,
                'checkpoint_sha256': digest(checkpoint), 'thresholds': dict.fromkeys(LABELS, .8)}
    target = tmp_path / 'three_actions.json'
    target.write_text(json.dumps(metadata))
    _, heads = load_three_actions(tmp_path)
    assert len(heads) == 3
    metadata['thresholds']['fall'] = float('nan')
    target.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='thresholds'):
        load_three_actions(tmp_path)
    metadata['thresholds']['fall'] = .8
    target.write_text(json.dumps(metadata))
    checkpoint.write_bytes(b'wrong weights')
    with pytest.raises(ValueError, match='Incompatible'):
        load_three_actions(tmp_path)
