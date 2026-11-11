import sys
from types import ModuleType

import numpy as np
import pytest
import torch

from mmaction2_demo.pose_specialists import SPEC, PoseSpecialists, candidate_bags, track_window


def detection(left=10):
    points = np.ones((17, 3), np.float32)
    points[:, :2] = [left + 10, 30]
    return {'box': [left, 10, left + 20, 60], 'points': points.tolist()}


def test_dense_tracks_keep_missing_frames_masked_and_pair_identity_distinct():
    frames = [[detection(), detection(70)] for _ in range(SPEC['frames'])]
    frames[5] = []
    tracks, boxes, coverage = track_window(frames, (100, 120))
    assert tracks.shape == (2, 32, 17, 3)
    assert len(boxes) == 2 and min(coverage) > .9
    assert (tracks[:, 5] == 0).all()
    assert tracks[0, 16, 0, 0] < tracks[1, 16, 0, 0]
    bags, pairs, candidates = candidate_bags(tracks)
    assert bags.shape == (3, 2, 32, 17, 3)
    assert pairs.tolist() == [False, False, True]
    assert candidates == [(0,), (1,), (0, 1)]


def test_missing_or_discontinuous_tracks_are_not_invented():
    frames = [[] for _ in range(32)]
    frames[16] = [detection()]
    tracks, _, _ = track_window(frames, (100, 120))
    assert len(tracks) == 0
    bags, pairs, _ = candidate_bags(tracks)
    assert bags.shape == (0, 2, 32, 17, 3) and not pairs.any()
    with pytest.raises(ValueError, match='Dense'):
        track_window(frames[:9], (100, 120))


def test_temporal_pair_head_is_order_invariant_and_single_person_cannot_grapple(monkeypatch):
    class Backbone(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.gcn = torch.nn.ModuleList([torch.nn.Linear(1, 1)] * 3)

        def forward(self, inputs):
            values = inputs.mean(dim=(2, 3, 4))
            return values[:, :, None, None, None].expand(-1, -1, 256, 4, 17)

    module = ModuleType('mmaction.models')
    module.STGCN = Backbone
    monkeypatch.setitem(sys.modules, 'mmaction.models', module)
    model = PoseSpecialists().eval()
    inputs = torch.rand(3, 2, 32, 17, 3)
    pairs = torch.tensor([True, True, False])
    output = model(inputs, pairs)
    swapped = model(inputs.flip(1), pairs)
    assert torch.allclose(output, swapped, atol=1e-6)
    assert output[2, 0] == -20
    assert output.shape == (3, 2) and torch.isfinite(output).all()
