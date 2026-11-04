"""Small checks for the distance-aware warm start and temporal sampling."""

import numpy as np
import torch

from action_engine import AUX_NAMES, InteractionDistanceHead, InteractionHead
from train_distance_fall_v3 import anchors, guarded_threshold


def test_distance_head_starts_with_exact_parent_predictions():
    torch.manual_seed(17)
    parent = InteractionHead().eval()
    child = InteractionDistanceHead().warm_start(parent).eval()
    bag = torch.randn(5, 4096)
    auxiliary = torch.randn(5, len(AUX_NAMES))
    assert torch.allclose(parent(bag), child(bag, auxiliary), atol=1e-6)
    assert torch.count_nonzero(child.layers[0].weight[:, 4096:]) == 0


def test_fight_and_fall_anchors_are_dense_near_boundaries():
    fight = {"dataset": "reviewed", "video": "fight.mp4", "action": "fight",
             "start": 1., "end": 4., "label": 1}
    fall = {"dataset": "gmdcsa24", "video": "fall.mp4", "action": None,
            "start": 2., "end": 6., "label": 1}
    fight_anchors = anchors(fight, 5.)
    fall_anchors = anchors(fall, 7.)
    assert len(fight_anchors) >= 8
    assert fight_anchors[-1] > 3.5
    assert len(fall_anchors) == 10
    assert fall_anchors[0] < 2.
    assert fall_anchors[0] < 2.1


def test_guarded_threshold_keeps_detected_near_fight_and_fall_recall():
    rows = [
        {"dataset": "gmdcsa24", "action": None, "label": 1, "people": 1,
         "nearest_pair_distance": None},
        {"dataset": "reviewed", "action": "fight", "label": 1, "people": 2,
         "nearest_pair_distance": .2},
        {"dataset": "reviewed", "action": "after_fight", "label": 0, "people": 2,
         "nearest_pair_distance": .7},
    ]
    threshold, _ = guarded_threshold(rows, [.9, .8, .3], [.95, .9, .4], .8)
    assert threshold <= .8
    assert np.isfinite(threshold)
