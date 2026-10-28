import numpy as np
import pytest
import torch

from fight.ava import action_scores, ava_inputs


def test_ava_preserves_aspect_ratio_person_proposals_and_rgb_normalization():
    frames = np.zeros((16, 48, 64, 3), dtype=np.uint8)
    frames[..., 2] = 255
    inputs, boxes = ava_inputs(frames, [[0, 0, 63, 47], [10, 10, 30, 30]])
    assert inputs.shape == (1, 3, 4, 256, 341)
    assert boxes.shape == (2, 5) and torch.all(boxes[:, 0] == 0)
    assert inputs[0, 0, 0, 0, 0] == pytest.approx((1 - 0.45) / 0.225)
    assert inputs[0, 2, 0, 0, 0] == pytest.approx(-0.45 / 0.225)
    assert boxes[1, 1] == pytest.approx(10 * 256 / 48)


def test_ava_class_ids_are_one_indexed_and_not_softmax_or_resigmoided():
    predictions = torch.zeros(2, 80)
    predictions[0, 63], predictions[0, 69], predictions[1, 67] = 0.8, 0.7, 0.9
    actions = action_scores(predictions)
    assert actions[0]["fight_hit_person"] == pytest.approx(0.8)
    assert actions[0]["hug_person"] == pytest.approx(0.7)
    assert actions[1]["hand_shake"] == pytest.approx(0.9)
    predictions[0, 70] = 0.95
    assert action_scores(predictions)[0]["kick_person"] == pytest.approx(0.95)


def test_slowfast_uses_official_32_frame_temporal_subsampling():
    frames = np.zeros((12, 48, 64, 3), dtype=np.uint8)
    inputs, boxes = ava_inputs(frames, [[0, 0, 63, 47]], num_frames=32)
    assert inputs.shape == (1, 3, 32, 256, 341)
    assert boxes.shape == (1, 5)
    with pytest.raises(ValueError, match="4/32"):
        ava_inputs(frames, [[0, 0, 63, 47]], num_frames=16)


def test_ava_rejects_invalid_outputs_and_empty_regions():
    with pytest.raises(ValueError, match="80"):
        action_scores(torch.zeros(1, 81))
    with pytest.raises(ValueError, match="sigmoid"):
        action_scores(torch.full((1, 80), float("nan")))
    with pytest.raises(ValueError, match="Nonempty"):
        ava_inputs(np.zeros((4, 48, 64, 3), dtype=np.uint8), [[10, 10, 5, 5]])
