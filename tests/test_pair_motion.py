import json
from pathlib import Path

import numpy as np
import pytest
import torch

from fight.pair_motion import (
    MotionConfig,
    PairMotionClassifier,
    motion_inputs,
    pair_descriptors,
)
from scripts.fight_pair_motion_train import balanced_order, source_cache_key


def test_motion_requires_exact_timebase_and_masks_static_background():
    clips = np.full((2, 16, 48, 64, 3), 128, np.uint8)
    motion = motion_inputs(clips)
    assert motion.shape == (2, 2, 15, 32, 32)
    assert motion.dtype == np.float32 and not motion.any()
    with pytest.raises(ValueError):
        motion_inputs(clips[:, :15])
    with pytest.raises(ValueError):
        MotionConfig(output_size=64)


def test_directed_pair_uses_opponent_torso_and_correct_sparse_time():
    poses = torch.zeros(1, 8, 2, 17, 3)
    poses[..., 2] = 1
    poses[:, :, 1, :, 0] = 2
    positions = torch.tensor([0, 2, 4, 6, 8, 10, 12, 15]) / 12
    poses[0, :, 0, 9, 0] = positions
    deltas = torch.diff(positions).reshape(1, 7, 1, 1, 1)
    described = pair_descriptors(poses, deltas)
    assert described.shape == (1, 2, 21, 8)
    torch.testing.assert_close(described[0, 0, 0], positions - 2)
    torch.testing.assert_close(described[0, 0, 4, 1:], torch.ones(7))
    torch.testing.assert_close(described[0, 0, 6, 1:], torch.ones(7))
    assert described[:, :, 19].min() == 1
    poses[:, :, 1, :, 2] = 0
    assert not pair_descriptors(poses, deltas)[:, :, 19].any()


def test_pair_model_is_slot_invariant_and_starts_at_rgb_baseline():
    torch.set_num_threads(2)
    torch.manual_seed(751)
    model = PairMotionClassifier().eval()
    features = torch.randn(2, 2048)
    poses = torch.randn(2, 8, 2, 17, 3)
    poses[..., 2] = 1
    motion = torch.randn(2, 2, 15, 32, 32)
    logits, actions = model.classify_features(features, poses, motion)
    torch.testing.assert_close(logits, model.rgb.head(features).flatten())
    assert actions.shape == (2, 6)
    assert not any(parameter.requires_grad for parameter in model.rgb.parameters())
    torch.nn.init.normal_(model.delta.weight)
    original, auxiliary = model.classify_features(features, poses, motion)
    swapped, swapped_auxiliary = model.classify_features(
        features, poses.flip(2), motion
    )
    torch.testing.assert_close(original, swapped)
    torch.testing.assert_close(auxiliary, swapped_auxiliary)
    missing, unused = model.classify_features(features, torch.zeros_like(poses), motion)
    assert torch.isfinite(missing).all()
    (original.sum() + auxiliary.sum()).backward()
    assert model.pair_encoder[0].weight.grad.abs().sum() > 0
    assert model.motion_encoder[0].weight.grad.abs().sum() > 0


def test_three_input_export_accepts_dynamic_region_count(tmp_path):
    from fight.model import prepare_export

    torch.set_num_threads(2)
    model = prepare_export(PairMotionClassifier().eval())
    inputs = (
        torch.zeros(1, 3, 16, 182, 182),
        torch.zeros(1, 8, 2, 17, 3),
        torch.zeros(1, 2, 15, 32, 32),
    )
    with torch.inference_mode():
        traced = torch.jit.trace(model, inputs, check_trace=False)
        traced.save(str(tmp_path / "model.ts"))
        restored = torch.jit.load(str(tmp_path / "model.ts"))
        doubled = tuple(value.repeat(2, *([1] * (value.ndim - 1))) for value in inputs)
        torch.testing.assert_close(restored(*doubled), model(*doubled))


def test_source_cache_keys_reuse_existing_public_qa_examples():
    manifest_path = Path("logs/fusion-v1/source_manifest.json")
    if not manifest_path.exists():
        pytest.skip("Optional existing public cache manifest not installed")
    manifest = json.loads(manifest_path.read_text())
    keys = {source_cache_key(row, manifest) for row in manifest["rows"]}
    assert {
        "e152ff7e4caa97cb58c80f31",
        "814883c6d42573b6646152bc",
        "4139b5df5ee4f86e2809bd28",
    } <= keys


def test_balanced_sampler_does_not_create_new_unique_examples():
    import random

    rows = [
        {"dataset": dataset, "label": label}
        for dataset in ("airtlab", "rlvs")
        for label in (0, 1)
    ]
    sampled = balanced_order(rows, random.Random(751))
    assert len(sampled) == 512
    assert all(sum(row is original for row in sampled) == 128 for original in rows)


def test_runtime_three_inputs_share_identical_upload_and_live_preprocessing():
    from fight.runtime import FightRuntime
    from fight.vision import ClipConfig
    from tests.test_fight_fusion import MockPose

    class Predictor(torch.nn.Module):
        def forward(self, clips, poses, motion):
            assert motion.shape[1:] == (2, 15, 32, 32)
            return (
                clips.mean(dim=(1, 2, 3, 4)) * 0
                + poses.mean(dim=(1, 2, 3, 4)) * 0
                + motion.mean(dim=(1, 2, 3, 4))
                + 5
            )

    runtimes = [
        FightRuntime(
            predictor=Predictor(),
            config=ClipConfig(frames=16, fps=12, size=32),
            pose_detector=MockPose(),
            input_kind="rgb_pose_motion",
        )
        for index in range(2)
    ]
    frames = [np.full((120, 160, 3), index, np.uint8) for index in range(36)]
    traces = [
        [
            runtime.push(frame, index / 12, sampled=bool(mode))
            for index, frame in enumerate(frames)
        ]
        for mode, runtime in enumerate(runtimes)
    ]
    assert [row["score"] for row in traces[0]] == [row["score"] for row in traces[1]]
    assert traces[0][-1]["state"] == "fighting"
