import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mmaction2_demo"))

from action_engine import InteractionHead, digest  # noqa: E402
from train_general_fall_v1 import (  # noqa: E402
    PARENT_SHA, PREVIOUS_TRAINER_SHA, anchors, compatible_cache,
    duration_decision, sampled_clip, sources,
)


def test_public_sources_split_and_labels(tmp_path):
    manifest = json.loads((ROOT / "benchmarks/gmdcsa24_v2_1_manifest.json").read_text(encoding="utf-8"))
    scfd = tmp_path / "scfd"
    for category in ("fight", "noFight"):
        folder = scfd / category
        folder.mkdir(parents=True)
        for number in range(150):
            (folder / f"{number:03}.mp4").touch()
    rows = sources(manifest, tmp_path / "fall", scfd)
    assert len(rows) == 459
    assert {row["split"] for row in rows if row["dataset"] == "gmdcsa24" and row["group"] == "gmdcsa24/subject_4"} == {"test"}
    assert not any(row["video"] == "gmdcsa24/Subject 4/Fall/15.mp4" for row in rows)
    for split in ("train", "development", "test"):
        assert {row["label"] for row in rows if row["split"] == split} == {0, 1}
    assert all(row["label"] == 1 for row in rows if row["video"].startswith("scfd/fight/"))


def test_fall_anchors_only_inside_publisher_interval_and_short_clips_pad():
    fall = {"dataset": "gmdcsa24", "label": 1, "start": 2.0, "end": 3.0, "video": "fall"}
    assert all(2.0 < t < 3.0 for t in anchors(fall, 5.0))
    frames = [np.full((2, 2, 3), index, np.uint8) for index in range(4)]
    sampled = sampled_clip(frames, fps=2.0, anchor=1.0)
    assert sampled.shape == (9, 2, 2, 3)
    assert int(sampled[0, 0, 0, 0]) == 0
    assert int(sampled[-1, 0, 0, 0]) == 3


def test_installed_parent_head_matches_v1_metadata():
    weights = ROOT / "mmaction2_demo/weights"
    metadata = json.loads((weights / "interaction.json").read_text(encoding="utf-8"))
    assert digest(weights / "interaction.pt") == metadata["checkpoint_sha256"] == PARENT_SHA
    head = InteractionHead()
    head.load_state_dict(torch.load(weights / "interaction.pt", map_location="cpu", weights_only=True), strict=True)


def test_bad_publisher_duration_uses_decoded_adl_but_excludes_unsafe_fall():
    manifest = json.loads((ROOT / "benchmarks/gmdcsa24_v2_1_manifest.json").read_text(encoding="utf-8"))
    item = next(row for row in manifest["sources"] if row["video"] == "Subject 2/ADL/04.mp4")
    adl = {"dataset": "gmdcsa24", "label": 0,
           "publisher_duration": item["publisher_duration_seconds"]}
    assert duration_decision(adl, 7.0) == "use_decoded"
    assert duration_decision({**adl, "label": 1}, 7.0) == "skip"


def test_previous_feature_cache_is_reusable_when_only_trainer_revision_changed():
    current = {"trainer_sha256": "updated", "engine_sha256": "same", "manifest_sha256": "same"}
    previous = {**current, "trainer_sha256": PREVIOUS_TRAINER_SHA}
    assert compatible_cache(previous, current)
    assert not compatible_cache({**previous, "engine_sha256": "different"}, current)
