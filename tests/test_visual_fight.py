import cv2
import numpy as np
import pytest
import torch

from anomaly.fight_data import VideoItem
from anomaly.visual_fight import (
    CLIP_FRAMES,
    CLIP_STRIDE,
    FEATURE_DIM,
    INPUT_SIZE,
    ClipFeatureExtractor,
    ClipFeatures,
    FightTemporalHead,
    StreamingFightScorer,
    clip_starts,
    extract_clips,
    predict_video,
    prepare_frame,
    sample_video,
)
from scripts.train_visual_fight import VideoData, fight_loss, frame_level, make_batch, selection_score


class MeanExtractor(torch.nn.Module):
    """Stand-in backbone: a feature that depends on every frame of the clip."""

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        per_frame = clips.float().mean(dim=(2, 3, 4))                     # (B, T)
        weights = torch.linspace(1.0, 2.0, per_frame.shape[1])
        return (per_frame * weights).sum(dim=1, keepdim=True).repeat(1, FEATURE_DIM) / 1000.0


def frames(count: int, size=(48, 64)) -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    return [rng.integers(0, 255, (*size, 3), dtype=np.uint8) for _ in range(count)]


def stream(video_frames, source_fps=30.0):
    """What sample_video yields for in-memory frames (same CLIP_FPS grid)."""
    step = max(source_fps / 15.0, 1.0)
    next_sample = 0.0
    for index, frame in enumerate(video_frames):
        if index + 1e-6 >= next_sample:
            next_sample += step
            yield (index, *prepare_frame(frame))
    yield ("end", len(video_frames), source_fps)


def test_clip_starts_cover_short_and_long_videos():
    assert clip_starts(0).tolist() == []
    assert clip_starts(5).tolist() == [0]
    assert clip_starts(CLIP_FRAMES).tolist() == [0]
    assert clip_starts(CLIP_FRAMES + 2 * CLIP_STRIDE).tolist() == [0, CLIP_STRIDE, 2 * CLIP_STRIDE]


def test_sample_video_follows_the_15_fps_grid(tmp_path):
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (64, 48))
    for frame in frames(60):
        writer.write(frame)
    writer.release()
    samples = list(sample_video(path))
    end = samples[-1]
    assert end[0] == "end" and end[1] == 60 and end[2] == pytest.approx(30.0)
    assert [sample[0] for sample in samples[:-1]] == list(range(0, 60, 2))
    assert samples[0][1].shape == (INPUT_SIZE, INPUT_SIZE, 3) and samples[0][1].dtype == np.uint8


def test_extract_clips_streams_clips_with_frame_spans():
    video = frames(120)                                       # 4 s at 30 FPS -> 60 samples
    payload = extract_clips(stream(video), MeanExtractor(), torch.device("cpu"), batch_size=2)
    starts = clip_starts(60)
    assert payload["features"].shape == (len(starts), FEATURE_DIM)
    assert payload["features"].dtype == np.float16
    assert payload["first_frames"].tolist() == [2 * start for start in starts]
    assert payload["last_frames"].tolist() == [2 * (start + CLIP_FRAMES - 1) for start in starts]
    assert payload["frame_count"] == 120 and np.all(payload["motion"] > 0)

    short = extract_clips(stream(frames(10)), MeanExtractor(), torch.device("cpu"))
    assert short["features"].shape == (1, FEATURE_DIM)       # padded single clip
    assert short["last_frames"].tolist() == [8]


def test_clip_labels_use_majority_and_guard_boundaries():
    clips = ClipFeatures(
        np.zeros((5, FEATURE_DIM)), np.zeros(5),
        first_frames=[0, 10, 20, 30, 40], last_frames=[9, 19, 29, 39, 49], frame_count=50, fps=10,
    )
    labels = np.zeros(50, dtype=np.int32)
    labels[22:30] = 1                                         # 8 of clip 2's 10 frames
    labels[30:33] = 1                                         # 3 of clip 3's frames
    assert clips.clip_labels(labels).tolist() == [0, 0, 1, -1, 0]
    assert clips.clip_labels(labels, guard_frames=5).tolist() == [0, -1, 1, -1, 0]


def test_head_is_causal_and_round_trips(tmp_path):
    torch.manual_seed(0)
    head = FightTemporalHead(hidden=32, layers=3).eval()
    assert head.receptive_field == 15
    features = torch.randn(1, 40, FEATURE_DIM)
    changed = features.clone()
    changed[:, 30:] = torch.randn(1, 10, FEATURE_DIM)        # only the future changes
    with torch.no_grad():
        before, after = head(features), head(changed)
    torch.testing.assert_close(before[:, :30], after[:, :30])
    assert not torch.allclose(before[:, 30:], after[:, 30:])

    path = tmp_path / "head.pt"
    head.save(path, {"epoch": 3})
    loaded = FightTemporalHead.load(path)
    with torch.no_grad():
        torch.testing.assert_close(loaded(features), before)


def test_streaming_scorer_matches_offline_scores():
    torch.manual_seed(0)
    head = FightTemporalHead(hidden=32, layers=2).eval()     # history of 7 clips < 12 clips below
    video = frames(30 * 13)                                   # 13 s at 30 FPS
    offline = extract_clips(stream(video), MeanExtractor(), torch.device("cpu"))
    expected = predict_video(head, offline["features"], torch.device("cpu"))

    scorer = StreamingFightScorer(MeanExtractor(), head, source_fps=30.0)
    live = [score for frame in video if (score := scorer.push(frame)) is not None]
    np.testing.assert_allclose(live, expected, rtol=1e-5, atol=1e-6)


def test_s3d_backbone_gives_2048_features():
    extractor = ClipFeatureExtractor(pretrained=False)
    clips = torch.randint(0, 255, (2, CLIP_FRAMES, INPUT_SIZE, INPUT_SIZE, 3), dtype=torch.uint8)
    output = extractor(clips)
    assert output.shape == (2, FEATURE_DIM) and torch.isfinite(output).all()
    assert not any(parameter.requires_grad for parameter in extractor.parameters())


def test_fight_loss_rewards_ranking_fights_above_calm_clips():
    labels = torch.tensor([[1, 1, 0, 0, -1]])
    good = torch.tensor([[3.0, 2.0, -2.0, -3.0, 9.0]])
    bad = torch.tensor([[-2.0, -3.0, 3.0, 2.0, 9.0]])
    good_loss, detail = fight_loss(good, labels, rank_weight=1.0)
    bad_loss, _ = fight_loss(bad, labels, rank_weight=1.0)
    assert good_loss < bad_loss and detail["ranked_rows"] == 1
    ignored = fight_loss(torch.tensor([[0.0, 0.0]]), torch.tensor([[-1, -1]]), 1.0)[0]
    assert ignored.item() == 0.0


def fake_video(key: str, fight: tuple[int, int] | None, clips: int = 30, fps: float = 10.0) -> VideoData:
    frame_count = clips * 10
    labels = np.zeros(frame_count, dtype=np.int32)
    if fight:
        labels[fight[0]:fight[1]] = 1
    item = VideoItem("ubi", key, None, "train", frame_labels=labels)
    first = np.arange(clips) * 10
    features = np.random.default_rng(len(key)).normal(size=(clips, FEATURE_DIM))
    motion = labels[first].astype(np.float32) + 0.1
    return VideoData(item, ClipFeatures(features, motion, first, first + 9, frame_count, fps), guard_seconds=1.0)


def test_make_batch_pads_and_frame_level_scores_whole_videos():
    videos = [fake_video("F1", (100, 200)), fake_video("N1", None, clips=10)]
    features, labels = make_batch(videos, [0], [1], batch_size=6, length=16, fight_share=0.5,
                                  rng=np.random.default_rng(0))
    assert features.shape == (6, 16, FEATURE_DIM) and labels.shape == (6, 16)
    assert (labels == 1).any() and (labels == 0).any()
    short_rows = (labels == -1).all(dim=1)
    assert not short_rows.any()

    perfect = frame_level(videos, [video.clips.motion.astype(np.float64) for video in videos])
    assert perfect["auc"] > 0.95 and perfect["video_auc_median"] > 0.95
    assert selection_score(perfect, "mean") == pytest.approx((perfect["auc"] + perfect["video_auc_median"]) / 2)
    assert selection_score({"auc": 0.8, "video_auc_median": float("nan")}, "video_auc") == 0.0


def test_scene_removal_is_causal_and_zero_shot_scores_are_probabilities():
    from scripts.diagnose_visual_fight import kinetics_fight_score, remove_scene

    features = np.random.default_rng(0).normal(size=(6, FEATURE_DIM)).astype(np.float16)
    scene = remove_scene(features, "scene").astype(np.float32)
    assert np.allclose(scene[0], 0.0)
    expected = features[3].astype(np.float32) - features[:3].astype(np.float32).mean(axis=0)
    np.testing.assert_allclose(scene[3], expected, atol=1e-2)
    changed = features.copy()
    changed[5] += 3.0                                           # a later clip does not move earlier ones
    np.testing.assert_array_equal(remove_scene(changed, "scene")[:5], remove_scene(features, "scene")[:5])
    assert np.allclose(remove_scene(features, "videomean").astype(np.float32).mean(axis=0), 0.0, atol=1e-2)

    weight = np.random.default_rng(1).normal(size=(400, 1024)).astype(np.float32) / 30
    scores = kinetics_fight_score(features, weight, np.zeros(400, np.float32), [3, 7])
    assert scores.shape == (6,) and np.all((scores > 0) & (scores < 1))
    assert np.allclose(kinetics_fight_score(features, weight, np.zeros(400, np.float32), list(range(400))), 1.0)


def test_person_groups_merge_close_people_and_rank_by_size():
    from anomaly.visual_fight import person_groups

    boxes = np.array([
        [100, 100, 140, 200], [150, 100, 190, 200],          # two people side by side
        [100, 105, 140, 205],                                 # the first one, a moment later
        [600, 100, 640, 200],                                 # someone alone far away
    ], dtype=np.float32)
    groups = person_groups(boxes, max_groups=3, gap=0.5)
    np.testing.assert_allclose(groups[0], [100, 100, 190, 205])
    np.testing.assert_allclose(groups[1], [600, 100, 640, 200])
    assert len(person_groups(boxes, max_groups=1)) == 1 and person_groups(np.zeros((0, 4))) == []


def test_square_crops_stay_square_inside_the_frame():
    from anomaly.visual_fight import MIN_CROP, crop_frame, square_crop_box

    region = square_crop_box(np.array([0, 10, 40, 110]), frame_w=640, frame_h=360, margin=0.25)
    assert region[2] - region[0] == pytest.approx(region[3] - region[1]) == pytest.approx(150)
    assert region[0] == pytest.approx(0) and region[1] >= 0          # shifted inside, not cut
    tiny = square_crop_box(np.array([300, 200, 310, 220]), 640, 360)
    assert tiny[2] - tiny[0] == pytest.approx(MIN_CROP)
    frame = np.full((360, 640, 3), 255, dtype=np.uint8)
    crop = crop_frame(frame, np.array([-50, -50, 150, 150]))            # partly outside: black border
    assert crop.shape == (INPUT_SIZE, INPUT_SIZE, 3) and crop[0, 0].sum() == 0 and crop[-1, -1].sum() == 765


def test_extract_crop_clips_uses_person_groups_and_falls_back_to_the_whole_frame():
    from anomaly.visual_fight import MAX_GROUPS, extract_crop_clips

    video = frames(120, size=(360, 640))

    def grid():
        for index in range(0, 120, 2):
            yield index, video[index]
        yield ("end", 120, 30.0)

    def boxes_for(first, last):                               # people only in the second half
        if first < 60:
            return np.zeros((0, 4), np.float32)
        return np.array([[100, 100, 140, 200], [150, 100, 190, 200], [500, 50, 530, 110]], np.float32)

    payload = extract_crop_clips(grid(), boxes_for, MeanExtractor(), torch.device("cpu"), batch_size=3)
    count = len(clip_starts(60))
    assert payload["features"].shape == (count, FEATURE_DIM)
    assert payload["crop_features"].shape == (count, MAX_GROUPS, FEATURE_DIM)
    assert payload["crop_count"].tolist() == [1, 1, 2]       # whole frame, whole frame, two groups
    np.testing.assert_allclose(payload["crop_boxes"][0, 0], [0, 0, 640, 360])
    last = payload["crop_features"][2, :2].astype(np.float32)
    np.testing.assert_allclose(payload["features"][2].astype(np.float32), last.max(axis=0))
    assert payload["first_frames"].tolist() == [0, 30, 60]


def test_pose_boxes_select_rows_in_the_frame_span(tmp_path):
    from scripts.extract_visual import pose_boxes

    np.savez(tmp_path / "v.npz", frame_index=np.array([9, 0, 3, 3]),
             boxes=np.array([[9, 9, 9, 9], [0, 0, 0, 0], [3, 3, 3, 3], [4, 4, 4, 4]], np.float32))
    boxes_for = pose_boxes(tmp_path / "v.npz")
    assert boxes_for(1, 5).tolist() == [[3, 3, 3, 3], [4, 4, 4, 4]]
    assert boxes_for(0, 9).shape == (4, 4)
    assert pose_boxes(tmp_path / "missing.npz")(0, 9).shape == (0, 4)
