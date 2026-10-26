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
