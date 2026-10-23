import numpy as np
import pytest
import torch

from anomaly.fight_data import (
    VideoItem,
    load_items,
    parse_ubi_annotation,
    parse_ucf_annotation,
    training_ranges,
)
from anomaly.skeleton import (
    BODY_TARGETS,
    LIMB_ENDS,
    NUM_CHANNELS,
    NUM_NODES,
    PoseTrack,
    _track_stats,
    proximity_matrix,
    pair_tensor,
    select_pairs,
    two_person_adjacency,
    window_bag,
    window_pairs,
    window_starts,
)
from anomaly.stgcn import PairBagClassifier, collate_bags, predict_windows
from scripts.dedup_videos import hamming, match_fraction
from scripts.train_fight import best_f1_threshold, split_videos, window_labels


def standing_person(x, y, height=100.0, frames=24):
    """Synthetic COCO skeleton: joints spread vertically around (x, y)."""
    keypoints = np.zeros((frames, 17, 3), dtype=np.float32)
    offsets = np.linspace(-height / 2, height / 2, 17)
    keypoints[:, :, 0] = x
    keypoints[:, :, 1] = y + offsets
    keypoints[:, :, 2] = 0.9
    return keypoints, np.full(frames, height, dtype=np.float32)


def test_parse_ubi_annotation_handles_crlf_and_floats():
    labels = parse_ubi_annotation("0\r\n1\r\n1.0\r\n0.0\r\n")
    np.testing.assert_array_equal(labels, [0, 1, 1, 0])


def test_parse_ucf_annotation_skips_missing_intervals():
    parsed = parse_ucf_annotation(
        "Fighting003_x264.mp4  Fighting  1820  3103  -1  -1\n"
        "Normal_Videos_003_x264.mp4  Normal  -1  -1  -1  -1\n"
    )
    assert parsed["Fighting003_x264"] == ("Fighting", [(1820, 3103)])
    assert parsed["Normal_Videos_003_x264"] == ("Normal", [])


def test_labels_for_unifies_granularities():
    frame = VideoItem("ubi", "a", None, "train", frame_labels=np.array([0, 1, 1]))
    np.testing.assert_array_equal(frame.labels_for(5), [0, 1, 1, 1, 1])
    interval = VideoItem("ucf", "b", None, "test", intervals=[(1, 2)])
    np.testing.assert_array_equal(interval.labels_for(4), [0, 1, 1, 0])
    clip = VideoItem("scf", "c", None, "test", clip_label=1)
    np.testing.assert_array_equal(clip.labels_for(3), [1, 1, 1])
    assert not VideoItem("scf", "d", None, "test", clip_label=0).has_positive


def test_training_ranges_keep_fight_runs_with_margin():
    labels = np.zeros(100, dtype=np.int32)
    labels[20:30] = 1
    labels[36:40] = 1
    labels[80:85] = 1
    item = VideoItem("ubi", "f", None, "train", frame_labels=labels)
    # runs 20-29 and 36-39 merge once widened by 5; 80-84 stays separate
    assert training_ranges(item, 5) == [(15, 44), (75, 89)]
    assert training_ranges(item, 30) == [(0, 114)]
    test_item = VideoItem("ubi", "t", None, "test", frame_labels=labels)
    assert training_ranges(test_item, 5) is None                 # test videos are never cropped
    normal = VideoItem("ubi", "n", None, "train", frame_labels=np.zeros(10))
    assert training_ranges(normal, 5) is None
    assert training_ranges(VideoItem("ucf", "u", None, "train", intervals=[(50, 60)]), 10) == [(40, 70)]


def test_extraction_options_follow_split_and_label():
    from types import SimpleNamespace
    from scripts.extract_pose import extraction_options

    args = SimpleNamespace(max_seconds_normal=20, max_seconds_test_normal=60, fight_margin_seconds=10)
    fight_train = VideoItem("ubi", "a", None, "train", frame_labels=np.array([0, 1]))
    normal_train = VideoItem("ubi", "b", None, "train", frame_labels=np.array([0, 0]))
    normal_test = VideoItem("ubi", "c", None, "test", frame_labels=np.array([0, 0]))
    fight_test = VideoItem("ubi", "d", None, "test", frame_labels=np.array([0, 1]))
    assert extraction_options(fight_train, args) == {"max_seconds": 0.0, "fight_margin_seconds": 10.0}
    assert extraction_options(normal_train, args) == {"max_seconds": 20.0, "fight_margin_seconds": -1.0}
    assert extraction_options(normal_test, args) == {"max_seconds": 60.0, "fight_margin_seconds": -1.0}
    assert extraction_options(fight_test, args) == {"max_seconds": 0.0, "fight_margin_seconds": -1.0}
    final = SimpleNamespace(max_seconds_normal=20, max_seconds_test_normal=None, fight_margin_seconds=10)
    assert extraction_options(normal_test, final)["max_seconds"] == 0.0   # final run re-extracts it


def test_cache_matches_ignores_caps_that_never_cut_the_video(tmp_path):
    from scripts.extract_pose import cache_matches

    def cache(name, cap, truncated, frames, margin=-1.0):
        path = tmp_path / f"{name}.npz"
        np.savez(path, max_seconds=np.float32(cap), truncated=np.bool_(truncated),
                 frame_count=np.int32(frames), fps=np.float32(10), fight_margin_seconds=np.float32(margin))
        return path

    short_clip = cache("short", cap=60, truncated=False, frames=20)        # 2 s clip, cap never applied
    cut_video = cache("cut", cap=60, truncated=True, frames=600)           # 60 s of a longer video
    full = {"max_seconds": 0.0, "fight_margin_seconds": -1.0}
    dev = {"max_seconds": 60.0, "fight_margin_seconds": -1.0}
    assert cache_matches(short_clip, full) and cache_matches(short_clip, dev)
    assert cache_matches(cut_video, dev) and not cache_matches(cut_video, full)
    assert cache_matches(cut_video, {"max_seconds": 20.0, "fight_margin_seconds": -1.0})   # longer cap covers shorter
    whole = cache("whole", cap=0, truncated=False, frames=6000)
    assert cache_matches(whole, dev)                                       # back to dev phase: keep it
    cropped = cache("crop", cap=0, truncated=False, frames=900, margin=10)
    assert cache_matches(cropped, {"max_seconds": 0.0, "fight_margin_seconds": 5.0})
    assert not cache_matches(cropped, {"max_seconds": 0.0, "fight_margin_seconds": 15.0})
    assert not cache_matches(cropped, full)                                # whole video wanted


def test_load_items_reads_ubi_layout_and_split(tmp_path):
    base = tmp_path / "UBI_FIGHTS"
    (base / "annotation").mkdir(parents=True)
    for folder, name in (("fight", "F_0_1_0_0_0"), ("normal", "N_0_0_0_1_0")):
        (base / "videos" / folder).mkdir(parents=True)
        (base / "videos" / folder / f"{name}.mp4").write_bytes(b"")
        (base / "annotation" / f"{name}.csv").write_text("0\n1\n" if folder == "fight" else "0\n0\n")
    (base / "test_videos.csv").write_text("F_0_1_0_0_0\n")

    items = {item.key: item for item in load_items(f"ubi:{tmp_path}")}
    assert items["F_0_1_0_0_0"].split == "test"
    assert items["N_0_0_0_1_0"].split == "train"
    assert items["F_0_1_0_0_0"].has_positive
    assert all(item.split == "train" for item in load_items(f"ubi:{tmp_path}:train"))


def test_adjacency_is_symmetric_and_links_people():
    adjacency = two_person_adjacency()
    assert adjacency.shape == (3, NUM_NODES, NUM_NODES)
    for subset in adjacency:
        np.testing.assert_allclose(subset, subset.T, atol=1e-6)
    assert adjacency[2, 9, 17 + 0] > 0       # wrist of person A to head of B
    assert adjacency[1, 9, 17 + 0] == 0      # never an intra-person edge


def test_select_pairs_prefers_close_people_and_falls_back_to_singles():
    near_a, near_b, far = standing_person(100, 100), standing_person(180, 100), standing_person(2000, 100)
    assert select_pairs([near_a, far, near_b], max_pairs=4, radius=2.5) == [(0, 2)]
    singles = select_pairs([near_a, far], max_pairs=4, radius=2.5)
    assert sorted(i for i, _ in singles) == [0, 1]
    assert all(j is None for _, j in singles)


def flailing_person(x, y, height=100.0, frames=24):
    """Standing skeleton whose wrists swing 30 px back and forth every sample."""
    keypoints, heights = standing_person(x, y, height, frames)
    swing = np.where(np.arange(frames) % 2 == 0, 30.0, -30.0)
    keypoints[:, [9, 10], 0] += swing[:, None]
    return keypoints, heights


def test_select_pairs_ranks_interaction_above_mere_closeness():
    # Two calm pairs standing very close, one active pair a little further apart.
    tracks = [
        standing_person(0, 0), standing_person(50, 0),
        standing_person(1000, 0), standing_person(1050, 0),
        flailing_person(2000, 0), flailing_person(2120, 0),
    ]
    assert select_pairs(tracks, max_pairs=1, radius=2.5) == [(4, 5)]
    assert len(select_pairs(tracks, max_pairs=8, radius=2.5)) == 3


def naive_proximity(first, second, stats_first, stats_second):
    """Pair-by-pair reference for proximity_matrix (the original loop version)."""
    together = ~np.isnan(stats_first.center[:, 0]) & ~np.isnan(stats_second.center[:, 0])
    if not together.any():
        return np.inf
    scale = (stats_first.height + stats_second.height) / 2
    centre = np.linalg.norm(stats_first.center[together] - stats_second.center[together], axis=1)
    closeness = float(np.quantile(centre, 0.2)) / scale
    reach = np.inf
    for kp_a, kp_b in ((first[0], second[0]), (second[0], first[0])):
        limbs = kp_a[together][:, LIMB_ENDS]
        bodies = kp_b[together][:, BODY_TARGETS]
        ok = (limbs[:, :, None, 2] >= 0.1) & (bodies[:, None, :, 2] >= 0.1)
        if ok.any():
            gaps = np.linalg.norm(limbs[:, :, None, :2] - bodies[:, None, :, :2], axis=-1)
            reach = min(reach, float(gaps[ok].min()) / scale)
    return min(closeness, reach)


def test_proximity_matrix_matches_pairwise_reference():
    rng = np.random.default_rng(3)
    tracks = []
    for _ in range(7):
        keypoints, heights = standing_person(*rng.uniform(0, 400, 2), height=rng.uniform(60, 140))
        keypoints[:, :, :2] += rng.normal(0, 8, (24, 17, 2))
        keypoints[:, :, 2] = rng.uniform(0, 1, (24, 17))          # some below 0.1
        absent = rng.random(24) < 0.3
        keypoints[absent] = 0
        heights[absent] = 0
        tracks.append((keypoints, heights))
    stats = [_track_stats(kp, h) for kp, h in tracks]
    matrix = proximity_matrix(np.stack([kp for kp, _ in tracks]), stats)
    for i in range(7):
        for j in range(7):
            if i != j:
                assert matrix[i, j] == pytest.approx(naive_proximity(tracks[i], tracks[j], stats[i], stats[j]), rel=1e-5)


def test_select_pairs_training_explores_beyond_top_ranks():
    crowd = [standing_person(60 * i, 0) for i in range(8)]   # many close pairs
    ranked = select_pairs(crowd, max_pairs=100, radius=2.5)
    assert len(ranked) > 8
    seen = set()
    for seed in range(20):
        chosen = select_pairs(crowd, max_pairs=8, radius=2.5, rng=np.random.default_rng(seed))
        assert len(chosen) == 8 and len(set(chosen)) == 8
        assert chosen[:6] == ranked[:6]                  # top ranks always kept
        seen.update(chosen[6:])
    assert len(seen) > 2                                 # the extra slots vary


def test_pair_tensor_is_resolution_invariant():
    small = pair_tensor(standing_person(100, 100, 50), standing_person(140, 100, 50))
    large = pair_tensor(standing_person(200, 200, 100), standing_person(280, 200, 100))
    assert small.shape == (NUM_CHANNELS, 24, NUM_NODES)
    np.testing.assert_allclose(small, large, atol=1e-5)


def test_pair_tensor_single_person_leaves_partner_empty():
    tensor = pair_tensor(standing_person(100, 100), None, rng=np.random.default_rng(0))
    occupied = [np.abs(tensor[:, :, :17]).sum(), np.abs(tensor[:, :, 17:]).sum()]
    assert sum(value > 0 for value in occupied) == 1


def make_pose(frames=40, people=((100, 100), (170, 100))):
    rows_frame, rows_id, boxes, keypoints = [], [], [], []
    for frame in range(frames):
        for track_id, (x, y) in enumerate(people, start=1):
            kp, _ = standing_person(x + frame, y, frames=1)
            rows_frame.append(frame)
            rows_id.append(track_id)
            boxes.append([x - 20, y - 50, x + 20, y + 50])
            keypoints.append(kp[0])
    sample_frames = np.arange(frames)
    return PoseTrack(
        sample_frames=sample_frames,
        sample_index=np.searchsorted(sample_frames, np.asarray(rows_frame)),
        track_ids=np.asarray(rows_id),
        boxes=np.asarray(boxes, dtype=np.float32),
        keypoints=np.asarray(keypoints, dtype=np.float32),
        frame_count=frames,
        fps=10.0,
    )


def test_window_bag_and_model_forward():
    pose = make_pose()
    bag = window_bag(pose, 0, 24, min_presence=8, max_pairs=4, radius=2.5)
    assert bag.shape == (1, NUM_CHANNELS, 24, NUM_NODES)
    empty = window_bag(make_pose(people=()), 0, 24, 8, 4, 2.5)
    assert empty.shape[0] == 0

    model = PairBagClassifier(channels=(16, 16, 32), strides=(1, 2, 1))
    bags, mask = collate_bags([bag, np.concatenate([bag, bag])])
    assert bags.shape == (2, 2, NUM_CHANNELS, 24, NUM_NODES)
    assert mask.tolist() == [[True, False], [True, True]]
    logits = model(bags, mask)
    assert logits.shape == (2, 1) and torch.isfinite(logits).all()

    window = {"length": 24, "min_presence": 8, "max_pairs": 4, "pair_radius": 2.5}
    starts = window_starts(pose.num_samples, 24, 4)
    probabilities = predict_windows(model, pose, starts, window, torch.device("cpu"), batch_size=2)
    assert probabilities.shape == starts.shape
    assert np.all((probabilities > 0) & (probabilities < 1))

    again, pairs = predict_windows(
        model, pose, starts, window, torch.device("cpu"), batch_size=2, return_pairs=True
    )
    np.testing.assert_allclose(again, probabilities)
    assert all(pair == (1, 2) for pair in pairs)


def test_windows_spanning_two_extracted_ranges_are_not_contiguous():
    pose = make_pose(frames=40)
    pose.segment_ids = np.array([0] * 20 + [1] * 20)
    assert pose.contiguous(0, 20)
    assert not pose.contiguous(10, 20)
    assert pose.contiguous(20, 24)                 # runs past the end inside one range
    pose.segment_ids = None
    assert pose.contiguous(10, 20)                 # old caches: one continuous range


def test_window_pairs_reports_track_ids():
    _, ids = window_pairs(make_pose(), 0, 24, min_presence=8, max_pairs=4, radius=2.5)
    assert ids == [(1, 2)]


def test_model_save_and_load_roundtrip(tmp_path):
    model = PairBagClassifier(channels=(16, 32), strides=(1, 2))
    path = tmp_path / "fight.pt"
    model.save(path, {"window": {"length": 24}})
    loaded = PairBagClassifier.load(path)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, loaded.state_dict()[key])


def test_window_starts_and_labels():
    np.testing.assert_array_equal(window_starts(10, 24, 4), [0])
    np.testing.assert_array_equal(window_starts(32, 24, 4), [0, 4, 8])
    samples = np.array([0] * 24 + [1] * 24)
    np.testing.assert_array_equal(window_labels(samples, np.array([0, 12, 6, 24]), 24), [0, 1, -1, 1])


def test_split_videos_keeps_fights_in_both_sides():
    items = [VideoItem("ubi", f"F{i}", None, "train", clip_label=1) for i in range(4)]
    items += [VideoItem("ubi", f"N{i}", None, "train", clip_label=0) for i in range(6)]
    train, val = split_videos(items, 0.25, seed=0)
    assert {item.key for item in train}.isdisjoint({item.key for item in val})
    assert any(item.has_positive for item in val) and any(item.has_positive for item in train)


def test_best_f1_threshold_matches_strict_rule():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.6, 0.9])
    threshold, f1 = best_f1_threshold(labels, scores)
    assert f1 == pytest.approx(1.0)
    np.testing.assert_array_equal(scores > threshold, labels.astype(bool))


def test_dedup_hamming_and_match_fraction():
    a = np.array([0b1011, 0xFFFF], dtype=np.uint64)
    np.testing.assert_array_equal(hamming(a, a), [[0, 13], [13, 0]])
    hashes = np.array(
        [[11, 22, 33, 44], [11, 22, 33, 44], [1 << 40, 1 << 41, 1 << 42, 1 << 43]],
        dtype=np.uint64,
    )
    fraction = match_fraction(hashes, max_distance=0)
    assert fraction[0, 1] == pytest.approx(1.0)
    assert fraction[0, 2] == 0.0
