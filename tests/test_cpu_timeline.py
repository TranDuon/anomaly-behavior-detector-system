from scripts.fight_cpu_acceptance import compare_modes


def sample(times, intervals):
    return {
        "video": "public",
        "predictions": [{"prediction_timestamp": time, "score": 0.9} for time in times],
        "events": [{"started_at": start, "ended_at": end} for start, end in intervals],
    }


def test_equal_counts_and_matching_subset_do_not_prove_full_parity():
    result = compare_modes(sample([1, 1.5, 2], [(1, 2)]), sample([1, 2], [(1.5, 2)]))
    assert result["matching_prediction_timestamps"] == 2
    assert result["maximum_matching_score_difference"] == 0
    assert result["offline_events"] == result["replay_events"]
    assert not result["all_prediction_timestamps_match"]
    assert not result["event_intervals_match"]
    assert not result["full_timeline_parity"]


def test_identical_timeline_and_sample_quantized_end_pass():
    result = compare_modes(
        sample([1, 1.5, 2], [(1, 2)]), sample([1, 1.5, 2], [(1, 2.03)])
    )
    assert result["full_timeline_parity"]


def test_empty_predictions_cannot_pass():
    assert not compare_modes(sample([], []), sample([], []))["full_timeline_parity"]
