import numpy as np
import pytest

from scripts.evaluation_metrics import aggregate_frame_scores, binary_metrics


def test_aggregate_frame_scores_uses_max_and_interpolates():
    scores = np.asarray([0.1, 0.3, 0.5], dtype=np.float64)
    frame_numbers = np.asarray([2, 2, 4], dtype=np.int32)

    frame_scores, observed = aggregate_frame_scores(scores, frame_numbers, 5)

    np.testing.assert_allclose(frame_scores, [0.3, 0.3, 0.4, 0.5, 0.5])
    np.testing.assert_array_equal(observed, [False, True, False, True, False])


def test_binary_metrics_reports_perfect_predictions():
    labels = np.asarray([0, 0, 1, 1], dtype=np.int32)
    scores = np.asarray([0.1, 0.4, 0.6, 0.9], dtype=np.float64)

    metrics = binary_metrics(labels, scores, threshold=0.5)

    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["specificity"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["average_precision"] == 1.0
    assert metrics["true_negative"] == 2
    assert metrics["true_positive"] == 2


def test_aggregate_frame_scores_rejects_mismatched_metadata():
    with pytest.raises(ValueError, match="equal length"):
        aggregate_frame_scores(
            np.asarray([0.1, 0.2]),
            np.asarray([1]),
            frame_count=2,
        )
