"""Tests for metrics."""
import pytest  # noqa

from mouse.utils import data_util
from mouse.utils import metrics

ground_truth = [
    data_util.SqueakBox(freq_start=1,
                        freq_end=7,
                        t_start=2,
                        t_end=8,
                        label='sh'),
    data_util.SqueakBox(freq_start=3,
                        freq_end=5,
                        t_start=9,
                        t_end=10,
                        label='fm'),
    data_util.SqueakBox(freq_start=7,
                        freq_end=10,
                        t_start=9,
                        t_end=11,
                        label='fm')
]
prediction = [
    data_util.SqueakBox(freq_start=0,
                        freq_end=5,
                        t_start=4,
                        t_end=8,
                        label=None),
    data_util.SqueakBox(freq_start=8,
                        freq_end=9,
                        t_start=9,
                        t_end=11,
                        label=None)
]


def test_iou_global_no_axes():
    """Test global iou computation for both axes."""
    assert metrics.intersection_over_union_global(prediction,
                                                  ground_truth) == 31 / 72


def test_iou_global_axis_0():
    """Test global iou computation for time axis."""
    assert metrics.intersection_over_union_global(prediction,
                                                  ground_truth,
                                                  axis=0) == 8 / 10


def test_iou_global_axis_1():
    """Test global iou computation for frequency axis."""
    assert metrics.intersection_over_union_global(prediction,
                                                  ground_truth,
                                                  axis=1) == 7 / 11


def test_iou_elementwise():
    """Test elementwise iou computation."""
    prediction_given_ground_truth_iou_elementwise = {
        ground_truth[0]: {
            prediction[0]: 25 / 54
        },
        ground_truth[1]: {},
        ground_truth[2]: {
            prediction[1]: 6 / 12
        }
    }

    ground_truth_given_prediction_iou_elementwise = {
        prediction[0]: {
            ground_truth[0]: 25 / 54
        },
        prediction[1]: {
            ground_truth[2]: 6 / 12
        }
    }
    assert metrics.intersection_over_union_elementwise(
        ground_truth,
        prediction) == prediction_given_ground_truth_iou_elementwise
    assert metrics.intersection_over_union_elementwise(
        prediction,
        ground_truth) == ground_truth_given_prediction_iou_elementwise


def test_coverage_axis_0():
    """Test coverage computation for time axis."""
    assert metrics.coverage(prediction, ground_truth,
                            axis=0) == (10 / 10, 8 / 10)


def test_coverage_axis_1():
    """Test coverage computation for frequency axis."""
    assert metrics.coverage(prediction, ground_truth, axis=1) == (7 / 8, 7 / 10)


def test_coverage_both_axes():
    """Test coverage computation for frequency axis."""
    assert metrics.coverage(prediction, ground_truth,
                            axis=None) == (31 / 36, 31 / 67)


def test_precision():
    """Test precision."""
    assert metrics.detection_precision(ground_truth=ground_truth,
                                       prediction=prediction,
                                       threshold=0.4) == 1
    assert metrics.detection_precision(ground_truth=ground_truth,
                                       prediction=prediction,
                                       threshold=0.5) == 0.5


def test_recall():
    """Test recall."""
    overall_recall, labels_recalls = metrics.detection_recall(
      ground_truth=ground_truth,
      prediction=prediction,
      threshold=0.4,
      mode='sum')
    assert overall_recall == 2 / 3
    assert labels_recalls == {'sh': 1, 'fm': 1 / 2}
