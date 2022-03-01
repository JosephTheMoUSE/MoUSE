"""Tests for metrics."""
import numpy as np
import pytest  # noqa

from mouse.utils import data_util, sound_util


def _get_box(ts, te, fs, fe, label=None) -> data_util.SqueakBox:
    """Create a squeakbox quickly."""
    return data_util.SqueakBox(freq_start=fs,
                               freq_end=fe,
                               t_start=ts,
                               t_end=te,
                               label=label)


def test_box_finding():
    """Tests box finding."""
    box = _get_box(ts=3, te=5, fs=10, fe=12)
    mask = np.zeros((20, 20))
    mask[box.freq_start:box.freq_end + 1, box.t_start:box.t_end + 1] = 1
    box_from_mask = data_util.find_bounding_boxes(mask=mask, min_side_length=0)

    assert len(box_from_mask) == 1
    assert box_from_mask[0] == box


def test_distance():
    """Tests distance calculation between `SqueakBox`es."""
    b_1 = _get_box(1, 1, 5, 5)
    b_2 = _get_box(2, 5, 7, 10)
    assert (1, 2) == data_util.get_abs_distances(b_1, b_2)

    b_3 = _get_box(1, 20, 11, 20)
    assert (0, 1) == data_util.get_abs_distances(b_3, b_2)

    b_4 = _get_box(20, 35, 1, 11)
    assert (0, 0) == data_util.get_abs_distances(b_3, b_4)

    b_5 = _get_box(25, 25, 0, 15)
    assert (0, 0) == data_util.get_abs_distances(b_5, b_4)
    assert (5, 0) == data_util.get_abs_distances(b_5, b_3)
    assert (20, 0) == data_util.get_abs_distances(b_5, b_2)
    assert (24, 0) == data_util.get_abs_distances(b_5, b_1)


def test_merging():
    """Test merging."""
    not_merging_boxes = [
        _get_box(ts, te, fs, fe) for ((ts, te), (fs, fe)) in [
            ((0, 1), (0, 1)),
            ((58, 59), (8, 9)),
            ((5, 6), (1, 4)),
            ((9, 10), (1, 4)),
            ((5, 10), (8, 9)),
        ]
    ]
    merging_boxes = [
        _get_box(ts, te, fs, fe) for ((ts, te), (fs, fe)) in [
            ((20, 26), (2, 3)),
            ((28, 29), (7, 8)),
            ((27, 28), (4, 5)),
        ]
    ]

    merged_boxes = set([_get_box(ts=20, te=29, fs=2, fe=8)] + not_merging_boxes)

    spec = sound_util.SpectrogramData(spec=None,
                                      freqs=np.linspace(0, 10000, 10),
                                      times=np.linspace(0, 60, 60))

    merged = data_util.merge_boxes(spec=spec,
                                   squeaks=merging_boxes + not_merging_boxes,
                                   delta_time=1.,
                                   delta_freq=2000)
    assert set(merged) == merged_boxes
