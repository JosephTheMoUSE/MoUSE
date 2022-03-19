"""Module containing metrics utilities."""
import warnings
from typing import Dict, List, Tuple, Union

import numpy as np

from mouse.utils import data_util

# Squeak detection metrics


def _box_iou(target_box: data_util.SqueakBox, cover_box: data_util.SqueakBox) -> float:
    """Compute intersection over union for two squeak boxes.

    Parameters
    ----------
    target_box : data_util.SqueakBox
        First squeak box to process.
    cover_box : data_util.SqueakBox
        Second squeak box to process.

    Returns
    -------
    double
        intersection over union of `first_box` and `second_box`.

    Raises
    ------
    ValueError
        If any of the boxes doesn't have a positive area.
    """
    target_area = (target_box.t_end - target_box.t_start +
                   1) * (target_box.freq_end - target_box.freq_start + 1)
    cover_area = (cover_box.t_end - cover_box.t_start + 1) * (cover_box.freq_end -
                                                              cover_box.freq_start + 1)

    if target_area <= 0 or cover_area <= 0:
        raise ValueError("boxes should have positive area")

    intersection_height = (min(target_box.freq_end, cover_box.freq_end) -
                           max(target_box.freq_start, cover_box.freq_start) + 1)
    intersection_width = (min(target_box.t_end, cover_box.t_end) -
                          max(target_box.t_start, cover_box.t_start) + 1)
    intersection_area = intersection_height * intersection_width

    intersection = intersection_area
    union = target_area + cover_area - intersection_area

    iou = intersection / union

    return iou


def _to_bit_map(
    squeaks: List[data_util.SqueakBox],
    freq_start: int,
    freq_end: int,
    t_start: int,
    t_end: int,
) -> np.ndarray:
    """Transform squeak list to 2D bit map.

    Ranges bounding the bit map are passed separately due to the use cases
    in functions computing intersection over union ratio.

    Parameters
    ----------
    squeaks : List[data_util.SqueakBox]
        List of squeaks to be placed on bit map.
    freq_start : int
        Start of the frequency range of the bit map.
    freq_end : int
        End of the frequency range of the bit map.
    t_start : int
        Start of the time range of the bit map.
    t_end : int
        End of the time range of the bit map.

    Returns
    -------
    np.array(bool)
        2D bit map, where squeak presence is marked as True.
    """
    height = freq_end - freq_start + 1
    width = t_end - t_start + 1

    bit_map = np.zeros((height, width), dtype=bool)

    for squeak in squeaks:
        bit_map[squeak.freq_start - freq_start:squeak.freq_end + 1 - freq_start,
                squeak.t_start - t_start:squeak.t_end + 1 - t_start,] = True

    return bit_map


def _to_bit_map_1d(squeaks: List[data_util.SqueakBox],
                   start: int,
                   end: int,
                   axis: int = 0) -> np.ndarray:
    """Transform squeak list to 1D bit map.

    Ranges bounding the bit map are passed separately due to the use cases
    in functions computing intersection over union ratio.

    Parameters
    ----------
    squeaks : List[data_util.SqueakBox]
        List of squeaks to be placed on bit map.
    start : int
        Start of the range for which bit map is created.
    end : int
        End of the range for which bit map is created.
    axis : int
        Axis for which 1D bit map will be created. Frequency axis is indicated
        by 0, time axis is indicated by 1. Any other value will raise exception.

    Returns
    -------
    np.array(bool)
        1D bit map, where squeak presence is marked as True.

    Raises
    ------
    ValueError
        If `axis` is different than 0 or 1.
    """
    length = end - start
    bit_map = np.zeros(length + 1, dtype=bool)

    for squeak in squeaks:
        if axis == 0:
            bit_map[squeak.t_start - start:squeak.t_end + 1 - start] = True
        elif axis == 1:
            bit_map[squeak.freq_start - start:squeak.freq_end + 1 - start] = True
        else:
            raise ValueError("Axis must be 0 or 1")

    return bit_map


def _get_squeaks_range(squeaks: List[data_util.SqueakBox],
                       axis: int) -> Tuple[int, int]:
    """Extract start and end of squeaks' time/frequency range.

    Parameters
    ----------
    squeaks : List[data_util.SqueakBox]
        List of squeaks to process.
    axis : int
        Axis along which to extract the range's start and end. Frequency
        axis is indicated by 0, time axis is indicated by 1. Any other value
        will raise exception.

    Returns
    -------
    int, int
        start of the range, end of the range

    Raises
    ------
    ValueError
        If `axis` is different than 0 or 1.
    """
    if axis == 0:
        start = min(map(lambda squeak: squeak.t_start, squeaks))
        end = max(map(lambda squeak: squeak.t_end, squeaks))
    elif axis == 1:
        start = min(map(lambda squeak: squeak.freq_start, squeaks))
        end = max(map(lambda squeak: squeak.freq_end, squeaks))
    else:
        raise ValueError("Axis must be 0 or 1")
    return start, end


def intersection_over_union_elementwise(
    target: List[data_util.SqueakBox], cover: List[data_util.SqueakBox]
) -> Dict[data_util.SqueakBox, Dict[data_util.SqueakBox, float]]:
    """Compute elementwise intersection over union ratio.

    For each element in `target` the function finds elements
    in `cover` that intersects with given element and computes
    the intersection over union ratio.

    Parameters
    ----------
    target : List[data_util.SqueakBox]
        List of squeaks for which the function looks for intersecting squeaks
        in `cover`.
    cover : List[data_util.SqueakBox]
        List of squeaks amongst which the function looks for squeaks
        intersecting with given squeak from `target`.

    Returns
    -------
    Dict[data_util.SqueakBox, Dict[data_util.SqueakBox, float]]
        Dictionary mapping squeaks from `target` to another dictionary with
        squeaks from `cover` that intersect with squeaks from `target`
        and corresponding intersection over union ratio.
    """
    target_sorted = sorted(target, key=lambda squeak: (squeak.t_start, squeak.t_end))
    cover_sorted = sorted(cover, key=lambda squeak: (squeak.t_end, squeak.t_start))

    result: Dict[data_util.SqueakBox, dict] = \
        {target_squeak: dict() for target_squeak in target_sorted}

    j_start = 0
    for target_squeak in target_sorted:

        while (j_start < len(cover_sorted) and
               cover_sorted[j_start].t_end < target_squeak.t_start):
            j_start += 1

        for j in range(j_start, len(cover_sorted)):
            cover_squeak = cover_sorted[j]
            if cover_squeak.t_start > target_squeak.t_end:
                break
            iou = _box_iou(target_squeak, cover_squeak)

            if iou > 0:
                result[target_squeak][cover_squeak] = iou

        if j_start >= len(cover_sorted):
            break

    return result


def intersection_over_union_global(
    prediction: List[data_util.SqueakBox],
    ground_truth: List[data_util.SqueakBox],
    axis: Union[int, None] = None,
) -> float:
    """Compute overall intersection over union ratio along given axis.

    Parameters
    ----------
    prediction : List[data_util.SqueakBox]
        List of predicted squeaks.
    ground_truth : List[data_util.SqueakBox]
        List of manually annotated squeaks.
    axis : Union[int, None]
        Axis along which intersection over union ratio will be computed.
        Frequency axis is indicated by 0, time axis is indicated by 1.
        None represents both axes at once.

    Returns
    -------
    float
        Value of intersection over union ratio.

    Raises
    ------
    ValueError
        If `axis` is different than 0, 1 or None.
    """
    squeaks = prediction + ground_truth
    if len(squeaks) == 0:
        warnings.warn("no squeaks provided for iou computation", RuntimeWarning)
        return 0

    if axis is None:
        freq_start, freq_end = _get_squeaks_range(squeaks, axis=1)
        t_start, t_end = _get_squeaks_range(squeaks, axis=0)

        prediction_bit = _to_bit_map(prediction, freq_start, freq_end, t_start, t_end)
        ground_truth_bit = _to_bit_map(ground_truth,
                                       freq_start,
                                       freq_end,
                                       t_start,
                                       t_end)

    elif axis == 0:
        t_start, t_end = _get_squeaks_range(squeaks, axis=0)

        prediction_bit = _to_bit_map_1d(prediction, t_start, t_end, axis=axis)
        ground_truth_bit = _to_bit_map_1d(ground_truth, t_start, t_end, axis=axis)

    elif axis == 1:
        freq_start, freq_end = _get_squeaks_range(squeaks, axis=1)

        prediction_bit = _to_bit_map_1d(prediction, freq_start, freq_end, axis=axis)
        ground_truth_bit = _to_bit_map_1d(ground_truth, freq_start, freq_end, axis=axis)
    else:
        raise ValueError("axis must be 0, 1 or None")

    intersection = np.logical_and(prediction_bit, ground_truth_bit)
    union = np.logical_or(prediction_bit, ground_truth_bit)
    return intersection.sum() / union.sum()


def coverage(
    squeaks_1: List[data_util.SqueakBox],
    squeaks_2: List[data_util.SqueakBox],
    axis: Union[int, None] = None,
):
    """Compute coverage of boxes from both lists.

    For both lists a percentage of coverage along `axis` is computed.
    Both lists are treated once as a list that is to be covered and once the
    list that covers the other.
    The order in which `squeaks_1` and `squeaks_2` are passed only affects
    the order of returned values.

    Parameters
    ----------
    squeaks_1 : List[data_util.SqueakBox]
        First list of squeaks.
    squeaks_2 : List[data_util.SqueakBox]
        Second list of squeaks.
    axis : int
        Axis that the cover is calculated along. Frequency axis is indicated
        by 0, time axis is indicated by 1. None represents both axes at once.
        Any other value will raise exception.

    Returns
    -------
    Tuple[float, float]
        Tuple with coverages for both lists. The exact format is:
              (Coverage of `squeaks_1` by `squeaks_2`,
               Coverage of `squeaks_2` by `squeaks_1`)

    Raises
    ------
    ValueError
        If `axis` is different than 0, 1 or None.
    """
    if axis in {0, 1}:
        start, end = _get_squeaks_range(squeaks_1 + squeaks_2, axis=axis)
        squeaks_1_bit_map = _to_bit_map_1d(squeaks_1, start=start, end=end, axis=axis)
        squeaks_2_bit_map = _to_bit_map_1d(squeaks_2, start=start, end=end, axis=axis)
    elif axis is None:
        freq_start, freq_end = _get_squeaks_range(squeaks_1 + squeaks_2, axis=1)
        t_start, t_end = _get_squeaks_range(squeaks_1 + squeaks_2, axis=0)

        squeaks_1_bit_map = _to_bit_map(squeaks_1, freq_start, freq_end, t_start, t_end)
        squeaks_2_bit_map = _to_bit_map(squeaks_2, freq_start, freq_end, t_start, t_end)
    else:
        raise ValueError("axis must be 0, 1 or None")

    squeaks_cover_bit_map = np.logical_and(squeaks_1_bit_map, squeaks_2_bit_map)
    squeaks_cover_sum = np.sum(squeaks_cover_bit_map)
    squeaks_1_sum = np.sum(squeaks_1_bit_map)
    squeaks_2_sum = np.sum(squeaks_2_bit_map)

    if np.sum(squeaks_1_bit_map) == 0:
        warnings.warn("`squeaks_1` have 0 area in total", RuntimeWarning)
        cov_1 = 0
    else:
        cov_1 = squeaks_cover_sum / squeaks_1_sum

    if np.sum(squeaks_2_bit_map) == 0:
        warnings.warn("`squeaks_2` have 0 area in total", RuntimeWarning)
        cov_2 = 0
    else:
        cov_2 = squeaks_cover_sum / squeaks_2_sum

    return cov_1, cov_2


def iou_dict_to_plain_list(
    iou_dict: Dict[data_util.SqueakBox, Dict[data_util.SqueakBox, float]],
    mode: str = "max",
) -> List[float]:
    """Transform elementwise iou dict to aggregating list.

    Parameters
    ----------
    iou_dict : Dict[data_util.SqueakBox,
                    Dict[data_util.SqueakBox,
                         float]]
        Elementwise iou dictionary.
    mode : str
        Specifies whether sum or max of iou should be computed.

    Returns
    -------
    List[float]
        Aggregated iou for squeaks in iou_dict keys.

    Raises
    ------
    ValueError
        If `mode` is different max or sum.
    """
    if mode != "max" and mode != "sum":
        raise ValueError("Mode must be 'sum' or 'max'")

    result = []
    if mode == "max":
        for (squeak, intersections) in iou_dict.items():
            if len(intersections.values()) > 0:
                result.append(max(intersections.values()))
            else:
                result.append(0.0)
    else:
        # todo: if still slow, change call to intersection_over_union_global
        for (squeak, intersections) in iou_dict.items():
            if len(intersections.values()) > 0:
                result.append(
                    intersection_over_union_global([squeak],
                                                   list(intersections.keys())))
            else:
                result.append(0.0)

    return result


def iou_dict_to_label_lists(
    iou_dict: Dict[data_util.SqueakBox, Dict[data_util.SqueakBox, float]],
    mode: str = "max",
) -> Dict[str, List[float]]:
    """Transform elementwise iou dict to aggregating lists grouped by label.

    Parameters
    ----------
    iou_dict : Dict[data_util.SqueakBox,
                    Dict[data_util.SqueakBox,
                         float]]
        Elementwise iou dictionary.
    mode : str
        Specifies whether sum or max of iou should be computed.

    Returns
    -------
    Dict[str, List[float]]
        Aggregated iou for squeaks in iou_dict keys grouped by squeak labels.

    Raises
    ------
    ValueError
        If `mode` is different max or sum.
    """
    if mode != "max" and mode != "sum":
        raise ValueError("Mode must be 'sum' or 'max'")

    result = {}
    if mode == "max":
        for (squeak, intersections) in iou_dict.items():
            if squeak.label not in result:
                result[squeak.label] = []

            if len(intersections.values()) > 0:
                result[squeak.label].append(max(intersections.values()))
            else:
                result[squeak.label].append(0.0)
    else:
        # todo: if still slow, change call to intersection_over_union_global
        for (squeak, intersections) in iou_dict.items():
            if squeak.label not in result:
                result[squeak.label] = []

            if len(intersections.values()) > 0:
                result[squeak.label].append(
                    intersection_over_union_global([squeak],
                                                   list(intersections.keys())))
            else:
                result[squeak.label].append(0.0)

    return result


def _count_hits(
    iou_dict: Dict[data_util.SqueakBox, Dict[data_util.SqueakBox, float]],
    threshold: float = 0.5,
    mode: str = "max",
) -> int:
    """Count valid detections.

    Parameters
    ----------
    iou_dict : Dict[data_util.SqueakBox,
                    Dict[data_util.SqueakBox,
                         float]]
        Elementwise iou dictionary.
    threshold : float
        If sum/max of elementwise iou (value of iou_dict) for given target
        squeak (key of iou_dict) is greater than
        this value the detection is considered as valid.
    mode : str
        Specifies whether sum or max of iou should be considered.

    Returns
    -------
    int
        Number of valid detections.

    Raises
    ------
    ValueError
        If `mode` is different max or sum.
    """
    if mode != "max" and mode != "sum":
        raise ValueError("Mode must be 'sum' or 'max'")

    result = 0
    if mode == "max":
        for (squeak, intersections) in iou_dict.items():

            if (len(intersections.values()) > 0 and
                    max(intersections.values()) >= threshold):
                result += 1
    else:
        # todo: if still slow, change call to intersection_over_union_global
        for (squeak, intersections) in iou_dict.items():

            if (len(intersections.values()) > 0 and intersection_over_union_global(
                [squeak], list(intersections.keys())) >= threshold):
                result += 1

    return result


def _count_hits_per_label(
    iou_dict: Dict[data_util.SqueakBox, Dict[data_util.SqueakBox, float]],
    threshold: float = 0.5,
    mode: str = "max",
) -> Dict[str, int]:
    """Count valid detections and group them by squeak label.

    Parameters
    ----------
    iou_dict : Dict[data_util.SqueakBox,
                    Dict[data_util.SqueakBox,
                         float]]
        Elementwise iou dictionary. Labels are extracted from squeaks
        that are outer dictionary keys.
    threshold : float
        If sum/max of elementwise iou (value of `iou_dict`) for given target
        squeak (key of `iou_dict`) is greater than
        this value the detection is considered as valid.
    mode : str
        Specifies whether sum or max of iou should be considered.

    Returns
    -------
    Dict[str, int]
        Number of valid detections grouped by squeak label.

    Raises
    ------
    ValueError
        If `mode` is different max or sum.
    """
    if mode != "max" and mode != "sum":
        raise ValueError("Mode must be 'sum' or 'max'")

    result = {}
    if mode == "max":
        for (squeak, intersections) in iou_dict.items():

            if squeak.label not in result:
                result[squeak.label] = 0

            if (len(intersections.values()) > 0 and
                    max(intersections.values()) >= threshold):
                result[squeak.label] += 1
    else:
        # todo: if still slow, change call to intersection_over_union_global
        for (squeak, intersections) in iou_dict.items():

            if squeak.label not in result:
                result[squeak.label] = 0

            if (len(intersections.values()) > 0 and intersection_over_union_global(
                [squeak], list(intersections.keys())) >= threshold):
                result[squeak.label] += 1

    return result


def detection_precision(
    ground_truth: Union[List[data_util.SqueakBox], None] = None,
    prediction: Union[List[data_util.SqueakBox], None] = None,
    pred_given_truth_iou_dict: Union[Dict[data_util.SqueakBox,
                                          Dict[data_util.SqueakBox, float]],
                                     None] = None,
    truth_given_pred_iou_dict: Union[Dict[data_util.SqueakBox,
                                          Dict[data_util.SqueakBox, float]],
                                     None] = None,
    threshold: float = 0.5,
    mode: str = "max",
) -> float:
    """Compute overall detection precision.

    Parameters
    ----------
    ground_truth : Union[List[data_util.SqueakBox], None]
        List of ground truth detection of squeaks.
        If None then `pred_given_truth_iou_dict`
        and `truth_given_pred_iou_dict` must be specified.
    prediction : Union[List[data_util.SqueakBox], None]
        List of detected squeaks. If None then `pred_given_truth_iou_dict`
        and `truth_given_pred_iou_dict` must be specified.
    pred_given_truth_iou_dict : Union[Dict[data_util.SqueakBox,
                                           Dict[data_util.SqueakBox,
                                                float]],
                                      None]
        Elementwise iou dictionary computed for target=ground_truth
        and cover=prediction. If None then `ground_truth` and `prediction`
        must be specified.
    truth_given_pred_iou_dict : Union[Dict[data_util.SqueakBox,
                                           Dict[data_util.SqueakBox,
                                                float]],
                                      None]
        Elementwise iou dictionary computed for target=prediction
        and cover=ground_truth. If None then `ground_truth` and `prediction`
        must be specified.
    threshold : float
        If sum/max of elementwise iou for given target squeak is greater than
        this value the detection is considered as valid.
    mode : str
        Specifies whether sum or max of iou should be considered.

    Returns
    -------
    float
        Precision of detections.

    Raises
    ------
    ValueError
        If (`prediction` or `ground_truth) and (`pred_given_truth_iou_dict`
        or `truth_given_pred_iou_dict`) are not specified.
    """
    if (prediction is None or
            ground_truth is None) and (pred_given_truth_iou_dict is None or
                                       truth_given_pred_iou_dict is None):
        raise ValueError(("prediction and ground_truth "
                          "or pred_given_truth_iou_dict and truth_given_pred_iou_dict "
                          "must be defined"))

    if pred_given_truth_iou_dict is None or truth_given_pred_iou_dict is None:
        pred_given_truth_iou_dict = intersection_over_union_elementwise(
            target=ground_truth, cover=prediction)
        truth_given_pred_iou_dict = intersection_over_union_elementwise(
            target=prediction, cover=ground_truth)

    tp = _count_hits(pred_given_truth_iou_dict, threshold=threshold, mode=mode)
    fp = len(truth_given_pred_iou_dict) - _count_hits(
        truth_given_pred_iou_dict, threshold=threshold, mode=mode)

    if fp + tp == 0:
        warnings.warn("no positives present for precision computation", RuntimeWarning)
        return 0

    return tp / (fp + tp)


def detection_recall(
    ground_truth: Union[List[data_util.SqueakBox], None] = None,
    prediction: Union[List[data_util.SqueakBox], None] = None,
    pred_given_truth_iou_dict: Union[Dict[data_util.SqueakBox,
                                          Dict[data_util.SqueakBox, float]],
                                     None] = None,
    threshold: float = 0.5,
    mode: str = "max",
) -> Tuple[float, Dict[str, float]]:
    """Compute overall detection recall and label recalls.

    Parameters
    ----------
    ground_truth : Union[List[data_util.SqueakBox], None]
        List of ground truth detection of squeaks.
        If None then `pred_given_truth_iou_dict` must be specified.
    prediction : Union[List[data_util.SqueakBox], None]
        List of detected squeaks. If None then `pred_given_truth_iou_dict`
        must be specified.
    pred_given_truth_iou_dict: Union[Dict[data_util.SqueakBox,
                                          Dict[data_util.SqueakBox,
                                               float]],
                                     None]
        Elementwise iou dictionary computed for target=ground_truth
        and cover=prediction. If None then `ground_truth` and `prediction`
        must be specified.
    threshold : float
        If sum/max of elementwise iou for given target squeak is greater than
        this value the detection is considered as valid.
    mode : str
        Specifies whether sum or max of iou should be considered.

    Returns
    -------
    float
        Recall of all provided detections.
    Dict[str, float]
        Recall for squeaks grouped by label.

    Raises
    ------
    ValueError
        If (`prediction` or `ground_truth) and `pred_given_truth_iou_dict`
        are not specified.
    """
    if (prediction is None or
            ground_truth is None) and pred_given_truth_iou_dict is None:
        raise ValueError(("prediction and ground_truth "
                          "or pred_given_truth_iou_dict must be defined"))

    if pred_given_truth_iou_dict is None:
        pred_given_truth_iou_dict = intersection_over_union_elementwise(
            target=ground_truth, cover=prediction)

    if len(pred_given_truth_iou_dict) == 0:
        warnings.warn("no true detections for recall computation", RuntimeWarning)
        overall_recall = 0
    else:
        tp = _count_hits(pred_given_truth_iou_dict, threshold=threshold, mode=mode)
        overall_recall = tp / len(pred_given_truth_iou_dict)

    labels_tp = _count_hits_per_label(pred_given_truth_iou_dict,
                                      threshold=threshold,
                                      mode=mode)

    labels_positives = {}
    for (squeak, intersections) in pred_given_truth_iou_dict.items():

        if squeak.label not in labels_positives:
            labels_positives[squeak.label] = 1
        else:
            labels_positives[squeak.label] += 1

    labels_recalls = {}
    for label in labels_positives.keys():
        labels_recalls[label] = labels_tp[label] / labels_positives[label]

    return overall_recall, labels_recalls
