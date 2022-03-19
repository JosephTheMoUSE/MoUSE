"""(Deprecated) Module implementing entropy-energy USVs search."""
from typing import List, Union

import numpy as np
from scipy import signal, stats

from mouse.utils import data_util
from mouse.utils import sound_util


def _entropy(x):
    x = np.array(x)
    return stats.mstats.gmean(x, axis=0) / np.mean(x, axis=0)


def _moving_average(x, w):
    x = np.pad(x, (w // 2, (w - 1) // 2), "constant", constant_values=(1,))
    return np.convolve(x, np.ones(w), "valid") / w


def _moving_average_2d(x, w_height, w_width):
    x = np.pad(x, ((w_height // 2, (w_height - 1) // 2), (0, 0)), "mean")
    return signal.convolve2d(x, np.ones(
        (w_height, w_width)), "valid").flatten() / (w_height * w_width)


def _binary_mask_to_ranges(binary_mask):
    beginnings = [
        i + 1
        for i in range(binary_mask.shape[0] - 1)
        if not binary_mask[i] and binary_mask[i + 1]
    ]

    ends = [
        i - 1
        for i in range(1, binary_mask.shape[0])
        if binary_mask[i - 1] and not binary_mask[i]
    ]

    if binary_mask[0]:
        beginnings = [0] + beginnings

    if binary_mask[-1]:
        ends.append(binary_mask.shape[0])

    return list(zip(beginnings, ends))


def _mark_low_entropy(spectrogram, window, threshold):
    avg_entropy = _moving_average(_entropy(spectrogram), window)
    low_entropy = avg_entropy < threshold

    return _binary_mask_to_ranges(low_entropy)


def _mark_high_energy_rows(spectrogram, energy_window, energy_threshold):
    if not isinstance(spectrogram, np.ndarray):
        spectrogram = spectrogram.numpy()
    mean_energy = np.mean(np.power(spectrogram, 2))
    width = spectrogram.shape[1]
    high_energy = (_moving_average_2d(np.power(spectrogram, 2), energy_window, width) >
                   energy_threshold * mean_energy)
    return _binary_mask_to_ranges(high_energy)


def _mark_low_entropy_high_energy(
    spectrogram,
    freqs,
    freq_cutoff,
    entropy_window,
    entropy_threshold,
    energy_window,
    energy_threshold,
):
    spectrogram_ = spectrogram[freqs > freq_cutoff, :]
    delta_freq = spectrogram.shape[0] - spectrogram_.shape[0]
    columns_to_check = _mark_low_entropy(spectrogram_,
                                         entropy_window,
                                         entropy_threshold)
    squeak_boxes = []
    for cols in columns_to_check:
        row_ranges = _mark_high_energy_rows(spectrogram_[:, cols[0]:cols[1] + 1],
                                            energy_window,
                                            energy_threshold)
        for rows in row_ranges:
            squeak_boxes.append(
                data_util.SqueakBox(
                    freq_start=rows[0] + delta_freq,
                    freq_end=rows[1] + delta_freq,
                    t_start=cols[0],
                    t_end=cols[1],
                    label=None,
                ))
    return squeak_boxes


def _filter_by_ratio(squeaks, ratio_cutoff):
    result = []
    for squeak in squeaks:
        width = squeak.t_end - squeak.t_start
        height = squeak.freq_end - squeak.freq_start

        if width > 0 and height / width < ratio_cutoff:
            result.append(squeak)
    return result


def find_USVs(
    spec: sound_util.SpectrogramData,
    freq_cutoff: int = 22000,
    entropy_window: int = 5,
    entropy_threshold: float = 0.71,
    energy_window: int = 18,
    energy_threshold: float = 2.1,
    filter: bool = False,
    filter_ratio: Union[float, None] = None,
    merge: bool = False,
    merge_delta_freq: Union[int, None] = None,
    merge_delta_time: Union[int, None] = None,
) -> List[data_util.SqueakBox]:
    """Find USVs on spectrogram `spec` using entropy-energy filtering.

    Parameters
    ----------
    spec : sound_util.SpectrogramData
        A spectrogram that is searched for USVs.
    freq_cutoff : int
        Frequencies lower than this value will be ignored.
    entropy_window : int
        Window size for entropy moving average.
    entropy_threshold : float
        Spectrogram fragments with entropy level below this value will
        be considered as possible squeaks.
    energy_window : int
        Window size for energy moving average.
    energy_threshold : float
        Spectrogram fragments with energy level above this value will
        be considered as possible squeaks.
    filter : bool
        If True, USVs' bounding boxes will be filtered based on boxes'
        sides ratio. Ratio may be specified by `filter_ratio` parameter.
    filter_ratio : Union[float, None]
        If `filter` is set to True then only bounding boxes with
        height / width ration smaller than filter_ratio will be returned.
    merge : bool
        If True, USVs' bounding boxes closer than `merge_delta_freq`
        in frequency domain or `merge_delta_time` in time domain
        will be merged.
    merge_delta_freq: int
        Defines merging details as described in `merge` argument.
    merge_delta_time: int
        Defines merging details as described in `merge` argument.

    Returns
    -------
    List[data_util.SqueakBox]
        List of detected USVs' bounding boxes.

    Raises
    ------
    ValueError
        If `merge` is set to True and `merge_delta_freq` or `merge_delta_time`
        are not specified. If `filter` is set to True and `filter_ratio`
        is not specified.
    """
    marked_boxes = _mark_low_entropy_high_energy(
        spec.spec,
        spec.freqs,
        freq_cutoff=freq_cutoff,
        entropy_window=entropy_window,
        entropy_threshold=entropy_threshold,
        energy_window=energy_window,
        energy_threshold=energy_threshold,
    )

    if merge:
        if merge_delta_time is None or merge_delta_freq is None:
            raise ValueError("merge_delta_time and merge_delta_freq must be specified")
        else:
            marked_boxes = data_util.merge_boxes(
                spec=spec,
                squeaks=marked_boxes,
                delta_freq=merge_delta_freq * spec.get_freq_pixel_span(),
                delta_time=merge_delta_time * spec.get_time_pixel_span(),
            )

    if filter:
        if filter_ratio is None:
            raise ValueError("filter_ratio must be specified")
        else:
            marked_boxes = _filter_by_ratio(marked_boxes, ratio_cutoff=filter_ratio)

    return marked_boxes
