"""Module containing utilities for loading and processing sound data."""
from __future__ import annotations

import time
import pathlib
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Dict
import requests

import numpy as np
import pandas as pd
from rtree import Index
from skimage import measure

import mouse.utils.constants as const
from mouse.utils import sound_util

# Storage classes


@dataclass
class DataFolder:
    """Storage class for a set of recordings."""

    wavs: List[pathlib.Path]
    df: pd.DataFrame
    folder_path: pathlib.Path
    signals: List[sound_util.SignalData]

    def get_signal(self, name: str) -> Optional[sound_util.SignalData]:
        """Get signal based on its filename `name`."""
        for signal in self.signals:
            if signal.name == name:
                return signal
        return None


@dataclass
class SqueakBox:
    """Storage class for bounding boxes.

    This class stores time and frequency values described by pixels on an associated
    spectrogram.
    """

    freq_start: int
    freq_end: int
    t_start: int
    t_end: int
    label: Optional[str] = None

    def __hash__(self):
        """Calculate hash."""
        return hash(
            (self.freq_start, self.freq_end, self.t_start, self.t_end, self.label))

    def __iter__(self):
        """Create iterator."""
        return iter(
            (self.freq_start, self.freq_end, self.t_start, self.t_end, self.label))


@dataclass(frozen=True)
class SqueakBoxSI:
    """Storage class for bounding boxes.

    This class stores time and frequency values described by SI units ([s] and [Hz]).
    """

    freq_start: float
    freq_end: float
    t_start: float
    t_end: float
    label: Optional[str] = None

    def to_dict(self):
        """Convert `SqueakBoxSI` to dict."""
        return {
            "time_start": self.t_start,
            "time_end": self.t_end,
            "freq_start": self.freq_start,
            "freq_end": self.freq_end,
            "label": self.label,
        }

    def to_squeak_box(self, spec_data: sound_util.SpectrogramData):
        """Convert `SqueakBoxSI` to `SqueakBox`."""
        t_start = np.abs(spec_data.times - self.t_start).argmin()
        t_end = np.abs(spec_data.times - self.t_end).argmin()
        freq_start = np.abs(spec_data.freqs - self.freq_start).argmin()
        freq_end = np.abs(spec_data.freqs - self.freq_end).argmin()
        return SqueakBox(
            freq_start=int(freq_start),
            freq_end=int(freq_end),
            t_start=int(t_start),
            t_end=int(t_end),
            label=self.label,
        )

    @staticmethod
    def from_dict(**values) -> SqueakBoxSI:
        """Load `SqueakBoxSI` from dict."""
        annotation = SqueakBoxSI(**values)
        return annotation

    @staticmethod
    def from_squeak_box(squeak_box: SqueakBox,
                        spec_data: sound_util.SpectrogramData) -> SqueakBoxSI:
        """Load `SqueakBoxSI` from `SqueakBox`."""
        t_start, t_end = spec_data.times[[squeak_box.t_start, squeak_box.t_end]]
        freq_start, freq_end = spec_data.freqs[
            [squeak_box.freq_start, squeak_box.freq_end]
        ]
        annotation = SqueakBoxSI(
            t_start=float(t_start),
            t_end=float(t_end),
            freq_start=float(freq_start),
            freq_end=float(freq_end),
            label=squeak_box.label if squeak_box.label else None,
        )

        return annotation


@dataclass(frozen=True)
class SignalNoise:
    """Storage class for noise information.

    Time for `start` and `end` of the noise is given in seconds.
    """

    config_id: str
    start: float
    end: float


# data loading utilities


def load_data_paths(
    folder: pathlib.Path,) -> Tuple[List[pathlib.Path], List[pathlib.Path]]:
    """Load paths to data-files from `folder`.

    Parameters
    ----------
    folder : pathlib.Path
        Folder containing data

    Returns
    -------
    list
        Files that end with 'wav'
    list
        Files that end with 'txt'
    """
    txt = []
    wav = []

    for file in folder.iterdir():
        if file.is_file():
            if file.name.endswith("wav"):
                wav.append(file)
            elif file.name.endswith("txt"):
                txt.append(file)

    return sorted(wav), sorted(txt)


def load_table(csv_path: pathlib.Path) -> pd.DataFrame:
    """Load and preprocess csv file with label information.

    Parameters
    ----------
    csv_path : `pathlib.Path`
    """
    df = pd.read_csv(csv_path, delimiter="\t")
    df.rename(columns={"NOTE": const.COL_USV_TYPE}, inplace=True)
    df.rename(columns=lambda x: x.replace(" ", "_") if isinstance(x, str) else x,
              inplace=True)

    def map_usv_type(val):
        """Make names of USVs uniform."""
        result = str(val).lower()
        if result == "22-khz call":
            result = "22-khz"
        return result

    df[const.COL_USV_TYPE] = df[const.COL_USV_TYPE].map(map_usv_type)
    return df


def load_data(sources: List[pathlib.Path],
              with_labels: bool = True) -> List[DataFolder]:
    """Load .wav files and (optional) dataframe with metadata.

    Loaded label data is preprocessed, this includes renaming columns,
    unifying USVs' names, changing USVs' start and end times.

    Parameters
    ----------
    sources:
        List of data-folders' paths.
    with_labels:
        Flag indicating whether labels should be loaded. If labels aren't
        present `with_labels` should be false.

    Returns
    -------
    List[DataFolder]
    """
    data_folders = []
    paths = [(load_data_paths(folder), folder) for folder in sources]
    for (wavs, txt), folder in paths:
        signals = [sound_util.SignalData(wav) for wav in wavs]

        if with_labels:
            df = load_table(txt[0])

            # fix recordings' start/end times
            file_time_df = df[[const.COL_BEGIN_FILE, const.COL_BEGIN_TIME]]
            # recordings are ordered by the beginnings of their recorded USVs
            order = (file_time_df.groupby(by=const.COL_BEGIN_FILE).agg(
                np.median).sort_values(const.COL_BEGIN_TIME).index)

            durations = {s.name: s.duration for s in signals}
            shift = durations[order[0]]
            for i, name in enumerate(order[1:], start=1):
                idx = df[const.COL_BEGIN_FILE] == name
                df.loc[idx, const.COL_BEGIN_TIME] -= shift
                assert np.all(df.loc[idx, const.COL_BEGIN_TIME] > 0)
                df.loc[idx, const.COL_END_TIME] -= shift
                shift += durations[order[i]]

        else:
            df = None

        data_folders.append(
            DataFolder(wavs=wavs, df=df, folder_path=folder, signals=signals))

    return data_folders


def load_squeak_boxes(
    df: Union[DataFolder, pd.DataFrame],
    filename: Union[str, pathlib.Path],
    spec: sound_util.SpectrogramData,
) -> List[SqueakBox]:
    """Calculate bounding boxes for each squeak based on `df`.

    Bounding boxes will consist of indices that bound the squeaks on the
    spectrogram `spec`.

    Parameters
    ----------
    df : Union[DataFolder, pd.DataFrame]
        Object to extract bounding boxes from. If `DataFolder` is passed
        `df.df` can't be `None`.
    filename : Union[str, pathlib.Path]
        Name of the file or path to the file associated with `spec`.
    spec: sound_data.SpectrogramData
        The spectrogram with squeaks.

    Returns
    -------
    List[SqueakBox]
        Bounding boxes for each squeak.
    """
    if isinstance(df, DataFolder):
        if df.df is None:
            raise ValueError("DataFolder must have DataFrame attribute (`df`).")
        df = df.df

    if isinstance(filename, pathlib.Path):
        name = filename.name
    else:
        name = filename

    file_idx = df[const.COL_BEGIN_FILE] == name

    if not file_idx.any():
        raise ValueError("`filename` not present in DataFrame.")

    file_df = df[file_idx]

    low_frequency = file_df[const.COL_LOW_FREQ].to_numpy().squeeze()
    high_frequency = file_df[const.COL_HIGH_FREQ].to_numpy().squeeze()
    begin_time = file_df[const.COL_BEGIN_TIME].to_numpy().squeeze()
    end_time = file_df[const.COL_END_TIME].to_numpy().squeeze()

    # move bounding boxes to 'target spectrogram' area
    low_frequency = np.clip(low_frequency, spec.freq_start, spec.freq_end)
    high_frequency = np.clip(high_frequency, spec.freq_start, spec.freq_end)
    begin_time = np.clip(begin_time, spec.t_start, spec.t_end)
    end_time = np.clip(end_time, spec.t_start, spec.t_end)

    # determining indices of squeaks that are inside 'target spectrogram'
    good_frequency_idx = low_frequency < high_frequency
    good_time_idx = begin_time < end_time
    good_idx = np.logical_and(good_frequency_idx, good_time_idx)

    labels = file_df.loc[good_idx, const.COL_USV_TYPE]

    # frequency/time to index conversion
    shifted_low_freq = low_frequency[good_idx] - spec.freq_start
    low_freq_idx = np.floor(shifted_low_freq / spec.freq_pixel_span).astype(np.int32)
    shifted_high_freq = high_frequency[good_idx] - spec.freq_start
    high_freq_idx = np.ceil(shifted_high_freq / spec.freq_pixel_span).astype(np.int32)
    shifted_begin_time = begin_time[good_idx] - spec.t_start
    begin_time_idx = np.floor(shifted_begin_time / spec.time_pixel_span).astype(
        np.int32)
    shifted_end_time = end_time[good_idx] - spec.t_start
    end_time_idx = np.ceil(shifted_end_time / spec.time_pixel_span).astype(np.int32)

    return [
        SqueakBox(
            freq_start=low_freq_idx[i],
            freq_end=high_freq_idx[i],
            t_start=begin_time_idx[i],
            t_end=end_time_idx[i],
            label=labels.iloc[i],
        ) for i in range(len(low_freq_idx))
    ]


# processing utilities


def filter_boxes(
    spec: sound_util.SpectrogramData,
    boxes: List[SqueakBox],
    min_time_span: float = 0.002,
    min_freq_span: int = 2000,
) -> List[SqueakBox]:
    """Filter boxes by their width and length.

    If a box is too short or to narrow it will be removed.

    Parameters
    ----------
    spec : sound_util.SpectrogramData
        Spectrogram on which `boxes` were detected.
    boxes : List[SqueakBox]
        Boxes that were detected on spectrogram `spec`.
    min_time_span : float
        Minimum required width (in seconds) of a box.
    min_freq_span : int
        Minimum required height (in Hz) of a box.

    Returns
    -------
    List[SqueakBox]
        A list of filtered boxes.
    """

    def is_valid(box: SqueakBox):
        t_span = spec.times[box.t_end] - spec.times[box.t_start]
        f_span = spec.freqs[box.freq_end] - spec.freqs[box.freq_start]
        return t_span > min_time_span and f_span > min_freq_span

    return [b for b in boxes if is_valid(b)]


def get_abs_distances(first: SqueakBox, second: SqueakBox) -> Tuple[int, int]:
    """Calculate absolute difference between `first` and `second`.

    Calculations are based on pixels.
    Distances in time and frequency are calculated separately.

    Returns
    -------
        int: absolute difference in time
        int: absolute difference in frequency
    """
    t_distance = min(abs(first.t_start - second.t_end),
                     abs(second.t_start - first.t_end))
    if (first.t_start <= second.t_start <= first.t_end or
            second.t_start <= first.t_start <= second.t_end):
        t_distance = 0

    freq_distance = min(abs(first.freq_start - second.freq_end),
                        abs(second.freq_start - first.freq_end))
    if (first.freq_start <= second.freq_start <= first.freq_end or
            second.freq_start <= first.freq_start <= second.freq_end):
        freq_distance = 0

    return (t_distance, freq_distance)


def merge_boxes(
    spec: sound_util.SpectrogramData,
    squeaks: List[SqueakBox],
    delta_freq: float,
    delta_time: float,
    label: Optional[Union[int, str]] = None,
    scores: Optional[Union[np.ndarray, List[float]]] = None,
) -> Union[List[SqueakBox], Tuple[List[SqueakBox], Union[np.ndarray, List]]]:
    """Merge boxes that are sufficiently close.

    Parameters
    ----------
    spec : sound_util.SpectrogramData
        Used to define "sufficiently close".
    squeaks : List[SqueakBox]
        List of detected squeaks to be merged.
    delta_freq : float
        Maximum distance between boxes in
        frequency direction to still be considered "overlapping".
    delta_time : float
        Maximum distance between boxes in
        time direction to still be considered "overlapping".
    label : Optional[Union[int, str]]
        If all squeaks are considered to be from same class
        this parameter sets merged boxes label to this value.
    scores : Optional[Union[np.ndarray, List]]
        If present this function will return score for each
        merged box equal to mean of `scores` that create this merge.

    Returns
    -------
        List[SqueakBox]: merged boxes
            or (if scores are present)
        List[SqueakBox]: merged boxes
        Union[np.ndarray, List] : scores for merged boxes
    """
    delta_time_pixels = spec.time_to_pixels(delta_time)
    delta_freq_pixels = spec.freq_to_pixels(delta_freq)

    index = Index()
    for idx, squeak in enumerate(squeaks):
        index.insert(
            id=idx,
            coordinates=[
                squeak.t_start - delta_time_pixels,
                squeak.freq_start - delta_freq_pixels,
                squeak.t_end + delta_time_pixels,
                squeak.freq_end + delta_freq_pixels,
            ],
        )
    parents = [i for i in range(len(squeaks))]

    # we couldn't find real example that overflows recurrence stack
    # if you can, please open an issue
    # chain of boxes will be accumulated immediately
    # due to merging implementation
    def get_parent(i):
        if parents[i] == i:
            return i
        parents[i] = get_parent(parents[i])
        return parents[i]

    def merge_parents(i, j):
        i = get_parent(i)
        j = get_parent(j)
        if i != j:
            parents[j] = i

    for i, squeak in enumerate(squeaks):
        intersections = index.intersection(
            [squeak.t_start, squeak.freq_start, squeak.t_end, squeak.freq_end])
        for j in intersections:
            merge_parents(i, j)

    aggregated_scores = defaultdict(lambda: list())
    aggregated_squeaks = defaultdict(lambda: list())
    for i, squeak in enumerate(squeaks):
        aggregated_squeaks[get_parent(i)].append(squeak)
        if scores is not None:
            aggregated_scores[get_parent(i)].append(scores[i])

    merged_scores = []
    merged_squeaks = []

    for key, squeaks in aggregated_squeaks.items():
        freq_starts, freq_ends, time_starts, time_ends, _ = list(zip(*squeaks))
        merged_squeaks.append(
            SqueakBox(
                freq_start=min(freq_starts),
                freq_end=max(freq_ends),
                t_start=min(time_starts),
                t_end=max(time_ends),
                label=str(label) if label else None,
            ))
        if scores is not None:
            s = aggregated_scores[key]
            merged_scores.append(sum(s) / len(s))

    return merged_squeaks if scores is None else (merged_squeaks, scores)


def clip_boxes(
    spec: sound_util.SpectrogramData,
    boxes: List[SqueakBox],
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
    freq_start: Optional[float] = None,
    freq_end: Optional[float] = None,
) -> List[SqueakBox]:
    """Adjust `boxes` for clipped spectrogram.

    Returned boxes correctly index squeaks on a spectrogram calculated with
    `sound_util.clip_spectrogram(spec, t_start, t_end, freq_start, freq_end).`
    Boxes that would be outside the clipped spectrogram are removed.
    """
    if t_start is None:
        t_start = spec.t_start
    if t_end is None:
        t_end = spec.t_end
    if freq_start is None:
        freq_start = spec.freq_start
    if freq_end is None:
        freq_end = spec.freq_end

    t_pixel_start = spec.time_to_pixels(t_start - spec.t_start)
    t_pixel_end = spec.time_to_pixels(t_end - spec.t_start)
    freq_pixel_start = spec.freq_to_pixels(freq_start - spec.freq_start)
    freq_pixel_end = spec.freq_to_pixels(freq_end - spec.freq_start)
    filtered_boxes = filter(
        lambda box: (box.t_end > t_pixel_start and box.t_start < t_pixel_end and box.
                     freq_start < freq_pixel_end and box.freq_end > freq_pixel_start),
        boxes,
    )
    return [
        SqueakBox(
            t_start=max(box.t_start, t_pixel_start) - t_pixel_start,
            t_end=min(box.t_end, t_pixel_end) - t_pixel_start,
            freq_start=max(box.freq_start, freq_pixel_start) - freq_pixel_start,
            freq_end=min(box.freq_end, freq_pixel_end) - freq_pixel_start,
            label=box.label,
        ) for box in filtered_boxes
    ]


def clip_spec_and_boxes(
    spec: sound_util.SpectrogramData,
    boxes: Union[List[SqueakBox], List[List[SqueakBox]]],
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
    freq_start: Optional[float] = None,
    freq_end: Optional[float] = None,
) -> Tuple[sound_util.SpectrogramData, List[SqueakBox]]:
    """Clip spectrogram and boxes according to specified values."""
    if len(boxes) > 0 and isinstance(boxes[0], list):
        result_nested = True
        _boxes: List[List[SqueakBox]] = boxes
    else:
        result_nested = False
        _boxes: List[List[SqueakBox]] = [boxes]

    clipped_spec = sound_util.clip_spectrogram(spec,
                                               freq_start=freq_start,
                                               freq_end=freq_end,
                                               t_start=t_start,
                                               t_end=t_end)
    result_boxes = []
    for unclipped_boxes in _boxes:
        clipped_boxes = clip_boxes(
            spec=spec,
            boxes=unclipped_boxes,
            t_start=clipped_spec.t_start,
            t_end=clipped_spec.t_end,
            freq_start=clipped_spec.freq_start,
            freq_end=clipped_spec.freq_end,
        )
        result_boxes.append(clipped_boxes)
    if result_nested:
        return clipped_spec, result_boxes
    else:
        return clipped_spec, result_boxes[0]


def find_bounding_boxes(mask: np.ndarray, min_side_length: int = 1) -> List[SqueakBox]:
    """Find bounding boxes that bound interesting areas on mask.

    Parameters
    ----------
    mask : np.ndarray
        A binary mask with interesting areas marked by ones.
    min_side_length : int
        Only boxes greater or equal than a square with sides of length
         `min_side_length` will be returned.

    Returns
    -------
    List[SqueakBox]
    """
    label_mask, region_count = measure.label(
        mask, background=0, connectivity=1, return_num=True
    )
    result = []
    regions = defaultdict(list)
    with np.nditer(label_mask, flags=["multi_index"], op_flags=["readonly"]) as it:
        for label in it:
            if label != -1:
                regions[label.item()].append(it.multi_index)

    for label in range(1, region_count + 1):
        region_idx = np.array(regions[label])
        lower_left = np.min(region_idx, axis=0)
        upper_right = np.max(region_idx, axis=0)
        if np.all((upper_right - lower_left + 1) >= min_side_length):
            result.append(
                SqueakBox(
                    freq_start=lower_left[0],
                    freq_end=upper_right[0],
                    t_start=lower_left[1],
                    t_end=upper_right[1],
                    label=None,
                ))
    return result


def download_file(url: str, output_path: str):
    """Download file from specified url.

    Parameters
    ----------
    url : str
        Link to file.
    output_path : str
        Path under which to save the file pointed by `url`
    """
    file_content = requests.get(url)
    for retry_timeout in [1, 30, 30]:
        try:
            file_content.raise_for_status()
            break
        except requests.exceptions.HTTPError as e:
            if e.response.status_code in [429, 500, 502, 503, 504]:
                time.sleep(retry_timeout)
                file_content = requests.get(url)
                continue
            raise
    file_content.raise_for_status()

    with open(output_path, 'wb') as f:
        f.write(file_content.content)
