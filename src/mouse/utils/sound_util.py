"""Module containing utilities for dealing with sound data."""

import pathlib
from typing import Optional

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T

# Storage classes


class SpectrogramData:
    """A class used to manage spectrogram and it's metadata."""

    def __init__(self, spec: torch.Tensor, times: np.array, freqs: np.array):
        """Create SpectrogramData object.

        Parameters
        ----------
        spec : torch.Tensor
            The 2d spectrogram array.
        times : np.array
            Times along the second axis of the spectrogram `spec`.
        freqs : np.array
            Frequencies along the first axis of the spectrogram `spec`.
        """
        self.spec: torch.Tensor = spec
        self.times: np.array = times
        self.freqs: np.array = freqs

    def get_height(self) -> int:
        """Get the height (frequency span) of the spectrogram."""
        if self.spec is not None:
            return self.spec.shape[0]
        return self.freqs.shape[0]

    def get_width(self) -> int:
        """Get the width (time span) of the spectrogram."""
        if self.spec is not None:
            return self.spec.shape[1]
        return self.times.shape[0]

    def time_to_pixels(self, t: float) -> int:
        """Calculate the minimum number of pixels that would span time `t`."""
        t_pixel = self.time_pixel_span
        return int(np.ceil(t / t_pixel))

    def freq_to_pixels(self, f: float) -> int:
        """Calculate the minimum number of pixels that would span `f`."""
        f_pixel = self.freq_pixel_span
        return int(np.ceil(f / f_pixel))

    @property
    def freq_end(self) -> float:
        """Return maximum frequency present on the spectrogram."""
        return self.freqs[-1]

    @property
    def freq_start(self) -> float:
        """Return minimum frequency present on the spectrogram."""
        return self.freqs[0]

    @property
    def t_end(self) -> float:
        """Return maximum time present on the spectrogram."""
        return self.times[-1]

    @property
    def t_start(self) -> float:
        """Return minimum time present on the spectrogram."""
        return self.times[0]

    @property
    def freq_span(self) -> float:
        """Return frequency span of the spectrogram."""
        return self.freqs[-1] - self.freqs[0]

    @property
    def time_span(self) -> float:
        """Return time span of the spectrogram."""
        return self.times[-1] - self.times[0]

    @property
    def freq_pixel_span(self) -> float:
        """Return span of frequencies represented by one pixel."""
        return self.freq_span / self.get_height()

    @property
    def time_pixel_span(self) -> float:
        """Return span of time represented by one pixel."""
        return self.time_span / self.get_width()


class SignalData:
    """A class used to manage signal and it's metadata."""

    def __init__(self, path: pathlib.Path):
        """Create `SignalData` object based on waveform file.

        Parameters
        ----------
        path : `pathlib.Path`
            path to waveform (.wav) file
        """
        signal, sample_rate = torchaudio.load(path)
        signal = signal.squeeze()
        self.signal: torch.Tensor = signal  # raw signal data
        self.sample_rate: int = sample_rate

        # length of the recording in seconds
        self.duration = self.signal.shape[0] / sample_rate

        self.name: str = path.name
        self.folder: str = path.parent.name

    def __repr__(self):
        """Represent `SignalData` with its filename and duration."""
        return f"name: {self.name}, length: {self.duration} [s]"


def spectrogram(
    signal: torch.Tensor,
    sample_rate=250000,
    spec_calculator: T.Spectrogram = None,
    n_fft=512,
    win_length=512,
    hop_length=256,
    power=1,
    **kwargs,
) -> SpectrogramData:
    """Create spectrogram from signal array."""
    if spec_calculator is None:
        spec_calculator = T.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            power=power,
            **kwargs,
        )

    signal = signal.squeeze()
    spec = spec_calculator(signal)
    times = np.arange(spec.shape[1]) * hop_length / sample_rate
    freqs = np.fft.rfftfreq(n_fft, 1 / sample_rate)

    return SpectrogramData(spec=spec, times=times, freqs=freqs)


def signal_spectrogram(
    signal_data: SignalData,
    start: float = 0.0,
    end: float = 1.0,
    spec_calculator: T.Spectrogram = None,
    n_fft=512,
    win_length=512,
    hop_length=256,
    power=1,
    **kwargs,
) -> SpectrogramData:
    """Create spectrogram from SignalData object.

    Parameters
    ----------
    signal_data : `SignalData`
        object to create spectrogram from
    start : float
        number from range [0., 1.] indicating which part of signal to use
    end : float
        number from range [0., 1.] indicating which part of signal to use

    Returns
    -------
    SpectrogramData
        Container with the spectrogram
    """
    if not (0.0 <= start <= 1.0 and 0.0 <= end <= 1.0 and start < end):
        raise ValueError(
            "'start' and 'end' should be from range [0., 1.] and 'start' must "
            "be smaller than 'end'")

    sample_rate = signal_data.sample_rate
    start_idx = int(signal_data.signal.shape[0] * start)
    end_idx = int(signal_data.signal.shape[0] * end)
    signal = signal_data.signal[start_idx:end_idx]

    spec_data = spectrogram(
        signal,
        sample_rate,
        spec_calculator,
        n_fft,
        win_length,
        hop_length,
        power,
        **kwargs,
    )
    time_shift = signal_data.duration * start
    spec_data.times += time_shift

    return spec_data


def clip_spectrogram(
    spec: SpectrogramData,
    freq_start: Optional[float] = None,
    freq_end: Optional[float] = None,
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
) -> SpectrogramData:
    """Extract a smaller spectrogram from `spec`."""
    if t_start is None:
        t_start = spec.t_start
    if t_end is None:
        t_end = spec.t_end
    if freq_start is None:
        freq_start = spec.freq_start
    if freq_end is None:
        freq_end = spec.freq_end

    if not (spec.t_start <= t_start <= spec.t_end and
            spec.t_start <= t_end <= spec.t_end and
            spec.freq_start <= freq_end <= spec.freq_end and
            spec.freq_start <= freq_start <= spec.freq_end):
        raise ValueError("Clipped values must be inside spectrogram `spec`!")

    if t_start is None:
        min_t_pixel = 0
    else:
        min_t_pixel = spec.time_to_pixels(t_start - spec.t_start)

    if t_end is None:
        max_t_pixel = spec.get_width()
    else:
        max_t_pixel = spec.time_to_pixels(t_end - spec.t_start)

    if freq_start is None:
        min_freq_pixel = 0
    else:
        min_freq_pixel = spec.freq_to_pixels(freq_start - spec.freq_start)

    if freq_end is None:
        max_freq_pixel = spec.get_height()
    else:
        max_freq_pixel = spec.freq_to_pixels(freq_end - spec.freq_start)

    clipped_freqs = spec.freqs[min_freq_pixel:max_freq_pixel]
    clipped_times = spec.times[min_t_pixel:max_t_pixel]
    clipped_spec = spec.spec[min_freq_pixel:max_freq_pixel, min_t_pixel:max_t_pixel]
    return SpectrogramData(spec=clipped_spec, freqs=clipped_freqs, times=clipped_times)
