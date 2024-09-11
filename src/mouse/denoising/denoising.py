"""Module implementing spectrogram denoising methods."""
import cv2
import librosa
import numpy as np
from scipy.ndimage import correlate1d, correlate
from scipy.signal import fftconvolve
import torch

from mouse.utils import sound_util

def gaussian_kernel1d(sigma, order, radius):
    """Generate a Gaussian kernel.
    A function to generate a Gaussian kernel, given sigma and radius.
    """
    kernel_size = 2*radius + 1
    x = np.arange(-radius, radius+1)
    kernel = np.exp(-(x**2) / (2 * sigma**2))
    kernel = kernel / np.sum(kernel)
    return kernel

# Bilateral filter.
def bilateral_filter(
    spectrogram: sound_util.SpectrogramData,
    d: int,
    sigma_color: float,
    sigma_space: float,
):
    """Perform in-place bilateral filtering on the given spectrogram.

    Parameters
    ----------
    spectrogram: sound_util.SpectrogramData
        The spectrogram to perform the denoising on.
    d : int
        A parameter passed to cv2.bilateralFilter.
    sigma_color : float
         A parameter passed to cv2.bilateralFilter.
    sigma_space : float
        A parameter passed to cv2.bilateralFilter.
    """
    spectrogram.spec = torch.tensor(
        cv2.bilateralFilter(
            src=spectrogram.spec.numpy(),
            d=d,
            sigmaColor=sigma_color,
            sigmaSpace=sigma_space,
        ))


# Short-duration transient suppression filter.
def _gaussian_filter(image, sigma_x, sigma_y, m):
    if m % 2 == 0:
        raise ValueError

    # Standard Gaussian filer is separable - we can compute it based on
    # two 1d Gaussian kernels instead of one 2d Gaussian kernel
    # which results in better complexity.
    gaussian_kernel_x = gaussian_kernel1d(sigma=sigma_x, order=0, radius=(m - 1) / 2)
    gaussian_kernel_y = gaussian_kernel1d(sigma=sigma_y, order=0, radius=(m - 1) / 2)
    result = correlate1d(input=image, weights=gaussian_kernel_x, axis=1)
    correlate1d(input=result, weights=gaussian_kernel_y, axis=0, output=result)
    return result


def _diagonal_gaussian_filter(image, sigma_x, sigma_y, m):
    if m % 2 == 0:
        raise ValueError

    r = range(-1 * (m - 1) // 2, (m + 1) // 2)
    # Paper suggests to have ony positive numbers for p and q but it doesn't
    # produce a diagonal kernel so both positive and negative are considered.
    diagonal_kernel = np.array([[
        np.exp(-(np.power((q - p) / sigma_x, 2) + np.power((q + p) / sigma_y, 2)))
        for p in r
    ]
                                for q in r])
    diagonal_kernel = diagonal_kernel / np.sum(diagonal_kernel)
    return correlate(image, diagonal_kernel)


def short_duration_transient_suppression_filter(spectrogram: sound_util.SpectrogramData,
                                                alpha: float,
                                                m: int = 3,
                                                a: float = None):
    """Perform in-place SDTS filtering on the given spectrogram.

    Based on:
    "Spectrogram denoising and automated extraction of the fundamental
    frequency variation of dolphin whistles".

    Parameters
    ----------
    spectrogram: sound_util.SpectrogramData
        The spectrogram to perform the denosing on.
    alpha : float
        Specifies "how much" of the unchanged spectrogram should be present in
        the filtered spectrogram. Should be between zero and one.
    m : int
        Size of the Gaussian kernels used in the method.
        Should be an odd number.
    a : float
        Needed for computing stds for Gaussian kernels. Default value is m / 10.

    Raises
    ------
    ValueError
        If m is an even number.
    """
    if m % 2 == 0:
        raise ValueError

    a = m / 10 if a is None else a

    horizontal_intermediate = _gaussian_filter(image=spectrogram.spec,
                                               sigma_x=a,
                                               sigma_y=6 * a,
                                               m=m)
    vertical_intermediate = _gaussian_filter(image=spectrogram.spec,
                                             sigma_x=6 * a,
                                             sigma_y=a,
                                             m=m)
    diagonal_intermediate_1 = _diagonal_gaussian_filter(image=spectrogram.spec,
                                                        sigma_x=a,
                                                        sigma_y=6 * a,
                                                        m=m)
    diagonal_intermediate_2 = _diagonal_gaussian_filter(image=spectrogram.spec,
                                                        sigma_x=6 * a,
                                                        sigma_y=a,
                                                        m=m)

    maximum = np.maximum(
        np.maximum(horizontal_intermediate, diagonal_intermediate_1),
        diagonal_intermediate_2,
    )
    spectrogram.spec = alpha * spectrogram.spec + (1 - alpha) * (maximum -
                                                                 vertical_intermediate)


# Noise gate denoising
def noise_gate_filter(
    spectrogram: sound_util.SpectrogramData,
    noise_spectrogram: sound_util.SpectrogramData,
    n_grad_freq: int,
    n_grad_time: int,
    n_std_thresh: float,
    noise_decrease: float,
    ref: float = 1.0,
    amin: float = 1e-20,
    top_db: float = 80.0,
):
    """Perform in-place noise gate filtering on the given spectrogram.

    The algorithm computes mean noise level for each frequency based on
    provided noise spectrogram and for each frequency attenuates values that
    are smaller than computed noise mean plus chosen number of noise stds.

    Parameters
    ----------
    spectrogram: sound_util.SpectrogramData
        The spectrogram to perform the denosing on.
    noise_spectrogram: sound_util.SpectrogramData
        Spectrogram with noise sample needed for computing noise stats.
    n_grad_freq: int
        How many frequency channels to smooth over with the mask.
    n_grad_time: int
        How many time channels to smooth over with the mask.
    n_std_thresh: float
        How many standard deviations louder than the mean dB of the noise
        (at each frequency level) the threshold should be set.
    noise_decrease: float
        To what extent the noise should be attenuated
        (1 means that all noise is attenuated).
    ref: float
        Parameter of dB conversion.
    amin: float
        Parameter of dB conversion.
    top_db: float
        Parameter of dB conversion.
    """
    noise_spec_db = librosa.core.amplitude_to_db(noise_spectrogram.spec.numpy(),
                                                 ref=ref,
                                                 amin=amin,
                                                 top_db=top_db)
    mean_freq_noise = np.mean(noise_spec_db, axis=1, keepdims=True)
    std_freq_noise = np.std(noise_spec_db, axis=1, keepdims=True)
    noise_threshold = mean_freq_noise + std_freq_noise * n_std_thresh
    spec_db = librosa.core.amplitude_to_db(spectrogram.spec.numpy(),
                                           ref=ref,
                                           amin=amin,
                                           top_db=top_db)

    # Calculate value which will be substituting noise
    mask_gain_db = np.min(spec_db)

    # Create a smoothing filter for the mask in time and frequency
    smoothing_filter = np.outer(
        np.concatenate([
            np.linspace(0, 1, n_grad_freq + 1, endpoint=False),
            np.linspace(1, 0, n_grad_freq + 1, endpoint=False),
        ])[1:],
        np.concatenate([
            np.linspace(0, 1, n_grad_time + 1, endpoint=False),
            np.linspace(1, 0, n_grad_time + 1, endpoint=False),
        ])[1:],
    )
    smoothing_filter = smoothing_filter / np.sum(smoothing_filter)

    # Calculate the threshold for each frequency/time bin
    db_threshold = np.repeat(
        noise_threshold,
        np.shape(spec_db)[1],
        axis=1,
    )

    # Mask if the signal is above the threshold
    mask = spec_db < db_threshold

    # Convolve the mask with a smoothing filter
    mask = fftconvolve(mask, smoothing_filter, mode="same")
    mask = mask * noise_decrease

    # Mask the signal
    spec_db_masked = spec_db * (1 - mask) + mask_gain_db * mask

    spectrogram.spec = torch.tensor(
        librosa.core.db_to_amplitude(spec_db_masked, ref=ref) *
        np.sign(spectrogram.spec.numpy()))
