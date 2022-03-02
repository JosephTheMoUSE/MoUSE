"""Tests for denoising helpers."""
import pytest  # noqa

import numpy as np
from mouse.denoising.denoising import _diagonal_gaussian_filter, _gaussian_filter


# Helper functions
def _g1_exp(p, q, sigma_p, sigma_q=None):
    sigma_q = sigma_p if sigma_q is None else sigma_q
    return np.exp(-0.5 * ((p / sigma_p)**2 + (q / sigma_q)**2))


def _get_single_std_gaussian_kernel_size_3():
    s = 1
    return np.array([
        [_g1_exp(1, 1, s), _g1_exp(0, 1, s), _g1_exp(1, 1, s)],
        [_g1_exp(1, 0, s), _g1_exp(0, 0, s), _g1_exp(1, 0, s)],
        [_g1_exp(1, 1, s), _g1_exp(0, 1, s), _g1_exp(1, 1, s)],
    ])


def _get_double_std_gaussian_kernel_size_3():
    s_p = 1
    s_q = 2
    return np.array([
        [_g1_exp(1, 1, s_p, s_q), _g1_exp(0, 1, s_p, s_q), _g1_exp(1, 1, s_p, s_q)],
        [_g1_exp(1, 0, s_p, s_q), _g1_exp(0, 0, s_p, s_q), _g1_exp(1, 0, s_p, s_q)],
        [_g1_exp(1, 1, s_p, s_q), _g1_exp(0, 1, s_p, s_q), _g1_exp(1, 1, s_p, s_q)],
    ])


def _g2_exp(p, q, sigma_p, sigma_q=None):
    sigma_q = sigma_p if sigma_q is None else sigma_q
    return np.exp(-1 * (((q - p) / sigma_p)**2 + ((p + q) / sigma_q)**2))


def _get_single_std_diagonal_gaussian_kernel_size_3():
    s = 1
    return np.array([
        [_g2_exp(1, 1, s), _g2_exp(0, 1, s), _g2_exp(1, 1, s)],
        [_g2_exp(1, 0, s), _g2_exp(0, 0, s), _g2_exp(1, 0, s)],
        [_g2_exp(1, 1, s), _g2_exp(0, 1, s), _g2_exp(1, 1, s)],
    ])


def _get_double_std_diagonal_gaussian_kernel_size_3():
    s_p = 1
    s_q = 2
    return np.array([
        [
            _g2_exp(-1, -1, s_p, s_q),
            _g2_exp(0, -1, s_p, s_q),
            _g2_exp(1, -1, s_p, s_q),
        ],
        [
            _g2_exp(-1, 0, s_p, s_q),
            _g2_exp(0, 0, s_p, s_q),
            _g2_exp(1, 0, s_p, s_q),
        ],
        [
            _g2_exp(-1, 1, s_p, s_q),
            _g2_exp(0, 1, s_p, s_q),
            _g2_exp(1, 1, s_p, s_q),
        ],
    ])


# Tests
def test_symmetric_gaussian_filter():
    """Test if standard Gaussian kernel with single std works."""
    image = np.array([[1.0, 2.0, 30.0, 4.0], [5.0, 66.0, 7.0, 8.0],
                      [90.0, 10.0, 91.0, 12.0]])
    gaussian_kernel = _get_single_std_gaussian_kernel_size_3()
    result = _gaussian_filter(image, sigma_x=1, sigma_y=1, m=3)
    gaussian_kernel = gaussian_kernel / np.sum(gaussian_kernel)
    # Border pixels don't matter so don't bother checking them
    middle_pixel_1 = np.sum(image[:, :3] * gaussian_kernel) / np.sum(gaussian_kernel)
    middle_pixel_2 = np.sum(image[:, 1:] * gaussian_kernel) / np.sum(gaussian_kernel)
    assert np.isclose(middle_pixel_1, result[1][1])
    assert np.isclose(middle_pixel_2, result[1][2])


def test_asymmetric_gaussian_filter():
    """Test if standard Gaussian kernel with two stds works."""
    image = np.array([[1.0, 2.0, 30.0, 4.0], [5.0, 66.0, 7.0, 8.0],
                      [90.0, 10.0, 91.0, 12.0]])
    gaussian_kernel = _get_double_std_gaussian_kernel_size_3()
    result = _gaussian_filter(image, sigma_x=1, sigma_y=2, m=3)
    # Border pixels don't matter so don't bother checking them
    middle_pixel_1 = np.sum(image[:, :3] * gaussian_kernel) / np.sum(gaussian_kernel)
    middle_pixel_2 = np.sum(image[:, 1:] * gaussian_kernel) / np.sum(gaussian_kernel)
    assert np.isclose(middle_pixel_1, result[1][1])
    assert np.isclose(middle_pixel_2, result[1][2])


def test_symmetric_diagonal_gaussian_filter():
    """Test if diagonal Gaussian kernel with single std works."""
    image = np.array([[1.0, 2.0, 30.0, 4.0], [5.0, 66.0, 7.0, 8.0],
                      [90.0, 10.0, 91.0, 12.0]])
    gaussian_kernel = _get_single_std_diagonal_gaussian_kernel_size_3()
    result = _diagonal_gaussian_filter(image, sigma_x=1, sigma_y=1, m=3)
    # Border pixels don't matter so don't bother checking them
    middle_pixel_1 = np.sum(image[:, :3] * gaussian_kernel) / np.sum(gaussian_kernel)
    middle_pixel_2 = np.sum(image[:, 1:] * gaussian_kernel) / np.sum(gaussian_kernel)
    assert np.isclose(middle_pixel_1, result[1][1])
    assert np.isclose(middle_pixel_2, result[1][2])


def test_asymmetric_diagonal_gaussian_filter():
    """Test if diagonal Gaussian kernel with two stds works."""
    image = np.array([[1.0, 2.0, 30.0, 4.0], [5.0, 66.0, 7.0, 8.0],
                      [90.0, 10.0, 91.0, 12.0]])
    gaussian_kernel = _get_double_std_diagonal_gaussian_kernel_size_3()
    result = _diagonal_gaussian_filter(image, sigma_x=1, sigma_y=2, m=3)
    # Border pixels don't matter so don't bother checking them
    middle_pixel_1 = np.sum(image[:, :3] * gaussian_kernel) / np.sum(gaussian_kernel)
    middle_pixel_2 = np.sum(image[:, 1:] * gaussian_kernel) / np.sum(gaussian_kernel)
    assert np.isclose(middle_pixel_1, result[1][1])
    assert np.isclose(middle_pixel_2, result[1][2])
