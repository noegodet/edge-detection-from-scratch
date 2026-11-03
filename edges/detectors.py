"""Edge detectors: Gaussian derivative (grey and RGB) and Gb (RGB)."""

import numpy as np

from .convolution import convolve2d
from .kernels import gaussian_derivative_kernel, gb_kernel

GB_EPS = 1.0  # Gb step parameter, in pixels


def rgb_to_grey(img_bgr):
    """Luminance from an OpenCV BGR image, as float64."""
    img = img_bgr.astype(np.float64)
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    return 0.299 * r + 0.587 * g + 0.114 * b


def split_rgb(img_bgr):
    """The three colour layers R, G, B as float64 arrays."""
    img = img_bgr.astype(np.float64)
    return [img[:, :, 2], img[:, :, 1], img[:, :, 0]]


def gd_grey(img_grey, size, sigma):
    """Gradient magnitude of a grey image with Gaussian derivative filters."""
    i0 = convolve2d(img_grey, gaussian_derivative_kernel(size, sigma, 0))
    i1 = convolve2d(img_grey, gaussian_derivative_kernel(size, sigma, 1))
    return np.sqrt(i0**2 + i1**2)


def boundary_strength(layers, kernel_0, kernel_1):
    """Combine the gradients of several layers into one boundary map.

    Builds the 2x2 structure tensor [[a, c], [c, d]] summed over layers and
    returns the square root of its largest eigenvalue.
    """
    a = c = d = 0
    for layer in layers:
        j0 = convolve2d(layer, kernel_0)
        j1 = convolve2d(layer, kernel_1)
        a = a + j0 * j0
        d = d + j1 * j1
        c = c + j0 * j1
    lam = (a + d) / 2 + np.sqrt(((a - d) / 2) ** 2 + c**2)
    return np.sqrt(lam)


def gd_rgb(img_bgr, size, sigma):
    """Gaussian derivative detector on the three colour layers."""
    return boundary_strength(
        split_rgb(img_bgr),
        gaussian_derivative_kernel(size, sigma, 0),
        gaussian_derivative_kernel(size, sigma, 1),
    )


def gb_rgb(img_bgr, size, sigma=None):
    """Gb detector on the three colour layers.

    Sigma is fixed to half the window radius, so the `sigma` argument is
    ignored. It is accepted only to share the signature of `gd_rgb`.
    """
    sigma_g = max((size // 2) / 2, 0.5)
    return boundary_strength(
        split_rgb(img_bgr),
        gb_kernel(size, sigma_g, GB_EPS, 0),
        gb_kernel(size, sigma_g, GB_EPS, 1),
    )
