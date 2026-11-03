"""Filter kernels built by hand: Gaussian, Gaussian derivative and Gb step kernel."""

import numpy as np
from numba import njit


def sigma_for_size(size):
    """Sigma such that the kernel covers about +/- 3 sigma."""
    return size / 6


@njit
def gaussian_kernel(size, sigma):
    """2D Gaussian kernel of shape (size, size), normalised to sum to 1."""
    kernel = np.zeros((size, size))
    offset = size // 2
    for i in range(size):
        for j in range(size):
            u = i - offset
            v = j - offset
            kernel[i, j] = np.exp(-(u**2 + v**2) / (2 * sigma**2))
    return kernel / np.sum(kernel)


@njit
def gaussian_derivative_kernel(size, sigma, axis):
    """Derivative of a Gaussian along `axis` (0 = rows, 1 = columns).

    Normalised so that the sum of absolute values is 1.
    """
    kernel = np.zeros((size, size))
    offset = size // 2
    for i in range(size):
        for j in range(size):
            u = i - offset
            v = j - offset
            g = np.exp(-(u**2 + v**2) / (2 * sigma**2))
            d = u if axis == 0 else v
            kernel[i, j] = -d * g
    total = np.sum(np.abs(kernel))
    if total > 0:
        kernel = kernel / total
    return kernel


@njit
def gb_kernel(size, sigma, eps, axis):
    """Gb step kernel along `axis`.

    Each offset is projected on the disk of radius `eps`, which models a
    sharp step instead of a smooth Gaussian edge.
    """
    kernel = np.zeros((size, size))
    offset = size // 2
    for i in range(size):
        for j in range(size):
            u = i - offset
            v = j - offset
            r = np.sqrt(u**2 + v**2)
            d = u if axis == 0 else v
            if r > eps:
                d = d * eps / r
            g = np.exp(-(u**2 + v**2) / (2 * sigma**2))
            kernel[i, j] = -d * g**2
    total = np.sum(np.abs(kernel))
    if total > 0:
        kernel = kernel / total
    return kernel
