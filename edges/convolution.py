"""2D convolution written by hand, with zero padding."""

import numpy as np
from numba import njit


@njit
def correlate2d(image, kernel):
    """Slide `kernel` over `image` without flipping it (cross-correlation).

    Pixels outside the image count as 0. The output has the image's shape.
    """
    rows, cols = image.shape
    size = kernel.shape[0]
    offset = size // 2
    output = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            acc = 0.0
            for di in range(size):
                ii = i + di - offset
                if ii < 0 or ii >= rows:
                    continue
                for dj in range(size):
                    jj = j + dj - offset
                    if jj < 0 or jj >= cols:
                        continue
                    acc += image[ii, jj] * kernel[di, dj]
            output[i, j] = acc
    return output


def convolve2d(image, kernel):
    """True convolution: correlation with the kernel flipped on both axes."""
    return correlate2d(image, np.ascontiguousarray(kernel[::-1, ::-1]))
