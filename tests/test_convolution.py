"""Check the hand-written convolution against OpenCV.

Run from the project root:  python -m tests.test_convolution
"""

import cv2
import numpy as np

from edges.convolution import convolve2d
from edges.kernels import gaussian_derivative_kernel, gaussian_kernel, gb_kernel


def max_diff_with_opencv(image, kernel):
    # cv2.filter2D computes a correlation, so flip the kernel to get a convolution
    flipped = np.ascontiguousarray(kernel[::-1, ::-1])
    reference = cv2.filter2D(image, -1, flipped, borderType=cv2.BORDER_CONSTANT)
    return np.abs(convolve2d(image, kernel) - reference).max()


def main():
    rng = np.random.default_rng(0)
    image = rng.random((60, 80)) * 255
    kernels = {
        "gaussian 7x7": gaussian_kernel(7, 7 / 6),
        "gaussian derivative 7x7": gaussian_derivative_kernel(7, 7 / 6, 0),
        "Gb 15x15": gb_kernel(15, 3.5, 1.0, 1),
    }
    for name, kernel in kernels.items():
        diff = max_diff_with_opencv(image, kernel)
        assert diff < 1e-9, f"{name}: difference {diff}"
        print(f"{name}: max difference with cv2.filter2D = {diff:.2e}")
    print("All checks passed.")


if __name__ == "__main__":
    main()
