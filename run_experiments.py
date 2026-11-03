"""Run the edge detection experiments and save every figure as a PNG.

Part A: Gaussian derivative detector on grey images.
Part B: Gaussian derivative (GD) and Gb detectors on RGB images.

Each part runs two experiments on every image:
  1. bigger filter, same image  (filter sizes 3, 7, 15)
  2. same filter, smaller image (scales 1, 1/2, 1/4)

Usage:
    python run_experiments.py --images images --out outputs
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save figures without opening windows
import cv2
import matplotlib.pyplot as plt

from edges.detectors import gb_rgb, gd_grey, gd_rgb, rgb_to_grey
from edges.image_io import list_images, load_image, resize
from edges.kernels import sigma_for_size

FILTER_SIZES = [3, 7, 15]
FIXED_SIZE = 7
SCALE_FACTORS = [1, 0.5, 0.25]
MAX_WIDTH = 800
RGB_DETECTORS = {"gd": ("GD RGB", gd_rgb), "gb": ("Gb RGB", gb_rgb)}


def save_row(panels, suptitle, path, first_is_rgb=False):
    """Plot (image, title) panels side by side and save the figure."""
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4.5))
    for k, (ax, (img, title)) in enumerate(zip(axes, panels)):
        if k == 0 and first_is_rgb:
            ax.imshow(img)
        else:
            ax.imshow(img, cmap="gray", interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle(suptitle)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def part_a(name, img, out):
    """Gaussian derivative detector on the grey image."""
    grey = rgb_to_grey(img)

    panels = [(grey, f"Grey image ({grey.shape[1]}x{grey.shape[0]})")]
    for size in FILTER_SIZES:
        sigma = sigma_for_size(size)
        panels.append((gd_grey(grey, size, sigma), f"Filter {size}x{size}, sigma = {sigma:.2f}"))
    save_row(panels, f"{name} - Part A - bigger filter, same image",
             out / "part_a" / f"{name}_1_filter_size.png")

    sigma = sigma_for_size(FIXED_SIZE)
    panels = []
    for factor in SCALE_FACTORS:
        small = resize(grey, factor)
        panels.append((gd_grey(small, FIXED_SIZE, sigma),
                       f"Scale {factor} ({small.shape[1]}x{small.shape[0]} px)"))
    save_row(panels, f"{name} - Part A - same {FIXED_SIZE}x{FIXED_SIZE} filter "
                     f"(sigma = {sigma:.2f}), smaller image",
             out / "part_a" / f"{name}_2_image_scale.png")


def part_b(name, img, out):
    """GD and Gb detectors on the three colour layers."""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for key, (label, detector) in RGB_DETECTORS.items():
        panels = [(img_rgb, f"RGB image ({img.shape[1]}x{img.shape[0]})")]
        for size in FILTER_SIZES:
            panels.append((detector(img, size, sigma_for_size(size)),
                           f"{label} - filter {size}x{size}"))
        save_row(panels, f"{name} - Part B - {label} - bigger filter, same image",
                 out / "part_b" / f"{name}_{key}_1_filter_size.png", first_is_rgb=True)

        sigma = sigma_for_size(FIXED_SIZE)
        panels = []
        for factor in SCALE_FACTORS:
            small = resize(img, factor)
            panels.append((detector(small, FIXED_SIZE, sigma),
                           f"{label} - scale {factor} ({small.shape[1]}x{small.shape[0]} px)"))
        save_row(panels, f"{name} - Part B - {label} - same {FIXED_SIZE}x{FIXED_SIZE} "
                         f"filter, smaller image",
                 out / "part_b" / f"{name}_{key}_2_image_scale.png")

    # Side-by-side comparison of both detectors with the largest filter
    size = FILTER_SIZES[-1]
    sigma = sigma_for_size(size)
    panels = [(img_rgb, "RGB image")]
    for label, detector in RGB_DETECTORS.values():
        panels.append((detector(img, size, sigma), f"{label} - filter {size}x{size}"))
    save_row(panels, f"{name} - Part B - GD vs Gb, filter {size}x{size}",
             out / "part_b" / f"{name}_3_gd_vs_gb.png", first_is_rgb=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", default="images", help="folder with the input images")
    parser.add_argument("--out", default="outputs", help="folder for the saved figures")
    parser.add_argument("--parts", default="ab", help="parts to run: a, b or ab")
    args = parser.parse_args()

    paths = list_images(args.images)
    if not paths:
        raise SystemExit(f"No images found in {args.images}")
    out = Path(args.out)

    for path in paths:
        img = load_image(path, MAX_WIDTH)
        if "a" in args.parts:
            part_a(path.stem, img, out)
        if "b" in args.parts:
            part_b(path.stem, img, out)


if __name__ == "__main__":
    main()
