"""Image loading and resizing."""

from pathlib import Path

import cv2

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".jfif", ".bmp"}


def list_images(folder):
    """Image files of `folder`, sorted by name."""
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def resize(img, factor):
    """Resize by `factor` with area interpolation (best for shrinking)."""
    if factor == 1:
        return img
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w * factor), int(h * factor)), interpolation=cv2.INTER_AREA)


def load_image(path, max_width):
    """Read a BGR image and shrink it so its width is at most `max_width`."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read {path}")
    if img.shape[1] > max_width:
        img = resize(img, max_width / img.shape[1])
    return img
