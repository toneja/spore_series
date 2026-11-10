"""Following a clicked spore through the aligned frames with template matching."""
from __future__ import annotations
import cv2
import numpy as np

from .config import MIN_TRACKING_SCORE
from .imaging import to_gray_uint8


def extract_crop(image: np.ndarray, x: int, y: int, size: int) -> np.ndarray:
    """Return a size x size crop centered on (x, y). Pixels outside the image are zero."""
    half = size // 2
    left, top = x - half, y - half
    h, w = image.shape[:2]
    pad_left = max(0, -left)
    pad_top = max(0, -top)
    pad_right = max(0, left + size - w)
    pad_bottom = max(0, top + size - h)
    if any((pad_left, pad_top, pad_right, pad_bottom)):
        image = cv2.copyMakeBorder(
            image,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=0,
        )
        left += pad_left
        top += pad_top
    return image[top : top + size, left : left + size].copy()


def track_spore(frames: list[np.ndarray], x: int, y: int, size: int):
    """Follow one spore through the registered frames by template matching.

    The template is cut from the reference frame at (x, y). Each later frame is searched
    within about +/- size/4 pixels of (x, y). Where the best match scores below
    MIN_TRACKING_SCORE, the clicked position is used for that frame instead.

    Returns one (center_x, center_y) and one match score per frame. The reference frame
    is always (x, y) with score 1.0.
    """
    template_size = max(16, int(size * 0.6))
    half_template = template_size // 2
    search_radius = max(8, size // 4)

    grays = [to_gray_uint8(f) for f in frames]
    template = extract_crop(grays[0], x, y, template_size)
    centers, scores = [(x, y)], [1.0]

    for gray in grays[1:]:
        left = max(0, x - search_radius - half_template)
        top = max(0, y - search_radius - half_template)
        right = min(gray.shape[1], x + search_radius + half_template + 1)
        bottom = min(gray.shape[0], y + search_radius + half_template + 1)
        search = gray[top:bottom, left:right]
        if search.shape[0] < template_size or search.shape[1] < template_size:
            centers.append((x, y))
            scores.append(0.0)
            continue

        result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        score = float(max_val) if np.isfinite(max_val) else 0.0
        if score < MIN_TRACKING_SCORE:
            centers.append((x, y))
        else:
            centers.append(
                (left + max_loc[0] + half_template, top + max_loc[1] + half_template)
            )
        scores.append(score)

    return centers, scores
