"""Reading, fitting and brightness handling for image frames."""
from __future__ import annotations
from pathlib import Path
import re

from PIL import Image
import cv2
import numpy as np


def natural_key(path: Path) -> list:
    """Sort key so that 'frame2' comes before 'frame10'."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def read_image(path: Path) -> np.ndarray:
    """Read an image at its native bit depth. Alpha is dropped; only the first page of
    a multi-page TIFF is read."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read image: {path.name}")
    if image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0].copy()
    elif image.ndim == 3 and image.shape[2] != 3:
        raise ValueError(f"{path.name}: unsupported channel count {image.shape[2]}")
    return image


def _channel_count(image: np.ndarray) -> int:
    return 1 if image.ndim == 2 else image.shape[2]


def prepare_series(paths: list[Path], images: list[np.ndarray]) -> list[np.ndarray]:
    """Check that all frames share bit depth and channel count, then fit every frame
    to the first frame's size."""
    reference = images[0]
    for path, image in zip(paths, images):
        if image.dtype != reference.dtype or _channel_count(image) != _channel_count(
            reference
        ):
            raise ValueError(
                f"{path.name} differs from {paths[0].name} in bit depth or channel "
                "count. All frames must match."
            )
    return [fit_to_shape(image, reference.shape) for image in images]


def fit_to_shape(image: np.ndarray, target_shape) -> np.ndarray:
    """Center-crop and zero-pad an image to the target (height, width).

    Pixels are never rescaled. Padding is zero rather than mirrored, so no invented
    image content can influence registration or the output figures.
    """
    target_h, target_w = target_shape[:2]
    h, w = image.shape[:2]
    if h > target_h:
        top = (h - target_h) // 2
        image = image[top : top + target_h]
    if w > target_w:
        left = (w - target_w) // 2
        image = image[:, left : left + target_w]

    h, w = image.shape[:2]
    pad_top = (target_h - h) // 2
    pad_bottom = target_h - h - pad_top
    pad_left = (target_w - w) // 2
    pad_right = target_w - w - pad_left
    if any((pad_top, pad_bottom, pad_left, pad_right)):
        image = cv2.copyMakeBorder(
            image,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=0,
        )
    return image


def shared_intensity_range(frames: list[np.ndarray]) -> tuple[float, float]:
    """1st and 99th percentile pooled over all frames, so every frame is scaled identically."""
    samples = [f.ravel()[:: max(1, f.size // 100_000)] for f in frames]
    lo, hi = np.percentile(np.concatenate(samples), (1, 99))
    return float(lo), float(hi)


def to_uint8(image: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Scale to 8-bit using the range [lo, hi]. 8-bit input is returned unchanged."""
    if image.dtype == np.uint8:
        return image
    scale = 255.0 / max(hi - lo, 1.0)
    return np.clip((image.astype(np.float32) - lo) * scale, 0, 255).astype(np.uint8)


def frames_to_pil(frames: list[np.ndarray]) -> list[Image.Image]:
    """Convert frames to RGB PIL images on one shared brightness scale.

    8-bit frames are shown as stored. Higher bit depths are stretched between the pooled
    1st and 99th percentiles of all frames, so a dim timepoint looks dim in the GIF and
    grid rather than being stretched to full contrast on its own.
    """
    if all(f.dtype == np.uint8 for f in frames):
        lo, hi = 0.0, 255.0
    else:
        lo, hi = shared_intensity_range(frames)

    pil_frames = []
    for frame in frames:
        rgb = to_uint8(frame, lo, hi)
        if rgb.ndim == 3:
            pil_frames.append(Image.fromarray(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)))
        else:
            pil_frames.append(Image.fromarray(rgb).convert("RGB"))
    return pil_frames


def to_gray_uint8(image: np.ndarray) -> np.ndarray:
    """8-bit grayscale copy for feature detection and template matching.

    Each frame is stretched on its own 1st-99th percentile range, even when it is already
    8-bit. Dim frames otherwise have too little contrast for SIFT to find features. This
    only affects matching, never the saved pixels, and both methods are insensitive to
    overall brightness.
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lo, hi = np.percentile(gray, (1, 99))
    scale = 255.0 / max(float(hi - lo), 1.0)
    return np.clip((gray.astype(np.float32) - lo) * scale, 0, 255).astype(np.uint8)


def _raw_gray_uint8(image: np.ndarray) -> np.ndarray:
    """8-bit grayscale with the original contrast. Only non-8-bit input is stretched,
    because stretching 8-bit frames amplifies background noise into false features."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if gray.dtype != np.uint8:
        lo, hi = np.percentile(gray, (1, 99))
        gray = to_uint8(gray, float(lo), float(hi))
    return gray
