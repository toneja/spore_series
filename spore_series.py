#!/usr/bin/env python3
"""Spore time-series extractor.

Registers a series of microscope images to the first frame, lets you click spores
on that reference frame, tracks each kept spore through the series, and saves per-spore
TIFF crops, an animated GIF, and a labelled grid JPG suitable for article figures.

Usage:
    python spore_series.py

See README.md for setup, inputs, and output details.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import queue
import re
import subprocess
import sys
import threading
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageSequence, ImageTk

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
OUTPUT_FOLDER_NAME = "spore_output"
DEFAULT_CROP_SIZE = 180
MIN_CROP_SIZE = 16

# Registration
MAX_FEATURES = 5000
MIN_FEATURE_MATCHES = 4  # fewest matches that can support a transform estimate
LOWE_RATIO = 0.72  # keep a match only if it beats its runner-up by this factor
RANSAC_THRESHOLD_PX = 3.0
MIN_ACCEPTED_INLIERS = 8  # a transform needs at least this many inliers to be trusted
MAX_SCALE_DEVIATION = 0.2  # accepted |scale - 1|
MAX_ROTATION_DEG = 15.0  # accepted rotation magnitude

# Translation registration (phase correlation refined by ECC; insensitive to brightness)
ECC_MIN_CORRELATION = 0.5  # ECC correlation a translation fit must reach to be accepted
ECC_MAX_ITERATIONS = 100

# Spore-centroid registration (fallback for sparse fields)
SPORE_MIN_AREA_PX = 150
SPORE_MAX_AREA_PX = 6000
SPORE_MIN_SIDE_PX = 14
SPORE_MIN_MATCHES = 6
SPORE_MATCH_PX = 4.0
SPORE_ANGLE_RANGE_DEG = 10.0
SPORE_ANGLE_STEP_DEG = 0.5
SPORE_SHIFT_CANDIDATES = 20  # shift hypotheses scored per rotation

# Tracking: below this template-match score, a frame falls back to the clicked position
MIN_TRACKING_SCORE = 0.35

# Image view (zoom and pan)
ZOOM_MIN = 1.0  # 1.0 = whole image fits in the view
ZOOM_MAX = 40.0
ZOOM_STEP = 1.1  # multiplier for one wheel notch or one +/- button press
DRAG_THRESHOLD_PX = 4  # movement below this counts as a click, not a drag
MARKER_RADIUS_PX = 7
KEPT_MARKER_COLOR = "#1e9e3c"  # green
REJECTED_MARKER_COLOR = "red"
REFERENCE_MAX_FIT_SCALE = 1.0  # the reference is never enlarged to fill the view
PREVIEW_MAX_FIT_SCALE = 3.0  # small GIF and grid previews may be enlarged to fit

# Preview, GIF and grid appearance
PREVIEW_SIZE = 230
PREVIEW_COLUMNS = 4
GIF_FRAME_MS = 700
GIF_LABEL_BAR_PX = 24
GRID_PANEL_PX = 400
GRID_PANEL_PAD_PX = 14
GRID_LABEL_HEIGHT_PX = 40
GRID_LABEL_FONT_PX = 26
GRID_JPEG_QUALITY = 95
GRID_DPI = 300

FONT_CANDIDATES = (
    "arial.ttf",
    "Arial.ttf",
    "DejaVuSans.ttf",
    "LiberationSans-Regular.ttf",
)
IDENTITY_AFFINE = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

GIF_NAME_PATTERN = re.compile(r"(S\d{3,})\.gif")
GRID_NAME_PATTERN = re.compile(r"(S\d{3,})_grid\.jpg")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class RegistrationError(RuntimeError):
    """Raised when a frame cannot be aligned to the reference."""


@dataclass
class Spore:
    """A kept ROI: its clicked position on the reference frame and its per-frame tracking."""

    spore_id: str
    x: int
    y: int
    size: int
    centers: list[tuple[int, int]]
    scores: list[float]


@dataclass
class FrameRegistration:
    """Registration diagnostics for one frame. Match counts are None for the reference."""

    method: str
    matches: int | None
    inliers: int | None
    matrix: np.ndarray

    @property
    def inlier_fraction(self) -> float | None:
        if not self.matches:
            return None
        return self.inliers / self.matches


# ---------------------------------------------------------------------------
# Loading and preparing images
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Intensity scaling for display and figures
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def _make_detector():
    if hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create(nfeatures=MAX_FEATURES), cv2.NORM_L2, "SIFT"
    return cv2.ORB_create(nfeatures=MAX_FEATURES), cv2.NORM_HAMMING, "ORB"


def _lowe_filter(knn_pairs):
    return [
        pair[0]
        for pair in knn_pairs
        if len(pair) == 2 and pair[0].distance < LOWE_RATIO * pair[1].distance
    ]


def _raw_gray_uint8(image: np.ndarray) -> np.ndarray:
    """8-bit grayscale with the original contrast. Only non-8-bit input is stretched,
    because stretching 8-bit frames amplifies background noise into false features."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if gray.dtype != np.uint8:
        lo, hi = np.percentile(gray, (1, 99))
        gray = to_uint8(gray, float(lo), float(hi))
    return gray


def _rotation(angle_deg: float) -> np.ndarray:
    a = math.radians(angle_deg)
    return np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])


def is_plausible_transform(matrix: np.ndarray | None, shape) -> bool:
    """Reject degenerate or implausible fits. Microscope stages move by shifts and small
    rotations, so the scale must stay near 1 and the rotation must stay small."""
    if matrix is None:
        return False
    a, b = float(matrix[0, 0]), float(matrix[1, 0])
    scale = math.hypot(a, b)
    rotation = math.degrees(math.atan2(b, a))
    shift = math.hypot(float(matrix[0, 2]), float(matrix[1, 2]))
    h, w = shape[:2]
    return (
        abs(scale - 1.0) <= MAX_SCALE_DEVIATION
        and abs(rotation) <= MAX_ROTATION_DEG
        and shift <= math.hypot(w, h)
    )


def _high_pass_normalized(gray: np.ndarray) -> np.ndarray:
    """Remove slow background variation and normalise contrast, so frames taken with
    different illumination can be compared directly."""
    g = gray.astype(np.float32)
    g = g - cv2.GaussianBlur(g, (0, 0), 25)
    return (g - g.mean()) / (g.std() + 1e-6)


def _translation_transform(ref_norm: np.ndarray, frame_gray: np.ndarray):
    """Estimate frame -> reference as a shift (plus small rotation) by image correlation.

    Phase correlation gives a starting shift, which ECC then refines. ECC's correlation
    coefficient measures how well the aligned images agree, and it is unaffected by
    brightness and contrast changes. Returns (matrix, correlation), or (None, -1.0) if ECC
    fails to converge.
    """
    frame_norm = _high_pass_normalized(frame_gray)
    window = cv2.createHanningWindow((ref_norm.shape[1], ref_norm.shape[0]), cv2.CV_32F)
    (dx, dy), _ = cv2.phaseCorrelate(ref_norm, frame_norm, window)
    warp = np.array(
        [[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32
    )  # reference -> frame
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        ECC_MAX_ITERATIONS,
        1e-5,
    )
    try:
        correlation, warp = cv2.findTransformECC(
            ref_norm, frame_norm, warp, cv2.MOTION_EUCLIDEAN, criteria, None, 5
        )
    except cv2.error:
        return None, -1.0
    return cv2.invertAffineTransform(warp), float(correlation)


def _feature_transform(detector, norm, ref_kp, ref_des, frame_gray):
    """Estimate frame -> reference from feature matches. Returns (matrix, matches, inliers)."""
    kp, des = detector.detectAndCompute(frame_gray, None)
    if des is None or ref_des is None or len(kp) < MIN_FEATURE_MATCHES:
        return None, 0, 0
    good = _lowe_filter(cv2.BFMatcher(norm).knnMatch(des, ref_des, k=2))
    if len(good) < MIN_FEATURE_MATCHES:
        return None, len(good), 0
    src = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([ref_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    matrix, mask = cv2.estimateAffinePartial2D(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=RANSAC_THRESHOLD_PX,
        maxIters=5000,
        confidence=0.99,
        refineIters=10,
    )
    if matrix is None or mask is None:
        return None, len(good), 0
    return matrix, len(good), int(mask.sum())


def detect_spore_centroids(gray: np.ndarray) -> np.ndarray:
    """Centroids of dark, compact blobs (the spores), found with Otsu's threshold."""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(binary)
    kept = []
    for i in range(1, count):
        area = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        if (
            SPORE_MIN_AREA_PX <= area <= SPORE_MAX_AREA_PX
            and max(w, h) >= SPORE_MIN_SIDE_PX
            and 0.5 <= w / h <= 2.0
        ):
            kept.append(centroids[i])
    return np.array(kept, dtype=np.float64).reshape(-1, 2)


def _spore_transform(ref_spores: np.ndarray, spores: np.ndarray, shape):
    """Estimate frame -> reference from spore centroids.

    Searches small rotations; for each, the most common shifts between the two centroid
    sets are scored by how many spores land within SPORE_MATCH_PX of a reference spore.
    The best candidate is then refined with a robust similarity fit to its inliers.
    Returns (matrix, matches, inliers).
    """
    if len(ref_spores) < SPORE_MIN_MATCHES or len(spores) < SPORE_MIN_MATCHES:
        return None, 0, 0
    center = np.array([shape[1] / 2.0, shape[0] / 2.0])

    def count_inliers(points, reference):
        d = np.linalg.norm(points[:, None, :] - reference[None, :, :], axis=2).min(
            axis=1
        )
        return int(np.sum(d <= SPORE_MATCH_PX))

    best_count, best_points = 0, None
    for angle in np.arange(
        -SPORE_ANGLE_RANGE_DEG, SPORE_ANGLE_RANGE_DEG + 1e-9, SPORE_ANGLE_STEP_DEG
    ):
        rotated = (spores - center) @ _rotation(angle).T + center
        diffs = (ref_spores[None, :, :] - rotated[:, None, :]).reshape(-1, 2)
        bins = np.round(diffs / 2.0).astype(int)
        keys, counts = np.unique(bins, axis=0, return_counts=True)
        for idx in np.argsort(-counts)[:SPORE_SHIFT_CANDIDATES]:
            shift = diffs[np.all(bins == keys[idx], axis=1)].mean(axis=0)
            points = rotated + shift
            n = count_inliers(points, ref_spores)
            if n > best_count:
                best_count, best_points = n, (angle, shift)
    if best_points is None or best_count < SPORE_MIN_MATCHES:
        return None, 0, 0

    angle, shift = best_points
    rotated = (spores - center) @ _rotation(angle).T + center + shift
    d = np.linalg.norm(rotated[:, None, :] - ref_spores[None, :, :], axis=2)
    nearest = d.argmin(axis=1)
    close = d.min(axis=1) <= SPORE_MATCH_PX
    src = spores[close].astype(np.float32).reshape(-1, 1, 2)
    dst = ref_spores[nearest[close]].astype(np.float32).reshape(-1, 1, 2)
    matrix, mask = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=SPORE_MATCH_PX
    )
    if matrix is None or mask is None:
        return None, int(close.sum()), 0
    return matrix, int(close.sum()), int(mask.sum())


def refine_with_spores(matrix, ref_spores, spores, shape, iterations: int = 5):
    """Tighten a transform by pairing each spore with its nearest reference spore and refitting.

    The refined transform is kept only if the median spore residual goes down, so this
    can improve a fit but never makes it worse.
    """
    if matrix is None or len(spores) == 0 or len(ref_spores) == 0:
        return matrix

    def median_residual(m):
        pts = spores @ m[:, :2].T + m[:, 2]
        return float(
            np.median(
                np.linalg.norm(pts[:, None] - ref_spores[None], axis=2).min(axis=1)
            )
        )

    best, best_residual = matrix, median_residual(matrix)
    current = matrix
    for _ in range(iterations):
        pts = spores @ current[:, :2].T + current[:, 2]
        d = np.linalg.norm(pts[:, None] - ref_spores[None], axis=2)
        nearest = d.argmin(axis=1)
        close = d.min(axis=1) < SPORE_MATCH_PX
        if close.sum() < SPORE_MIN_MATCHES:
            break
        refit, _ = cv2.estimateAffinePartial2D(
            spores[close].astype(np.float32),
            ref_spores[nearest[close]].astype(np.float32),
            method=cv2.LMEDS,
        )
        if refit is None or not is_plausible_transform(refit, shape):
            break
        current = refit
        residual = median_residual(current)
        if residual < best_residual:
            best, best_residual = current, residual
    return best


def _feature_registration(name, detector, norm, ref_kp, ref_des, frame_gray, shape):
    """Return a FrameRegistration if SIFT-style matching gives a trustworthy transform."""
    matrix, matches, inliers = _feature_transform(
        detector, norm, ref_kp, ref_des, frame_gray
    )
    if inliers >= MIN_ACCEPTED_INLIERS and is_plausible_transform(matrix, shape):
        return FrameRegistration(name, matches, inliers, matrix)
    return None


def _spore_registration(ref_spores, frame_gray, shape):
    """Return a FrameRegistration if the spore centroids give a trustworthy transform."""
    matrix, matches, inliers = _spore_transform(
        ref_spores, detect_spore_centroids(frame_gray), shape
    )
    if inliers >= MIN_ACCEPTED_INLIERS and is_plausible_transform(matrix, shape):
        return FrameRegistration("spores", matches, inliers, matrix)
    return None


def _translation_registration(ref_norm, frame_gray, shape):
    """Return a FrameRegistration if the image-correlation translation fit is trustworthy."""
    matrix, correlation = _translation_transform(ref_norm, frame_gray)
    if correlation >= ECC_MIN_CORRELATION and is_plausible_transform(matrix, shape):
        return FrameRegistration(
            f"translation (ECC r={correlation:.2f})", None, None, matrix
        )
    return None


def register_images(frames: list[np.ndarray]):
    """Align every frame onto the first frame.

    Each frame is aligned by the first method that gives a trustworthy transform:
      1. SIFT feature matches on the original image contrast, with RANSAC.
      2. Translation by phase correlation refined with ECC. Robust to brightness changes
         and to sparse or changing features, as long as the frame is shifted, not rotated.
      3. Dark spore centroids, which are stable landmarks on sparse, featureless fields.
      4. SIFT on a contrast-stretched copy, for very dim frames.
    Each candidate must have a plausible scale and rotation, and enough support (inliers
    for feature methods, correlation for ECC).
    Returns (aligned_frames, registrations). aligned_frames[0] is the reference itself.
    Raises RegistrationError if no method succeeds for some frame.
    """
    detector, norm, method = _make_detector()
    reference = frames[0]
    ref_h, ref_w = reference.shape[:2]
    shape = reference.shape

    ref_raw = _raw_gray_uint8(reference)
    ref_stretched = to_gray_uint8(reference)
    ref_norm = _high_pass_normalized(ref_raw)
    ref_kp_raw, ref_des_raw = detector.detectAndCompute(ref_raw, None)
    ref_kp_str, ref_des_str = detector.detectAndCompute(ref_stretched, None)
    ref_spores = detect_spore_centroids(ref_raw)
    if ref_des_raw is None or len(ref_kp_raw) < MIN_FEATURE_MATCHES:
        raise RegistrationError("Not enough features in the reference image.")

    aligned = [reference]
    registrations = [FrameRegistration("reference", None, None, IDENTITY_AFFINE.copy())]

    for frame_number, frame in enumerate(frames[1:], start=2):
        raw = _raw_gray_uint8(frame)
        attempts = [
            lambda: _feature_registration(
                "SIFT", detector, norm, ref_kp_raw, ref_des_raw, raw, shape
            ),
            lambda: _translation_registration(ref_norm, raw, shape),
            lambda: _spore_registration(ref_spores, raw, shape),
            lambda: _feature_registration(
                "SIFT (stretched)",
                detector,
                norm,
                ref_kp_str,
                ref_des_str,
                to_gray_uint8(frame),
                shape,
            ),
        ]
        chosen = None
        for attempt in attempts:
            chosen = attempt()
            if chosen is not None:
                break
        if chosen is None:
            raise RegistrationError(
                f"Frame {frame_number}: no method could align this frame. Check that it "
                "shares features or spores with the reference, or remove it from the folder."
            )

        refined = refine_with_spores(
            chosen.matrix, ref_spores, detect_spore_centroids(raw), shape
        )
        if refined is not chosen.matrix:
            chosen.method += " + spore refinement"
            chosen.matrix = refined

        aligned.append(
            cv2.warpAffine(
                frame,
                chosen.matrix,
                (ref_w, ref_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        )
        registrations.append(chosen)

    return aligned, registrations


# ---------------------------------------------------------------------------
# Spore cropping and tracking
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Figures and file output
# ---------------------------------------------------------------------------


def load_label_font(size: int) -> ImageFont.ImageFont:
    """Prefer a scalable system font for crisp labels; fall back to Pillow's default."""
    for name in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow older than 10.1 has no sized default font
        return ImageFont.load_default()


def make_contact_sheet(crops: list[np.ndarray], labels: list[str]) -> Image.Image:
    """Preview grid shown when reviewing a candidate spore."""
    frames = frames_to_pil(crops)
    rows = math.ceil(len(frames) / PREVIEW_COLUMNS)
    cell_h = PREVIEW_SIZE + 34
    sheet = Image.new("RGB", (PREVIEW_COLUMNS * PREVIEW_SIZE, rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    font = load_label_font(14)

    for i, (frame, label) in enumerate(zip(frames, labels)):
        thumb = frame.copy()
        thumb.thumbnail(
            (PREVIEW_SIZE - 12, PREVIEW_SIZE - 12), Image.Resampling.LANCZOS
        )
        x = (i % PREVIEW_COLUMNS) * PREVIEW_SIZE
        y = (i // PREVIEW_COLUMNS) * cell_h
        sheet.paste(
            thumb,
            (
                x + (PREVIEW_SIZE - thumb.width) // 2,
                y + (PREVIEW_SIZE - thumb.height) // 2,
            ),
        )
        draw.text((x + 6, y + PREVIEW_SIZE + 5), label, fill="black", font=font)
    return sheet


def save_gif(crops: list[np.ndarray], labels: list[str], output_path: Path) -> None:
    """Animated GIF with each timepoint's label in a bar across the top of the frame."""
    font = load_label_font(16)
    frames = []
    for frame, label in zip(frames_to_pil(crops), labels):
        draw = ImageDraw.Draw(frame)
        draw.rectangle((0, 0, frame.width, GIF_LABEL_BAR_PX), fill="black")
        draw.text((6, 4), label, fill="white", font=font)
        frames.append(frame)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=GIF_FRAME_MS,
        loop=0,
    )


def grid_layout(count: int, columns: int | None = None) -> tuple[int, int]:
    """Return (columns, rows) for a grid of `count` panels.

    The default is the smallest near-square grid: 2 panels give 2x1, 3 give 2x2 with one
    empty cell, 6 give 3x2, 7 give 3x3 with two empty cells. Pass `columns` to force a width.
    """
    if count < 1:
        raise ValueError("A grid needs at least one panel.")
    columns = columns or math.ceil(math.sqrt(count))
    return columns, math.ceil(count / columns)


def grid_cell_origin(index: int, count: int, columns: int) -> tuple[int, int]:
    """Top-left pixel of panel `index` in the grid cell layout (before the panel padding).

    A partial last row is centered under the full rows above it, so it reads as
    intentional in a figure rather than as an unfinished row.
    """
    cell_w = GRID_PANEL_PX + 2 * GRID_PANEL_PAD_PX
    cell_h = cell_w + GRID_LABEL_HEIGHT_PX
    row, col = divmod(index, columns)
    row_start = row * columns
    in_row = min(columns, count - row_start)
    row_offset = (columns - in_row) * cell_w // 2
    return row_offset + col * cell_w, row * cell_h


def save_publication_grid(
    crops: list[np.ndarray],
    labels: list[str],
    output_path: Path,
    columns: int | None = None,
) -> None:
    """Labelled grid JPG for article figures.

    Each panel has a thin black frame and its timepoint label below it, on white.
    Works for any number of timepoints; see grid_layout() for how rows and columns are chosen.
    """
    frames = frames_to_pil(crops)
    count = len(frames)
    columns, rows = grid_layout(count, columns)

    cell_w = GRID_PANEL_PX + 2 * GRID_PANEL_PAD_PX
    cell_h = cell_w + GRID_LABEL_HEIGHT_PX
    sheet = Image.new("RGB", (columns * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    font = load_label_font(GRID_LABEL_FONT_PX)

    for i, (frame, label) in enumerate(zip(frames, labels)):
        cell_x, cell_y = grid_cell_origin(i, count, columns)
        x0 = cell_x + GRID_PANEL_PAD_PX
        y0 = cell_y + GRID_PANEL_PAD_PX
        panel = frame.resize((GRID_PANEL_PX, GRID_PANEL_PX), Image.Resampling.LANCZOS)
        sheet.paste(panel, (x0, y0))
        draw.rectangle(
            (x0 - 1, y0 - 1, x0 + GRID_PANEL_PX, y0 + GRID_PANEL_PX),
            outline="black",
            width=1,
        )
        text_w = draw.textlength(label, font=font)
        draw.text(
            (x0 + (GRID_PANEL_PX - text_w) / 2, y0 + GRID_PANEL_PX + 6),
            label,
            fill="black",
            font=font,
        )

    sheet.save(
        output_path,
        format="JPEG",
        quality=GRID_JPEG_QUALITY,
        dpi=(GRID_DPI, GRID_DPI),
    )


def list_saved_outputs(output_dir: Path | None) -> dict[str, dict[str, Path]]:
    """Find saved GIFs and grid JPGs in the output folder.

    Returns {spore_id: {"gif": path, "grid": path}}, with only the files that exist.
    Read from disk, so it also finds outputs saved in an earlier session.
    """
    if output_dir is None or not output_dir.is_dir():
        return {}
    found: dict[str, dict[str, Path]] = {}
    for path in output_dir.iterdir():
        gif = GIF_NAME_PATTERN.fullmatch(path.name)
        grid = GRID_NAME_PATTERN.fullmatch(path.name)
        if gif:
            found.setdefault(gif.group(1), {})["gif"] = path
        elif grid:
            found.setdefault(grid.group(1), {})["grid"] = path
    return dict(sorted(found.items()))


def delete_output_files(output_dir: Path | None, spore_ids=None) -> list[Path]:
    """Delete saved GIFs and grid JPGs, for the given spore IDs or for all of them.

    TIFF crops, CSV files, and registration diagnostics are never touched.
    Returns the paths that were deleted.
    """
    deleted = []
    for spore_id, files in list_saved_outputs(output_dir).items():
        if spore_ids is not None and spore_id not in spore_ids:
            continue
        for path in files.values():
            path.unlink()
            deleted.append(path)
    return deleted


def load_gif_frames(path: Path) -> tuple[list[Image.Image], list[int]]:
    """Decode every frame of a GIF as RGB, with its display time in milliseconds."""
    frames, durations = [], []
    with Image.open(path) as gif:
        for frame in ImageSequence.Iterator(gif):
            frames.append(frame.convert("RGB"))
            durations.append(
                int(frame.info.get("duration", GIF_FRAME_MS)) or GIF_FRAME_MS
            )
    return frames, durations


def open_in_file_manager(path: Path) -> None:
    """Open a folder in the system file browser."""
    if sys.platform.startswith("win"):
        os.startfile(path)  # opens the user's own output folder
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


def _write_image(path: Path, image: np.ndarray) -> None:
    # cv2.imwrite returns False instead of raising on many failures, so check it.
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write {path}")


def write_registration_diagnostics(
    diag_dir: Path,
    paths: list[Path],
    aligned: list[np.ndarray],
    registrations: list[FrameRegistration],
) -> None:
    """Save a 50/50 overlay of each aligned frame on the reference, and the transforms."""
    diag_dir.mkdir(parents=True, exist_ok=True)
    frames = frames_to_pil(aligned)
    reference = frames[0].convert("RGBA")
    for path, frame in zip(paths[1:], frames[1:]):
        overlay = Image.blend(reference, frame.convert("RGBA"), 0.5)
        overlay.save(diag_dir / f"{path.stem}_overlay.png")

    with open(diag_dir / "transformations.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "filename",
                "method",
                "matches",
                "inliers",
                "inlier_fraction",
                "m00",
                "m01",
                "m02",
                "m10",
                "m11",
                "m12",
            ]
        )
        for path, reg in zip(paths, registrations):
            fraction = reg.inlier_fraction
            writer.writerow(
                [
                    path.name,
                    reg.method,
                    "" if reg.matches is None else reg.matches,
                    "" if reg.inliers is None else reg.inliers,
                    "" if fraction is None else f"{fraction:.4f}",
                    *reg.matrix.flatten().tolist(),
                ]
            )


SESSION_FILE_NAME = "session.json"


def read_spores_csv(output_dir: Path) -> list[tuple[str, int, int, int]]:
    """Read (spore_id, x, y, crop_size) rows from spores.csv. Empty if there is no file."""
    path = output_dir / "spores.csv"
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return [
            (
                row["spore_id"],
                int(row["reference_x"]),
                int(row["reference_y"]),
                int(row["crop_size"]),
            )
            for row in csv.DictReader(f)
        ]


def existing_spore_numbers(output_dir: Path | None) -> set[int]:
    """Numbers of the spores already saved in the output folder (from spores.csv, the crop
    folders, and the GIF and grid names). New spores take numbers above these, so an
    earlier spore's files are never overwritten."""
    if output_dir is None or not output_dir.is_dir():
        return set()
    names = set(list_saved_outputs(output_dir))
    crops_dir = output_dir / "crops"
    if crops_dir.is_dir():
        names |= {p.name for p in crops_dir.iterdir() if p.is_dir()}
    try:
        names |= {row[0] for row in read_spores_csv(output_dir)}
    except (OSError, ValueError, KeyError):
        pass
    numbers = set()
    for name in names:
        match = re.fullmatch(r"S(\d+)", name)
        if match:
            numbers.add(int(match.group(1)))
    return numbers


def folder_fingerprint(paths: list[Path]) -> list[dict]:
    """Identify the input images by name and content hash. A saved selection is restored
    only onto the same images it was made on."""
    fingerprint = []
    for path in paths:
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
        fingerprint.append({"name": path.name, "sha256": digest.hexdigest()})
    return fingerprint


def save_session(output_dir: Path, fingerprint: list[dict], rejected) -> None:
    """Record the images the selection was made on, and the rejected candidates."""
    data = {
        "images": fingerprint,
        "rejected": [[int(x), int(y), int(size)] for x, y, size in rejected],
    }
    with open(output_dir / SESSION_FILE_NAME, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_session(output_dir: Path) -> dict | None:
    """Read session.json. Returns None if it is missing or unreadable."""
    path = output_dir / SESSION_FILE_NAME
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        rejected = [(int(x), int(y), int(size)) for x, y, size in data["rejected"]]
        return {"images": data["images"], "rejected": rejected}
    except (OSError, ValueError, KeyError, TypeError):
        return None


def restore_selection(
    output_dir: Path, fingerprint: list[dict], aligned: list[np.ndarray]
):
    """Work out what an earlier session left in the output folder.

    Returns (kept_spores, rejected, message). The kept spores are restored only when the
    images match the ones they were saved from; their tracking is recomputed on the
    registered frames. The message is empty when the folder has no earlier outputs.
    """
    numbers = existing_spore_numbers(output_dir)
    if not numbers:
        return [], [], ""
    next_id = f"S{max(numbers) + 1:03d}"
    session = load_session(output_dir)
    if session is None:
        return (
            [],
            [],
            (
                "Earlier outputs were found in this folder. They are kept, and new spores "
                f"start at {next_id}."
            ),
        )
    if session["images"] != fingerprint:
        return (
            [],
            [],
            (
                "The earlier selection was not restored because the images have changed since "
                f"it was saved. Earlier outputs are kept, and new spores start at {next_id}."
            ),
        )
    try:
        saved = read_spores_csv(output_dir)
    except (OSError, ValueError, KeyError):
        return (
            [],
            [],
            (
                "The earlier spores.csv could not be read, so no spores were restored. "
                f"New spores start at {next_id}."
            ),
        )
    spores = []
    for spore_id, x, y, size in saved:
        centers, scores = track_spore(aligned, x, y, size)
        spores.append(Spore(spore_id, x, y, size, centers, scores))
    message = (
        f"Restored {len(spores)} kept spores and {len(session['rejected'])} rejected candidates "
        f"from the earlier session. New spores start at {next_id}."
    )
    return spores, session["rejected"], message


def save_spore_outputs(
    output_dir: Path,
    paths: list[Path],
    aligned: list[np.ndarray],
    spores: list[Spore],
) -> None:
    """Write spores.csv and, for each spore: TIFF crops, a GIF, a grid JPG, and alignment CSV.

    Uses the centers stored when each spore was reviewed, so the saved files match exactly
    what the user approved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = [p.stem for p in paths]

    # Merge with spores already listed, so earlier spores stay listed after a reopen.
    rows = {
        spore_id: (x, y, size) for spore_id, x, y, size in read_spores_csv(output_dir)
    }
    for spore in spores:
        rows[spore.spore_id] = (spore.x, spore.y, spore.size)
    with open(output_dir / "spores.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["spore_id", "reference_x", "reference_y", "crop_size"])
        for spore_id in sorted(rows):
            writer.writerow([spore_id, *rows[spore_id]])

    for spore in spores:
        crops = [
            extract_crop(frame, cx, cy, spore.size)
            for frame, (cx, cy) in zip(aligned, spore.centers)
        ]
        spore_dir = output_dir / "crops" / spore.spore_id
        spore_dir.mkdir(parents=True, exist_ok=True)

        for crop, label in zip(crops, labels):
            _write_image(spore_dir / f"{label}.tif", crop)
        save_gif(crops, labels, output_dir / f"{spore.spore_id}.gif")
        save_publication_grid(crops, labels, output_dir / f"{spore.spore_id}_grid.jpg")

        with open(
            spore_dir / "crop_alignment.csv", "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.writer(f)
            writer.writerow(
                ["filename", "crop_center_x", "crop_center_y", "template_match_score"]
            )
            for label, (cx, cy), score in zip(labels, spore.centers, spore.scores):
                writer.writerow([label, cx, cy, score])


# ---------------------------------------------------------------------------
# Zoom and pan geometry (no Tk, so it can be tested directly)
# ---------------------------------------------------------------------------


def _clamp_axis(pan: float, content: float, available: float) -> float:
    """Keep an image axis in view. A smaller-than-view image is centered; a larger one
    may move only far enough that its edges do not pull inside the view."""
    if content <= available:
        # Whole pixels, so the drawn image and the click mapping share the same origin.
        return float(round((available - content) / 2))
    return min(0.0, max(available - content, pan))


@dataclass
class ViewTransform:
    """Maps between image pixels and canvas pixels for a zoomable, pannable view.

    At zoom 1 the whole image is visible. `zoom` multiplies the fit scale, and `pan_x`
    and `pan_y` give the canvas position of image pixel (0, 0).
    """

    image_w: int
    image_h: int
    canvas_w: int = 1
    canvas_h: int = 1
    max_fit_scale: float = 1.0
    zoom: float = ZOOM_MIN
    pan_x: float = 0.0
    pan_y: float = 0.0

    @property
    def fit_scale(self) -> float:
        return min(
            self.canvas_w / self.image_w,
            self.canvas_h / self.image_h,
            self.max_fit_scale,
        )

    @property
    def scale(self) -> float:
        """Canvas pixels per image pixel."""
        return self.fit_scale * self.zoom

    def to_canvas(self, u: float, v: float) -> tuple[float, float]:
        return self.pan_x + u * self.scale, self.pan_y + v * self.scale

    def to_image(self, cx: float, cy: float) -> tuple[float, float]:
        return (cx - self.pan_x) / self.scale, (cy - self.pan_y) / self.scale

    def reset(self) -> None:
        """Fit the whole image and center it."""
        self.zoom = ZOOM_MIN
        self._clamp_pan()

    def resize_canvas(self, width: int, height: int) -> None:
        self.canvas_w, self.canvas_h = max(1, width), max(1, height)
        self._clamp_pan()

    def zoom_at(self, cx: float, cy: float, factor: float) -> None:
        """Zoom by `factor`, keeping the image point under (cx, cy) fixed on screen."""
        u, v = self.to_image(cx, cy)
        self.zoom = min(ZOOM_MAX, max(ZOOM_MIN, self.zoom * factor))
        self.pan_x = cx - u * self.scale
        self.pan_y = cy - v * self.scale
        self._clamp_pan()

    def pan_by(self, dx: float, dy: float) -> None:
        self.pan_x += dx
        self.pan_y += dy
        self._clamp_pan()

    def _clamp_pan(self) -> None:
        self.pan_x = _clamp_axis(self.pan_x, self.image_w * self.scale, self.canvas_w)
        self.pan_y = _clamp_axis(self.pan_y, self.image_h * self.scale, self.canvas_h)


# ---------------------------------------------------------------------------
# Zoomable image view (Tk)
# ---------------------------------------------------------------------------


class ZoomView(tk.Canvas):
    """A canvas that shows one image with zoom and pan, plus optional markers.

    Zoom: mouse wheel or trackpad scroll and pinch, or the +/- buttons in ZoomControls.
    Pan: drag with the left button while zoomed in, or drag with the right or middle button.
    Select: a left click without dragging reports the image pixel under the cursor.
    """

    def __init__(self, master, *, max_fit_scale=1.0, on_click=None):
        super().__init__(master, background="#252525", highlightthickness=0)
        self._max_fit_scale = max_fit_scale
        self._on_click = on_click
        self.on_zoom_changed = None
        self._pil: Image.Image | None = None
        self._view: ViewTransform | None = None
        self._markers: list[tuple[float, float, str, str]] = []
        self._photo = None
        self._press_xy: tuple[int, int] | None = None
        self._last_xy: tuple[int, int] | None = None
        self._dragged = False

        self.bind("<Configure>", self._on_configure)
        self.bind("<MouseWheel>", self._on_wheel)
        self.bind(
            "<Button-4>", lambda e: self._zoom_at(e.x, e.y, ZOOM_STEP)
        )  # Linux up
        self.bind(
            "<Button-5>", lambda e: self._zoom_at(e.x, e.y, 1 / ZOOM_STEP)
        )  # Linux down
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)
        for press in ("<ButtonPress-2>", "<ButtonPress-3>"):
            self.bind(press, self._on_pan_press)
        for motion in ("<B2-Motion>", "<B3-Motion>"):
            self.bind(motion, self._on_pan_drag)

    @property
    def zoom(self) -> float:
        return self._view.zoom if self._view else ZOOM_MIN

    # -- public API ---------------------------------------------------------

    def set_image(self, pil: Image.Image, keep_view: bool = False) -> None:
        """Show an image. The view resets to fit unless keep_view is True and the size
        is unchanged, which lets an animation keep the user's zoom and position."""
        self._pil = pil
        size_changed = (
            self._view is None or (self._view.image_w, self._view.image_h) != pil.size
        )
        if size_changed or not keep_view:
            self._view = ViewTransform(
                pil.width,
                pil.height,
                max(self.winfo_width(), 1),
                max(self.winfo_height(), 1),
                self._max_fit_scale,
            )
            self._view.reset()
            self._notify_zoom()
        self._render()

    def clear(self) -> None:
        self._pil = None
        self._view = None
        self._markers = []
        self.delete("all")
        self._notify_zoom()

    def set_markers(self, markers) -> None:
        """Markers are (image_x, image_y, color, label). An empty label draws a plain ring."""
        self._markers = list(markers)
        self._render()

    def reset_view(self) -> None:
        if self._view:
            self._view.reset()
            self._render()
            self._notify_zoom()

    def zoom_by(self, factor: float) -> None:
        if self._view:
            self._zoom_at(self.winfo_width() / 2, self.winfo_height() / 2, factor)

    # -- rendering ----------------------------------------------------------

    def _render(self) -> None:
        self.delete("all")
        if self._pil is None or self._view is None:
            return
        v = self._view
        s = v.scale

        # Draw only the visible part of the image, so large images stay fast at high zoom.
        u0 = max(0, math.floor(-v.pan_x / s))
        v0 = max(0, math.floor(-v.pan_y / s))
        u1 = min(v.image_w, math.ceil((v.canvas_w - v.pan_x) / s))
        v1 = min(v.image_h, math.ceil((v.canvas_h - v.pan_y) / s))
        if u1 > u0 and v1 > v0:
            region = self._pil.crop((u0, v0, u1, v1))
            size = (max(1, round((u1 - u0) * s)), max(1, round((v1 - v0) * s)))
            # Nearest-neighbour when enlarging keeps real pixels visible; smooth when shrinking.
            resample = Image.Resampling.NEAREST if s >= 2 else Image.Resampling.LANCZOS
            self._photo = ImageTk.PhotoImage(region.resize(size, resample))
            x, y = v.to_canvas(u0, v0)
            self.create_image(round(x), round(y), image=self._photo, anchor="nw")

        for mx, my, color, label in self._markers:
            cx, cy = v.to_canvas(mx + 0.5, my + 0.5)
            r = MARKER_RADIUS_PX
            self.create_oval(cx - r, cy - r, cx + r, cy + r, outline=color, width=2)
            if label:
                self.create_text(
                    cx + r + 3, cy - r - 3, text=label, fill=color, anchor="sw"
                )

    def _notify_zoom(self) -> None:
        if self.on_zoom_changed:
            self.on_zoom_changed(self.zoom)

    def _zoom_at(self, x: float, y: float, factor: float) -> None:
        if self._view is None:
            return
        self._view.zoom_at(x, y, factor)
        self._render()
        self._notify_zoom()

    # -- events -------------------------------------------------------------

    def _on_configure(self, event) -> None:
        if self._view is not None:
            self._view.resize_canvas(event.width, event.height)
            self._render()

    def _on_wheel(self, event) -> None:
        # Windows and most mice send 120 per notch. macOS trackpads send small values, so scale them.
        if not event.delta:
            return
        steps = event.delta / 120 if abs(event.delta) >= 120 else event.delta / 6
        self._zoom_at(event.x, event.y, ZOOM_STEP**steps)

    def _on_press(self, event) -> None:
        self._press_xy = (event.x, event.y)
        self._last_xy = (event.x, event.y)
        self._dragged = False

    def _on_drag(self, event) -> None:
        # Left-drag pans only when zoomed in, so a click at fit zoom still selects.
        if self._press_xy is None or self._view is None or self._view.zoom <= ZOOM_MIN:
            return
        if not self._dragged:
            dx = event.x - self._press_xy[0]
            dy = event.y - self._press_xy[1]
            if math.hypot(dx, dy) < DRAG_THRESHOLD_PX:
                return
        self._dragged = True
        self._pan_to(event.x, event.y)

    def _on_release(self, event) -> None:
        if self._press_xy is None:
            return
        dragged = self._dragged
        self._press_xy = None
        if dragged or self._view is None or self._on_click is None:
            return
        u, v = self._view.to_image(event.x, event.y)
        if 0 <= u < self._view.image_w and 0 <= v < self._view.image_h:
            self._on_click(math.floor(u), math.floor(v))

    def _on_pan_press(self, event) -> None:
        self._last_xy = (event.x, event.y)

    def _on_pan_drag(self, event) -> None:
        if self._view is not None and self._last_xy is not None:
            self._pan_to(event.x, event.y)

    def _pan_to(self, x: int, y: int) -> None:
        lx, ly = self._last_xy
        self._view.pan_by(x - lx, y - ly)
        self._last_xy = (x, y)
        self._render()


class ZoomControls(ttk.Frame):
    """Zoom out, fit, zoom in, and a readout of the current zoom for a ZoomView."""

    def __init__(self, master, view: ZoomView):
        super().__init__(master)
        self._view = view
        self._label = ttk.Label(self, text="1.0×", width=6)
        ttk.Button(
            self, text="−", width=3, command=lambda: view.zoom_by(1 / ZOOM_STEP)
        ).pack(side="left")
        ttk.Button(self, text="Fit", command=view.reset_view).pack(side="left", padx=4)
        ttk.Button(
            self, text="+", width=3, command=lambda: view.zoom_by(ZOOM_STEP)
        ).pack(side="left")
        self._label.pack(side="left", padx=6)
        view.on_zoom_changed = self._show_zoom

    def _show_zoom(self, zoom: float) -> None:
        self._label.config(text=f"{zoom:.1f}×")


# ---------------------------------------------------------------------------
# Output preview window (Tk)
# ---------------------------------------------------------------------------


class OutputPreviewWindow(tk.Toplevel):
    """Browse saved GIFs and grid JPGs for any spore, with zoom, and delete them."""

    def __init__(self, app: SporeApp):
        super().__init__(app.root)
        self.app = app
        self.title("Preview outputs")
        self.geometry("920x720")
        self.protocol("WM_DELETE_WINDOW", self._close)

        self._outputs: dict[str, dict[str, Path]] = {}
        self._current_id: str | None = None
        self._frames: list[Image.Image] = []
        self._durations: list[int] = []
        self._frame_index = 0
        self._playing = True
        self._after_id = None

        top = ttk.Frame(self, padding=8)
        top.pack(fill="x")
        ttk.Label(top, text="Spore:").pack(side="left")
        self.spore_var = tk.StringVar()
        self.spore_box = ttk.Combobox(
            top, textvariable=self.spore_var, state="readonly", width=10
        )
        self.spore_box.pack(side="left", padx=6)
        self.spore_box.bind(
            "<<ComboboxSelected>>", lambda e: self._show(self.spore_var.get())
        )
        self.message = ttk.Label(top, text="")
        self.message.pack(side="left", padx=8)
        ttk.Button(top, text="Open output folder", command=self._open_folder).pack(
            side="right"
        )
        ttk.Button(
            top, text="Delete this spore's GIF + grid", command=self._delete_current
        ).pack(side="right", padx=6)

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        grid_tab = ttk.Frame(notebook)
        gif_tab = ttk.Frame(notebook)
        notebook.add(grid_tab, text="Grid JPG")  # first tab, shown on open
        notebook.add(gif_tab, text="Animated GIF")

        grid_bar = ttk.Frame(grid_tab, padding=(0, 4))
        grid_bar.pack(fill="x")
        self.grid_view = ZoomView(grid_tab, max_fit_scale=PREVIEW_MAX_FIT_SCALE)
        self.grid_view.pack(fill="both", expand=True)
        ZoomControls(grid_bar, self.grid_view).pack(side="right")

        gif_bar = ttk.Frame(gif_tab, padding=(0, 4))
        gif_bar.pack(fill="x")
        self.play_button = ttk.Button(
            gif_bar, text="Pause", width=8, command=self._toggle_play
        )
        self.play_button.pack(side="left")
        self.frame_label = ttk.Label(gif_bar, text="")
        self.frame_label.pack(side="left", padx=8)
        self.gif_view = ZoomView(gif_tab, max_fit_scale=PREVIEW_MAX_FIT_SCALE)
        self.gif_view.pack(fill="both", expand=True)
        ZoomControls(gif_bar, self.gif_view).pack(side="right")

        self.refresh()

    # -- contents -----------------------------------------------------------

    def refresh(self, select: str | None = None) -> None:
        """Re-read the output folder and show the chosen spore, or the current one if it still exists."""
        self._outputs = list_saved_outputs(self.app.output_dir)
        ids = list(self._outputs)
        self.spore_box["values"] = ids
        if not ids:
            self._stop_animation()
            self._current_id = None
            self.spore_var.set("")
            self.gif_view.clear()
            self.grid_view.clear()
            self.frame_label.config(text="")
            self.message.config(text="No GIFs or grids saved.")
            return
        self.message.config(text="")
        if select in self._outputs:
            chosen = select
        elif self._current_id in self._outputs:
            chosen = self._current_id
        else:
            chosen = ids[0]
        self.spore_var.set(chosen)
        self._show(chosen)

    def _show(self, spore_id: str) -> None:
        self._stop_animation()
        self._current_id = spore_id
        files = self._outputs.get(spore_id, {})

        if "grid" in files:
            with Image.open(files["grid"]) as image:
                self.grid_view.set_image(image.convert("RGB"))
        else:
            self.grid_view.clear()

        if "gif" in files:
            self._frames, self._durations = load_gif_frames(files["gif"])
            self._frame_index = 0
            self._playing = True
            self.play_button.config(text="Pause")
            self._show_frame(keep_view=False)
            self._schedule()
        else:
            self._frames, self._durations = [], []
            self.gif_view.clear()
            self.frame_label.config(text="No GIF for this spore")

    def _show_frame(self, keep_view: bool = True) -> None:
        self.gif_view.set_image(self._frames[self._frame_index], keep_view=keep_view)
        self.frame_label.config(
            text=f"Frame {self._frame_index + 1} of {len(self._frames)}"
        )

    # -- animation ----------------------------------------------------------

    def _schedule(self) -> None:
        if self._playing and len(self._frames) > 1:
            self._after_id = self.after(
                self._durations[self._frame_index], self._advance
            )

    def _advance(self) -> None:
        self._after_id = None
        self._frame_index = (self._frame_index + 1) % len(self._frames)
        self._show_frame()
        self._schedule()

    def _stop_animation(self) -> None:
        if self._after_id is not None:
            self.after_cancel(self._after_id)
            self._after_id = None

    def _toggle_play(self) -> None:
        self._playing = not self._playing
        self.play_button.config(text="Pause" if self._playing else "Play")
        if self._playing:
            self._schedule()
        else:
            self._stop_animation()

    # -- actions ------------------------------------------------------------

    def _delete_current(self) -> None:
        spore_id = self._current_id
        if spore_id is None:
            return
        if not messagebox.askyesno(
            "Delete outputs",
            f"Delete the GIF and grid JPG for {spore_id}?\n\n"
            "TIFF crops and CSV files are kept. Saving again recreates them.",
            parent=self,
        ):
            return
        delete_output_files(self.app.output_dir, [spore_id])
        self.app.status.set(f"Deleted the GIF and grid for {spore_id}.")
        self.refresh()

    def _open_folder(self) -> None:
        try:
            open_in_file_manager(self.app.output_dir)
        except OSError as exc:
            messagebox.showerror("Could not open folder", str(exc), parent=self)

    def _close(self) -> None:
        self._stop_animation()
        self.app.preview_window = None
        self.destroy()


# ---------------------------------------------------------------------------
# Main window (Tk)
# ---------------------------------------------------------------------------


class SporeApp:
    """Tkinter front end: load frames, select spores, review, save, and preview outputs."""

    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Spore time-series extractor")
        root.geometry("1050x800")

        self.paths: list[Path] = []
        self.images: list[np.ndarray] = []  # fitted to the reference shape
        self.aligned: list[np.ndarray] = []  # registered frames; empty until registered
        self.registrations: list[FrameRegistration] = []
        self.spores: list[Spore] = []  # kept
        self.rejected: list[
            tuple[int, int, int]
        ] = []  # (x, y, crop size) of rejected candidates
        self.output_dir: Path | None = None
        self.preview_window: OutputPreviewWindow | None = None
        self._fingerprint: list[dict] = []  # identifies the loaded images on disk
        self._busy = False  # a background task is running
        self._busy_window: tk.Toplevel | None = None

        self.crop_size_var = tk.StringVar(value=str(DEFAULT_CROP_SIZE))
        self.status = tk.StringVar(
            value="Load a folder containing one image per timepoint."
        )
        self._reference_pil: Image.Image | None = None

        self._build_toolbar()
        self._build_view()
        self._build_footer()

    # -- layout -------------------------------------------------------------

    def _build_toolbar(self):
        bar = ttk.Frame(self.root, padding=(8, 8, 8, 2))
        bar.pack(fill="x")
        ttk.Button(bar, text="Load image folder", command=self.load_folder).pack(
            side="left"
        )
        ttk.Button(
            bar, text="Save selected spores + outputs", command=self.save_all
        ).pack(side="left", padx=6)
        ttk.Button(bar, text="Preview outputs", command=self.open_preview).pack(
            side="left"
        )
        ttk.Button(
            bar, text="Delete GIFs + grids", command=self.delete_all_outputs
        ).pack(side="left", padx=6)
        ttk.Label(bar, text="Crop size (px):").pack(side="left", padx=(18, 4))
        ttk.Entry(bar, textvariable=self.crop_size_var, width=7).pack(side="left")
        ttk.Label(self.root, textvariable=self.status, padding=(10, 2)).pack(fill="x")

    def _build_view(self):
        zoom_bar = ttk.Frame(self.root, padding=(8, 2))
        zoom_bar.pack(fill="x")
        body = ttk.Frame(self.root, padding=(8, 0, 8, 8))
        body.pack(fill="both", expand=True)

        self.view = ZoomView(
            body, max_fit_scale=REFERENCE_MAX_FIT_SCALE, on_click=self.on_view_click
        )
        self.view.pack(fill="both", expand=True)
        ZoomControls(zoom_bar, self.view).pack(side="left")
        ttk.Label(
            zoom_bar,
            text="Scroll or pinch to zoom · drag to pan when zoomed · click to select",
        ).pack(side="left", padx=12)

    def _build_footer(self):
        footer = ttk.Frame(self.root, padding=8)
        footer.pack(fill="x")
        ttk.Label(
            footer,
            text="Green = kept spore. Red = rejected; click there again to re-select it. "
            "Review each candidate through all timepoints, then keep or reject it.",
        ).pack(anchor="w")
        self.spore_list = tk.Listbox(footer, height=5)
        self.spore_list.pack(fill="x", pady=5)

    # -- loading and registration -------------------------------------------

    def load_folder(self):
        if self._busy:
            return
        folder = filedialog.askdirectory(title="Choose image folder")
        if not folder:
            return
        paths = sorted(
            (
                p
                for p in Path(folder).iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            ),
            key=natural_key,
        )
        if len(paths) < 2:
            messagebox.showerror(
                "Images required", "Choose a folder with at least two image files."
            )
            return
        if len({p.stem for p in paths}) != len(paths):
            messagebox.showerror(
                "Duplicate names",
                "Two images share a file name apart from the extension (for example "
                "a.tif and a.png). Rename them so each timepoint has a unique label.",
            )
            return
        try:
            images = prepare_series(paths, [read_image(p) for p in paths])
        except (OSError, ValueError) as exc:
            messagebox.showerror("Image loading error", str(exc))
            return

        self.paths, self.images = paths, images
        self.aligned, self.registrations, self._fingerprint = [], [], []
        self.spores, self.rejected = [], []
        self.spore_list.delete(0, tk.END)
        self.output_dir = Path(folder) / OUTPUT_FOLDER_NAME
        self.status.set(
            f"Loaded {len(paths)} images. Reference: {paths[0].name}. Registering..."
        )

        self._reference_pil = frames_to_pil([images[0]])[0]
        self.view.set_image(self._reference_pil)
        self._refresh_markers()
        if self.preview_window is not None:
            self.preview_window.refresh()
        self.register()

    def register(self, then=None) -> None:
        """Register all frames to the reference in the background, then restore any earlier
        selection saved in the output folder. Calls then() on success, if given."""
        if not self.images:
            messagebox.showinfo("Load images", "Load an image folder first.")
            return
        if self._busy:
            return
        images, paths, output_dir = self.images, self.paths, self.output_dir

        def work():
            aligned, registrations = register_images(images)
            fingerprint = folder_fingerprint(paths)
            restored = restore_selection(output_dir, fingerprint, aligned)
            return aligned, registrations, fingerprint, restored

        def done(result):
            (
                aligned,
                registrations,
                fingerprint,
                (spores, rejected, restore_message),
            ) = result
            self.aligned, self.registrations, self._fingerprint = (
                aligned,
                registrations,
                fingerprint,
            )
            self.spores, self.rejected = [], list(rejected)
            self.spore_list.delete(0, tk.END)
            for spore in spores:
                self._add_kept(spore)
            self._refresh_markers()
            summary = "; ".join(
                f"{p.name}: {r.method}"
                + ("" if r.inliers is None else f" ({r.inliers}/{r.matches} inliers)")
                for p, r in zip(paths[1:], registrations[1:])
            )
            self.status.set(
                "Registration complete. Review alignment before selecting spores. "
                + summary
                + (f" {restore_message}" if restore_message else "")
            )
            if then is not None:
                then()

        def failed(exc):
            if isinstance(exc, RegistrationError):
                messagebox.showerror("Registration failed", str(exc))
            else:
                messagebox.showerror("Registration error", str(exc))
            self.status.set("Registration failed. Original images remain loaded.")

        self._run_in_background(
            "Registering images to the first timepoint.\nThis can take a little while.",
            work,
            done,
            failed,
        )

    def _run_in_background(self, message, work, on_done, on_error):
        """Run work() in a thread while a 'please wait' window is shown. Only the main thread
        touches Tk: the result is passed back through a queue and handled by polling."""
        self._busy = True
        self._show_busy(message)
        results: queue.Queue = queue.Queue()

        def runner():
            try:
                results.put((True, work()))
            except Exception as exc:  # reported to the user on the main thread
                results.put((False, exc))

        threading.Thread(target=runner, daemon=True).start()
        self.root.after(100, lambda: self._poll_background(results, on_done, on_error))

    def _poll_background(self, results, on_done, on_error):
        try:
            ok, payload = results.get_nowait()
        except queue.Empty:
            self.root.after(
                100, lambda: self._poll_background(results, on_done, on_error)
            )
            return
        self._busy = False
        self._hide_busy()
        if ok:
            on_done(payload)
        else:
            on_error(payload)

    def _show_busy(self, message):
        win = tk.Toplevel(self.root)
        win.title("Please wait")
        win.transient(self.root)
        win.resizable(False, False)
        win.protocol("WM_DELETE_WINDOW", lambda: None)  # cannot be closed while working
        ttk.Label(win, text=message, padding=(20, 16, 20, 8), justify="center").pack()
        bar = ttk.Progressbar(win, mode="indeterminate", length=260)
        bar.pack(padx=20, pady=(0, 18))
        bar.start(12)
        self._busy_window = win
        self.root.config(cursor="watch")
        try:
            win.grab_set()  # ignore clicks on the main window until the task finishes
        except tk.TclError:
            pass  # some window managers refuse a grab before the window is mapped

    def _hide_busy(self):
        if self._busy_window is not None:
            self._busy_window.grab_release()
            self._busy_window.destroy()
            self._busy_window = None
        self.root.config(cursor="")

    def _spore_line(self, spore: Spore) -> str:
        return f"{spore.spore_id}: x={spore.x}, y={spore.y}, size={spore.size}"

    def _add_kept(self, spore: Spore) -> None:
        self.spores.append(spore)
        self.spore_list.insert(tk.END, self._spore_line(spore))

    def _new_spore_id(self) -> str:
        """Next free ID. Numbers already on disk count too, so earlier files are not overwritten."""
        used = existing_spore_numbers(self.output_dir) | {
            int(s.spore_id[1:]) for s in self.spores
        }
        return f"S{max(used, default=0) + 1:03d}"

    # -- spore selection ----------------------------------------------------

    def _refresh_markers(self):
        markers = [(s.x, s.y, KEPT_MARKER_COLOR, s.spore_id) for s in self.spores]
        markers += [(x, y, REJECTED_MARKER_COLOR, "") for x, y, _size in self.rejected]
        self.view.set_markers(markers)

    def _read_crop_size(self) -> int | None:
        try:
            size = int(self.crop_size_var.get())
        except ValueError:
            size = 0
        if size < MIN_CROP_SIZE:
            messagebox.showerror(
                "Invalid crop size",
                f"Enter a whole number of at least {MIN_CROP_SIZE} pixels.",
            )
            return None
        return size

    def _forget_rejections_near(self, x: int, y: int, size: int) -> None:
        """Drop red markers that a new decision at (x, y) replaces, so re-selecting an area
        does not leave a stack of red circles behind."""
        radius = max(size / 2, MARKER_RADIUS_PX)
        self.rejected = [
            r for r in self.rejected if math.hypot(r[0] - x, r[1] - y) >= radius
        ]

    def on_view_click(self, x: int, y: int):
        if not self.images or self._busy:
            return
        if not self.aligned:
            self.register(then=lambda: self.on_view_click(x, y))
            return
        size = self._read_crop_size()
        if size is None:
            return

        centers, scores = track_spore(self.aligned, x, y, size)
        crops = [
            extract_crop(frame, cx, cy, size)
            for frame, (cx, cy) in zip(self.aligned, centers)
        ]
        labels = [p.stem for p in self.paths]
        candidate = Spore("", x, y, size, centers, scores)
        self.review_candidate(candidate, crops, labels)

    def review_candidate(self, candidate: Spore, crops, labels):
        win = tk.Toplevel(self.root)
        win.title(f"Review candidate at ({candidate.x}, {candidate.y})")

        sheet = make_contact_sheet(crops, labels)
        photo = ImageTk.PhotoImage(sheet)
        canvas = tk.Canvas(
            win, width=sheet.width, height=sheet.height, highlightthickness=0
        )
        canvas.pack(fill="both", expand=True)
        canvas.create_image(0, 0, image=photo, anchor="nw")
        canvas.image = photo  # keep a reference so Tk does not discard the image

        controls = ttk.Frame(win, padding=8)
        controls.pack(fill="x")
        ttk.Label(
            controls,
            text="Review the chronological crops. Keep this spore if "
            "the sequence is useful for your figure.",
        ).pack(side="left")

        def keep():
            # The candidate carries its own crop size, so changing the size box later
            # cannot make the saved spore differ from the crops that were reviewed.
            self._forget_rejections_near(candidate.x, candidate.y, candidate.size)
            candidate.spore_id = self._new_spore_id()
            self._add_kept(candidate)
            self._refresh_markers()
            self.status.set(
                f"Selected {len(self.spores)} spores. Click another spore or save."
            )
            win.destroy()

        def reject():
            self._forget_rejections_near(candidate.x, candidate.y, candidate.size)
            self.rejected.append((candidate.x, candidate.y, candidate.size))
            self._refresh_markers()
            self.status.set(
                f"Rejected the candidate at ({candidate.x}, {candidate.y}). "
                "Click there again to re-select it."
            )
            win.destroy()

        ttk.Button(controls, text="Keep spore", command=keep).pack(side="right", padx=4)
        ttk.Button(controls, text="Reject", command=reject).pack(side="right")
        win.protocol(
            "WM_DELETE_WINDOW", reject
        )  # closing the window counts as rejecting

    # -- saving and outputs -------------------------------------------------

    def save_all(self):
        if not self.spores:
            messagebox.showinfo(
                "No spores selected", "Select and keep at least one spore first."
            )
            return
        if not self.aligned:
            messagebox.showerror("Not registered", "Register the images before saving.")
            return
        paths, aligned, registrations = self.paths, self.aligned, self.registrations
        spores, rejected = list(self.spores), list(self.rejected)
        output_dir, fingerprint = self.output_dir, self._fingerprint

        def work():
            save_spore_outputs(output_dir, paths, aligned, spores)
            write_registration_diagnostics(
                output_dir / "registration", paths, aligned, registrations
            )
            save_session(output_dir, fingerprint, rejected)

        def done(_):
            self.status.set(
                f"Saved {len(spores)} spore sequences to {output_dir}. "
                "Registration diagnostics are in registration/."
            )
            if self.preview_window is not None:
                self.preview_window.refresh()

        def failed(exc):
            messagebox.showerror("Save failed", str(exc))
            self.status.set("Save failed. See the error message.")

        self._run_in_background(
            "Saving spores, GIFs and grids.\nThis can take a little while.",
            work,
            done,
            failed,
        )

    def open_preview(self):
        if self.preview_window is not None:
            self.preview_window.refresh()
            self.preview_window.lift()
            return
        if not list_saved_outputs(self.output_dir):
            messagebox.showinfo(
                "No outputs yet",
                "Save selected spores first. Their GIFs and grid JPGs can then be previewed here.",
            )
            return
        self.preview_window = OutputPreviewWindow(self)

    def delete_all_outputs(self):
        outputs = list_saved_outputs(self.output_dir)
        file_count = sum(len(files) for files in outputs.values())
        if file_count == 0:
            messagebox.showinfo(
                "Nothing to delete", "No GIFs or grid JPGs are saved yet."
            )
            return
        if not messagebox.askyesno(
            "Delete GIFs + grids",
            f"Delete {file_count} GIF and grid JPG file(s) for {len(outputs)} spore(s) in\n"
            f"{self.output_dir}?\n\nTIFF crops and CSV files are kept. "
            "Saving again recreates the GIFs and grids.",
        ):
            return
        deleted = delete_output_files(self.output_dir)
        self.status.set(f"Deleted {len(deleted)} GIF and grid files.")
        if self.preview_window is not None:
            self.preview_window.refresh()


def main():
    root = tk.Tk()
    SporeApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
