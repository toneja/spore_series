"""Aligning every frame to the reference frame. Each method proposes a transform,
and register_images accepts the first one that passes the checks."""
from __future__ import annotations
import math

import cv2
import numpy as np

from .config import (
    ECC_MAX_ITERATIONS,
    ECC_MIN_CORRELATION,
    IDENTITY_AFFINE,
    LOWE_RATIO,
    MAX_FEATURES,
    MAX_ROTATION_DEG,
    MAX_SCALE_DEVIATION,
    MIN_ACCEPTED_INLIERS,
    MIN_FEATURE_MATCHES,
    RANSAC_THRESHOLD_PX,
    SPORE_ANGLE_RANGE_DEG,
    SPORE_ANGLE_STEP_DEG,
    SPORE_MATCH_PX,
    SPORE_MAX_AREA_PX,
    SPORE_MIN_AREA_PX,
    SPORE_MIN_MATCHES,
    SPORE_MIN_SIDE_PX,
    SPORE_SHIFT_CANDIDATES,
)
from .imaging import _raw_gray_uint8, to_gray_uint8
from .models import FrameRegistration, RegistrationError


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
