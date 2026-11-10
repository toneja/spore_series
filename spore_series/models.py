"""Data types shared by the spore series modules."""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np


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
