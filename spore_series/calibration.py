"""Reading the pixel size (microns per pixel) from image metadata.

Microscope software writes the calibration into different places, so several are checked:
TIFF resolution tags (inch or centimetre units), OME-TIFF physical sizes, ImageJ's
"unit=micron" convention, and the DPI stored in PNG and JPEG files. Values that look like
software defaults (72 or 96 dpi) are ignored, because they are not calibrations.
"""
from __future__ import annotations

import math
import re

from PIL import Image

from .config import (
    DEFAULT_DPI_VALUES,
    SCALE_MAX_UM_PER_PX,
    SCALE_MIN_UM_PER_PX,
)

TIFF_DESCRIPTION = 270
TIFF_X_RESOLUTION = 282
TIFF_RESOLUTION_UNIT = 296
OME_PHYSICAL_SIZE_X = re.compile(
    r'PhysicalSizeX="([0-9.eE+-]+)"(?:\s+PhysicalSizeXUnit="([^"]*)")?'
)
# "?m" is how an ASCII-only writer stores a micro sign, so it is read as micrometres too.
UNIT_TO_UM = {
    "µm": 1.0,
    "μm": 1.0,
    "?m": 1.0,
    "um": 1.0,
    "micron": 1.0,
    "microns": 1.0,
    "nm": 1e-3,
    "mm": 1e3,
    "cm": 1e4,
    "m": 1e6,
}
MICRONS_PER_INCH = 25400.0
MICRONS_PER_CM = 10000.0


def read_pixel_size(path) -> tuple[float | None, str]:
    """Return (microns_per_pixel, description) for one image file.

    microns_per_pixel is None when the file has no usable calibration. The description says
    where the value came from, or why it was not used.
    """
    try:
        with Image.open(path) as im:
            if im.format == "TIFF":
                return _from_tiff(im)
            return _from_dpi(im)
    except (OSError, ValueError) as exc:
        return None, f"could not read metadata ({exc})"


def _checked(um_per_px: float, source: str) -> tuple[float | None, str]:
    if not math.isfinite(um_per_px) or not (
        SCALE_MIN_UM_PER_PX <= um_per_px <= SCALE_MAX_UM_PER_PX
    ):
        return (
            None,
            f"{source}: {um_per_px:g} µm/px is outside the plausible range and was ignored",
        )
    return um_per_px, source


def _as_float(value) -> float:
    """TIFF resolution tags may be rationals stored as (numerator, denominator) pairs."""
    if isinstance(value, tuple):
        return value[0] / value[1]
    return float(value)


def _from_tiff(im: Image.Image) -> tuple[float | None, str]:
    tags = im.tag_v2
    description = str(tags.get(TIFF_DESCRIPTION, ""))

    ome = OME_PHYSICAL_SIZE_X.search(description)
    if ome:
        value = float(ome.group(1))
        unit = ome.group(2) or "µm"
        if unit in UNIT_TO_UM:
            return _checked(
                value * UNIT_TO_UM[unit], f"OME-TIFF PhysicalSizeX = {value:g} {unit}"
            )

    if TIFF_X_RESOLUTION not in tags:
        return None, "no resolution tags in the TIFF"
    x_res = _as_float(tags[TIFF_X_RESOLUTION])
    if x_res <= 0:
        return None, "the TIFF resolution tag is zero"
    unit = tags.get(TIFF_RESOLUTION_UNIT, 2)

    if unit == 1 and "unit=micron" in description:  # ImageJ: pixels per micron
        return _checked(1.0 / x_res, "ImageJ metadata (unit=micron)")
    if unit == 3:  # pixels per centimetre
        return _checked(MICRONS_PER_CM / x_res, f"TIFF resolution {x_res:g} pixels/cm")
    if unit == 2:  # pixels per inch
        if x_res in DEFAULT_DPI_VALUES:
            return (
                None,
                f"TIFF resolution {x_res:g} dpi looks like a software default and was ignored",
            )
        return _checked(MICRONS_PER_INCH / x_res, f"TIFF resolution {x_res:g} dpi")
    return None, "the TIFF has no calibrated resolution unit"


def _from_dpi(im: Image.Image) -> tuple[float | None, str]:
    dpi = im.info.get("dpi")
    if not dpi:
        return None, "no resolution stored in the file"
    x_dpi = float(dpi[0])
    if x_dpi in DEFAULT_DPI_VALUES:
        return None, f"{x_dpi:g} dpi looks like a software default and was ignored"
    return _checked(MICRONS_PER_INCH / x_dpi, f"{im.format} resolution {x_dpi:g} dpi")
