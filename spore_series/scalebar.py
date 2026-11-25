"""Scale bars for the GIF frames and the grid panels."""
from __future__ import annotations

from PIL import Image, ImageDraw

from .config import SCALE_BAR_CANDIDATES_UM, SCALE_BAR_MAX_FRACTION


def choose_bar_um(field_um: float) -> float:
    """Longest round bar length that fits within SCALE_BAR_MAX_FRACTION of the field."""
    limit = SCALE_BAR_MAX_FRACTION * field_um
    fitting = [c for c in SCALE_BAR_CANDIDATES_UM if c <= limit]
    return max(fitting) if fitting else min(SCALE_BAR_CANDIDATES_UM)


def bar_label(bar_um: float) -> str:
    if bar_um >= 1000:
        return f"{bar_um / 1000:g} mm"
    return f"{bar_um:g} µm"


def draw_scale_bar(
    image: Image.Image,
    um_per_px: float,
    output_px_per_source_px: float,
    font,
) -> None:
    """Draw a scale bar in the bottom-right corner of `image`, in place.

    um_per_px is the calibration of the source pixels. output_px_per_source_px is how many
    output pixels one source pixel occupies: 1.0 for the GIF frames, and the panel size divided
    by the crop size for grid panels. The bar is white with a black edge, so it reads on
    both light and dark images.
    """
    um_per_out_px = um_per_px / output_px_per_source_px
    bar_um = choose_bar_um(image.width * um_per_out_px)
    bar_px = bar_um / um_per_out_px
    if bar_px > 0.5 * image.width:
        return  # the image is too small for any bar to fit sensibly

    draw = ImageDraw.Draw(image)
    thickness = max(3, image.height // 100)
    margin = max(6, image.width // 40)
    x1 = image.width - margin
    x0 = x1 - bar_px
    y1 = image.height - margin
    y0 = y1 - thickness

    draw.rectangle((x0 - 1, y0 - 1, x1 + 1, y1 + 1), fill="black")
    draw.rectangle((x0, y0, x1, y1), fill="white")

    label = bar_label(bar_um)
    left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
    text_w, text_h = right - left, bottom - top
    text_x = (x0 + x1) / 2 - text_w / 2
    text_y = y0 - text_h - 4
    draw.text(
        (text_x, text_y),
        label,
        fill="white",
        font=font,
        stroke_width=1,
        stroke_fill="black",
    )
