"""Writing the per-spore outputs: TIFF crops, GIF, labelled grid JPG, CSV rows and
registration diagnostics."""
from __future__ import annotations
from pathlib import Path
import csv
import math

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import cv2
import numpy as np

from .config import (
    FONT_CANDIDATES,
    GIF_FRAME_MS,
    GIF_LABEL_BAR_PX,
    GRID_DPI,
    GRID_JPEG_QUALITY,
    GRID_LABEL_FONT_PX,
    GRID_LABEL_HEIGHT_PX,
    GRID_PANEL_PAD_PX,
    GRID_PANEL_PX,
    PREVIEW_COLUMNS,
    PREVIEW_SIZE,
)
from .files import read_spores_csv
from .imaging import frames_to_pil
from .models import FrameRegistration, Spore
from .tracking import extract_crop


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
