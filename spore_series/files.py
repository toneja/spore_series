"""Finding, reading and deleting the files in a spore_output folder."""
from __future__ import annotations
from pathlib import Path
import csv
import os
import re
import subprocess
import sys

from PIL import Image
from PIL import ImageSequence

from .config import GIF_FRAME_MS, GIF_NAME_PATTERN, GRID_NAME_PATTERN


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
