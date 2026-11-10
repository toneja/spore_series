"""Recording and restoring an earlier selection, so a reopened folder continues where it
left off."""
from __future__ import annotations
from pathlib import Path
import hashlib
import json

import numpy as np

from .config import SESSION_FILE_NAME
from .files import existing_spore_numbers, read_spores_csv
from .models import Spore
from .tracking import track_spore


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
