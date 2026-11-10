"""Window for previewing and deleting saved GIFs and grids."""
from __future__ import annotations

from typing import TYPE_CHECKING
from pathlib import Path
from tkinter import messagebox
from tkinter import ttk
import tkinter as tk

from PIL import Image

from .config import PREVIEW_MAX_FIT_SCALE
from .files import (
    delete_output_files,
    list_saved_outputs,
    load_gif_frames,
    open_in_file_manager,
)
from .zoom import ZoomControls, ZoomView


if TYPE_CHECKING:  # the type is only needed for annotations; avoids an import cycle
    from .app import SporeApp


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
