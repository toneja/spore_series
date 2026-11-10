"""The main window: load images, select and review spores, save, and open the preview."""
from __future__ import annotations
from pathlib import Path
from tkinter import filedialog
from tkinter import messagebox
from tkinter import ttk
import math
import queue
import threading
import tkinter as tk

from PIL import Image
from PIL import ImageTk
import numpy as np

from .config import (
    DEFAULT_CROP_SIZE,
    IMAGE_EXTENSIONS,
    KEPT_MARKER_COLOR,
    MARKER_RADIUS_PX,
    MIN_CROP_SIZE,
    OUTPUT_FOLDER_NAME,
    REFERENCE_MAX_FIT_SCALE,
    REJECTED_MARKER_COLOR,
)
from .files import delete_output_files, existing_spore_numbers, list_saved_outputs
from .imaging import frames_to_pil, natural_key, prepare_series, read_image
from .models import FrameRegistration, RegistrationError, Spore
from .outputs import (
    make_contact_sheet,
    save_spore_outputs,
    write_registration_diagnostics,
)
from .preview import OutputPreviewWindow
from .registration import register_images
from .session import folder_fingerprint, restore_selection, save_session
from .tracking import extract_crop, track_spore
from .zoom import ZoomControls, ZoomView


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
