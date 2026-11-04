#!/usr/bin/env python3
"""Semi-automated spore time-series registration, ROI extraction, GIF creation,
and publication-ready grid (JPG) figure creation.

Dependencies:
    pip install opencv-python pillow numpy

Usage:
    python spore_series.py

Select a folder containing one image per timepoint. Images are naturally sorted
by filename; the first image is the registration reference.
"""
from __future__ import annotations

import csv
import re
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
PREVIEW_SIZE = 230
DEFAULT_CROP_SIZE = 180
GIF_DURATION_MS = 700

# Publication grid (JPG) settings
GRID_PANEL_SIZE = 400  # px per panel edge in the output JPG
GRID_PANEL_PAD = 14  # white margin around each panel
GRID_LABEL_HEIGHT = 40  # px reserved below each panel for the timepoint label
GRID_LABEL_FONT_SIZE = 26
GRID_JPEG_QUALITY = 95
GRID_DPI = 300


def natural_key(path: Path):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def read_image(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def display_rgb(image):
    """Convert an OpenCV image to an 8-bit RGB image for Tk/Pillow display."""
    if image.ndim == 2:
        gray = image
        if gray.dtype != np.uint8:
            lo, hi = np.percentile(gray, (1, 99))
            gray = np.clip(
                (gray.astype(np.float32) - lo) / max(float(hi - lo), 1.0) * 255, 0, 255
            ).astype(np.uint8)
        return Image.fromarray(gray).convert("RGB")

    arr = image
    if arr.dtype != np.uint8:
        lo, hi = np.percentile(arr, (1, 99))
        arr = np.clip(
            (arr.astype(np.float32) - lo) / max(float(hi - lo), 1.0) * 255, 0, 255
        ).astype(np.uint8)
    if arr.shape[2] == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(arr)


def to_gray(image):
    if image.ndim == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if gray.dtype != np.uint8:
        lo, hi = np.percentile(gray, (1, 99))
        gray = np.clip(
            (gray.astype(np.float32) - lo) / max(float(hi - lo), 1.0) * 255, 0, 255
        ).astype(np.uint8)
    return gray


def match_image_shape(image, target_shape):
    """Pad or center-trim an image to match the reference; never rescale pixels."""
    target_h, target_w = target_shape[:2]
    height, width = image.shape[:2]

    if height > target_h:
        top = (height - target_h) // 2
        image = image[top : top + target_h, :]
    if width > target_w:
        left = (width - target_w) // 2
        image = image[:, left : left + target_w]

    height, width = image.shape[:2]
    pad_top = max(0, (target_h - height) // 2)
    pad_bottom = max(0, target_h - height - pad_top)
    pad_left = max(0, (target_w - width) // 2)
    pad_right = max(0, target_w - width - pad_left)
    if any((pad_top, pad_bottom, pad_left, pad_right)):
        image = cv2.copyMakeBorder(
            image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT
        )
    return image


def align_spore_centers(images, x, y, size):
    """Track the central spore body locally; crop windows move, image pixels do not."""
    template_size = max(16, int(size * 0.6))
    half = template_size // 2
    template = to_gray(extract_crop(images[0], x, y, template_size))
    radius = max(8, size // 4)
    centers, scores = [(x, y)], [1.0]

    for image in images[1:]:
        gray = to_gray(image)
        left = max(0, x - radius - half)
        top = max(0, y - radius - half)
        right = min(gray.shape[1], x + radius + half + 1)
        bottom = min(gray.shape[0], y + radius + half + 1)
        search = gray[top:bottom, left:right]
        if search.shape[0] < template_size or search.shape[1] < template_size:
            centers.append((x, y))
            scores.append(0.0)
            continue

        result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, location = cv2.minMaxLoc(result)
        if not np.isfinite(score) or score < 0.35:
            centers.append((x, y))
            scores.append(float(score) if np.isfinite(score) else 0.0)
        else:
            centers.append((left + location[0] + half, top + location[1] + half))
            scores.append(float(score))
    return centers, scores


def register_images(images):
    """Register each image directly to image 0 using background feature matches."""
    reference = images[0]
    ref_gray = to_gray(reference)
    if hasattr(cv2, "SIFT_create"):
        detector = cv2.SIFT_create(nfeatures=5000)
        norm = cv2.NORM_L2
        method = "SIFT"
    else:
        detector = cv2.ORB_create(nfeatures=5000)
        norm = cv2.NORM_HAMMING
        method = "ORB"

    kp_ref, des_ref = detector.detectAndCompute(ref_gray, None)
    if des_ref is None or len(kp_ref) < 4:
        raise RuntimeError("Not enough features in the reference image.")

    registered = [reference]
    transforms = [np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])]
    reports = [("reference", len(kp_ref), len(kp_ref), 1.0)]

    matcher = cv2.BFMatcher(norm)
    for index, image in enumerate(images[1:], start=1):
        kp_img, des_img = detector.detectAndCompute(to_gray(image), None)
        if des_img is None or len(kp_img) < 4:
            raise RuntimeError(f"Not enough features in image {index + 1}.")
        pairs = matcher.knnMatch(des_img, des_ref, k=2)
        good = [
            m
            for pair in pairs
            if len(pair) == 2
            for m, n in [pair]
            if m.distance < 0.72 * n.distance
        ]
        if len(good) < 4:
            raise RuntimeError(
                f"Only {len(good)} feature matches in {index + 1} images; "
                "registration could not be estimated."
            )

        src = np.float32([kp_img[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp_ref[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        matrix, inliers = cv2.estimateAffinePartial2D(
            src,
            dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=5000,
            confidence=0.99,
            refineIters=10,
        )
        if matrix is None or inliers is None:
            raise RuntimeError(f"Could not estimate alignment for image {index + 1}.")
        inlier_count = int(inliers.sum())
        inlier_fraction = inlier_count / len(good)
        if inlier_count < 4:
            raise RuntimeError(f"Too few reliable matches in image {index + 1}.")

        aligned = cv2.warpAffine(
            image,
            matrix,
            (reference.shape[1], reference.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        registered.append(aligned)
        transforms.append(matrix)
        reports.append((method, len(good), inlier_count, inlier_fraction))

    return registered, transforms, reports


def extract_crop(image, x, y, size):
    """Extract a fixed-size crop, padding at image edges if necessary."""
    half = size // 2
    left, top = x - half, y - half
    right, bottom = left + size, top + size
    pad_left = max(0, -left)
    pad_top = max(0, -top)
    pad_right = max(0, right - image.shape[1])
    pad_bottom = max(0, bottom - image.shape[0])
    if any((pad_left, pad_top, pad_right, pad_bottom)):
        image = cv2.copyMakeBorder(
            image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT
        )
        left += pad_left
        top += pad_top
    return image[top : top + size, left : left + size].copy()


def make_contact_sheet(crops, labels, title):
    columns = 4
    rows = (len(crops) + columns - 1) // columns
    cell_w, cell_h = PREVIEW_SIZE, PREVIEW_SIZE + 34
    sheet = Image.new("RGB", (columns * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (crop, label) in enumerate(zip(crops, labels)):
        thumb = display_rgb(crop)
        thumb.thumbnail((PREVIEW_SIZE - 12, PREVIEW_SIZE - 12))
        x = (i % columns) * cell_w
        y = (i // columns) * cell_h
        sheet.paste(
            thumb,
            (
                x + (PREVIEW_SIZE - thumb.width) // 2,
                y + (PREVIEW_SIZE - thumb.height) // 2,
            ),
        )
        draw.text((x + 6, y + PREVIEW_SIZE + 5), label, fill="black")
    return sheet


def save_gif(crops, labels, output_path):
    frames = []
    for crop, label in zip(crops, labels):
        frame = display_rgb(crop).convert("RGB")
        draw = ImageDraw.Draw(frame)
        draw.rectangle((0, 0, frame.width, 24), fill="black")
        draw.text((6, 5), label, fill="white")
        frames.append(frame)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=GIF_DURATION_MS,
        loop=0,
        optimize=False,
    )


def _grid_font(size):
    """Prefer a scalable font for crisp labels; fall back to PIL's default."""
    for name in (
        "arial.ttf",
        "Arial.ttf",
        "DejaVuSans.ttf",
        "LiberationSans-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def make_publication_grid(crops, labels, output_path, columns=None):
    """Save timepoint crops as a labeled grid JPG suitable for article figures.

    Layout defaults to a near-square grid (e.g. 6 images -> 3 columns x 2 rows).
    Each panel is labeled beneath with its timepoint name on a white background
    with thin frames, so the figure reads cleanly in print.
    """
    n = len(crops)
    if columns is None:
        columns = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / columns))

    pad = GRID_PANEL_PAD
    label_h = GRID_LABEL_HEIGHT
    panel = GRID_PANEL_SIZE
    cell_w = panel + 2 * pad
    cell_h = panel + 2 * pad + label_h

    sheet = Image.new("RGB", (columns * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    font = _grid_font(GRID_LABEL_FONT_SIZE)

    for i, (crop, label) in enumerate(zip(crops, labels)):
        thumb = display_rgb(crop).convert("RGB")
        thumb = thumb.resize((panel, panel), Image.Resampling.LANCZOS)
        col, row = i % columns, i // columns
        x0 = col * cell_w + pad
        y0 = row * cell_h + pad
        sheet.paste(thumb, (x0, y0))
        draw.rectangle(
            (x0 - 1, y0 - 1, x0 + panel, y0 + panel), outline="black", width=1
        )
        text_w = draw.textlength(label, font=font)
        draw.text(
            (x0 + (panel - text_w) / 2, y0 + panel + 6),
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


class SporeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Spore time-series extractor")
        self.root.geometry("1050x800")
        self.paths = []
        self.images = []
        self.registered = []
        self.transforms = []
        self.registration_reports = []
        self.spores = []
        self.reference_photo = None
        self.canvas_image_id = None
        self.scale = 1.0
        self.offset = (0, 0)
        self.output_dir = None

        top = ttk.Frame(root, padding=8)
        top.pack(fill="x")
        ttk.Button(top, text="1. Load image folder", command=self.load_folder).pack(
            side="left"
        )
        ttk.Button(top, text="2. Register images", command=self.register).pack(
            side="left", padx=6
        )
        ttk.Button(
            top, text="Save selected spores + GIFs + grids", command=self.save_all
        ).pack(side="left")
        ttk.Label(top, text="Crop size (px):").pack(side="left", padx=(18, 4))
        self.crop_size_var = tk.StringVar(value=str(DEFAULT_CROP_SIZE))
        ttk.Entry(top, textvariable=self.crop_size_var, width=7).pack(side="left")

        self.status = tk.StringVar(
            value="Load a folder containing one image per timepoint."
        )
        ttk.Label(root, textvariable=self.status, padding=(10, 2)).pack(fill="x")

        body = ttk.Frame(root, padding=8)
        body.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(body, background="#252525", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Button-1>", self.select_spore)
        self.canvas.bind("<Configure>", self.redraw_reference)

        bottom = ttk.Frame(root, padding=8)
        bottom.pack(fill="x")
        ttk.Label(
            bottom,
            text="Click the center of a spore in the registered reference image. "
            "Review all timepoints, then keep or reject the candidate.",
        ).pack(anchor="w")
        self.spore_list = tk.Listbox(bottom, height=5)
        self.spore_list.pack(fill="x", pady=5)

    def load_folder(self):
        folder = filedialog.askdirectory(title="Choose image folder")
        if not folder:
            return
        paths = sorted(
            [p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS],
            key=natural_key,
        )
        if len(paths) < 2:
            messagebox.showerror(
                "Images required", "Choose a folder with at least two image files."
            )
            return
        try:
            images = [read_image(p) for p in paths]
            shape = images[0].shape
            if any(
                im.ndim != images[0].ndim
                or (im.ndim == 3 and im.shape[2] != shape[2])
                or im.dtype != images[0].dtype
                for im in images
            ):
                raise ValueError(
                    "Images must have the same number of channels and bit depth."
                )
            images = [match_image_shape(im, shape) for im in images]
        except Exception as exc:
            messagebox.showerror("Image loading error", str(exc))
            return
        self.paths, self.images = paths, images
        self.registered = []
        self.spores = []
        self.spore_list.delete(0, tk.END)
        self.output_dir = Path(folder) / "spore_output"
        self.status.set(f"Loaded {len(paths)} images. Reference: {paths[0].name}")
        self.show_image(images[0])

    def register(self):
        if not self.images:
            messagebox.showinfo("Load images", "Load an image folder first.")
            return
        self.status.set("Registering images to the first timepoint...")
        self.root.update_idletasks()
        try:
            registered, transforms, reports = register_images(self.images)
        except Exception as exc:
            messagebox.showerror("Registration failed", str(exc))
            self.status.set("Registration failed. Original images remain loaded.")
            return
        self.registered = registered
        self.transforms = transforms
        self.registration_reports = reports
        self.show_image(self.registered[0])
        summary = "; ".join(
            f"{self.paths[i].name}: {r[2]}/{r[1]} inliers"
            for i, r in enumerate(reports[1:], start=1)
        )
        self.status.set(
            "Registration complete. Review alignment before selecting spores. "
            + summary
        )

        if messagebox.askyesno(
            "Registration diagnostics",
            "Create registration diagnostics in the output folder? "
            "This saves reference/aligned overlays and transformation parameters.",
        ):
            self.save_registration_diagnostics()

    def save_registration_diagnostics(self):
        diag = self.output_dir / "registration"
        diag.mkdir(parents=True, exist_ok=True)
        ref = display_rgb(self.registered[0]).convert("RGBA")
        for path, image in zip(self.paths[1:], self.registered[1:]):
            aligned = display_rgb(image).convert("RGBA")
            overlay = Image.blend(ref, aligned, 0.5)
            overlay.save(diag / f"{path.stem}_overlay.png")
        with open(diag / "transformations.csv", "w", newline="", encoding="utf-8") as f:
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
            for path, matrix, report in zip(
                self.paths, self.transforms, self.registration_reports
            ):
                writer.writerow([path.name, *report, *matrix.flatten().tolist()])
        messagebox.showinfo("Diagnostics saved", f"Saved to:\n{diag}")

    def show_image(self, image):
        pil = display_rgb(image)
        cw = max(self.canvas.winfo_width(), 100)
        ch = max(self.canvas.winfo_height(), 100)
        self.scale = min(cw / pil.width, ch / pil.height, 1.0)
        new_size = (
            max(1, int(pil.width * self.scale)),
            max(1, int(pil.height * self.scale)),
        )
        shown = pil.resize(new_size, Image.Resampling.LANCZOS)
        self.reference_photo = ImageTk.PhotoImage(shown)
        self.offset = ((cw - new_size[0]) // 2, (ch - new_size[1]) // 2)
        self.canvas.delete("all")
        self.canvas_image_id = self.canvas.create_image(
            self.offset[0], self.offset[1], image=self.reference_photo, anchor="nw"
        )
        for spore in self.spores:
            x = self.offset[0] + spore["x"] * self.scale
            y = self.offset[1] + spore["y"] * self.scale
            self.canvas.create_oval(
                x - 7, y - 7, x + 7, y + 7, outline="yellow", width=2
            )
            self.canvas.create_text(
                x + 10, y - 10, text=spore["id"], fill="yellow", anchor="sw"
            )

    def redraw_reference(self, _event=None):
        if self.registered:
            self.show_image(self.registered[0])
        elif self.images:
            self.show_image(self.images[0])

    def select_spore(self, event):
        if not self.images:
            return
        if not self.registered:
            if not messagebox.askyesno(
                "Images not registered",
                "Register the images before selecting spores? Registration is recommended.",
            ):
                return
            self.register()
            if not self.registered:
                return
        x = int((event.x - self.offset[0]) / self.scale)
        y = int((event.y - self.offset[1]) / self.scale)
        if not (
            0 <= x < self.registered[0].shape[1]
            and 0 <= y < self.registered[0].shape[0]
        ):
            return
        try:
            size = int(self.crop_size_var.get())
            if size < 16:
                raise ValueError
        except ValueError:
            messagebox.showerror(
                "Invalid crop size", "Enter a crop size of at least 16 pixels."
            )
            return

        centers, scores = align_spore_centers(self.registered, x, y, size)
        crops = [
            extract_crop(im, cx, cy, size)
            for im, (cx, cy) in zip(self.registered, centers)
        ]
        labels = [p.stem for p in self.paths]
        sheet = make_contact_sheet(crops, labels, f"Candidate at {x}, {y}")
        self.review_candidate(x, y, crops, labels, sheet, centers, scores)

    def review_candidate(self, x, y, crops, labels, sheet, centers, scores):
        win = tk.Toplevel(self.root)
        win.title(f"Review candidate at ({x}, {y})")
        photo = ImageTk.PhotoImage(sheet)
        canvas = tk.Canvas(
            win,
            width=sheet.width,
            height=sheet.height,
            scrollregion=(0, 0, sheet.width, sheet.height),
        )
        canvas.pack(fill="both", expand=True)
        canvas.create_image(0, 0, image=photo, anchor="nw")
        canvas.image = photo

        controls = ttk.Frame(win, padding=8)
        controls.pack(fill="x")
        ttk.Label(
            controls,
            text="Review the chronological crops. Keep this spore if "
            "the sequence is useful for your figure.",
        ).pack(side="left")

        def accept():
            spore_id = f"S{len(self.spores) + 1:03d}"
            self.spores.append(
                {
                    "id": spore_id,
                    "x": x,
                    "y": y,
                    "size": int(self.crop_size_var.get()),
                    "centers": centers,
                    "scores": scores,
                }
            )
            self.spore_list.insert(tk.END, f"{spore_id}: x={x}, y={y}")
            self.show_image(self.registered[0])
            self.status.set(
                f"Selected {len(self.spores)} spores. Click another spore or save."
            )
            win.destroy()

        ttk.Button(controls, text="Keep spore", command=accept).pack(
            side="right", padx=4
        )
        ttk.Button(controls, text="Reject", command=win.destroy).pack(side="right")

    def save_all(self):
        if not self.spores:
            messagebox.showinfo(
                "No spores selected", "Select and keep at least one spore first."
            )
            return
        if not self.registered:
            messagebox.showerror(
                "Not registered", "Register the images before saving crops."
            )
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        labels = [p.stem for p in self.paths]
        with open(
            self.output_dir / "spores.csv", "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.writer(f)
            writer.writerow(["spore_id", "reference_x", "reference_y", "crop_size"])
            for spore in self.spores:
                writer.writerow([spore["id"], spore["x"], spore["y"], spore["size"]])

        for spore in self.spores:
            folder = self.output_dir / "crops" / spore["id"]
            folder.mkdir(parents=True, exist_ok=True)
            centers, scores = align_spore_centers(
                self.registered, spore["x"], spore["y"], spore["size"]
            )
            crops = [
                extract_crop(im, cx, cy, spore["size"])
                for im, (cx, cy) in zip(self.registered, centers)
            ]
            for crop, label in zip(crops, labels):
                cv2.imwrite(str(folder / f"{label}.tif"), crop)
            save_gif(crops, labels, self.output_dir / f"{spore['id']}.gif")
            make_publication_grid(
                crops, labels, self.output_dir / f"{spore['id']}_grid.jpg"
            )
            with open(
                folder / "crop_alignment.csv", "w", newline="", encoding="utf-8"
            ) as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "filename",
                        "crop_center_x",
                        "crop_center_y",
                        "template_match_score",
                    ]
                )
                for label, center, score in zip(labels, centers, scores):
                    writer.writerow([label, *center, score])

        messagebox.showinfo(
            "Extraction complete",
            f"Saved {len(self.spores)} spore sequences, TIFF crops, GIFs, grid JPGs, "
            f"and coordinates to:\n"
            f"{self.output_dir}",
        )
        self.status.set(
            f"Saved {len(self.spores)} spore sequences to {self.output_dir}"
        )


def main():
    root = tk.Tk()
    SporeApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
