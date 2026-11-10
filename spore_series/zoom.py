"""Zoomable image view: pan, mouse-wheel and trackpad zoom, and mapping clicks to image pixels."""
from __future__ import annotations
from dataclasses import dataclass
from tkinter import ttk
import math
import tkinter as tk

from PIL import Image
from PIL import ImageTk

from .config import DRAG_THRESHOLD_PX, MARKER_RADIUS_PX, ZOOM_MAX, ZOOM_MIN, ZOOM_STEP


def _clamp_axis(pan: float, content: float, available: float) -> float:
    """Keep an image axis in view. A smaller-than-view image is centered; a larger one
    may move only far enough that its edges do not pull inside the view."""
    if content <= available:
        # Whole pixels, so the drawn image and the click mapping share the same origin.
        return float(round((available - content) / 2))
    return min(0.0, max(available - content, pan))


@dataclass
class ViewTransform:
    """Maps between image pixels and canvas pixels for a zoomable, pannable view.

    At zoom 1 the whole image is visible. `zoom` multiplies the fit scale, and `pan_x`
    and `pan_y` give the canvas position of image pixel (0, 0).
    """

    image_w: int
    image_h: int
    canvas_w: int = 1
    canvas_h: int = 1
    max_fit_scale: float = 1.0
    zoom: float = ZOOM_MIN
    pan_x: float = 0.0
    pan_y: float = 0.0

    @property
    def fit_scale(self) -> float:
        return min(
            self.canvas_w / self.image_w,
            self.canvas_h / self.image_h,
            self.max_fit_scale,
        )

    @property
    def scale(self) -> float:
        """Canvas pixels per image pixel."""
        return self.fit_scale * self.zoom

    def to_canvas(self, u: float, v: float) -> tuple[float, float]:
        return self.pan_x + u * self.scale, self.pan_y + v * self.scale

    def to_image(self, cx: float, cy: float) -> tuple[float, float]:
        return (cx - self.pan_x) / self.scale, (cy - self.pan_y) / self.scale

    def reset(self) -> None:
        """Fit the whole image and center it."""
        self.zoom = ZOOM_MIN
        self._clamp_pan()

    def resize_canvas(self, width: int, height: int) -> None:
        self.canvas_w, self.canvas_h = max(1, width), max(1, height)
        self._clamp_pan()

    def zoom_at(self, cx: float, cy: float, factor: float) -> None:
        """Zoom by `factor`, keeping the image point under (cx, cy) fixed on screen."""
        u, v = self.to_image(cx, cy)
        self.zoom = min(ZOOM_MAX, max(ZOOM_MIN, self.zoom * factor))
        self.pan_x = cx - u * self.scale
        self.pan_y = cy - v * self.scale
        self._clamp_pan()

    def pan_by(self, dx: float, dy: float) -> None:
        self.pan_x += dx
        self.pan_y += dy
        self._clamp_pan()

    def _clamp_pan(self) -> None:
        self.pan_x = _clamp_axis(self.pan_x, self.image_w * self.scale, self.canvas_w)
        self.pan_y = _clamp_axis(self.pan_y, self.image_h * self.scale, self.canvas_h)


class ZoomView(tk.Canvas):
    """A canvas that shows one image with zoom and pan, plus optional markers.

    Zoom: mouse wheel or trackpad scroll and pinch, or the +/- buttons in ZoomControls.
    Pan: drag with the left button while zoomed in, or drag with the right or middle button.
    Select: a left click without dragging reports the image pixel under the cursor.
    """

    def __init__(self, master, *, max_fit_scale=1.0, on_click=None):
        super().__init__(master, background="#252525", highlightthickness=0)
        self._max_fit_scale = max_fit_scale
        self._on_click = on_click
        self.on_zoom_changed = None
        self._pil: Image.Image | None = None
        self._view: ViewTransform | None = None
        self._markers: list[tuple[float, float, str, str]] = []
        self._photo = None
        self._press_xy: tuple[int, int] | None = None
        self._last_xy: tuple[int, int] | None = None
        self._dragged = False

        self.bind("<Configure>", self._on_configure)
        self.bind("<MouseWheel>", self._on_wheel)
        self.bind(
            "<Button-4>", lambda e: self._zoom_at(e.x, e.y, ZOOM_STEP)
        )  # Linux up
        self.bind(
            "<Button-5>", lambda e: self._zoom_at(e.x, e.y, 1 / ZOOM_STEP)
        )  # Linux down
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)
        for press in ("<ButtonPress-2>", "<ButtonPress-3>"):
            self.bind(press, self._on_pan_press)
        for motion in ("<B2-Motion>", "<B3-Motion>"):
            self.bind(motion, self._on_pan_drag)

    @property
    def zoom(self) -> float:
        return self._view.zoom if self._view else ZOOM_MIN

    # -- public API ---------------------------------------------------------

    def set_image(self, pil: Image.Image, keep_view: bool = False) -> None:
        """Show an image. The view resets to fit unless keep_view is True and the size
        is unchanged, which lets an animation keep the user's zoom and position."""
        self._pil = pil
        size_changed = (
            self._view is None or (self._view.image_w, self._view.image_h) != pil.size
        )
        if size_changed or not keep_view:
            self._view = ViewTransform(
                pil.width,
                pil.height,
                max(self.winfo_width(), 1),
                max(self.winfo_height(), 1),
                self._max_fit_scale,
            )
            self._view.reset()
            self._notify_zoom()
        self._render()

    def clear(self) -> None:
        self._pil = None
        self._view = None
        self._markers = []
        self.delete("all")
        self._notify_zoom()

    def set_markers(self, markers) -> None:
        """Markers are (image_x, image_y, color, label). An empty label draws a plain ring."""
        self._markers = list(markers)
        self._render()

    def reset_view(self) -> None:
        if self._view:
            self._view.reset()
            self._render()
            self._notify_zoom()

    def zoom_by(self, factor: float) -> None:
        if self._view:
            self._zoom_at(self.winfo_width() / 2, self.winfo_height() / 2, factor)

    # -- rendering ----------------------------------------------------------

    def _render(self) -> None:
        self.delete("all")
        if self._pil is None or self._view is None:
            return
        v = self._view
        s = v.scale

        # Draw only the visible part of the image, so large images stay fast at high zoom.
        u0 = max(0, math.floor(-v.pan_x / s))
        v0 = max(0, math.floor(-v.pan_y / s))
        u1 = min(v.image_w, math.ceil((v.canvas_w - v.pan_x) / s))
        v1 = min(v.image_h, math.ceil((v.canvas_h - v.pan_y) / s))
        if u1 > u0 and v1 > v0:
            region = self._pil.crop((u0, v0, u1, v1))
            size = (max(1, round((u1 - u0) * s)), max(1, round((v1 - v0) * s)))
            # Nearest-neighbour when enlarging keeps real pixels visible; smooth when shrinking.
            resample = Image.Resampling.NEAREST if s >= 2 else Image.Resampling.LANCZOS
            self._photo = ImageTk.PhotoImage(region.resize(size, resample))
            x, y = v.to_canvas(u0, v0)
            self.create_image(round(x), round(y), image=self._photo, anchor="nw")

        for mx, my, color, label in self._markers:
            cx, cy = v.to_canvas(mx + 0.5, my + 0.5)
            r = MARKER_RADIUS_PX
            self.create_oval(cx - r, cy - r, cx + r, cy + r, outline=color, width=2)
            if label:
                self.create_text(
                    cx + r + 3, cy - r - 3, text=label, fill=color, anchor="sw"
                )

    def _notify_zoom(self) -> None:
        if self.on_zoom_changed:
            self.on_zoom_changed(self.zoom)

    def _zoom_at(self, x: float, y: float, factor: float) -> None:
        if self._view is None:
            return
        self._view.zoom_at(x, y, factor)
        self._render()
        self._notify_zoom()

    # -- events -------------------------------------------------------------

    def _on_configure(self, event) -> None:
        if self._view is not None:
            self._view.resize_canvas(event.width, event.height)
            self._render()

    def _on_wheel(self, event) -> None:
        # Windows and most mice send 120 per notch. macOS trackpads send small values, so scale them.
        if not event.delta:
            return
        steps = event.delta / 120 if abs(event.delta) >= 120 else event.delta / 6
        self._zoom_at(event.x, event.y, ZOOM_STEP**steps)

    def _on_press(self, event) -> None:
        self._press_xy = (event.x, event.y)
        self._last_xy = (event.x, event.y)
        self._dragged = False

    def _on_drag(self, event) -> None:
        # Left-drag pans only when zoomed in, so a click at fit zoom still selects.
        if self._press_xy is None or self._view is None or self._view.zoom <= ZOOM_MIN:
            return
        if not self._dragged:
            dx = event.x - self._press_xy[0]
            dy = event.y - self._press_xy[1]
            if math.hypot(dx, dy) < DRAG_THRESHOLD_PX:
                return
        self._dragged = True
        self._pan_to(event.x, event.y)

    def _on_release(self, event) -> None:
        if self._press_xy is None:
            return
        dragged = self._dragged
        self._press_xy = None
        if dragged or self._view is None or self._on_click is None:
            return
        u, v = self._view.to_image(event.x, event.y)
        if 0 <= u < self._view.image_w and 0 <= v < self._view.image_h:
            self._on_click(math.floor(u), math.floor(v))

    def _on_pan_press(self, event) -> None:
        self._last_xy = (event.x, event.y)

    def _on_pan_drag(self, event) -> None:
        if self._view is not None and self._last_xy is not None:
            self._pan_to(event.x, event.y)

    def _pan_to(self, x: int, y: int) -> None:
        lx, ly = self._last_xy
        self._view.pan_by(x - lx, y - ly)
        self._last_xy = (x, y)
        self._render()


class ZoomControls(ttk.Frame):
    """Zoom out, fit, zoom in, and a readout of the current zoom for a ZoomView."""

    def __init__(self, master, view: ZoomView):
        super().__init__(master)
        self._view = view
        self._label = ttk.Label(self, text="1.0×", width=6)
        ttk.Button(
            self, text="−", width=3, command=lambda: view.zoom_by(1 / ZOOM_STEP)
        ).pack(side="left")
        ttk.Button(self, text="Fit", command=view.reset_view).pack(side="left", padx=4)
        ttk.Button(
            self, text="+", width=3, command=lambda: view.zoom_by(ZOOM_STEP)
        ).pack(side="left")
        self._label.pack(side="left", padx=6)
        view.on_zoom_changed = self._show_zoom

    def _show_zoom(self, zoom: float) -> None:
        self._label.config(text=f"{zoom:.1f}×")
