"""Settings for the spore series extractor. Change these to tune registration, tracking,
the image view and the figure outputs."""
from __future__ import annotations
import re

import numpy as np


IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
OUTPUT_FOLDER_NAME = "spore_output"
DEFAULT_CROP_SIZE = 180
MIN_CROP_SIZE = 16

# Registration
MAX_FEATURES = 5000
MIN_FEATURE_MATCHES = 4  # fewest matches that can support a transform estimate
LOWE_RATIO = 0.72  # keep a match only if it beats its runner-up by this factor
RANSAC_THRESHOLD_PX = 3.0
MIN_ACCEPTED_INLIERS = 8  # a transform needs at least this many inliers to be trusted
MAX_SCALE_DEVIATION = 0.2  # accepted |scale - 1|
MAX_ROTATION_DEG = 15.0  # accepted rotation magnitude

# Translation registration (phase correlation refined by ECC; insensitive to brightness)
ECC_MIN_CORRELATION = 0.5  # ECC correlation a translation fit must reach to be accepted
ECC_MAX_ITERATIONS = 100

# Spore-centroid registration (fallback for sparse fields)
SPORE_MIN_AREA_PX = 150
SPORE_MAX_AREA_PX = 6000
SPORE_MIN_SIDE_PX = 14
SPORE_MIN_MATCHES = 6
SPORE_MATCH_PX = 4.0
SPORE_ANGLE_RANGE_DEG = 10.0
SPORE_ANGLE_STEP_DEG = 0.5
SPORE_SHIFT_CANDIDATES = 20  # shift hypotheses scored per rotation

# Tracking: below this template-match score, a frame falls back to the clicked position
MIN_TRACKING_SCORE = 0.35

# Image view (zoom and pan)
ZOOM_MIN = 1.0  # 1.0 = whole image fits in the view
ZOOM_MAX = 40.0
ZOOM_STEP = 1.1  # multiplier for one wheel notch or one +/- button press
DRAG_THRESHOLD_PX = 4  # movement below this counts as a click, not a drag
MARKER_RADIUS_PX = 7
KEPT_MARKER_COLOR = "#1e9e3c"  # green
REJECTED_MARKER_COLOR = "red"
REFERENCE_MAX_FIT_SCALE = 1.0  # the reference is never enlarged to fill the view
PREVIEW_MAX_FIT_SCALE = 3.0  # small GIF and grid previews may be enlarged to fit

# Preview, GIF and grid appearance
PREVIEW_SIZE = 230
PREVIEW_COLUMNS = 4
GIF_FRAME_MS = 700
GIF_LABEL_BAR_PX = 24
GRID_PANEL_PX = 400
GRID_PANEL_PAD_PX = 14
GRID_LABEL_HEIGHT_PX = 40
GRID_LABEL_FONT_PX = 26
GRID_JPEG_QUALITY = 95
GRID_DPI = 300

FONT_CANDIDATES = (
    "arial.ttf",
    "Arial.ttf",
    "DejaVuSans.ttf",
    "LiberationSans-Regular.ttf",
)
IDENTITY_AFFINE = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

GIF_NAME_PATTERN = re.compile(r"(S\d{3,})\.gif")
GRID_NAME_PATTERN = re.compile(r"(S\d{3,})_grid\.jpg")

SESSION_FILE_NAME = "session.json"
