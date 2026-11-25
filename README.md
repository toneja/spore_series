# Spore Time-Series Extractor

The `spore_series` package aligns a series of microscope images taken over time, lets you click
individual spores on the first image, follows each kept spore through the whole series, and
saves the results as TIFF crops, an animated GIF, and a labelled grid for figures. The
window can also preview and delete the GIFs and grids it has saved.

## Requirements

- Python 3.9 or newer
- Packages: `opencv-python`, `numpy`, `pillow` (Pillow 9.1 or newer, OpenCV 4.4 or newer)
- Tkinter for the window. It ships with the python.org installers for Windows and macOS.
  On Debian or Ubuntu, install it with `sudo apt install python3-tk`.

```bash
pip install opencv-python numpy pillow
python spore_series.py
```

## Code layout

The program is a package, so each part can be read and changed on its own:

| File | Contents |
|---|---|
| `spore_series.py` | Launcher. Run this to start the window. |
| `spore_series/config.py` | All settings (the constants in the Settings table) |
| `spore_series/models.py` | Data types: `Spore`, `FrameRegistration`, `RegistrationError` |
| `spore_series/imaging.py` | Reading images, fitting frames to the reference size, brightness |
| `spore_series/registration.py` | Aligning frames to the reference (SIFT, translation, spores) |
| `spore_series/tracking.py` | Following each spore through the frames |
| `spore_series/outputs.py` | TIFF crops, GIF, grid (TIFF, PNG or JPG), CSV rows, registration diagnostics |
| `spore_series/session.py` | Recording and restoring an earlier selection (`session.json`) |
| `spore_series/files.py` | Listing, reading and deleting files in `spore_output` |
| `spore_series/zoom.py` | The zoomable image view and click mapping |
| `spore_series/preview.py` | The output preview window |
| `spore_series/calibration.py` | Reading the pixel size (µm per pixel) from image metadata |
| `spore_series/scalebar.py` | Drawing scale bars on the GIF frames and grid panels |
| `spore_series/app.py` | The main window |

Keep the `spore_series` folder next to `spore_series.py`. Both must stay in the same
folder for the launcher to find the package.

## Input

Put the images for one experiment in a single folder, with one image per timepoint.

- **Supported formats:** `.tif`, `.tiff`, `.png`, `.jpg`, `.jpeg`, `.bmp`. Only the first page
  of a multi-page TIFF is read.
- **Number of timepoints:** any number of two or more. The montage adapts to the count (see
  *Grid layout and format*).
- **Order:** files are sorted by natural order of their names, so `t2` comes before `t10`.
  Name them so that they sort chronologically, for example `t01.tif`, `t02.tif`, and so on.
- **Reference:** the first image in that order is the registration reference. All spores
  are clicked on it, and every other frame is aligned to it.
- **Labels:** each image's file name without its extension is used as its timepoint label
  in the GIF and grid. Two images cannot share a name apart from the extension
  (for example `a.tif` and `a.png`).
- **Consistency:** all images must have the same number of channels and the same bit depth.
  Images of a different size are centre-cropped or zero-padded to match the first image.
  Pixels are never rescaled.

## Workflow

1. **Load image folder.** Choose the folder. The first image appears in the window, and the
   frames are registered to it automatically. A "Please wait" window shows while this runs,
   and the status line reports the result. If the folder has outputs from an earlier session,
   they are picked up as described in *Reopening a folder*. The **Microns per pixel** box in the
   toolbar is filled in from the image metadata when the images have a calibration. Type a value
   there to set or change it (see *Scale bars*).
2. **Zoom in on the spore if needed** (see *Viewing and zooming* below), so the click lands
   on the centre you want.
3. **Click a spore's centre** on the reference image. A candidate window shows the spore
   through every timepoint, in order.
   - Set **Crop size (px)** in the toolbar before clicking. The size is fixed for each
     candidate when you click, so changing the box afterwards does not change what you
     reviewed.
   - Click **Keep spore** to add it, or **Reject** to discard it. Closing the window also
     counts as rejecting.
4. **Markers on the reference:**
   - **Green** circles with IDs (S001, S002, ...) are kept spores.
   - **Red** circles are rejected candidates. They do not block the area. Click there again
     to review it another time. If you then keep it, the red circle is removed. If you
     reject it again, it is replaced by a new red circle.
5. Repeat steps 2 to 4 for each spore.
6. **Save selected spores + outputs.** Writes all files for every kept spore, along with
   the registration diagnostics and the session record. A "Please wait" window shows while
   it works. Saving does not ask for confirmation. The grid is saved in the format chosen in
   **Grid format** (TIFF by default).
7. **Preview outputs** (optional) and **Delete GIFs + grids** (optional). See below.

### Viewing and zooming

| Action | Effect |
|---|---|
| Mouse wheel, or trackpad two-finger scroll or pinch | Zoom in or out, centred on the cursor |
| **−**, **Fit**, **+** buttons | Zoom out, reset to fit, zoom in, centred on the view |
| Left-drag while zoomed in | Pan |
| Right-drag or middle-drag, at any zoom | Pan |
| Left click (without dragging) | Select the spore under the cursor |

The zoom level is shown next to the buttons. At zoom 1.0 the whole image fits the window.
Clicks always report the image pixel under the cursor at the current zoom, so zooming in
makes it easier to hit a spore's centre precisely. Trackpad gestures depend on how your
operating system sends them. If your trackpad does not zoom, use the **+** and **−** buttons.

## Previewing and deleting outputs

**Preview outputs** opens a window that lists every spore with a saved GIF or grid in the
output folder. Choose a spore from the drop-down list.

- **Animated GIF** tab: plays the saved GIF. **Pause** and **Play** control the animation.
  The frame counter shows which timepoint is on screen. The GIF uses the same zoom controls
  as the main window.
- **Grid** tab (shown first): shows the saved grid, with zoom.
- **Delete this spore's GIF + grid** deletes that spore's GIF and grid after you confirm.
- **Open output folder** opens the folder in your file browser.

**Delete GIFs + grids** in the main window deletes every saved GIF and grid after you
confirm.

Deletion does not remove TIFF crops, `spores.csv`, `crop_alignment.csv`, or registration
diagnostics. Deleted files cannot be recovered. To recreate them, select the same spores
again and save, or keep the spore selection and save again.

## Scale bars

The GIF frames and the grid panels get a scale bar in the bottom-right corner. The TIFF
crops are never changed.

**Where the scale comes from.** When you load a folder, the first image's metadata is read:

- TIFF resolution tags in inches or centimetres, ImageJ's `unit=micron` convention, and
  OME-TIFF `PhysicalSizeX` values.
- The DPI stored in PNG and JPEG files.

Values of 72 or 96 dpi are ignored, because software writes those as defaults rather than as
calibrations. Values outside 0.01 to 1000 µm per pixel are also ignored. If the first image has
no usable calibration, the box stays empty.

**Setting it yourself.** Type the size of one pixel in microns in the **Microns per pixel**
box, for example `0.65`, and press Enter or **Set scale**. A typed value replaces the metadata
value. It is saved in `session.json`, so reopening the folder restores it. Leave the box empty
for no scale bars.

**Changing it later.** Outputs already saved keep their old scale bars. Save again to redraw
them. The status line reminds you after you change the value.

**Bar length.** The bar is the longest round length (0.5, 1, 2, 5, 10, 20, 50 … µm) that is at
most a quarter of the image width. In the grid, each panel is enlarged from its crop, and the
bar is drawn at the enlarged size, so it still shows the true length.

**Check the value.** The metadata is only as good as the software that wrote it. Confirm the
pixel size against your microscope's calibration, for example by imaging a stage micrometer,
before using the scale bars in a paper.

## Reopening a folder

If the input folder already contains a `spore_output` folder from an earlier session, the
program checks it when you load the folder. Images are compared by a content hash, not by
date.

- **Same images.** The kept spores and rejected candidates from the last save are restored,
  and their tracking is recomputed on the registered frames. The status line says how many
  were restored.
- **Changed images.** If any image differs from those the selection was made on, the
  selection is not restored. Earlier outputs are kept untouched.
- **Either way,** new spores are numbered after the highest number already in the output
  folder, and `spores.csv` keeps the rows from earlier saves. Earlier files are never
  overwritten by a new spore.

There is no button to remove a single restored spore. To start over, move the `spore_output`
folder out of the input folder and load the folder again. Deleting it removes the outputs.

## Outputs

Everything is written to a `spore_output` folder inside the input folder:

```
spore_output/
├── spores.csv                 # one row per kept spore: ID, clicked x/y, crop size
├── session.json               # image fingerprints, rejected candidates and scale, for reopening
├── S001.gif                   # animated GIF, one frame per timepoint, labelled, with scale bar
├── S001_grid.tif              # labelled grid for figures, with scale bars (TIFF, 300 dpi)
├── crops/
│   └── S001/
│       ├── t01.tif            # one TIFF crop per timepoint, from the aligned frame
│       ├── ...
│       └── crop_alignment.csv # per-frame tracked centre and match score
├── S002.gif
├── S002_grid.tif
├── crops/S002/...
└── registration/              # written on every save
    ├── t02_overlay.png        # 50/50 blend of each aligned frame over the reference
    ├── ...
    └── transformations.csv    # per-frame method, match counts, and 2x3 affine matrix
```

### Grid layout and format

The grid uses the smallest near-square arrangement that fits the timepoints, and each
panel is 400 px square with a thin black border and its label below it in black on white.

The grid is saved at 300 dpi in the format chosen in the **Grid format** box:

- **TIFF (lossless), the default.** LZW-compressed, so the pixels are exactly as drawn. Most
  journals accept TIFF for figures.
- **PNG (lossless).** The same pixels as TIFF, in a format most image tools open. PNG stores
  resolution as pixels per metre, so the 300 dpi tag reads back as 299.999 dpi, which is the
  same figure.
- **JPG (lossy).** Smallest file, with a small loss from compression. Use it only when a
  journal or a file-size limit requires it.

Each spore keeps one grid. Saving in a new format removes that spore's grid in the old
format.

| Timepoints | Layout (columns x rows) | Empty cells |
|---|---|---|
| 2 | 2 x 1 | 0 |
| 3 or 4 | 2 x 2 | 1 or 0 |
| 5 or 6 | 3 x 2 | 1 or 0 |
| 7, 8 or 9 | 3 x 3 | 2, 1 or 0 |
| 10, 11 or 12 | 4 x 3 | 2, 1 or 0 |

When the last row is partial, its panels are centred under the full rows, so the figure
looks intentional. To force a width, pass `columns=` to `save_publication_grid`.

### GIF

Each frame is 700 ms, and the GIF loops. A black bar across the top of each frame carries
the timepoint label. The GIF has 256 colours, so it is for viewing rather than figures.

### CSV columns

| File | Columns |
|---|---|
| `spores.csv` | `spore_id, reference_x, reference_y, crop_size` |
| `crop_alignment.csv` | `filename, crop_center_x, crop_center_y, template_match_score` |
| `transformations.csv` | `filename, method, matches, inliers, inlier_fraction, m00, m01, m02, m10, m11, m12`. The reference row and the translation rows have blank match counts. |

Coordinates are in pixels of the reference image, with (0, 0) at the top-left.

## How it works

- **Registration:** each frame is aligned to the reference by the first method that gives a
  trustworthy result:
  1. **SIFT** feature matches on the original image contrast, filtered with Lowe's ratio test
     and fitted with RANSAC. Needs at least 8 inliers.
  2. **Translation by image correlation.** A phase correlation gives a starting shift, and
     ECC (enhanced correlation coefficient) refines it. The method compares the high-pass
     filtered images, so it is not affected by changes in overall brightness or contrast.
     It is accepted when the correlation reaches 0.5. The method name in the CSV includes
     the correlation, for example `translation (ECC r=0.76)`.
  3. **Spore centroids**, when neither of the above gives a result. Dark, compact spots are
     matched between frames.
  4. **SIFT on a contrast-stretched copy**, only for very dim frames.
  Every candidate must also have a scale within 20% of 1 and a rotation of at most 15
  degrees. Stage moves are shifts and small rotations, so a fit that fails these checks is a
  fitting error, not a real move. Feature-based fits are then refined using the spores: each
  spore is paired with its nearest reference spore and the fit is updated. The refinement is
  kept only when it reduces the median spore residual.
  The "Method" column in `transformations.csv` records which method was used for each frame.
- **Tracking:** the spore is cut out of the reference as a template (60% of the crop size).
  In each later frame, the template is searched within about ±¼ of the crop size of the
  clicked position. The best match becomes that frame's centre. If the best score is below
  0.35, the clicked position is used for that frame.
- **Crops:** each crop is taken from the aligned frame at its tracked centre.
- **Brightness:** 8-bit images are shown and saved as they are. For higher bit depths,
  all frames are stretched on one shared scale (1st to 99th percentile across the whole
  series). A timepoint that is dimmer than the others therefore stays dimmer in the GIF and
  grid.

## Settings

Change these in `spore_series/config.py`:

| Constant | Default | Meaning |
|---|---|---|
| `DEFAULT_CROP_SIZE` | 180 | Crop size shown at startup, in pixels |
| `MIN_CROP_SIZE` | 16 | Smallest allowed crop size |
| `MIN_TRACKING_SCORE` | 0.35 | Below this match score, the clicked position is used |
| `LOWE_RATIO` | 0.72 | Lower values keep fewer feature matches |
| `RANSAC_THRESHOLD_PX` | 3.0 | Reprojection tolerance for registration inliers |
| `MIN_ACCEPTED_INLIERS` | 8 | Inliers a registration transform needs to be accepted |
| `MAX_SCALE_DEVIATION` | 0.2 | Largest allowed scale change between a frame and the reference |
| `MAX_ROTATION_DEG` | 15.0 | Largest allowed rotation between a frame and the reference |
| `ECC_MIN_CORRELATION` | 0.5 | Correlation a translation fit must reach to be accepted |
| `ECC_MAX_ITERATIONS` | 100 | Iteration limit for the ECC refinement |
| `SPORE_MATCH_PX` | 4.0 | Distance within which two spore centroids count as the same spore |
| `ZOOM_MAX` | 40.0 | Largest zoom, as a multiple of the fit-to-window size |
| `ZOOM_STEP` | 1.1 | Zoom change per wheel notch or button press |
| `DRAG_THRESHOLD_PX` | 4 | Movement below this counts as a click, not a drag |
| `PREVIEW_MAX_FIT_SCALE` | 3.0 | How much small previews may be enlarged to fit |
| `GIF_FRAME_MS` | 700 | Time per GIF frame |
| `GRID_PANEL_PX` | 400 | Panel size in the grid |
| `GRID_LABEL_FONT_PX` | 26 | Label font size in the grid |
| `GRID_JPEG_QUALITY` | 95 | JPEG quality, used only for the JPG format |
| `GRID_DPI` | 300 | Resolution written into the grid file |
| `GRID_DEFAULT_FORMAT` | tiff | Grid format selected at startup: `tiff`, `png` or `jpg` |
| `SCALE_BAR_CANDIDATES_UM` | 0.5 … 5000 | Round bar lengths a scale bar may use, in µm |
| `SCALE_BAR_MAX_FRACTION` | 0.25 | Longest scale bar, as a fraction of the image width |
| `SCALE_BAR_FONT_PX_GIF` | 12 | Font size of the scale bar label in the GIF |
| `SCALE_BAR_FONT_PX_GRID` | 20 | Font size of the scale bar label in the grid |
| `SCALE_MIN_UM_PER_PX` | 0.01 | Smallest pixel size accepted, in µm |
| `SCALE_MAX_UM_PER_PX` | 1000.0 | Largest pixel size accepted, in µm |
| `DEFAULT_DPI_VALUES` | 72, 96 | Resolutions treated as software defaults and ignored |

## Troubleshooting

- **"Only N feature matches... registration could not be estimated."** The frames have too
  little texture for feature matching, or the sample moved too far between timepoints. Look
  at the frames directly. Registration can sometimes succeed if you raise `LOWE_RATIO` a
  little (for example to 0.8), which keeps more matches, but check the overlays afterwards.
- **"Frame N: too few reliable matches."** Usually one frame is very different from the
  reference, such as a focus change or an out-of-plane shift. Remove that frame from the
  folder or check its overlay in `registration/`.
- **Registration failed.** The images stay loaded and you can still browse and zoom, but no
  spores can be kept until registration succeeds. Clicking the image tries registration again.
- **"Frame N: no method could align this frame."** None of the four methods gave a trustworthy
  transform. Check that the frame shares features or spores with the reference. The frame may
  be out of focus, empty, or far from the reference. Remove it, or check the frame by eye.
- **Spore drifts or jumps in the GIF.** See *Known limitations*. Check the GIF and grid by
  eye.
- **Crop size error.** Enter a whole number of at least 16.
- **"No scale in the file metadata."** The message gives the reason the metadata was not
  used. Type the microns per pixel in the box instead.
- **Scale bar length looks wrong.** The value in the box is the one used. Check it against
  the calibration (see *Scale bars*), correct it, and save again.
- **Save failed.** The usual cause is a read-only or full drive, or a file that is open in
  another program (for example a CSV open in Excel). Close it and save again.

## Known limitations

- **Tracking is anchored to the click.** Each frame is searched around the clicked position,
  not the previous frame's position, so a spore that moves more than about ¼ of the crop
  size from where it was clicked will not be followed.
- **Faint spores can lock onto the background.** The template comes from the reference
  frame, so a spore that is fainter than its static background can stay at the clicked
  position. The match score stays high in this case, so it does not flag the error. Spores
  brighter than their background tracked exactly in testing. Check the GIF and grid for
  each spore.
- **Frame edges.** Registration can leave a black margin where the warped image did not
  cover the reference. Crops near the edge include zeros for the same reason.
- **Overwriting.** Saving rewrites the files of any spore whose ID is in the current
  selection, and each spore keeps one grid. New spore IDs continue after the highest number
  already in the output folder,
  so earlier spores are not replaced. Files from earlier runs are never removed
  automatically, and the preview window lists them.
- **Scale bars need a correct pixel size.** A wrong value in the metadata gives wrong bars,
  and the program cannot check it. Scale bars appear only in the saved GIFs and grids, not
  in the main window.
- **Deletion is permanent.** There is no undo for GIF and grid deletion.
- **Figures.** The labels and scale bars are part of the pixels in the GIF and grid. The grid
  is saved as TIFF by default. Check your journal's figure requirements, since some ask for
  EPS or a particular resolution. The TIFF crops are unlabelled. The frames after the
  reference are resampled once during alignment, so their edges are slightly softer than the
  camera's pixels. The reference frame is unchanged.
