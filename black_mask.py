#!/usr/bin/env python3
"""
black_mask.py — highlight the non-fiber (black/background) regions of an image.

Everything darker than an automatically chosen threshold is painted red;
everything else is left as the original grayscale. Written for SEM fiber
micrographs but works on any TIFF/PNG/JPEG.

Usage
-----
    python black_mask.py input.tif
    python black_mask.py input.tif -o out.png
    python black_mask.py input.tif --threshold 60      # force a cutoff
    python black_mask.py input.tif --invert            # dark = fiber instead
    python black_mask.py input.tif --scale 3           # enlarge the output

Dependencies: numpy, Pillow, scipy
    pip install numpy pillow scipy
"""

import argparse
import os
import sys
import numpy as np
from PIL import Image

# ======================================================================
#  EASY-TO-EDIT SETTINGS  — change these, no need to touch the code below
# ======================================================================

# How close to black a pixel must be to count as "not fiber" (0-255).
# Any pixel darker than this becomes red. Lower = only very dark counts as
# background; higher = more of the dim/gray edges count as background.
# Set to None to auto-pick the cutoff for each image (Otsu's method).
BLACK_CUTOFF = None          # e.g. 60  (or None for automatic)

# Pixels to chop off each edge BEFORE processing — use this to remove the
# SEM info banner / scale bar so its bright text isn't mistaken for fiber.
# None detects the bottom banner; use an explicit crop when detection is uncertain.
CROP_BOTTOM = None
CROP_TOP = 0
CROP_LEFT = 0
CROP_RIGHT = 0

# Where outputs land when -o/--output isn't given.
DEFAULT_OUTPUT_DIR = "output/mask_output"

# ======================================================================

try:
    from scipy import ndimage
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


from sem_image import load_gray as shared_load_gray, crop_edges, detect_databar_top

def load_gray(path):
    return np.round(shared_load_gray(path)*255).astype(np.uint8)


def otsu_threshold(gray):
    """Classic Otsu: pick the cutoff that best separates the two brightness
    populations (here: dark background vs bright fiber)."""
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    weight_bg = np.cumsum(hist)
    weight_fg = total - weight_bg
    valid = (weight_bg > 0) & (weight_fg > 0)

    intensity = np.arange(256)
    cum_mean = np.cumsum(hist * intensity)
    mean_bg = np.divide(cum_mean, weight_bg, out=np.zeros(256), where=weight_bg > 0)
    grand = cum_mean[-1]
    mean_fg = np.divide(grand - cum_mean, weight_fg,
                        out=np.zeros(256), where=weight_fg > 0)

    between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
    between[~valid] = 0
    return int(np.argmax(between))


def clean_mask(mask):
    """Remove single-pixel speckle so the mask is solid regions, not noise."""
    if not HAVE_SCIPY:
        return mask
    mask = ndimage.binary_opening(mask, iterations=1)
    mask = ndimage.binary_closing(mask, iterations=2)
    return mask


def make_overlay(path, out_path=None, threshold=None, invert=False,
                 smooth=1.0, scale=1, color=(230, 30, 30),
                 crop=None):
    gray = load_gray(path)

    # strip the info banner / borders first, so nothing there skews the
    # threshold or gets mislabeled as fiber
    if crop is None:
        crop = (CROP_TOP, CROP_BOTTOM, CROP_LEFT, CROP_RIGHT)
    top,bottom,left,right = crop
    if bottom is None:
        bottom = gray.shape[0]-detect_databar_top(gray.astype(float)/255)
    gray = crop_edges(gray,top,bottom,left,right)

    if HAVE_SCIPY and smooth > 0:
        work = ndimage.gaussian_filter(gray.astype(float), smooth)
    else:
        work = gray

    # cutoff priority: explicit -t  >  top-of-file BLACK_CUTOFF  >  auto/Otsu
    if threshold is None:
        threshold = BLACK_CUTOFF if BLACK_CUTOFF is not None else otsu_threshold(np.round(work).astype(np.uint8))

    # default: darker-than-threshold is background ("not fiber")
    if not 0 <= threshold <= 255 or scale < 1 or smooth < 0:
        raise ValueError("Invalid threshold, scale or smoothing")
    mask = work <= threshold
    if invert:
        mask = ~mask
    mask = clean_mask(mask)

    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.uint8)
    rgb[mask] = color
    out = Image.fromarray(rgb)

    if scale != 1:
        out = out.resize((out.width * scale, out.height * scale), Image.NEAREST)

    if out_path is None:
        stem = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(DEFAULT_OUTPUT_DIR, f"{stem}_blackmask.png")

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)  # create the folder if needed
    out.save(out_path)
    from pathlib import Path
    Image.fromarray((~mask).astype(np.uint8)*255).save(Path(out_path).with_name(Path(out_path).stem+"_fiber_mask.png"))

    pct = 100.0 * mask.mean()
    return out_path, threshold, pct


def main():
    p = argparse.ArgumentParser(description="Paint non-fiber (black) regions red.")
    p.add_argument("input", help="input image (TIFF/PNG/JPEG)")
    p.add_argument("-o", "--output", help="output path (default: <input>_blackmask.png)")
    p.add_argument("-t", "--threshold", type=int,
                   help="force brightness cutoff 0-255 (default: auto/Otsu)")
    p.add_argument("--invert", action="store_true",
                   help="treat BRIGHT as background instead of dark")
    p.add_argument("--smooth", type=float, default=1.0,
                   help="Gaussian pre-smoothing sigma (default 1.0; 0 disables)")
    p.add_argument("--scale", type=int, default=1,
                   help="integer upscale factor for the output (default 1)")
    p.add_argument("--crop-bottom", type=int, default=None,
                   help=f"pixels to cut off the bottom (default {CROP_BOTTOM}, removes info banner)")
    p.add_argument("--crop-top", type=int, default=None, help="pixels to cut off the top")
    p.add_argument("--crop-left", type=int, default=None, help="pixels to cut off the left")
    p.add_argument("--crop-right", type=int, default=None, help="pixels to cut off the right")
    args = p.parse_args()

    crop = (
        CROP_TOP if args.crop_top is None else args.crop_top,
        CROP_BOTTOM if args.crop_bottom is None else args.crop_bottom,
        CROP_LEFT if args.crop_left is None else args.crop_left,
        CROP_RIGHT if args.crop_right is None else args.crop_right,
    )

    try:
        out_path, thr, pct = make_overlay(
            args.input, args.output, args.threshold,
            args.invert, args.smooth, args.scale, crop=crop)
    except FileNotFoundError:
        sys.exit(f"error: file not found: {args.input}")

    print(f"threshold = {thr}")
    print(f"masked (not-fiber) = {pct:.1f}% of pixels")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
