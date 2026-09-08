#!/usr/bin/env python3
"""
top_fiber_depth.py — highlight the fiber(s) closest to the viewer with a
pretrained monocular-depth model (Depth Anything V2). No training, CPU inference supported.

Experimental visualization: the model predicts relative depth from image
appearance. It is not validated here for SEMs and does not establish actual
fiber heights or crossing order. Always provide the original grayscale image;
painted red masks are rejected. Cropping and segmentation run separately.

Runs on CPU. Small model ~1-3s/image on a laptop; Base ~10-20s; Large slower.
First run downloads weights from Hugging Face (~100MB Small / ~400MB Base).

Usage
-----
    python top_fiber_depth.py input.png
    python top_fiber_depth.py input.png --model base
    python top_fiber_depth.py input.png --nearest 15      # keep nearest 15% depth
    python top_fiber_depth.py input.png --invert          # if near/far look flipped
    python top_fiber_depth.py input.png --save-depth      # also dump the depth map

Dependencies:
    pip install torch transformers pillow numpy
"""

import argparse
import os
import sys
import numpy as np
from PIL import Image
from functools import lru_cache
from sem_image import prepare, segment

# ======================================================================
#  EASY-TO-EDIT SETTINGS
# ======================================================================

MODEL = "small"            # "small" | "base" | "large"
NEAREST_PERCENT = 20       # keep the nearest this-% of depth as "top" fibers
MIN_BLOB = 1500            # ignore highlighted blobs smaller than this (px)
INVERT_DEPTH = False       # flip if the model's near/far comes out reversed
HIGHLIGHT_COLOR = (60, 220, 90)
OVERLAY_ALPHA = 0.55

# background red mask (these images have red = non-fiber); tune if red drifts
RED_R_MIN, RED_G_MAX, RED_B_MAX = 140, 90, 90

DEFAULT_OUTPUT_DIR = "output/depth_output"

_HF_IDS = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "base":  "depth-anything/Depth-Anything-V2-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Large-hf",
}
# ======================================================================


def fiber_mask(rgb):
    """Fiber = everything that isn't the red background. If there's no red
    (a raw grayscale SEM), treat the whole frame as fiber."""
    r = rgb[..., 0].astype(int); g = rgb[..., 1].astype(int); b = rgb[..., 2].astype(int)
    red = (r > RED_R_MIN) & (g < RED_G_MAX) & (b < RED_B_MAX)
    if red.mean() < 0.02:          # basically no red -> not a masked image
        return segment(np.mean(rgb,axis=2).astype(np.float32)/255)
    return ~red


@lru_cache(maxsize=3)
def depth_pipeline(model_key, revision=None):
    from transformers import pipeline
    return pipeline("depth-estimation", model=_HF_IDS[model_key], revision=revision, device=-1)


def run_depth(pil_img, model_key, revision=None):
    pipe = depth_pipeline(model_key, revision)
    out = pipe(pil_img)
    prediction = out['predicted_depth']
    if hasattr(prediction,'detach'):
        prediction = prediction.detach().cpu().numpy()
    depth = np.asarray(prediction, dtype=np.float32).squeeze()
    if depth.ndim != 2 or not np.isfinite(depth).all():
        raise ValueError('Depth model returned invalid dimensions or non-finite values')
    # pipeline returns depth at model resolution; resize to image size
    if depth.shape != (pil_img.height, pil_img.width):
        depth = np.asarray(
            Image.fromarray(depth).resize((pil_img.width, pil_img.height), Image.BILINEAR),
            dtype=np.float32)
    return depth


def main():
    p = argparse.ArgumentParser(description="Highlight closest fibers via Depth Anything V2.")
    p.add_argument("input")
    p.add_argument("-o", "--output")
    p.add_argument("--model", choices=list(_HF_IDS), default=MODEL)
    p.add_argument("--nearest", type=float, default=NEAREST_PERCENT,
                   help="keep the nearest this %% of depth (default 20)")
    p.add_argument("--min-blob", type=int, default=MIN_BLOB)
    p.add_argument("--invert", action="store_true", default=INVERT_DEPTH)
    p.add_argument("--save-depth", action="store_true")
    p.add_argument('--revision', help='Hugging Face model commit for reproducibility')
    p.add_argument('--crop-bottom', type=int, default=None)
    p.add_argument("--threads", type=int, default=12,
                   help="CPU threads for torch (default 12; you have 12 physical cores)")
    args = p.parse_args()
    if not 0 < args.nearest <= 100 or args.min_blob < 1 or args.threads < 1:
        p.error('nearest must be in (0,100]; min-blob and threads must be positive')

    try:
        import torch
        torch.set_num_threads(args.threads)
    except Exception:
        pass

    try:
        gray, geometry = prepare(args.input, max_dim=1600, crop_bottom=args.crop_bottom)
        pil = Image.fromarray(np.round(gray*255).astype(np.uint8)).convert('RGB')
    except FileNotFoundError:
        sys.exit(f"error: file not found: {args.input}")
    rgb = np.asarray(pil)

    try:
        depth = run_depth(pil, args.model, args.revision)
    except Exception as e:
        sys.exit(f"error running the model (network/proxy or install issue?): {e}")

    # Depth Anything: larger value = closer. Invert if it comes out flipped.
    if args.invert:
        depth = -depth

    mask = segment(gray)
    vals = depth[mask]
    if vals.size == 0:
        sys.exit("no fiber pixels found (check the red mask thresholds)")

    cutoff = np.percentile(vals, 100 - args.nearest)   # nearest N% of depth
    top = mask & (depth >= cutoff)

    # drop small speckle
    try:
        from scipy import ndimage
        lbl, n = ndimage.label(top)
        sizes = ndimage.sum(np.ones_like(lbl), lbl, range(1, n + 1))
        keep = np.isin(lbl, [i + 1 for i, s in enumerate(sizes) if s >= args.min_blob])
        top = keep
    except ImportError:
        pass

    out = rgb.copy()
    out[top] = (OVERLAY_ALPHA * np.array(HIGHLIGHT_COLOR)
                + (1 - OVERLAY_ALPHA) * out[top]).astype(np.uint8)

    if args.output is None:
        stem = os.path.splitext(os.path.basename(args.input))[0]
        outp = os.path.join(DEFAULT_OUTPUT_DIR, f"{stem}_topdepth.png")
    else:
        outp = args.output
    od = os.path.dirname(outp)
    if od:
        os.makedirs(od, exist_ok=True)
    Image.fromarray(out).save(outp)
    from pathlib import Path
    import json
    np.save(Path(outp).with_suffix('.depth.npy'),depth)
    Path(outp).with_suffix('.json').write_text(json.dumps({
        'experimental':True,'model':_HF_IDS[args.model],'requested_revision':args.revision,
        'resolved_revision':getattr(depth_pipeline(args.model,args.revision).model.config,'_commit_hash',None),
        'geometry':geometry,'nearest_percent':args.nearest,'invert':args.invert,
        'note':'Relative depth cue; not validated for SEM fiber ordering.'},indent=2),encoding='utf-8')
    print(f"model={args.model}  nearest={args.nearest}%  highlighted {top.mean()*100:.1f}% of pixels")
    print(f"saved -> {outp}")

    if args.save_depth:
        d = depth - depth.min()
        d = (d / (d.max() + 1e-9) * 255).astype(np.uint8)
        dp = str(Path(outp).with_name(Path(outp).stem + '_depth.png'))
        Image.fromarray(d).save(dp)
        print(f"saved depth map -> {dp}")


if __name__ == "__main__":
    main()
