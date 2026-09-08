"""
fiber_diameter.py — measure the diameter of a fiber whose edges are both
visible against dark background (not against another fiber).

The key change vs the previous version: instead of scoring candidate
points with a fixed-size perpendicular, we walk outward from each
candidate ONE PIXEL AT A TIME on both sides, stop at the first strong
intensity drop (the edge), and then check whether the pixel just past
that stop is BACKGROUND (mask == False) or still fiber (mask == True).

If either side stops inside another fiber, the "edge" is really another
fiber's boundary and the point is rejected. We only accept points where
both walks exit into background. Those are exactly the "you can see this
fiber against the sky on both sides at this latitude" points — matching
what a person picks by eye.

Pipeline:
  1. Segment fibers, crop SEM databar.
  2. Skeletonize; drop skeleton pixels near junctions and image borders.
  3. Structure tensor at each candidate -> local tangent.
  4. Walk the perpendicular outward on both sides. Reject candidates
     where an edge is a fiber-on-fiber boundary. Score the survivors
     by (a) contrast of the measured cross-section and (b) how much
     dark background sits past each edge (further = cleaner fiber).
  5. Best score wins.
  6. Diameter = distance between the two accepted edge points.

Overlay:
  red line   -> measured diameter (with black halo)
  yellow dots -> detected edges
  cyan dot   -> sample point
  white text -> diameter

Usage:
  python fiber_diameter.py image.tif -o output --nm-per-px 2.48
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from scipy.ndimage import map_coordinates, gaussian_filter1d
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import (closing, disk, skeletonize,
                                remove_small_holes, remove_small_objects)
from skimage.feature import structure_tensor


DEFAULT_NM_PER_PX = None
MAX_DIM = 1600


# ---------------------------------------------------- image preparation
from sem_image import load_gray, detect_databar_top, segment

def crop_databar(img):
    return img[:detect_databar_top(img)]

def maybe_downscale(img, max_dim=MAX_DIM):
    # Native pixels preserve diameter and coordinate accuracy.
    return img, 1.0


# ---------------------------------------------- candidate generation
def prune_skeleton(mask, dist, edge_margin):
    """Return skeleton pixels that are (a) inside the frame with margin,
    (b) not near a junction, (c) on a fiber of reasonable width."""
    skel = skeletonize(mask)
    if not skel.any():
        return np.zeros((0, 2), int)

    nb = ndi.convolve(skel.astype(int), np.ones((3, 3)), mode='constant') - 1
    junc = skel & (nb >= 3)
    junc_dist = ndi.distance_transform_edt(~junc) if junc.any() else np.full(mask.shape, np.inf)
    typical_hw = float(np.median(dist[skel]))
    keep_from_junc = max(6.0, 1.2 * typical_hw)

    H, W = mask.shape
    ok = skel.copy()
    ok &= junc_dist > keep_from_junc
    ok[:edge_margin] = False
    ok[-edge_margin:] = False
    ok[:, :edge_margin] = False
    ok[:, -edge_margin:] = False
    hw_of = dist * skel
    ok &= (hw_of > 0.5 * typical_hw)

    return np.argwhere(ok)


def _tangent_at(Arr, Arc, Acc, r, c):
    S = np.array([[Arr[r, c], Arc[r, c]],
                  [Arc[r, c], Acc[r, c]]])
    eigvals, eigvecs = np.linalg.eigh(S)
    coherence = ((eigvals[1] - eigvals[0]) /
                 (eigvals[1] + eigvals[0] + 1e-12))
    return eigvecs[:, 0], float(coherence)


# ---------------------------------------------- one-sided edge walk
def walk_to_edge(sm_img, mask, r, c, direction, max_steps, step=0.5):
    """Walk from (r, c) along `direction` until the intensity drops past
    the fiber edge. Returns a dict describing what we found, or None if
    we ran off the image.

    Returned dict:
      edge_r, edge_c : sub-pixel edge location
      past_r, past_c : integer pixel a bit past the edge (used to check
                       whether we exited into background or into another
                       fiber)
      past_in_mask   : True if past_r/past_c is inside the mask
                       (== "the thing on the far side is another fiber")
      travel         : distance from (r, c) to the edge
      contrast       : centre intensity minus intensity just past edge
    """
    H, W = sm_img.shape

    # Sample intensity along the ray with sub-pixel step
    ts = np.arange(0.0, max_steps + step / 2, step)
    rr = r + ts * direction[0]
    cc = c + ts * direction[1]
    inb = (rr >= 0) & (rr < H - 1) & (cc >= 0) & (cc < W - 1)
    if inb.sum() < 5:
        return None
    ts, rr, cc = ts[inb], rr[inb], cc[inb]

    prof = map_coordinates(sm_img, [rr, cc], order=1)
    prof = gaussian_filter1d(prof, 2.0)
    grad = np.gradient(prof)      # negative where intensity drops

    # Anchor to the FIRST mask exit, not a stronger edge on a neighbor.
    on_fiber = map_coordinates(mask, [rr,cc], order=0) > 0
    exits = np.flatnonzero(~on_fiber)
    if not on_fiber[0] or not len(exits):
        return None
    exit_i = int(exits[0])
    if exit_i < 2:
        return None
    past = exit_i + max(2, int(3.0/step))
    if past >= len(prof) or on_fiber[exit_i:past+1].any():
        return None
    # Search only in the neighborhood of this boundary. No ray may jump
    # across a gap and measure the outside of the next fiber instead.
    radius = max(2,int(2.0/step))
    lo,hi = max(1,exit_i-radius), min(len(grad)-2,exit_i+radius)
    i_edge = lo + int(np.argmin(grad[lo:hi+1]))
    if grad[i_edge] >= -0.005:
        return None
    # Quadratic interpolation of the derivative minimum.
    denom = grad[i_edge-1]-2*grad[i_edge]+grad[i_edge+1]
    offset = np.clip(0.5*(grad[i_edge-1]-grad[i_edge+1])/denom,-0.5,0.5) if abs(denom)>1e-12 else 0.
    travel = float(ts[i_edge]+offset*step)
    edge_r,edge_c = float(r+travel*direction[0]),float(c+travel*direction[1])
    past_r,past_c = int(round(rr[past])),int(round(cc[past]))
    past_in_mask = bool(mask[past_r,past_c])
    interior_mean = prof[:max(1,exit_i//2)].mean()
    contrast = float(interior_mean-prof[past])

    return {
        'edge_r': edge_r, 'edge_c': edge_c,
        'past_r': past_r, 'past_c': past_c,
        'past_in_mask': past_in_mask,
        'travel': travel,
        'contrast': contrast,
    }


# ---------------------------------------------- point picker + measure
def measure_at(sm_img, mask, r, c, tangent, half_width_hint):
    """Try to measure the diameter at (r, c). Returns dict on success,
    None if the point is invalid (edge lands inside another fiber, off
    image, etc.)."""
    normal = np.array([-tangent[1], tangent[0]])
    max_steps = max(30.0, 4.0 * half_width_hint)

    side_pos = walk_to_edge(sm_img, mask, r, c, normal, max_steps)
    side_neg = walk_to_edge(sm_img, mask, r, c, -normal, max_steps)
    if side_pos is None or side_neg is None:
        return None

    # REJECT if either walk lands inside another fiber.
    if side_pos['past_in_mask'] or side_neg['past_in_mask']:
        return None

    # REJECT if contrast is weak (not really an edge).
    if side_pos['contrast'] < 0.10 or side_neg['contrast'] < 0.10:
        return None

    e1 = (side_neg['edge_r'], side_neg['edge_c'])
    e2 = (side_pos['edge_r'], side_pos['edge_c'])
    diameter_px = float(np.hypot(e2[0] - e1[0], e2[1] - e1[1]))

    # score: contrast plus "how far into background did we get past each
    # edge" — deeper background = cleaner isolated fiber.
    # Both `past_r,past_c` are guaranteed to be background here.
    # Use distance-transform of the background as a proxy.
    return {
        'edge1_rc': e1,
        'edge2_rc': e2,
        'diameter_px': diameter_px,
        'contrast_l': side_neg['contrast'],
        'contrast_r': side_pos['contrast'],
        'past_left_rc': (side_neg['past_r'], side_neg['past_c']),
        'past_right_rc': (side_pos['past_r'], side_pos['past_c']),
    }


def pick_and_measure(img, mask, dist):
    """Score every clean candidate skeleton pixel by whether we can
    perform a valid measurement there. Return best (point, tangent,
    measurement)."""
    H, W = img.shape
    edge_margin = max(30, int(0.05 * min(H, W)))
    candidates = prune_skeleton(mask, dist, edge_margin)
    if len(candidates) == 0:
        return None, None, None

    sm = gaussian(img, 1.5)
    typical_hw = float(np.median(dist[mask & (dist > 0)]))
    tensor_sigma = max(3.0, typical_hw * 0.7)
    Arr, Arc, Acc = structure_tensor(sm, sigma=tensor_sigma, mode='reflect')

    # Background distance transform: how deep into the black background
    # does a pixel sit? Larger = further from any fiber.
    bg_dist = ndi.distance_transform_edt(~mask)

    if len(candidates) > 3000:
        stride = len(candidates) // 3000 + 1
        candidates = candidates[::stride]

    best_score = -1.0
    best = (None, None, None)
    for r, c in candidates:
        tangent, coherence = _tangent_at(Arr, Arc, Acc, int(r), int(c))
        if coherence < 0.4:
            continue
        hw = float(dist[r, c])
        m = measure_at(sm, mask, int(r), int(c), tangent, hw)
        if m is None:
            continue

        # Depth of background past each edge — how much "sky" you see
        # past the fiber. Larger = the fiber is genuinely isolated here.
        pl_r, pl_c = m['past_left_rc']
        pr_r, pr_c = m['past_right_rc']
        sky_l = float(bg_dist[pl_r, pl_c])
        sky_r = float(bg_dist[pr_r, pr_c])

        # Score: multiplicative so any weak factor kills the score.
        contrast = 0.5 * (m['contrast_l'] + m['contrast_r'])
        sky = min(sky_l, sky_r)                 # worst side dominates
        score = contrast * coherence * (1.0 + sky)

        if score > best_score:
            best_score = score
            best = ((int(r), int(c)), tangent, m)

    if best[0] is None:
        return None, None, None
    return best


# ---------------------------------------------- overlay
def render_overlay(img, mask, point, result, out_path,
                   diameter_px_orig, diameter_nm, nm_per_px):
    base = (np.stack([img] * 3, -1) * 255).astype(np.uint8)
    tint = base.copy()
    tint[mask] = (0.75 * base[mask] +
                  0.25 * np.array([70, 130, 200])).astype(np.uint8)

    im = Image.fromarray(tint)
    draw = ImageDraw.Draw(im)

    r0, c0 = point
    e1 = result['edge1_rc']
    e2 = result['edge2_rc']

    draw.line([(e1[1], e1[0]), (e2[1], e2[0])], fill=(0, 0, 0), width=8)
    draw.line([(e1[1], e1[0]), (e2[1], e2[0])], fill=(255, 0, 0), width=3)

    for e in (e1, e2):
        draw.ellipse([e[1] - 5, e[0] - 5, e[1] + 5, e[0] + 5],
                     fill=(255, 220, 0), outline=(0, 0, 0), width=1)

    draw.ellipse([c0 - 4, r0 - 4, c0 + 4, r0 + 4],
                 fill=(0, 220, 220), outline=(0, 0, 0), width=1)

    if nm_per_px is None:
        label = f'{diameter_px_orig:.1f} px'
    else:
        label = f'{diameter_nm:.0f} nm  ({diameter_px_orig:.1f} px)'
    text_x = int((e1[1] + e2[1]) / 2 + 20)
    text_y = int((e1[0] + e2[0]) / 2 - 8)
    try:
        font = ImageFont.truetype('arial.ttf', 20)
    except Exception:
        font = ImageFont.load_default()
    draw.text((text_x + 1, text_y + 1), label, fill=(0, 0, 0), font=font)
    draw.text((text_x, text_y), label, fill=(255, 255, 255), font=font)

    im.save(out_path)


# ---------------------------------------------- top-level
def analyze(path, outdir, nm_per_px=DEFAULT_NM_PER_PX):
    if nm_per_px is None:
        from sem_scale import analyze_tiff
        nm_per_px = analyze_tiff(path)['pixel_size_nm']
    if nm_per_px is None or not np.isfinite(nm_per_px) or nm_per_px <= 0:
        raise ValueError('A positive calibration is required; supply --nm-per-px')
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    full = load_gray(path)
    full = crop_databar(full)
    img, scale = maybe_downscale(full)
    if scale < 1.0:
        print(f'(downscaled {full.shape[1]}x{full.shape[0]} -> '
              f'{img.shape[1]}x{img.shape[0]} for analysis; '
              f'results scaled back to original pixels)')

    mask = segment(img)
    dist = ndi.distance_transform_edt(mask)

    point, tangent, result = pick_and_measure(img, mask, dist)
    if point is None:
        print(f'{path}: no fiber with both edges against background found')
        return

    diameter_px_orig = result['diameter_px'] / scale
    diameter_nm = diameter_px_orig * nm_per_px
    angle_deg = float(np.degrees(np.arctan2(tangent[0], tangent[1])))
    r, c = point
    sample_x_orig = int(round(c / scale))
    sample_y_orig = int(round(r / scale))

    lines = [
        f'File:         {Path(path).name}',
        f'Sample point: (x={sample_x_orig}, y={sample_y_orig}) in original image',
        f'Fiber angle:  {angle_deg:+.1f} deg  (0 = horizontal)',
        f'Diameter:     {diameter_px_orig:.2f} px  =  {diameter_nm:.2f} nm '
        f'({diameter_nm / 1000:.4f} um)',
        f'nm/px used:   {nm_per_px}',
    ]
    text = '\n'.join(lines)
    print(text)

    stem = Path(path).stem
    overlay_path = outdir / f'{stem}_diameter.png'
    text_path = outdir / f'{stem}_diameter.txt'
    render_overlay(img, mask, point, result, overlay_path,
                   diameter_px_orig, diameter_nm, nm_per_px)
    text_path.write_text(text + '\n', encoding='utf-8')
    print(f'\nOverlay: {overlay_path}')
    print(f'Text:    {text_path}')


def main():
    p = argparse.ArgumentParser(
        description='Measure fiber diameter where both edges are against background.')
    p.add_argument('image', help='input image (TIFF, PNG, ...)')
    p.add_argument('-o', '--outdir', default='output', help='output folder')
    p.add_argument('--nm-per-px', type=float, default=DEFAULT_NM_PER_PX,
                   help='pixel size in nm; e.g. 2.48 for JEOL 10000x')
    args = p.parse_args()
    analyze(args.image, args.outdir, nm_per_px=args.nm_per_px)


if __name__ == '__main__':
    main()