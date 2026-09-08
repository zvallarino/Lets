"""Shared image preparation. Measurements never consume colored overlays."""
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu, threshold_triangle, threshold_multiotsu


def load_gray(path):
    with Image.open(path) as im:
        im.seek(0)
        if im.mode in ('I', 'I;16', 'I;16B', 'I;16L', 'F'):
            a = np.asarray(im, dtype=np.float32)
            if not np.isfinite(a).all():
                raise ValueError('Image contains non-finite intensities')
            lo, hi = np.percentile(a, [0.5, 99.5])
            if hi <= lo:
                lo, hi = float(a.min()), float(a.max())
            return np.clip((a-lo)/max(hi-lo, 1e-12), 0, 1)
        rgb = np.asarray(im.convert('RGB'))
        # Generated masks/depth overlays must not become measurement inputs.
        chroma = rgb.max(axis=2).astype(float) - rgb.min(axis=2)
        if np.mean(chroma > 30) > 0.01:
            raise ValueError('Use the original grayscale micrograph, not a colored overlay')
        return np.asarray(im.convert('L'), dtype=np.float32)/255


def detect_databar_top(img):
    """Conservative bottom-band detector; do not crop at one dark specimen row.

    Requires a nearly black full-width separator and a predominantly black
    suffix. Returns full height if the evidence is insufficient. Always review
    the recorded ROI, or supply --crop-bottom for other acquisition layouts.
    """
    h, w = img.shape
    if h < 64 or w < 32 or float(np.ptp(img)) < 0.02:
        return h
    dark = (img <= 0.06).mean(axis=1)
    run = max(3, int(round(h*0.001)))
    for r in range(int(h*0.75), h-max(run, int(h*0.015))):
        if np.all(dark[r:r+run] > 0.985):
            above = dark[max(0,r-max(10,run*3)):r]
            if (np.median(dark[r:]) > 0.8 and
                    np.median(above) < 0.9 and np.mean(img[:r]) > 0.07):
                return r
    return h


def crop_edges(img, top=0, bottom=0, left=0, right=0):
    values = (top,bottom,left,right)
    if any(not isinstance(v, (int,np.integer)) or v < 0 for v in values):
        raise ValueError('Crop margins must be nonnegative integers')
    h,w = img.shape
    if top+bottom >= h or left+right >= w:
        raise ValueError('Crop removes the entire image')
    return img[top:h-bottom, left:w-right]


def prepare(path, max_dim=1600, crop_bottom=None):
    full = load_gray(path)
    original_shape = full.shape
    top = detect_databar_top(full) if crop_bottom is None else full.shape[0]-crop_bottom
    full = crop_edges(full, bottom=full.shape[0]-top)
    if max_dim is not None and (not isinstance(max_dim,int) or max_dim < 0):
        raise ValueError('max_dim must be nonnegative')
    h,w = full.shape
    if max_dim and max(h,w)>max_dim:
        s = max_dim/max(h,w)
        size = (max(1,round(w*s)),max(1,round(h*s)))
        img = np.asarray(Image.fromarray(full).resize(size,Image.Resampling.BILINEAR))
    else:
        img = full.copy()
    geometry = {'original_shape_hw':list(original_shape), 'roi_xywh':[0,0,w,h],
                'analysis_shape_hw':list(img.shape),
                'source_px_per_analysis_x':w/img.shape[1],
                'source_px_per_analysis_y':h/img.shape[0],
                'crop_method':'automatic' if crop_bottom is None else 'explicit'}
    return img, geometry


def segment(img, threshold=None, return_info=False):
    a = np.asarray(img,dtype=np.float32)
    if a.ndim != 2 or not a.size or not np.isfinite(a).all():
        raise ValueError('Expected a finite, nonempty grayscale image')
    sm = ndi.gaussian_filter(a,1.0)
    constant = float(np.ptp(sm)) < 1e-6
    otsu = float(threshold_otsu(sm)) if not constant else 0.
    triangle = float(threshold_triangle(sm)) if not constant else 0.
    # Separate dark background, dim fiber bodies and bright rims. A binary
    # Otsu split can mistake the dark core for background; Triangle can instead
    # merge whole dense fields. Retain both as independent sensitivity checks.
    # Three-class separation is still a heuristic, not semantic segmentation.
    try:
        automatic=float(threshold_multiotsu(sm,classes=3)[0]) if not constant else 0.
    except ValueError:
        automatic=otsu
    t = automatic if threshold is None else float(threshold)
    if not 0 <= t <= 1:
        raise ValueError('Threshold must lie between 0 and 1')
    raw = np.zeros(a.shape,bool) if constant else sm>t
    # Avoid closing adjacent fibers together or filling real narrow pores.
    labels,n = ndi.label(raw)
    sizes = np.bincount(labels.ravel())
    keep = sizes>=16; keep[0]=False
    mask = keep[labels]
    info = {'threshold':t,'method':'three_class_background_split' if threshold is None else 'explicit',
            'otsu_threshold':otsu,'triangle_threshold':triangle,
            'otsu_coverage_pct':float((sm>otsu).mean()*100) if not constant else 0.,
            'triangle_coverage_pct':float((sm>triangle).mean()*100) if not constant else 0.,
            'coverage_pct':float(mask.mean()*100),
            'coverage_threshold_minus_0_03_pct':float((sm>max(0,t-.03)).mean()*100) if not constant else 0.,
            'coverage_threshold_plus_0_03_pct':float((sm>min(1,t+.03)).mean()*100) if not constant else 0.,
            'constant_image':constant}
    return (mask, info) if return_info else mask
