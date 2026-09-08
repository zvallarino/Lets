"""
Nanofiber crossing analysis
===========================
 
Given a grayscale image of bright fibers on a dark background (SEM-style),
this script:
 
  1. Segments the fibers (Gaussian smooth -> Otsu -> morphological cleanup).
  2. Skeletonizes the segmentation, breaks the skeleton into branches, and
     labels junction clusters.
  3. Merges junction clusters linked by short "connector" branches into
     crossing ZONES (fiber-on-fiber blobs create two nearby junctions with
     a short bridge between them; those are really one crossing).
  4. Stitches individual fibers together by pairing branch-ends across
     each zone whose travel directions form the straightest continuation
     (undirected maximum-weight matching on a continuation-score matrix). This is what
     gives the fiber count.
  5. Detects crossings geometrically -- every pair of fibers whose
     centerlines approach within ~their combined half-widths.
  6. Infers z-order at each crossing with the "profile-step" method:
     resample the smoothed image along each fiber's own centerline through
     the crossing zone; the fiber that runs UNDER the other crosses the
     top fiber's boundary and shows a sharp intensity step, while the top
     fiber's profile stays smooth. Bigger step == under. Falls back to an
     intensity-match tiebreak when the two steps are close.
  7. Renders a color-coded overlay: each fiber's pixels are colored by
     "which centerline is closest, normalized by that fiber's half-width",
     and at each crossing the overlap is reassigned to the top fiber so
     the visual matches the inference. A legend strip is appended below.
 
Usage:
    python fiber_analysis.py input.png                # -> result_overlay.png
    python fiber_analysis.py input.tif my_prefix      # -> my_prefix_overlay.png
 
Works on anything PIL can open (PNG, TIFF, ...). RGB inputs are converted
to grayscale automatically.
"""
 
import sys
import warnings
 
# scikit-image is renaming a few keyword args across versions; the current
# call sites work on both old and new APIs, so silence the transitional noise.
warnings.filterwarnings('ignore', message='Parameter `area_threshold`.*')
warnings.filterwarnings('ignore', message='Parameter `min_size`.*')
 
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.ndimage import uniform_filter1d, gaussian_filter1d, map_coordinates
from scipy.spatial import cKDTree
from skimage.filters import threshold_otsu, gaussian, sobel, meijering
from skimage.morphology import (skeletonize, closing, disk,
                                remove_small_holes, remove_small_objects)
import networkx as nx
import cv2
from sem_image import prepare, segment as shared_segment, detect_databar_top
 
 
# ------------------------------------------------------------------- config
MIN_FIBER_LEN    = 15     # px of centerline; shorter components are spurs
PAIR_SCORE_MIN   = 0.55   # continuation quality threshold (1 = straight)
CONNECTOR_FACTOR = 3.0    # branch is a zone-internal connector if
                          #    len < FACTOR * local half-width
 
# up to ten distinct fiber colors before wrapping (BGR order for cv2)
PALETTE = [(66, 135, 245), (240, 84, 84), (46, 204, 113), (241, 196, 15),
           (155, 89, 182), (26, 188, 156), (230, 126, 34), (231, 84, 128),
           (52, 152, 219), (200, 200, 90)]
 
 
# ------------------------------------------------------------- load & mask
def _detect_databar_top(a):
    """Return the top row of an SEM databar (solid dark band at the bottom
    of the frame), or a.shape[0] if there isn't one."""
    return detect_databar_top(np.asarray(a, dtype=float)/255)

 
 
def load_gray(path, max_dim=1400, crop_databar=True):
    """Read image, convert to float grayscale in [0, 1]. Optionally crop
    an SEM databar and downscale to keep the pipeline responsive."""
    return prepare(path, max_dim, None if crop_databar else 0)[0]

 
 
def segment(img):
    """Bright-on-dark fiber mask."""
    return shared_segment(img)

 
 
def ridge_skeleton_mask(img, outer_mask):
    """Build a skeletonizable mask from RIDGES of the intensity image
    (bright tubes), restricted to the outer segmentation. This separates
    fibers that touch tangentially -- each fiber has its own intensity
    ridge even when they share a mask blob."""
    # Estimate fiber half-widths from the outer mask's distance transform,
    # then use those to set the ridge filter's sigma range.
    if not outer_mask.any():
        return np.zeros_like(outer_mask)
    dist = ndi.distance_transform_edt(outer_mask)
    hw_lo = max(1.0, np.percentile(dist[outer_mask], 40) * 0.5)
    hw_hi = max(hw_lo + 2.0, np.percentile(dist[outer_mask], 90) * 0.8)
    n_scales = min(12, max(3, int(round((hw_hi - hw_lo) / 2))))
    sigmas = np.linspace(hw_lo, hw_hi, n_scales)
 
    sm = gaussian(img, sigma=1.5)
    resp = meijering(sm, sigmas=sigmas, black_ridges=False) * outer_mask
    ridges = resp > 0.35 * resp.max()
    ridges = remove_small_objects(ridges, 50)
    # thin the ridge blob to a single-pixel-wide skeleton; downstream
    # skeleton_branches() expects a thin structure
    return skeletonize(ridges)
 
 
# --------------------------------------------------- skeleton -> branches
def skeleton_branches(mask_or_skel, already_skeleton=False):
    """Skeletonize the mask (unless it's already a skeleton), split at
    junction clusters, and return ordered branch polylines plus the
    junction-cluster centroids."""
    skel = mask_or_skel if already_skeleton else skeletonize(mask_or_skel)
    # 8-connected neighbor count of the skeleton
    nb = ndi.convolve(skel.astype(int), np.ones((3, 3)), mode='constant') - 1
    nb[~skel] = 0
    junctions = skel & (nb >= 3)
    jlab, njunc = ndi.label(junctions, structure=np.ones((3, 3)))
    blab, nbr = ndi.label(skel & ~junctions, structure=np.ones((3, 3)))
 
    branches = []
    slices = ndi.find_objects(blab)
    for b in range(1, nbr + 1):
        sl = slices[b-1]
        coords = np.argwhere(blab[sl] == b) + np.array([sl[0].start,sl[1].start])
        pts = set(map(tuple, coords))
 
        def nbrs(p, S):
            r, c = p
            return [(r + dr, c + dc)
                    for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                    if (dr, dc) != (0, 0) and (r + dr, c + dc) in S]
 
        # order the branch pixels by walking from one endpoint to the other
        ends = sorted(p for p in pts if len(nbrs(p, pts)) <= 1)
        start = ends[0] if ends else min(pts)
        order, seen, cur = [start], {start}, start
        while True:
            nxt = [q for q in nbrs(cur, pts) if q not in seen]
            if not nxt:
                break
            cur = nxt[0]
            order.append(cur)
            seen.add(cur)
 
        br = {'pix': np.array(order, float)}
        # attach adjacent junction id at each end (if any)
        for key, p in (('j0', order[0]), ('j1', order[-1])):
            r, c = int(p[0]), int(p[1])
            win = jlab[max(r - 1, 0):r + 2, max(c - 1, 0):c + 2]
            ids = np.unique(win[win > 0])
            br[key] = int(ids[0]) if len(ids) else None
        branches.append(br)
 
    centers = ndi.center_of_mass(junctions, jlab, range(1,njunc+1))
    jcent = {j: np.array(c) for j,c in enumerate(centers,1)}
    return skel, jcent, branches
 
 
# --------------------------------------------------- crossing-zone merging
def build_zones(branches, jcent, dist):
    """Union junctions connected by short 'connector' branches -- skeleton
    artifacts inside an occlusion blob -- into single crossing zones."""
    parent = {j: j for j in jcent}
    bounds = {j:(np.array(c),np.array(c)) for j,c in jcent.items()}
    members = {j:1 for j in jcent}
 
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
 
    def union(a, b, local_width):
        a,b=find(a),find(b)
        if a==b:return True
        lo=np.minimum(bounds[a][0],bounds[b][0])
        hi=np.maximum(bounds[a][1],bounds[b][1])
        # A chain of short branches in a dense mat must not collapse an
        # entire field into one crossing with thousands of matching ends.
        if members[a]+members[b]>16 or np.linalg.norm(hi-lo)>6*max(local_width,1):
            return False
        parent[a]=b; bounds[b]=(lo,hi);members[b]+=members[a]
        return True
 
    connectors = set()
    for i, br in enumerate(branches):
        j0, j1 = br['j0'], br['j1']
        if j0 is not None and j1 is not None and j0 != j1:
            pts = br['pix'].astype(int)
            local_w = float(np.max(dist[pts[:, 0], pts[:, 1]]))
            if len(pts) < CONNECTOR_FACTOR * local_w:
                if union(j0, j1,local_w):
                    connectors.add(i)
        elif j0 is not None and j0 == j1 and len(br['pix']) < 12:
            connectors.add(i)  # tiny loop at a single junction
 
    zone_of = {j: find(j) for j in jcent}
    zones = {}
    for j, z in zone_of.items():
        zones.setdefault(z, {'junctions': [], 'connectors': []})
        zones[z]['junctions'].append(j)
    for i in connectors:
        z = zone_of[branches[i]['j0']]
        zones[z]['connectors'].append(i)
    for z, d in zones.items():
        pts = [jcent[j] for j in d['junctions']]
        for i in d['connectors']:
            pts.append(branches[i]['pix'].mean(axis=0))
        d['center'] = np.mean(pts, axis=0)
    return zones, zone_of, connectors
 
 
# ------------------------------------------- pairing / fiber construction
def travel_dir(branch, end, n=10):
    """Unit direction of travel when EXITING the branch at the given end."""
    pix = branch['pix']
    if end == 0:
        a, b = pix[min(n, len(pix) - 1)], pix[0]
    else:
        a, b = pix[-min(n + 1, len(pix))], pix[-1]
    v = b - a
    nn = np.linalg.norm(v)
    return v / nn if nn else v
 
 
def stitch_fibers(branches, zones, zone_of, connectors):
    """Pair branch-ends across each zone by best straight continuation,
    then chain paired branches into individual fibers."""
    # ends of NON-connector branches attached to each zone
    zone_ends = {z: [] for z in zones}
    for i, br in enumerate(branches):
        if i in connectors:
            continue
        for e, jk in ((0, 'j0'), (1, 'j1')):
            j = br[jk]
            if j is not None:
                zone_ends[zone_of[j]].append((i, e))
 
    links = []  # (branch_a, end_a, branch_b, end_b, zone)
    for z, ends in zone_ends.items():
        if len(ends) < 2:
            continue
        n = len(ends)
        S = np.full((n, n), -2.0)
        for a in range(n):
            for b in range(a + 1, n):
                ia, ea = ends[a]
                ib, eb = ends[b]
                if ia == ib:
                    continue
                pa = branches[ia]['pix'][0 if ea == 0 else -1]
                pb = branches[ib]['pix'][0 if eb == 0 else -1]
                da = travel_dir(branches[ia], ea)  # exiting A
                db = travel_dir(branches[ib], eb)  # exiting B
                v = pb - pa
                nv = np.linalg.norm(v)
                v = v / nv if nv > 1e-6 else da
                # A travels toward B, B travels toward A, and the two
                # travel directions are anti-parallel -> straight through
                s = (np.dot(da, v) + np.dot(db, -v) + np.dot(da, -db)) / 3.0
                S[a, b] = S[b, a] = s
        # Undirected matching: an end can belong to exactly one pair.
        # Bipartite assignment can form cycles and its greedy conversion loses pairs.
        matching_graph = nx.Graph()
        for a in range(n):
            for b in range(a+1,n):
                if S[a,b] >= PAIR_SCORE_MIN:
                    matching_graph.add_edge(a,b,weight=float(S[a,b]))
        pairs = nx.max_weight_matching(matching_graph)
        for a, b in sorted(tuple(sorted(pair)) for pair in pairs):
            ia, ea = ends[a]
            ib, eb = ends[b]
            links.append((ia, ea, ib, eb, z))
 
    G = nx.Graph()
    for i in range(len(branches)):
        if i not in connectors:
            G.add_node(i)
    for ia, ea, ib, eb, z in links:
        G.add_edge(ia, ib, zone=z)
 
    fibers = []
    for comp in nx.connected_components(G):
        sub = G.subgraph(comp)
        endsn = [nn for nn in sub if sub.degree(nn) <= 1]
        start = endsn[0] if endsn else next(iter(comp))
        chain = list(nx.dfs_preorder_nodes(sub, source=start))
        pts, prev_tail = [], None
        zlist = [G.edges[e]['zone'] for e in sub.edges]
        for bi in chain:
            seg = list(branches[bi]['pix'])
            if prev_tail is None and len(chain)>1:
                next_bi = chain[1]
                first_link = next(link for link in links if {link[0],link[2]} == {bi,next_bi})
                linked_end = first_link[1] if first_link[0] == bi else first_link[3]
                if linked_end == 0:
                    seg = seg[::-1]
            if prev_tail is not None:
                if (np.linalg.norm(seg[0] - prev_tail)
                        > np.linalg.norm(seg[-1] - prev_tail)):
                    seg = seg[::-1]
                gap = np.linalg.norm(seg[0] - prev_tail)
                if gap > 1.5:  # straight bridge across the crossing zone
                    steps = int(gap)
                    bridge = [prev_tail + (seg[0] - prev_tail) * t
                              for t in np.linspace(0, 1, steps + 2)[1:-1]]
                    pts.extend(bridge)
            pts.extend(seg)
            prev_tail = seg[-1]
        P = np.array(pts)
        if len(P) >= 9:  # smooth the centerline (kills skeleton jaggies)
            P = np.column_stack([uniform_filter1d(P[:, 0], 9),
                                 uniform_filter1d(P[:, 1], 9)])
        fibers.append({'pts': P, 'zones': zlist})
    return [f for f in fibers if len(f['pts']) >= MIN_FIBER_LEN]
 
 
# ------------------------------------------------------------- geometry
def local_dir(fiber, target, k=12):
    """Unit tangent of `fiber` near the closest point to `target`."""
    pts = fiber['pts']
    i = int(np.linalg.norm(pts - target, axis=1).argmin())
    a, b = max(0, i - k), min(len(pts), i + k + 1)
    v = pts[b - 1] - pts[a]
    n = np.linalg.norm(v)
    return v / n if n else v
 
 
def fiber_halfwidth(fiber, dist, zone_centers=()):
    """Robust half-width along the centerline. Crossing zones inflate the
    distance transform and bridges pass through blob centers, so a low
    percentile of the distribution tracks the true fiber half-width."""
    pts = np.round(fiber['pts']).astype(int)
    H, W = dist.shape
    ok = ((pts[:, 0] >= 0) & (pts[:, 0] < H) &
          (pts[:, 1] >= 0) & (pts[:, 1] < W))
    vals = dist[pts[ok][:, 0], pts[ok][:, 1]]
    vals = vals[vals>0]
    return float(np.percentile(vals, 30)) if vals.size else 0.0
 
 
# ------------------------------------------------- geometric crossing det.
def find_crossings(fa, fb, dist):
    """Contiguous stretches where the two centerlines approach within
    ~their combined half-widths. Returns one center per crossing (a pair
    can cross more than once in a curly image)."""
    A, B = fa['pts'], fb['pts']
    wa, wb = fiber_halfwidth(fa, dist), fiber_halfwidth(fb, dist)
    thr = 0.75 * (wa + wb)
    tree = cKDTree(B)
    dmin, jidx = tree.query(A)
    close = dmin < thr
    if not close.any():
        return []
    centers = []
    lab, n = ndi.label(close)
    for k in range(1, n + 1):
        idx = np.where(lab == k)[0]
        i_best = idx[np.argmin(dmin[idx])]
        pa, pb = A[i_best], B[jidx[i_best]]
        centers.append((pa + pb) / 2.0)
    # merge centers closer than thr (wiggly closest-approach can split)
    merged = []
    for c in centers:
        for m in merged:
            if np.linalg.norm(c - m['c']) < thr:
                m['pts'].append(c)
                m['c'] = np.mean(m['pts'], axis=0)
                break
        else:
            merged.append({'c': c, 'pts': [c]})
    return [m['c'] for m in merged]
 
 
# ------------------------------------------------------- z-order inference
def centerline_profile(sm_img, fiber, center):
    """Smoothed intensity profile along the fiber, resampled at 1 px
    arclength steps, with s=0 at the crossing center."""
    P = fiber['pts']
    i0 = int(np.linalg.norm(P - center, axis=1).argmin())
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    arc = np.concatenate([[0], np.cumsum(seg)])
    arc -= arc[i0]
    s_grid = np.arange(arc[0], arc[-1], 1.0)
    rr = np.interp(s_grid, arc, P[:, 0])
    cc = np.interp(s_grid, arc, P[:, 1])
    prof = map_coordinates(sm_img, [rr, cc], order=1)
    return s_grid, gaussian_filter1d(prof, 2)
 
 
def occlusion_step(sm_img, fiber, center, t_in, t_out=25):
    """Max |d(intensity)/ds| along the centerline inside the crossing
    zone. The UNDER fiber's path crosses the top fiber's boundary and
    produces a sharp step; the TOP fiber's path stays smooth."""
    s_grid, prof = centerline_profile(sm_img, fiber, center)
    if len(prof)<2:
        return 0.0, 0.0
    d = np.abs(np.gradient(prof))
    inz = np.abs(s_grid) <= t_in
    outz = (np.abs(s_grid) > t_in) & (np.abs(s_grid) <= t_in + t_out)
    step_in = float(d[inz].max()) if inz.any() else 0.0
    step_out = float(d[outz].max()) if outz.any() else 0.0
    return step_in, step_out
 
 
def fiber_intensity(img, fiber, zone_centers, dist):
    """Median centerline intensity AWAY from every crossing zone."""
    cache_key=(id(img),id(dist),id(zone_centers))
    if fiber.get('_intensity_cache_key')==cache_key:
        return fiber['_intensity_cache_value']
    pts = np.round(fiber['pts']).astype(int)
    H, W = img.shape
    ok = ((pts[:, 0] >= 0) & (pts[:, 0] < H) &
          (pts[:, 1] >= 0) & (pts[:, 1] < W))
    fpts, pts = fiber['pts'][ok], pts[ok]
    vals = img[pts[:, 0], pts[:, 1]]
    keep = np.ones(len(vals), bool)
    if len(zone_centers):
        centers=np.asarray(zone_centers)
        rc=np.round(centers).astype(int)
        radii=dist[rc[:,0].clip(0,H-1),rc[:,1].clip(0,W-1)]+8
        # Restrict exact disc tests to centers near this path's bounding box.
        margin=float(radii.max())
        nearby=np.all((centers>=fpts.min(axis=0)-margin)&(centers<=fpts.max(axis=0)+margin),axis=1)
        for zc,rad in zip(centers[nearby],radii[nearby]):
            keep &= np.linalg.norm(fpts-zc,axis=1)>rad
    value=float(np.median(vals[keep] if keep.sum()>5 else vals)) if vals.size else 0.
    fiber['_intensity_cache_key']=cache_key;fiber['_intensity_cache_value']=value
    return value
 
 
def infer_z(img, sm_img, fibers, fa, fb, center, dist, all_zone_centers):
    """Decide which of two crossing fibers is on top."""
    da, db = local_dir(fibers[fa], center), local_dir(fibers[fb], center)
    wa = fiber_halfwidth(fibers[fa], dist)
    wb = fiber_halfwidth(fibers[fb], dist)
    # occlusion half-length along each fiber = (other width) / sin(angle)
    # sin of angle between the two centerlines at the crossing
    sin_t = max(abs(float(da[0] * db[1] - da[1] * db[0])), 0.35)
    step_a, _ = occlusion_step(sm_img, fibers[fa], center, wb / sin_t + 6)
    step_b, _ = occlusion_step(sm_img, fibers[fb], center, wa / sin_t + 6)
 
    sep = abs(step_a - step_b) / (max(step_a, step_b) + 1e-6)
    if sep > 0.25:
        top = fb if step_a > step_b else fa   # bigger step == UNDER
        method, conf = 'profile-step', sep
    else:
        # intensity-match tiebreak: zone brightness matches top fiber
        ia = fiber_intensity(img, fibers[fa], all_zone_centers, dist)
        ib = fiber_intensity(img, fibers[fb], all_zone_centers, dist)
        r0, c0 = int(round(center[0])), int(round(center[1]))
        zone_i = float(np.median(img[max(r0 - 3, 0):r0 + 4,
                                     max(c0 - 3, 0):c0 + 4]))
        top = fa if abs(zone_i - ia) <= abs(zone_i - ib) else fb
        method = 'intensity-match'
        conf = min(abs(abs(zone_i - ia) - abs(zone_i - ib))
                   / (abs(ia - ib) + 1e-6), 1.0)
    # These cue scores are not calibrated probabilities. Abstain on weak evidence.
    if max(step_a,step_b)<0.01 or conf<0.4 or abs(float(da[0]*db[1]-da[1]*db[0]))<0.35:
        top, method = None, 'unresolved'
    return top, {fa: step_a, fb: step_b, 'method': method, 'conf': conf,
                 'w' + str(fa): wa, 'w' + str(fb): wb}
 
 
# ------------------------------------------------------------- rendering
def render(img, mask, fibers, crossings, dist, zone_centers,
           out_path, alpha=0.45):
    H, W = img.shape
    base = (np.stack([img] * 3, -1) * 255).astype(np.uint8)
    if not fibers:
        Image.fromarray(base).save(out_path)
        return
 
    # One nearest-centerline transform avoids an N_fibers x H x W array.
    # This visualization is not used for measurement or segmentation.
    seeds = np.zeros((H,W),np.int32)
    widths = np.array([max(fiber_halfwidth(f,dist),2.) for f in fibers])
    for i,f in enumerate(fibers):
        p=np.round(f['pts']).astype(int)
        seeds[p[:,0].clip(0,H-1),p[:,1].clip(0,W-1)]=i+1
    distance,indices=ndi.distance_transform_edt(seeds==0,return_indices=True)
    owner=seeds[indices[0],indices[1]]-1
    owner[(~mask)|(distance>2.5*widths[owner])]=-1
    # Limit overlap recoloring to this crossing, since two paths can cross
    # repeatedly with different local orders.
    trees={}
    for zc,ia,ib,top,_ in crossings:
        if top is None:continue
        radius=int(2*(widths[ia]+widths[ib])+8)
        r,c=np.round(zc).astype(int)
        r0,r1=max(0,r-radius),min(H,r+radius+1)
        c0,c1=max(0,c-radius),min(W,c+radius+1)
        rr,cc=np.mgrid[r0:r1,c0:c1]
        points=np.column_stack([rr.ravel(),cc.ravel()])
        for i in (ia,ib):
            if i not in trees:trees[i]=cKDTree(fibers[i]['pts'])
        da=trees[ia].query(points)[0].reshape(rr.shape)
        db=trees[ib].query(points)[0].reshape(rr.shape)
        both=(da<=1.15*widths[ia])&(db<=1.15*widths[ib])&mask[r0:r1,c0:c1]
        owner[r0:r1,c0:c1][both]=top
    overlay=base.copy()
    palette=np.array(PALETTE,float)
    valid=owner>=0
    overlay[valid]=(alpha*palette[owner[valid]%len(PALETTE)]+(1-alpha)*base[valid]).astype(np.uint8)

    for (zc, fa, fb, top, _) in crossings:
        r, c = int(zc[0]), int(zc[1])
        cv2.circle(overlay, (c, r), 6, (255, 255, 255), 1, cv2.LINE_AA)
        label = f'F{top + 1} on top' if top is not None else 'order unresolved'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                      0.38, 1)
        tx = min(max(c - tw // 2, 2), W - tw - 2)
        ty = max(r - 10, th + 2)
        cv2.putText(overlay, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(overlay, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (255, 255, 255), 1, cv2.LINE_AA)
 
    # Legend: multi-column when there are many fibers
    n = len(fibers)
    if n == 0:
        Image.fromarray(overlay).save(out_path)
        return
    col_w = max(90, W // max(1, min(n, 6)))
    cols_n = max(1, W // col_w)
    rows_n = (n + cols_n - 1) // cols_n
    row_h = 20
    pad = row_h * rows_n + 12
    canvas = np.zeros((H + pad, W, 3), np.uint8)
    canvas[:H] = overlay
    for i in range(n):
        col = i // rows_n
        row = i % rows_n
        x0 = col * col_w + 6
        y = H + 15 + row * row_h
        cv2.rectangle(canvas, (x0, y - 9), (x0 + 12, y + 2),
                      tuple(int(x) for x in PALETTE[i % len(PALETTE)]), -1)
        cv2.putText(canvas, f'F{i + 1}', (x0 + 18, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (255, 255, 255), 1, cv2.LINE_AA)
    Image.fromarray(canvas).save(out_path)
 
 
# ---------------------------------------------------------------- main
def analyze(path, out_prefix='result', mode='auto',
            max_dim=1400, crop_databar=True):
    """Run the whole pipeline. `mode` is 'skeleton' (default fast path for
    fibers that cross in X patterns), 'ridge' (for fibers that touch
    tangentially without an X-crossing), or 'auto' (try skeleton, fall
    back to ridge if the skeleton yields no junctions but the mask has
    substantial coverage -- the tell-tale sign of tangential touching)."""
    img = load_gray(path, max_dim=max_dim, crop_databar=crop_databar)
    mask = segment(img)
    dist = ndi.distance_transform_edt(mask)
 
    if mode in ('skeleton', 'auto'):
        skel, jcent, branches = skeleton_branches(mask)
    if mode == 'auto' and (len(jcent) == 0 and mask.mean() > 0.25):
        print('  (auto) no junctions detected but mask is dense -> ridge mode')
        mode = 'ridge'
    if mode == 'ridge':
        ridge_sk = ridge_skeleton_mask(img, mask)
        skel, jcent, branches = skeleton_branches(ridge_sk, already_skeleton=True)
 
    zones, zone_of, connectors = build_zones(branches, jcent, dist)
    fibers = stitch_fibers(branches, zones, zone_of, connectors)
    # drop tiny fragments: length < 3 x its own width, or below 40 px absolute
    kept = []
    for f in fibers:
        hw = fiber_halfwidth(f, dist)
        min_len = max(40, 3 * 2 * hw)
        if len(f['pts']) >= min_len:
            kept.append(f)
    fibers = kept
    print(f'  after length filter: {len(fibers)} fibers')
    sm_img = gaussian(img, 1.5)
    zone_centers = [d['center'] for d in zones.values()]
 
    crossings = []
    for a in range(len(fibers)):
        for b in range(a + 1, len(fibers)):
            for zc in find_crossings(fibers[a], fibers[b], dist):
                top, dbg = infer_z(img, sm_img, fibers, a, b,
                                   zc, dist, zone_centers)
                crossings.append((zc, a, b, top, dbg))
 
    render(img, mask, fibers, crossings, dist, zone_centers,
           f'{out_prefix}_overlay.png')
 
    print(f'=== {path} ===')
    print(f'Image after crop/downscale: {img.shape[1]} x {img.shape[0]} px')
    print(f'Mode: {mode}')
    print(f'Fiber count: {len(fibers)}')
    for i, f in enumerate(fibers):
        print(f'  Fiber {i + 1}: centerline {len(f["pts"])} px, '
              f'width ~{2 * fiber_halfwidth(f, dist, zone_centers):.0f} px')
    print(f'Crossings: {len(crossings)}')
    for (zc, a, b, top, dbg) in crossings:
        if top is None:
            print(f'  @ {zc}: order unresolved (cue score {dbg["conf"]:.2f})')
            continue
        bot = b if top == a else a
        print(f'  @ (x={int(zc[1])}, y={int(zc[0])}): '
              f'Fiber {top + 1} OVER Fiber {bot + 1}   '
              f'[{dbg["method"]}, cue score {dbg["conf"]:.2f}; '
              f'step F{a + 1}={dbg[a]:.4f} vs F{b + 1}={dbg[b]:.4f}]')
    return fibers, crossings
 
 
if __name__ == '__main__':
    import argparse
    import os
    p = argparse.ArgumentParser(description='Nanofiber crossing analysis.')
    p.add_argument('image', help='input image (PNG, TIFF, ...)')
    p.add_argument('-o', '--outdir', default='output',
                   help='output folder for overlays (default: output/)')
    p.add_argument('-p', '--prefix', default=None,
                   help='overlay filename prefix (default: input filename stem)')
    p.add_argument('-m', '--mode', default='auto',
                   choices=('auto', 'skeleton', 'ridge'),
                   help='segmentation mode. skeleton = fast, works when '
                        'fibers cross in X patterns; ridge = detect centerlines '
                        'as intensity ridges, needed when fibers touch '
                        'tangentially; auto = try skeleton first, fall back '
                        'to ridge if needed (default)')
    p.add_argument('--max-dim', type=int, default=1400,
                   help='downscale so max(H, W) <= this. Default 1400 keeps '
                        'memory sane on large TIFFs. Set 0 to disable.')
    p.add_argument('--no-crop-databar', action='store_true',
                   help='keep the SEM databar (default is to auto-crop it)')
    args = p.parse_args()
 
    os.makedirs(args.outdir, exist_ok=True)
    stem = args.prefix or os.path.splitext(os.path.basename(args.image))[0]
    out_prefix = os.path.join(args.outdir, stem)
    analyze(args.image, out_prefix,
            mode=args.mode,
            max_dim=(None if args.max_dim == 0 else args.max_dim),
            crop_databar=not args.no_crop_databar)
    print(f'\nOverlay written to: {out_prefix}_overlay.png')
