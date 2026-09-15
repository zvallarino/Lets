"""Conservative source-boundary and continuity checks for geometry previews."""
import numpy as np
from scipy import ndimage as ndi
from scipy.signal import find_peaks
from sem_image import segment


class SourceBoundaryCheck:
    def __init__(self, gray, diagnostics=True, interpolate=False, sample_scale=(1.,1.)):
        self.mask, self.segmentation = segment(gray, return_info=True) if diagnostics else (None,None)
        self.interpolate=interpolate
        self.sample_scale=np.asarray(sample_scale,dtype=float)
        self.shape=np.asarray(gray.shape)/self.sample_scale
        self.sm = ndi.gaussian_filter(gray, self.sample_scale)

    def measure(self, point, tangent, half_width):
        # SEM fibers have bright rims and darker cores. Use the first supported
        # outward intensity fall, rather than requiring empty background in a
        # global mask (underlying fibers often occupy that background).
        normal = np.array([-tangent[1], tangent[0]])
        h, w = self.shape
        ts = np.arange(0., max(12., 1.7*half_width)+5., .5)
        r,c=np.round(point).astype(int)
        if not (0<=r<h and 0<=c<w):return None
        edges=[]
        body_supported=False
        for direction in (-normal,normal):
            coords=np.asarray(point)[:,None]+direction[:,None]*ts
            if (coords[0].min()<1 or coords[0].max()>=h-2 or
                    coords[1].min()<1 or coords[1].max()>=w-2):
                return None
            profile=ndi.map_coordinates(self.sm,(coords+.5)*self.sample_scale[:,None]-.5,order=1)
            profile=ndi.gaussian_filter1d(profile,1.)
            gradient=np.gradient(profile,.5)
            peaks,_=find_peaks(-gradient,height=.004,prominence=.004)
            edge=None
            for i in peaks:
                if ts[i]<max(3.,.30*half_width) or ts[i]>1.6*half_width:
                    continue
                lo=max(0,i-5);hi=min(len(profile),i+7)
                contrast=float(profile[lo:i+1].max()-profile[i:hi].min())
                # Reject a seed in an inter-fiber gap; bright rims alone must
                # not make two neighboring fibers look like one wide body.
                if contrast<.025:
                    continue
                body_supported |= profile[0] > profile[i:hi].min()+.015
                travel=ts[i]
                if self.interpolate and 0<i<len(gradient)-1:
                    curvature=gradient[i-1]-2*gradient[i]+gradient[i+1]
                    if abs(curvature)>1e-12:
                        travel+=.5*float(np.clip(.5*(gradient[i-1]-gradient[i+1])/curvature,-.5,.5))
                edge=np.asarray(point)+direction*travel
                break
            if edge is None:return None
            edges.append(edge)
        if not body_supported:return None
        width=float(np.linalg.norm(edges[1]-edges[0]))
        return {'edge1_rc':edges[0], 'edge2_rc':edges[1], 'diameter_px':width}

    def has_internal_seam(self, edge1, edge2, tangent):
        """Detect a narrow bright internal ridge persisting along the axis.

        A bright rim between two dark fiber bodies differs from a single dark
        core bounded by two outer rims. Require a prominent interior peak in at
        least four of five longitudinal profiles at a consistent lateral offset.
        This flags an ambiguous merged width; it does not invent two identities.
        """
        edge1,edge2=np.asarray(edge1),np.asarray(edge2)
        width=float(np.linalg.norm(edge2-edge1))
        if width<12:return False
        normal=(edge2-edge1)/width
        mid=(edge1+edge2)/2
        xs=np.arange(-width/2,width/2+.25,.5)
        profiles=[]
        span=max(10.,.6*width)
        for offset in np.linspace(-span,span,5):
            coords=mid[:,None]+normal[:,None]*xs+tangent[:,None]*offset
            if (coords[0].min()<1 or coords[0].max()>=self.shape[0]-2 or
                    coords[1].min()<1 or coords[1].max()>=self.shape[1]-2):
                return False
            profile=ndi.map_coordinates(self.sm,(coords+.5)*self.sample_scale[:,None]-.5,order=1)
            peaks,props=find_peaks(profile,prominence=.045)
            candidates=[]
            for i,prominence in zip(peaks,props['prominences']):
                if abs(xs[i])>.30*width:continue
                radius=max(4,int(.18*width/.5))
                if i-radius<0 or i+radius>=len(profile):continue
                left=profile[i-radius:i-max(1,radius//3)]
                right=profile[i+max(1,radius//3):i+radius+1]
                if min(profile[i]-np.median(left),profile[i]-np.median(right))>=.04:
                    candidates.append(float(xs[i]))
            profiles.append(candidates)
        for x in profiles[2]:
            if sum(any(abs(y-x)<=max(2.,.08*width) for y in row) for row in profiles)>=4:
                return True
        return False

    def check(self, depth_measurement, tangent, keep_rejected=False):
        """Require two source boundaries stable on both sides along the axis.

        The depth map only proposes the seed. A gap at the seed, missing edge,
        widening, or shifted neighboring section causes abstention. Never bridge
        a missing section or infer an occluded boundary.
        """
        seed = (np.asarray(depth_measurement['edge1_rc']) +
                np.asarray(depth_measurement['edge2_rc'])) / 2
        hint = depth_measurement['diameter_px'] / 2
        if self.has_internal_seam(depth_measurement['edge1_rc'],depth_measurement['edge2_rc'],tangent):
            return None, 'persistent_internal_seam'
        m = self.measure(seed, tangent, hint)
        if m is None:
            return None, 'source_edges_unresolved'
        width = m['diameter_px']
        mid = (np.asarray(m['edge1_rc']) + np.asarray(m['edge2_rc'])) / 2
        if self.has_internal_seam(m['edge1_rc'],m['edge2_rc'],tangent):
            return (m if keep_rejected else None), 'persistent_internal_seam'
        if width < 6:
            return (m if keep_rejected else None), 'source_underresolved'
        if np.linalg.norm(mid-seed) > max(3., .25*width):
            return (m if keep_rejected else None), 'source_center_disagreement'
        normal = np.array([-tangent[1], tangent[0]])
        # The span scales with width to cover more than a one-pixel edge glitch.
        span = max(8., .4*width)
        for offset in (-span, span):
            neighbor = self.measure(mid + offset*tangent, tangent, width/2)
            if neighbor is None:
                return (m if keep_rejected else None), 'crossing_or_missing_edge'
            neighbor_mid = (np.asarray(neighbor['edge1_rc']) +
                            np.asarray(neighbor['edge2_rc'])) / 2
            if abs(neighbor['diameter_px']-width) > max(2., .18*min(width,neighbor['diameter_px'])):
                return (m if keep_rejected else None), 'local_width_change'
            if abs(np.dot(neighbor_mid-mid, normal)) > max(2., .20*width):
                return (m if keep_rejected else None), 'local_direction_change'
        return m, None


def filter_track(samples, tangent):
    """Reject width outliers and lateral jumps without joining gaps.

    Samples are dictionaries or None in axis order. Rejected sites retain a
    reason for review. A robust axis width reference is used only with enough
    support; it is not a population diameter estimate.
    """
    valid = [s for s in samples if s is not None and not s.get('rejection')]
    if not valid:
        return samples
    reference = float(np.median([s['width'] for s in valid]))
    normal = np.array([-tangent[1], tangent[0]])
    if len(valid) >= 5:
        for s in valid:
            if abs(s['width']-reference) > max(2., .20*reference):
                s['rejection'] = 'track_width_outlier'
    # Compare with sections on either side beyond the immediate cross-section,
    # so a short widened plateau cannot validate itself using adjacent samples.
    for s in valid:
        if s.get('rejection'):continue
        before=[];after=[]
        for other in valid:
            if other is s or other.get('rejection'):continue
            delta=other['mid']-s['mid']
            along=float(np.dot(delta,tangent))
            if not max(12.,s['width'])<=abs(along)<=max(100.,4*s['width']):continue
            if abs(np.dot(delta,normal))>max(4.,.35*s['width']):continue
            (before if along<0 else after).append(other['width'])
        if len(before)>=3 and len(after)>=3:
            local=min(float(np.median(before)),float(np.median(after)))
            if s['width']-local>max(2.,.20*local):
                s['rejection']='neighbor_width_outlier'
    previous = None
    for s in samples:
        if s is None or s.get('rejection'):
            previous = None
            continue
        if previous is not None:
            delta = s['mid']-previous['mid']
            if abs(np.dot(delta, normal)) > max(2., .15*min(s['width'],previous['width'])):
                s['rejection'] = 'track_center_jump'
                previous = None
                continue
        previous = s
    return samples
