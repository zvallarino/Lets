"""Independent, non-AI diameter estimates from thresholded source-image sections.

Inspired by centerline/distance-map methods, not an implementation of DiameterJ.
No trained model, depth map, path stitching, or assumed hidden boundary is used.
"""
import argparse,csv,hashlib,json
from collections import Counter
from datetime import datetime
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
from sem_image import prepare,segment
from sem_metadata import check_sidecar
from sem_scale import analyze_tiff
from source_geometry import SourceBoundaryCheck


class MaskDiameter:
    def __init__(self,gray,threshold_delta=.03,min_width=10.):
        self.gray=gray;self.sm=ndi.gaussian_filter(gray,1.)
        initial,self.segmentation=segment(gray,return_info=True)
        # Histogram thresholds can land at the background edge of an empty bin
        # and systematically inflate widths. Use the class-intensity midpoint.
        otsu=self.segmentation['otsu_threshold']
        background=self.sm[~initial];foreground=self.sm[self.sm>otsu]
        midpoint=(float(np.median(background))+float(np.median(foreground)))/2 if background.size and foreground.size else otsu
        self.mask=segment(gray,threshold=midpoint)
        self.segmentation.update(classical_threshold=midpoint,classical_threshold_method='background_bright_class_midpoint',classical_coverage_pct=float(self.mask.mean()*100))
        self.threshold=midpoint;self.delta=threshold_delta;self.min_width=min_width
        self.skeleton=skeletonize(self.mask)
        self.distance=ndi.distance_transform_edt(np.pad(self.mask,1))[1:-1,1:-1]
        neighbors=ndi.convolve(self.skeleton.astype(np.uint8),np.ones((3,3),np.uint8),mode='constant')-self.skeleton
        junction=self.skeleton & (neighbors>2);ends=self.skeleton & (neighbors<2)
        self.junction_distance=ndi.distance_transform_edt(~junction) if junction.any() else np.full(gray.shape,np.inf)
        self.end_distance=ndi.distance_transform_edt(~ends) if ends.any() else np.full(gray.shape,np.inf)
        self.points=np.argwhere(self.skeleton)
        self.tree=cKDTree(self.points) if len(self.points) else None
        self.seam=SourceBoundaryCheck(gray,diagnostics=False)

    def tangent(self,point,radius):
        if self.tree is None:return None
        ids=self.tree.query_ball_point(point,max(8.,radius*1.5))
        if len(ids)<5:return None
        pts=self.points[ids].astype(float);pts-=pts.mean(axis=0)
        values,vectors=np.linalg.eigh(pts.T@pts)
        if values[1]<=0 or values[0]/values[1]>.08:return None
        return vectors[:,1]

    def section(self,point,tangent,threshold,hint):
        normal=np.array([-tangent[1],tangent[0]])
        ts=np.arange(0.,max(12.,3*hint)+.25,.25)
        edges=[]
        for direction in (-normal,normal):
            coords=np.asarray(point)[:,None]+direction[:,None]*ts
            inside=(coords[0]>=1)&(coords[0]<self.gray.shape[0]-2)&(coords[1]>=1)&(coords[1]<self.gray.shape[1]-2)
            coords=coords[:,inside];travel=ts[inside]
            if len(travel)<5:return None
            profile=ndi.map_coordinates(self.sm,coords,order=1)
            if profile[0]<=threshold:return None
            exits=np.flatnonzero(profile<=threshold)
            if not len(exits) or exits[0]==0:return None
            i=int(exits[0])
            # Require a real exterior gap; never continue to an edge on another fiber.
            if i+8>=len(profile) or np.any(profile[i:i+9]>threshold):return None
            if float(np.max(profile[:i])-np.median(profile[i+4:i+9]))<.025:return None
            denominator=profile[i-1]-profile[i]
            distance=travel[i-1]+.25*(profile[i-1]-threshold)/denominator
            edges.append(np.asarray(point)+direction*distance)
        return edges

    def measure(self,point,tangent=None):
        point=np.asarray(point,dtype=float);r,c=np.round(point).astype(int)
        if not (1<=r<self.mask.shape[0]-2 and 1<=c<self.mask.shape[1]-2) or not self.mask[r,c]:
            return None,'outside_segmented_fiber'
        radius=float(self.distance[r,c])
        local_tangent=self.tangent(point,radius)
        if local_tangent is None:return None,'ambiguous_axis'
        if tangent is None:tangent=local_tangent
        tangent=np.asarray(tangent,dtype=float);tangent/=np.linalg.norm(tangent)
        if abs(float(np.dot(tangent,local_tangent)))<np.cos(np.deg2rad(15)):
            return None,'axis_disagreement'
        edges=self.section(point,tangent,self.threshold,radius)
        if edges is None:return None,'missing_mask_boundary'
        a,b=edges;width=float(np.linalg.norm(b-a));mid=(a+b)/2
        result={'edge1_rc':a.tolist(),'edge2_rc':b.tolist(),'midpoint_rc':mid.tolist(),
                'width_analysis_px':width,'edt_width_analysis_px':2*radius,
                'tangent_rc':tangent.tolist()}
        if width<self.min_width:return result,'underresolved'
        if self.junction_distance[r,c]<1.25*width:return result,'near_junction'
        if self.end_distance[r,c]<.75*width:return result,'near_endpoint'
        if np.linalg.norm(mid-point)>max(1.5,.15*width):return result,'off_center'
        # EDT is diagnostic; unlike 2*EDT, interpolated sections avoid integer-radius bias.
        if abs(2*radius-width)>max(2.,.15*width):return result,'distance_map_disagreement'
        if self.seam.has_internal_seam(a,b,tangent):return result,'internal_seam'
        widths=[width]
        for threshold in (max(0.,self.threshold-self.delta),min(1.,self.threshold+self.delta)):
            variant=self.section(point,tangent,threshold,radius)
            if variant is None:return result,'threshold_unstable'
            widths.append(float(np.linalg.norm(variant[1]-variant[0])))
        result['threshold_width_min_px']=min(widths);result['threshold_width_max_px']=max(widths)
        if max(widths)-min(widths)>max(1.,.15*width):return result,'threshold_unstable'
        for offset in (-max(6.,.6*width),max(6.,.6*width)):
            pair=self.section(mid+offset*tangent,tangent,self.threshold,radius)
            if pair is None:return result,'neighbor_boundary_missing'
            other=float(np.linalg.norm(pair[1]-pair[0]));other_mid=(pair[0]+pair[1])/2
            if abs(other-width)>max(1.,.15*min(other,width)):return result,'neighbor_width_change'
            if abs(float(np.dot(other_mid-mid,np.array([-tangent[1],tangent[0]]))))>max(1.5,.15*width):
                return result,'neighbor_center_change'
        return result,None

    def sample(self,spacing=12,max_candidates=3000):
        # Deterministic image-wide sampling, not the first N sites in scan order.
        selected=[];cells=set()
        for r,c in self.points:
            key=(int(r//spacing),int(c//spacing))
            if key not in cells:cells.add(key);selected.append((r,c))
        if len(selected)>max_candidates:
            indices=np.linspace(0,len(selected)-1,max_candidates,dtype=int)
            selected=[selected[i] for i in indices]
        rows=[]
        for i,point in enumerate(selected,1):
            measured,reason=self.measure(point)
            rows.append(dict(measured or {},sample_id=i,seed_rc=list(map(int,point)),
                             review='CHECK' if reason else 'OK',reason=reason or ''))
        return rows


def write_csv(path,rows,fields):
    with Path(path).open('w',newline='',encoding='utf-8-sig') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)


def scaled_width(a,b,transform,nm_xy):
    delta=(np.asarray(b)-a)*[transform['source_px_per_analysis_y'],transform['source_px_per_analysis_x']]
    if nm_xy is not None:delta=delta*np.asarray(nm_xy[::-1])/1000
    return float(np.linalg.norm(delta))


def compare_geometry(checker,transform,geometry_output,out,source=None):
    """Same-location comparison. Unmeasurable sites are never called agreement."""
    folder=Path(geometry_output);notes=json.loads((folder/'preview_notes.json').read_text(encoding='utf-8'))
    if source is not None:
        if Path(notes.get('source','')).resolve()!=Path(source).resolve():
            raise ValueError('Geometry output belongs to a different source image')
        if notes.get('source_sha256') and notes['source_sha256']!=hashlib.sha256(Path(source).read_bytes()).hexdigest():
            raise ValueError('Source image changed since the geometry run')
    with (folder/'diameters_audit.csv').open(encoding='utf-8-sig') as handle:
        accepted={int(r['sample_id']) for r in csv.DictReader(handle) if r['review']=='OK'}
    if notes['geometry']['roi_xywh']!=transform['roi_xywh']:
        raise ValueError('Cannot compare methods with different crop regions')
    gs=np.array([notes['geometry']['source_px_per_analysis_y'],notes['geometry']['source_px_per_analysis_x']])
    cs=np.array([transform['source_px_per_analysis_y'],transform['source_px_per_analysis_x']])
    rows=[]
    for i,site in enumerate(notes['sites'],1):
        if i not in accepted:continue
        a=(np.array(site['edge1_rc'])+.5)*gs-.5;b=(np.array(site['edge2_rc'])+.5)*gs-.5
        mid=(a+b)/2;normal=(b-a)/np.linalg.norm(b-a)
        source_tangent=np.array([-normal[1],normal[0]])
        tangent=source_tangent/cs;tangent/=np.linalg.norm(tangent)
        measured,reason=checker.measure((mid+.5)/cs-.5,tangent)
        row={'geometry_sample_id':i,'x_source_px':float(mid[1]),'y_source_px':float(mid[0]),
             'geometry_width_source_px':float(np.linalg.norm(b-a)),
             'comparison':'not_comparable','reason':reason or ''}
        if measured is not None:
            ca=(np.array(measured['edge1_rc'])+.5)*cs-.5;cb=(np.array(measured['edge2_rc'])+.5)*cs-.5
            cw=float(np.linalg.norm(cb-ca));gw=row['geometry_width_source_px']
            row['non_ai_width_source_px']=cw
            row['relative_difference']=abs(cw-gw)/((cw+gw)/2)
            if reason is None:
                # Agreement must include BOTH edge locations, not just equal widths on adjacent fibers.
                edge_error=min(max(np.linalg.norm(ca-a),np.linalg.norm(cb-b)),max(np.linalg.norm(ca-b),np.linalg.norm(cb-a)))
                row['max_edge_difference_source_px']=float(edge_error)
                row['comparison']='agree' if row['relative_difference']<=.10 and edge_error<=max(float(cs.max()),.10*gw) else 'disagree'
        rows.append(row)
    fields=['geometry_sample_id','x_source_px','y_source_px','geometry_width_source_px','non_ai_width_source_px',
            'relative_difference','max_edge_difference_source_px','comparison','reason']
    write_csv(Path(out)/'method_comparison.csv',rows,fields)
    write_csv(Path(out)/'cross_checked_measurements.csv',[r for r in rows if r['comparison']=='agree'],fields)
    counts=Counter(r['comparison'] for r in rows)
    return {'method_agree':counts['agree'],'method_disagree':counts['disagree'],'method_not_comparable':counts['not_comparable']}


def analyze_classical(image,outdir='output/non_ai',crop_bottom=None,geometry_output=None):
    image=Path(image);metadata=check_sidecar(image)
    if metadata['status']!='not_available':
        expected=metadata['crop_bottom_px']
        if crop_bottom is not None and crop_bottom!=expected:raise ValueError('Crop conflicts with metadata')
        crop_bottom=expected
    gray,transform=prepare(image,1600,crop_bottom)
    checker=MaskDiameter(gray)
    rows=checker.sample()
    out=Path(outdir)/(image.stem+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir(parents=True,exist_ok=False)
    nm_xy=metadata.get('nm_per_source_pixel_xy')
    calibration='JEOL sidecar consistency check' if nm_xy else 'unavailable'
    if nm_xy is None:
        try:nominal=analyze_tiff(image)['pixel_size_nm']
        except (ValueError,OSError,KeyError):nominal=None
        if nominal is not None and np.isfinite(nominal) and nominal>0:
            nm_xy=[nominal,nominal];calibration='nominal JEOL magnification/reference calibration'
    unit='um' if nm_xy else 'source_px';diameter_key='diameter_'+unit
    for row in rows:
        if 'edge1_rc' in row:
            row[diameter_key]=scaled_width(row['edge1_rc'],row['edge2_rc'],transform,nm_xy)
            mid=np.asarray(row['midpoint_rc'])
            row['x_source_px']=float((mid[1]+.5)*transform['source_px_per_analysis_x']-.5)
            row['y_source_px']=float((mid[0]+.5)*transform['source_px_per_analysis_y']-.5)
    fields=['sample_id',diameter_key,'review','reason','x_source_px','y_source_px','width_analysis_px',
            'edt_width_analysis_px','threshold_width_min_px','threshold_width_max_px']
    write_csv(out/'non_ai_measurements.csv',rows,fields)
    good=[r for r in rows if r['review']=='OK']
    write_csv(out/'non_ai_ok.csv',good,fields)
    Image.fromarray(np.uint8(checker.mask)*255).save(out/'segmentation_mask.png')
    rgb=np.repeat(np.uint8(np.clip(gray,0,1)*255)[...,None],3,axis=2)
    boundary=checker.mask & ~ndi.binary_erosion(checker.mask)
    rgb[boundary]=[255,190,0]
    Image.fromarray(rgb).save(out/'segmentation_overlay.png')
    preview=Image.fromarray(np.uint8(np.clip(gray,0,1)*255)).convert('RGB');draw=ImageDraw.Draw(preview)
    for row in good:
        a,b=np.array(row['edge1_rc']),np.array(row['edge2_rc']);mid=(a+b)/2;t=np.array(row['tangent_rc'])
        draw.line([tuple((mid-3*t)[::-1]),tuple((mid+3*t)[::-1])],fill=(40,175,255),width=2)
        draw.line([tuple(a[::-1]),tuple(b[::-1])],fill=(255,50,50),width=2)
    preview.save(out/'non_ai_overlay.png')
    values=np.array([r[diameter_key] for r in good])
    stats={'n':len(values),'units':unit,'median':float(np.median(values)) if len(values) else None,
           'p10':float(np.percentile(values,10)) if len(values) else None,'p90':float(np.percentile(values,90)) if len(values) else None,
           'scope':'Sampled cross-section distribution, not independent-fiber statistics.'}
    if len(values):
        counts,edges=np.histogram(values,bins='auto')
        write_csv(out/'non_ai_histogram.csv',[{'lower':float(a),'upper':float(b),'count':int(n)} for a,b,n in zip(edges[:-1],edges[1:],counts)],['lower','upper','count'])
    report={'method':'classical_source_mask_v1','source':str(image.resolve()),'source_sha256':hashlib.sha256(image.read_bytes()).hexdigest(),
        'status':'researcher_review_required','geometry':transform,'metadata_check':metadata,'calibration':calibration,
        'nm_per_source_pixel_xy':nm_xy,'segmentation':checker.segmentation,
        'non_ai_candidates':len(rows),'non_ai_ok':len(good),'non_ai_check':len(rows)-len(good),
        'rejection_counts':dict(Counter(r['reason'] for r in rows if r['reason'])),
        'settings':{'min_width_analysis_px':10,'threshold_delta':.03,'width_sensitivity_fraction':.15,
                    'junction_exclusion_width_factor':1.25,'sampling_grid_px':12,'max_candidates':3000},
        'statistics':stats,'sites':rows,
        'scope':'Independent non-AI site selection and mask-based measurement. Not DiameterJ and not ground truth. Original source checks share seam rejection code with geometry.'}
    if geometry_output is not None:report.update(compare_geometry(checker,transform,geometry_output,out,image))
    (out/'non_ai_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(f"Non-AI: {len(good)} OK / {len(rows)} sampled candidates. Saved: {out.resolve()}",flush=True)
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('image');p.add_argument('-o','--outdir',default='output/non_ai')
    p.add_argument('--crop-bottom',type=int);p.add_argument('--geometry-output')
    args=p.parse_args();analyze_classical(**vars(args))


if __name__=='__main__':main()
