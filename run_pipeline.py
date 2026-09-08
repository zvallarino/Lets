"""Reproducible SEM fiber batch analysis; results require image review.

python run_pipeline.py pictures/TIFF -o output/review
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.feature import structure_tensor
import fiber_analysis as fa
import fiber_diameter as fd
from sem_image import prepare, load_gray, segment
from sem_scale import analyze_tiff, expand_paths

VERSION = '2.0.0'


@dataclass(frozen=True)
class Config:
    max_dim: int = 1600
    crop_bottom: int | None = None
    nm_per_px: float | None = None
    threshold: float | None = None
    min_length_px: float = 40
    max_samples: int = 300
    sample_spacing_px: float = 30
    depth_model: str | None = None
    depth_revision: str | None = None

    def validate(self):
        if self.max_dim < 128 or self.max_samples < 1:
            raise ValueError('max-dim >=128 and max-samples >=1 are required')
        if self.crop_bottom is not None and self.crop_bottom < 0:
            raise ValueError('crop-bottom must be nonnegative')
        for name in ('nm_per_px','min_length_px','sample_spacing_px'):
            value = getattr(self,name)
            if value is not None and (not math.isfinite(value) or value<=0):
                raise ValueError(f'{name} must be positive and finite')
        if self.threshold is not None and not 0<=self.threshold<=1:
            raise ValueError('threshold must lie between 0 and 1')


def write_csv(path, rows, fields):
    with Path(path).open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def original_points(points, geometry):
    # PIL resize uses pixel-center coordinates. Distances use the same exact
    # independent x/y factors rather than a rounded nominal resize ratio.
    factors=np.array([geometry['source_px_per_analysis_y'],geometry['source_px_per_analysis_x']])
    return (np.asarray(points)+0.5)*factors-0.5


def trace(img, mask, cfg):
    dist=ndi.distance_transform_edt(mask)
    _,centers,branches=fa.skeleton_branches(mask)
    zones,zone_of,connectors=fa.build_zones(branches,centers,dist)
    fibers=fa.stitch_fibers(branches,zones,zone_of,connectors)
    fibers=[f for f in fibers if np.linalg.norm(np.diff(f['pts'],axis=0),axis=1).sum() >=
            max(cfg.min_length_px,6*fa.fiber_halfwidth(f,dist))]
    return fibers,dist,zones


def measurements(path, geometry, img, mask, dist, fibers, cfg, nm_per_px, outdir=None):
    """Systematic spatial samples; final edge walks use native-resolution pixels.

    No best-score cherry-picking. Rejected positions are retained in the audit.
    Repeated measurements on a path are not independent biological replicates.
    """
    if not mask.any() or not fibers:
        return [],[],None
    candidates=fd.prune_skeleton(mask,dist,max(8,round(min(mask.shape)*.01)))
    # Evenly limit candidates across raster order, then enforce spatial spacing.
    if len(candidates)>cfg.max_samples*20:
        candidates=candidates[np.linspace(0,len(candidates)-1,cfg.max_samples*20,dtype=int)]
    selected=[]; bins={}; spacing=cfg.sample_spacing_px
    for r,c in candidates:
        key=(int(r//spacing),int(c//spacing))
        nearby=[p for dr in (-1,0,1) for dc in (-1,0,1)
                for p in bins.get((key[0]+dr,key[1]+dc),[])]
        if any((r-a)**2+(c-b)**2<spacing**2 for a,b in nearby):
            continue
        selected.append((int(r),int(c))); bins.setdefault(key,[]).append((r,c))
    if len(selected)>cfg.max_samples:
        selected=[selected[i] for i in np.linspace(0,len(selected)-1,cfg.max_samples,dtype=int)]
    full=load_gray(path)[:geometry['roi_xywh'][3],:geometry['roi_xywh'][2]]
    native_mask,native_seginfo=segment(full,threshold=cfg.threshold,return_info=True)
    if outdir is not None:
        Image.fromarray(native_mask.astype(np.uint8)*255).save(Path(outdir)/'native_fiber_mask.png')
    native_sm=ndi.gaussian_filter(full,1.0)
    tensor=structure_tensor(ndi.gaussian_filter(img,1),sigma=3,mode='reflect')
    allpts=np.concatenate([f['pts'] for f in fibers])
    ids=np.concatenate([np.full(len(f['pts']),i) for i,f in enumerate(fibers)])
    tree=cKDTree(allpts)
    factors=np.array([geometry['source_px_per_analysis_y'],geometry['source_px_per_analysis_x']])
    accepted=[]; audit=[]
    for sample_id,(r,c) in enumerate(selected,1):
        d,j=tree.query([r,c]); fid=int(ids[j])+1
        original=original_points([r,c],geometry)
        row={'sample_id':sample_id,'fiber_id':fid,'x_px':float(original[1]),'y_px':float(original[0])}
        tangent,coherence=fd._tangent_at(*tensor,r,c)
        reason=None
        if d>max(2,dist[r,c]*0.5): reason='no_close_retained_path'
        if coherence<0.5: reason='ambiguous_axis'
        tangent=tangent*factors; tangent/=max(np.linalg.norm(tangent),1e-12)
        rr,cc=np.round(original).astype(int)
        m=None if reason else fd.measure_at(native_sm,native_mask,rr,cc,tangent,dist[r,c]*max(factors))
        if reason is None and m is None: reason='edges_not_isolated_or_low_contrast'
        if m is not None:
            e1,e2=m['edge1_rc'],m['edge2_rc']
            row.update({'edge1_x_px':e1[1],'edge1_y_px':e1[0],'edge2_x_px':e2[1],'edge2_y_px':e2[0],
                        'diameter_px':m['diameter_px'],'diameter_nm':m['diameter_px']*nm_per_px if nm_per_px else None,
                        'axis_deg':float(np.degrees(np.arctan2(tangent[0],tangent[1]))%180),
                        'axis_coherence':coherence,'contrast_min':min(m['contrast_l'],m['contrast_r'])})
            if m['diameter_px']<6:
                reason='under_resolved_width'
        row['status']='accepted' if reason is None else reason
        audit.append(row)
        if reason is None: accepted.append(row)
    return accepted,audit,native_seginfo


def overlay(img, mask, fibers, samples, geometry, path):
    rgb=np.repeat(np.round(img*255).astype(np.uint8)[...,None],3,axis=2)
    rgb[mask]=(rgb[mask]*.8+np.array([20,120,200])*.2).astype(np.uint8)
    im=Image.fromarray(rgb); draw=ImageDraw.Draw(im)
    for i,f in enumerate(fibers,1):
        pts=[(float(c),float(r)) for r,c in f['pts']]
        color=tuple(fa.PALETTE[(i-1)%len(fa.PALETTE)])
        draw.line(pts,fill=color,width=1)
        draw.text(pts[len(pts)//2],str(i),fill=(255,255,0),stroke_width=1,stroke_fill=(0,0,0))
    sx,sy=geometry['source_px_per_analysis_x'],geometry['source_px_per_analysis_y']
    for s in samples:
        a=((s['edge1_x_px']+.5)/sx-.5,(s['edge1_y_px']+.5)/sy-.5)
        b=((s['edge2_x_px']+.5)/sx-.5,(s['edge2_y_px']+.5)/sy-.5)
        draw.line([a,b],fill=(255,40,40),width=2)
    im.save(path)


SAMPLE_FIELDS=['sample_id','fiber_id','status','x_px','y_px','edge1_x_px','edge1_y_px','edge2_x_px','edge2_y_px','diameter_px','diameter_nm','axis_deg','axis_coherence','contrast_min']
SUMMARY_FIELDS=['file','status','estimated_visible_paths','paths_per_um2','paths_per_megapixel','coverage_pct','roi_area_um2','accepted_diameter_samples','sample_median_diameter_nm','unresolved_crossings','review_flags','seconds']


def analyze_one(path, outdir, cfg):
    cfg.validate(); started=time.monotonic(); path=Path(path)
    outdir=Path(outdir); outdir.mkdir(parents=True,exist_ok=True)
    # A run always writes a new directory so failures cannot expose stale results.
    target=outdir/(path.stem+'_'+digest(path)[:8])
    target.mkdir(exist_ok=False)
    img,geometry=prepare(path,cfg.max_dim,cfg.crop_bottom)
    print('  crop and calibration',flush=True)
    flags=[]; scale_meta={}
    try: scale_meta=analyze_tiff(path)
    except Exception as e: flags.append('metadata_unavailable: '+str(e))
    nm_per_px=cfg.nm_per_px or scale_meta.get('pixel_size_nm')
    if cfg.nm_per_px is None and nm_per_px:
        flags.append('calibration_from_JEOL_reference_requires_independent_check')
    if nm_per_px is None: flags.append('missing_physical_calibration')
    if geometry['roi_xywh'][3]==geometry['original_shape_hw'][0] and cfg.crop_bottom is None:
        flags.append('no_banner_detected_verify_ROI')
    mask,seginfo=segment(img,cfg.threshold,True)
    fibers,dist,zones=trace(img,mask,cfg)
    print(f'  {len(fibers)} estimated paths; evaluating crossings',flush=True)
    zcent=[v['center'] for v in zones.values()]; sm=ndi.gaussian_filter(img,1.5)
    crossings=[]
    # Bounding boxes avoid testing clearly disjoint paths.
    widths=[fa.fiber_halfwidth(f,dist) for f in fibers]
    bounds=[(f['pts'].min(axis=0)-widths[i],f['pts'].max(axis=0)+widths[i]) for i,f in enumerate(fibers)]
    for a in range(len(fibers)):
        for b in range(a+1,len(fibers)):
            if np.any(bounds[a][1]<bounds[b][0]) or np.any(bounds[b][1]<bounds[a][0]): continue
            for center in fa.find_crossings(fibers[a],fibers[b],dist):
                top,dbg=fa.infer_z(img,sm,fibers,a,b,center,dist,zcent)
                rc=original_points(center,geometry)
                crossings.append({'fiber_a':a+1,'fiber_b':b+1,'x_px':float(rc[1]),'y_px':float(rc[0]),
                                  'top_fiber':None if top is None else top+1,'method':dbg['method'],'cue_score':dbg['conf']})
    samples,audit,native_seginfo=measurements(path,geometry,img,mask,dist,fibers,cfg,nm_per_px,target)
    print(f'  {len(samples)} accepted diameter samples',flush=True)
    fiber_rows=[]; all_points=[]
    for i,f in enumerate(fibers,1):
        pts=original_points(f['pts'],geometry)
        _,_,vh=np.linalg.svd(pts-pts.mean(axis=0),full_matrices=False)
        direction=vh[0]
        values=[s['diameter_nm'] for s in samples if s['fiber_id']==i and s['diameter_nm'] is not None]
        fiber_rows.append({'fiber_id':i,'length_px':float(np.linalg.norm(np.diff(pts,axis=0),axis=1).sum()),
                           'global_axis_deg':float(np.degrees(np.arctan2(direction[0],direction[1]))%180),
                           'diameter_samples':len([s for s in samples if s['fiber_id']==i]),
                           'median_diameter_nm':float(np.median(values)) if values else None})
        all_points.extend({'fiber_id':i,'point_index':j,'x_px':float(c),'y_px':float(r)} for j,(r,c) in enumerate(pts))
    area_px=geometry['roi_xywh'][2]*geometry['roi_xywh'][3]
    area_um2=area_px*(nm_per_px/1000)**2 if nm_per_px else None
    values=[s['diameter_nm'] for s in samples if s['diameter_nm'] is not None]
    sensitivity=seginfo['coverage_threshold_minus_0_03_pct']-seginfo['coverage_threshold_plus_0_03_pct']
    flags.append('path_count_requires_review_for_splits_merges_and_occlusion')
    if sensitivity>10: flags.append('segmentation_sensitive_to_threshold')
    if abs(seginfo['otsu_coverage_pct']-seginfo['triangle_coverage_pct'])>10:
        flags.append('segmentation_methods_disagree_review_mask')
    if len(samples)<10: flags.append('few_accepted_diameter_samples')
    if samples and np.median([s['diameter_px'] for s in samples])/max(geometry['source_px_per_analysis_x'],geometry['source_px_per_analysis_y'])<6:
        flags.append('thin_fibers_at_tracing_resolution_count_may_be_incomplete')
    if not mask.any(): flags.append('no_foreground')
    if mask.mean()>.85: flags.append('dense_or_poorly_separated_foreground')
    unresolved=sum(c['top_fiber'] is None for c in crossings)
    if unresolved: flags.append('unresolved_crossings')
    if cfg.depth_model:
        try:
            from top_fiber_depth import run_depth,depth_pipeline,_HF_IDS
            pil=Image.fromarray(np.round(img*255).astype(np.uint8)).convert('RGB')
            depth=run_depth(pil,cfg.depth_model,cfg.depth_revision)
            np.save(target/'experimental_depth.npy',depth)
            depth_metadata={'model':_HF_IDS[cfg.depth_model], 'requested_revision':cfg.depth_revision,
                'resolved_revision':getattr(depth_pipeline(cfg.depth_model,cfg.depth_revision).model.config,'_commit_hash',None)}
            (target/'depth_metadata.json').write_text(json.dumps(depth_metadata,indent=2),encoding='utf-8')
            flags.append('depth_experimental_not_used_as_measurement_truth')
        except Exception as e: flags.append('depth_failed: '+str(e))
    result={'file':path.name,'status':'review_required','estimated_visible_paths':len(fibers),
            'paths_per_um2':len(fibers)/area_um2 if area_um2 else None,
            'paths_per_megapixel':len(fibers)/area_px*1e6,
            'coverage_pct':seginfo['coverage_pct'],'roi_area_um2':area_um2,
            'accepted_diameter_samples':len(samples),
            'sample_median_diameter_nm':float(np.median(values)) if values else None,
            'unresolved_crossings':unresolved,'review_flags':'; '.join(flags),
            'seconds':round(time.monotonic()-started,2)}
    report={**result,'version':VERSION,'input_path':str(path.resolve()),'input_sha256':digest(path),
            'config':asdict(cfg),'geometry':geometry,'scale_metadata':scale_meta,
            'nm_per_source_px':nm_per_px,'segmentation':seginfo,'native_segmentation':native_seginfo,
            'diameter_statistic':'median of accepted systematically spaced visible cross-sections; not a fiber-weighted mean',
            'density_definition':'estimated visible stitched paths / cropped physical field area; includes edge fragments; not total 3D fibers',
            'crossings':crossings,'count_is_estimate':True}
    (target/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    write_csv(target/'diameters.csv',audit,SAMPLE_FIELDS)
    reference=[{'image_sha256':report['input_sha256'],'sample_id':s['sample_id'],
                'x_px':s['x_px'],'y_px':s['y_px'],'analyst':'','session':'','manual_diameter_nm':''} for s in samples]
    write_csv(target/'manual_reference_template.csv',reference,['image_sha256','sample_id','x_px','y_px','analyst','session','manual_diameter_nm'])
    write_csv(target/'fibers.csv',fiber_rows,['fiber_id','length_px','global_axis_deg','diameter_samples','median_diameter_nm'])
    write_csv(target/'centerlines.csv',all_points,['fiber_id','point_index','x_px','y_px'])
    write_csv(target/'crossings.csv',crossings,['fiber_a','fiber_b','x_px','y_px','top_fiber','method','cue_score'])
    Image.fromarray(mask.astype(np.uint8)*255).save(target/'fiber_mask.png')
    Image.fromarray(np.round(img*255).astype(np.uint8)).save(target/'cropped_analysis.png')
    overlay(img,mask,fibers,samples,geometry,target/'review_overlay.png')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('inputs',nargs='+');p.add_argument('-o','--outdir',default='output/review')
    p.add_argument('--max-dim',type=int,default=1600);p.add_argument('--crop-bottom',type=int)
    p.add_argument('--nm-per-px',type=float);p.add_argument('--threshold',type=float)
    p.add_argument('--max-samples',type=int,default=300);p.add_argument('--sample-spacing-px',type=float,default=30)
    p.add_argument('--depth-model',choices=['small','base','large']);p.add_argument('--depth-revision')
    args=p.parse_args()
    missing=[path for path in args.inputs if not Path(path).exists()]
    if missing:p.error('Input paths not found: '+', '.join(missing))
    cfg=Config(max_dim=args.max_dim,crop_bottom=args.crop_bottom,nm_per_px=args.nm_per_px,
               threshold=args.threshold,max_samples=args.max_samples,sample_spacing_px=args.sample_spacing_px,
               depth_model=args.depth_model,depth_revision=args.depth_revision)
    try:cfg.validate()
    except ValueError as e:p.error(str(e))
    files=expand_paths(args.inputs)
    if not files:p.error('No TIFF input files found')
    from datetime import datetime,timezone
    run=Path(args.outdir)/datetime.now(timezone.utc).strftime('run_%Y%m%dT%H%M%S_%fZ')
    run.mkdir(parents=True,exist_ok=False)
    versions={}
    for name in ['numpy','scipy','Pillow','scikit-image','networkx','opencv-python','tifffile','torch','transformers']:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:pass
    manifest={'version':VERSION,'python':sys.version,'platform':platform.platform(),'dependencies':versions,
              'config':asdict(cfg),'source_sha256':{f.name:digest(f) for f in Path(__file__).parent.glob('*.py')}}
    (run/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    results=[];failures=0
    for i,f in enumerate(files,1):
        print(f'[{i}/{len(files)}] {f.name}',flush=True)
        try:result=analyze_one(f,run,cfg)
        except Exception as e:
            failures+=1;result={'file':f.name,'status':'failed','review_flags':f'{type(e).__name__}: {e}'}
        results.append(result)
        write_csv(run/'summary.csv',results,SUMMARY_FIELDS)
        print(json.dumps(result),flush=True)
    print(f'Results: {run.resolve()}',flush=True)
    return 1 if failures else 0


if __name__=='__main__':
    raise SystemExit(main())
