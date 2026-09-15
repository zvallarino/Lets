"""Large-model visual geometry review. No fiber counts or density calculations.

Depth proposes axes; crossbars require stable boundaries in the source image.
These are preview widths, not validated physical diameters. Inspect the matching
source overlay. Straight prominent fibers only;
curved, hidden, weakly resolved, or ambiguous crossings are deliberately omitted.
"""
import argparse,json,csv,shutil,hashlib
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from skimage.transform import hough_line,hough_line_peaks
from fiber_diameter import measure_at
from source_geometry import SourceBoundaryCheck, filter_track
from collections import Counter
from sem_image import prepare
from sem_metadata import check_sidecar
from top_fiber_depth import run_depth

LARGE_REVISION='7581137eff8d4e94f6e796d3baea0e9fa79b22d2'

def geometry_preview(gray,depth,nearest_percent=15,sample_step=4.,crossbar_spacing=6.):
    if not all(np.isfinite(v) and v>0 for v in (sample_step,crossbar_spacing)):
        raise ValueError("Sampling step and crossbar spacing must be finite and positive")
    if gray.shape != depth.shape or not np.isfinite(depth).all():
        raise ValueError('Image and finite depth map must have identical dimensions')
    if not 0<nearest_percent<100:raise ValueError('nearest must be between 0 and 100')
    if np.ptp(depth)<1e-8:raise ValueError('Depth map is constant; no geometry can be inferred')
    d=(depth-depth.min())/np.ptp(depth)
    cutoff=float(np.percentile(d,100-nearest_percent))
    mask=d>=cutoff
    sk=skeletonize(mask)
    votes,angles,rhos=hough_line(sk)
    acc,theta,rho=hough_line_peaks(votes,angles,rhos,min_distance=35,min_angle=15,threshold=100,num_peaks=6)
    H,W=d.shape
    im=Image.fromarray(np.round(d*255).astype(np.uint8)).convert('RGB');draw=ImageDraw.Draw(im)
    source=Image.fromarray(np.round(gray*255).astype(np.uint8)).convert('RGB');source_draw=ImageDraw.Draw(source)
    sm=ndi.gaussian_filter(d,1)
    dist=ndi.distance_transform_edt(mask)
    source_check=SourceBoundaryCheck(gray)
    records=[]
    audit=[]
    reasons=Counter()
    for a,t,rh in zip(acc,theta,rho):
        normal=np.array([np.sin(t),np.cos(t)])
        tangent=np.array([-normal[1],normal[0]])
        origin=rh*normal
        samples=[];crossbars=[]
        for s in np.arange(-2*max(H,W),2*max(H,W),sample_step):
            p=origin+s*tangent;r,c=np.round(p).astype(int)
            if not (8<=r<H-8 and 8<=c<W-8) or not mask[r,c]:
                samples.append(None);continue
            # Do not measure merged silhouettes around another candidate axis.
            near_crossing=any(abs(float(np.dot(p,np.array([np.sin(t2),np.cos(t2)])))-rh2)<max(65.,2*dist[r,c])
                              for t2,rh2 in zip(theta,rho) if abs(t2-t)>.05)
            if near_crossing:
                reasons['candidate_axis_crossing']+=1
                samples.append(None);continue
            m=measure_at(sm,mask,r,c,tangent,max(float(dist[r,c]),10))
            if m is None:
                samples.append(None);continue
            m,reason=source_check.check(m,tangent,keep_rejected=True)
            if reason:
                reasons[reason]+=1
                rejected={'axis':len(records)+1,'seed_rc':p.tolist(),'status':'rejected','reason':reason}
                if m is not None:
                    rejected.update({'edge1_rc':np.asarray(m['edge1_rc']).tolist(),'edge2_rc':np.asarray(m['edge2_rc']).tolist(),
                        'midpoint_rc':((np.asarray(m['edge1_rc'])+np.asarray(m['edge2_rc']))/2).tolist(),
                        'width_analysis_px':m['diameter_px']})
                audit.append(rejected)
                samples.append(None);continue
            e1,e2=np.array(m['edge1_rc']),np.array(m['edge2_rc'])
            mid=(e1+e2)/2
            if np.linalg.norm(mid-p)>35:
                reasons['axis_center_disagreement']+=1
                samples.append(None);continue
            samples.append({'mid':mid,'e1':e1,'e2':e2,'width':m['diameter_px']})
        samples=filter_track(samples,tangent)
        positions=[]
        valid=[s for s in samples if s is not None and not s.get('rejection')]
        supported=len(valid)>=3
        for sample in samples:
            if sample is None:
                positions.append(None);continue
            reason=sample.get('rejection') or (None if supported else 'insufficient_axis_support')
            entry={'axis':len(records)+1,'midpoint_rc':sample['mid'].tolist(),
                   'edge1_rc':sample['e1'].tolist(),'edge2_rc':sample['e2'].tolist(),
                   'width_analysis_px':sample['width'],'status':'rejected' if reason else 'accepted'}
            if reason:
                reasons[reason]+=1
                entry['reason']=reason
                positions.append(None)
            else:
                mid=sample['mid']
                positions.append(mid)
                if not crossbars or np.linalg.norm(mid-crossbars[-1][0])>=crossbar_spacing:
                    crossbars.append((mid,sample['e1'],sample['e2']))
            audit.append(entry)
        previous=None
        for p in positions:
            if p is None:previous=None;continue
            if previous is not None and np.linalg.norm(p-previous)<25:
                points=[(float(previous[1]),float(previous[0])),(float(p[1]),float(p[0]))]
                draw.line(points,fill=(40,175,255),width=2)
                source_draw.line(points,fill=(40,175,255),width=2)
            previous=p
        for mid,e1,e2 in crossbars:
            points=[(float(e1[1]),float(e1[0])),(float(e2[1]),float(e2[0]))]
            draw.line(points,fill=(255,50,50),width=2)
            source_draw.line(points,fill=(255,50,50),width=2)
        if valid and supported:
            label=valid[len(valid)//2]['mid']
            for painter in (draw,source_draw):
                painter.text((float(label[1])+6,float(label[0])+4),f'F{len(records)+1}',fill=(255,230,40),stroke_width=1,stroke_fill=(0,0,0))
        records.append({'line_normal_angle':float(t),'line_offset_px':float(rh),'hough_support':int(a),'preview_crossbars':len(crossbars),'accepted_sites':len(valid) if supported else 0})
    return im,source,{'purpose':'Visual review only; source-checked preview widths are not validated physical diameters',
                      'geometry_method':'source_checked_v3',
                      'sample_step_analysis_px':sample_step,'crossbar_spacing_analysis_px':crossbar_spacing,
                      'measurement_space':'resized analysis image pixels; row,column coordinates',
                      'source_segmentation':source_check.segmentation,
                      'rejection_counts':dict(reasons),'sites':audit,
                      'depth_cutoff':cutoff,'nearest_percent':nearest_percent,'axes':records}

def input_images(path):
    path=Path(path)
    if path.is_file():return [path]
    if path.is_dir():
        return sorted((f for f in path.iterdir() if f.is_file() and f.suffix.lower() in ('.tif','.tiff')),key=lambda f:f.name.lower())
    raise ValueError('Input does not exist: '+str(path))


def export_measurements(out,notes,tolerance=.20):
    """Three-column researcher CSV, plus coordinates/reasons in an audit CSV.

    Geometry axis IDs are provisional tracks, not confirmed biological identities.
    Rejected measurements remain CHECK rows and are excluded from OK summaries.
    """
    from sem_scale import analyze_tiff
    transform=notes['geometry']
    metadata=notes.get('metadata_check',{})
    nm_xy=metadata.get('nm_per_source_pixel_xy')
    calibration='JEOL sidecar consistency check' if nm_xy else 'unavailable'
    if nm_xy is None:
        try:
            nominal=analyze_tiff(notes['source'])['pixel_size_nm']
        except (ValueError,OSError,KeyError):nominal=None
        if nominal is not None and np.isfinite(nominal) and nominal>0:
            nm_xy=[nominal,nominal];calibration='nominal JEOL magnification/reference calibration'
    sx,sy=transform['source_px_per_analysis_x'],transform['source_px_per_analysis_y']
    rows=[]
    for index,site in enumerate(notes['sites'],1):
        if 'edge1_rc' not in site or 'edge2_rc' not in site:continue
        e1,e2=np.asarray(site['edge1_rc']),np.asarray(site['edge2_rc'])
        delta=(e2-e1)*np.array([sy,sx])
        width=float(np.linalg.norm(delta*(np.array(nm_xy[::-1])/1000 if nm_xy else 1)))
        mid=(e1+e2)/2
        axis=notes['axes'][site['axis']-1]
        t=axis['line_normal_angle']
        along=float(np.dot(mid,np.array([-np.cos(t),np.sin(t)])))
        rows.append({'fiber_id':f"Fiber {site['axis']}",'axis_id':site['axis'],'sample_id':index,
                     'diameter':width,'review':'OK' if site['status']=='accepted' else 'CHECK',
                     'reason':site.get('reason',''),'position_along_axis':along,
                     'x_source_px':float((mid[1]+.5)*sx-.5),'y_source_px':float((mid[0]+.5)*sy-.5),
                     'status':site['status']})
    rows.sort(key=lambda r:(r['axis_id'],r['position_along_axis']))
    for row in rows:
        neighbors=sorted([r for r in rows if r['axis_id']==row['axis_id'] and r is not row and r['status']=='accepted'],
                         key=lambda r:abs(r['position_along_axis']-row['position_along_axis']))[:6]
        if len(neighbors)>=2:
            reference=float(np.median([r['diameter'] for r in neighbors]+([row['diameter']] if row['status']=='accepted' else [])))
            deviation=abs(row['diameter']-reference)/reference if reference>0 else 0
            if deviation>tolerance:
                row['review']='CHECK';row['reason']='; '.join(filter(None,[row['reason'],'neighbor_diameter_difference']))
        else:
            row['review']='CHECK';row['reason']='; '.join(filter(None,[row['reason'],'insufficient_neighbors']))
    unit='um' if nm_xy else 'source_px'
    diameter_key='diameter_'+unit
    with (Path(out)/'diameters.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f);writer.writerow(['fiber_id',diameter_key,'review'])
        writer.writerows((r['fiber_id'],round(r['diameter'],6),r['review']) for r in rows)
    with (Path(out)/'diameters_ok.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f);writer.writerow(['fiber_id',diameter_key,'review'])
        writer.writerows((r['fiber_id'],round(r['diameter'],6),r['review']) for r in rows if r['review']=='OK')
    fields=['fiber_id','sample_id',diameter_key,'review','reason','status','x_source_px','y_source_px','position_along_axis']
    with (Path(out)/'diameters_audit.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader()
        writer.writerows(dict(r,**{diameter_key:r['diameter']}) for r in rows)
    notes['diameter_export']={'units':unit,'calibration':calibration,'neighbor_tolerance_fraction':tolerance,
        'rows':len(rows),'check_rows':sum(r['review']=='CHECK' for r in rows),
        'fiber_id_definition':'Provisional geometry axis/track ID; not a confirmed individual fiber or count. IDs are independent of density module.',
        'scope':'All numerically measurable sites, including filtered-out candidates; CHECK rows require review. Widths use native-pixel edge verification when native_verification is present; otherwise preview estimates scaled to source units.'}
    return rows


MEASUREMENT_FIELDS=['measurements_total','measurements_ok','measurements_check','red_lines_drawn']
REVIEW_FIELDS=['review_status','target_ok_measurements','ok_fraction','top_check_reasons']

REVIEW_REASONS={
    'preview_width_disagreement':'Original-pixel width differed by more than 20% (with a 2-analysis-pixel floor) from the preview proposal.',
    'source_edges_unresolved':'Could not locate both fiber boundaries in the source image.',
    'crossing_or_missing_edge':'A nearby section lost a boundary, often at an overlap or image edge.',
    'local_width_change':'Width changed too much between neighboring sections.',
    'local_direction_change':'The center moved sideways between neighboring sections.',
    'persistent_internal_seam':'An internal bright ridge suggests a merged or ambiguous fiber width.',
    'source_center_disagreement':'Source-image center disagreed with the depth proposal.',
    'source_underresolved':'Width was below six analysis pixels.',
    'track_width_outlier':'Width differed too much from other sections on this track.',
    'neighbor_width_outlier':'A wider section disagreed with sections on both sides.',
    'track_center_jump':'The tracked center jumped sideways.',
    'insufficient_axis_support':'Fewer than three sections survived on this proposed axis.',
    'neighbor_diameter_difference':'Exported width differed from neighboring widths by over 20%.',
    'insufficient_neighbors':'Fewer than two accepted neighboring sections were available.',
    'candidate_axis_crossing':'Candidate was too close to another proposed axis.',
    'axis_center_disagreement':'Measured center was too far from the proposed axis.',
}


def write_review_summary(out,notes,rows,target=100):
    """Report sampling sufficiency separately from successful execution or accuracy."""
    if not isinstance(target,int) or target<1:raise ValueError('Measurement target must be a positive integer')
    counts=measurement_counts(notes)
    ok=counts['measurements_ok'];total=counts['measurements_total']
    reasons=Counter(reason for row in rows if row['review']=='CHECK'
                    for reason in row['reason'].split('; ') if reason)
    notes['measurement_review']={
        'review_status':'no_ok_measurements' if ok==0 else ('below_target' if ok<target else 'target_reached'),
        'target_ok_measurements':target,'ok_fraction':round(ok/total,4) if total else None,
        'top_check_reasons':'; '.join(f'{reason}: {count}' for reason,count in reasons.most_common(3)),
    }
    notes['check_reason_counts']=dict(reasons)
    with (Path(out)/'review_reasons.csv').open('w',newline='',encoding='utf-8-sig') as handle:
        writer=csv.writer(handle);writer.writerow(['scope','reason','count','explanation'])
        for scope,entries in [('exported_CHECK_rows',reasons),('geometry_candidates',notes.get('rejection_counts',{}))]:
            for reason,count in sorted(entries.items(),key=lambda item:(-item[1],item[0])):
                writer.writerow([scope,reason,count,REVIEW_REASONS.get(reason,('Original-pixel recheck: '+REVIEW_REASONS.get(reason[7:],reason[7:])) if reason.startswith('native_') else reason)])
    (Path(out)/'REVIEW_README.txt').write_text(
        'Completed means execution finished, not that measurement quality passed.\n'
        'OK means the software checks passed; inspect the source overlay before using results.\n'
        'CHECK means a numerical width needs review; do not include it in the accepted diameter distribution.\n'
        'diameters_ok.csv contains only OK rows. diameters.csv and diameters_audit.csv retain all measured candidates.\n'
        'The measurement target is a sampling warning, not a quality validation threshold.\n'
        'Repeated sections on one track are not independent fibers. More sections do not prove greater accuracy.\n'
        'Red lines are a spaced subset of final OK measurements after native-pixel and neighbor review.\n'
        'review_reasons.csv separates exported CHECK rows from geometry candidate rejections.\n'
        'A row may have multiple reasons; these reason totals must not be added together as a measurement count.\n',encoding='utf-8')
    return notes['measurement_review']


def measurement_counts(notes):
    """Count exported widths after all review checks, separately from display lines.

    Missing data stays blank for older outputs; an analyzed empty image has zeros.
    These are cross-section measurements, not individual fiber counts.
    """
    export=notes.get('diameter_export',{})
    total=export.get('rows')
    check=export.get('check_rows')
    return dict(zip(MEASUREMENT_FIELDS,[total,
        total-check if total is not None and check is not None else None,check,
        sum(a['preview_crossbars'] for a in notes['axes']) if 'axes' in notes else None]))


def analyze_image(image,outdir,nearest=15,crop_bottom=None,sample_step=4.,crossbar_spacing=6.,target_measurements=100):
    metadata_check=check_sidecar(image)
    if metadata_check['status']!='not_available':
        expected_crop=metadata_check['crop_bottom_px']
        if crop_bottom is not None and crop_bottom!=expected_crop:
            raise ValueError(f'Explicit crop {crop_bottom} conflicts with sidecar crop {expected_crop}')
        crop_bottom=expected_crop
        print('  Text metadata checked; banner crop '+str(crop_bottom)+' px.',flush=True)
    for warning in metadata_check.get('warnings',[]):
        print('  Metadata note: '+warning,flush=True)
    gray,transform=prepare(image,1600,crop_bottom)
    depth=run_depth(Image.fromarray(np.round(gray*255).astype(np.uint8)).convert('RGB'),'large',LARGE_REVISION)
    preview,source,notes=geometry_preview(gray,depth,nearest,sample_step,crossbar_spacing)
    from datetime import datetime
    out=Path(outdir)/(Path(image).stem+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True,exist_ok=False)
    np.save(out/'large_depth.npy',depth)
    notes.update({'model':'Depth Anything V2 Large','revision':LARGE_REVISION,'source':str(Path(image).resolve()),'source_sha256':hashlib.sha256(Path(image).read_bytes()).hexdigest(),'geometry':transform,'metadata_check':metadata_check})
    from native_geometry import refine_sites,render_final
    refine_sites(image,notes)
    rows=export_measurements(out,notes)
    preview,source=render_final(gray,depth,notes,rows,crossbar_spacing)
    preview.save(out/'geometry_preview.png');source.save(out/'source_geometry_preview.png')
    review=write_review_summary(out,notes,rows,target_measurements)
    counts=measurement_counts(notes)
    notes['measurement_summary']=counts
    with (out/'measurement_summary.csv').open('w',newline='',encoding='utf-8-sig') as handle:
        writer=csv.DictWriter(handle,fieldnames=['image']+MEASUREMENT_FIELDS+REVIEW_FIELDS)
        writer.writeheader();writer.writerow(dict(image=notes['source'],**counts,**review))
    (out/'preview_notes.json').write_text(json.dumps(notes,indent=2),encoding='utf-8')
    print(f"  {Path(image).name}: {counts['measurements_total']} measured widths "
          f"({counts['measurements_ok']} OK, {counts['measurements_check']} CHECK); "
          f"{counts['red_lines_drawn']} red lines drawn.",flush=True)
    print(f"  Review: {review['review_status']} ({counts['measurements_ok']}/{target_measurements} requested OK widths).",flush=True)
    if review['top_check_reasons']:print('  CHECK reasons: '+review['top_check_reasons'],flush=True)
    print('Preview saved: '+str(out.resolve()),flush=True)
    return out


def copy_review_images(result,review,prefix):
    """Keep researcher files intact and collect uniquely named annotated copies."""
    review=Path(review)
    review.mkdir(parents=True,exist_ok=True)
    paths={}
    for kind,filename in [('source','source_geometry_preview.png')]:
        target=review/(prefix+'__'+kind+'.png')
        shutil.copy2(Path(result)/filename,target)
        paths['review_'+kind]=str(target.resolve())
    return paths


def run_batch(images,outdir,nearest=15,crop_bottom=None,sample_step=4.,crossbar_spacing=6.,target_measurements=100):
    from datetime import datetime
    run=Path(outdir)/('batch_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    run.mkdir(parents=True,exist_ok=False)
    review=run/'quick_review'
    review.mkdir()
    rows=[]
    for index,image in enumerate(images,1):
        print(f'[{index}/{len(images)}] {image.name}',flush=True)
        try:
            result=analyze_image(image,run,nearest,crop_bottom,sample_step,crossbar_spacing,target_measurements)
            row={'image':str(image.resolve()),'status':'completed','output':str(result.resolve()),'error':'',
                 'review_source':''}
            notes=json.loads((result/'preview_notes.json').read_text(encoding='utf-8'))
            row.update(measurement_counts(notes))
            row.update(notes.get('measurement_review',{}))
            try:
                row.update(copy_review_images(result,review,f'{index:03d}__{image.name}'))
            except OSError as exc:
                row['status']='review_copy_failed'
                row['error']=f'{type(exc).__name__}: {exc}'
                print('Review copy failed: '+row['error'],flush=True)
        except Exception as exc:
            row={'image':str(image.resolve()),'status':'failed','output':'','error':f'{type(exc).__name__}: {exc}','review_source':''}
            print('Failed: '+row['error'],flush=True)
        rows.append(row)
        with (run/'batch_summary.csv').open('w',newline='',encoding='utf-8-sig') as handle:
            writer=csv.DictWriter(handle,fieldnames=['image','status']+MEASUREMENT_FIELDS+REVIEW_FIELDS+['output','error','review_source'])
            writer.writeheader();writer.writerows(rows)
    failed=sum(row['status']!='completed' for row in rows)
    print(f'Finished: {len(rows)-failed} completed, {failed} failed. Results: {run.resolve()}',flush=True)
    print('Annotated images: '+str(review.resolve()),flush=True)
    insufficient=sum(row.get('review_status') in ('no_ok_measurements','below_target') for row in rows)
    print(f'Sampling review: {insufficient} images below the {target_measurements} OK-width target. Inspect overlays even when the target is reached.',flush=True)
    return 1 if failed else 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('image',help='one image or a folder of TIFFs (non-recursive)')
    p.add_argument('-o','--outdir',default='output/geometry')
    p.add_argument('--nearest',type=float,default=15)
    p.add_argument('--crop-bottom',type=int)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--target-measurements',type=int,default=100,help='flag images below this many OK widths; does not relax geometry checks (default: 100)')
    p.add_argument('--sample-step',type=float,default=4.,help='candidate spacing in analysis pixels (default: 4)')
    p.add_argument('--crossbar-spacing',type=float,default=6.,help='minimum drawn width-line spacing in analysis pixels (default: 6; increase for fewer lines)')
    args=p.parse_args()
    if args.threads<1 or not 0<args.nearest<100:p.error('Invalid threads or nearest percentage')
    if args.target_measurements<1:p.error('target-measurements must be positive')
    if not all(np.isfinite(v) and v>0 for v in (args.sample_step,args.crossbar_spacing)):
        p.error('sample-step and crossbar-spacing must be finite and positive')
    if args.crop_bottom is not None and args.crop_bottom<0:p.error('crop-bottom must be nonnegative')
    try:images=input_images(args.image)
    except ValueError as exc:p.error(str(exc))
    if not images:p.error('No TIFF files found in the input folder')
    import torch
    torch.set_num_threads(args.threads)
    print('Running Depth Anything V2 Large. No count or density analysis.',flush=True)
    # run_depth caches the model pipeline, so a folder loads Large only once.
    if Path(args.image).is_dir():
        return run_batch(images,args.outdir,args.nearest,args.crop_bottom,args.sample_step,args.crossbar_spacing,args.target_measurements)
    result=analyze_image(images[0],args.outdir,args.nearest,args.crop_bottom,args.sample_step,args.crossbar_spacing,args.target_measurements)
    copy_review_images(result,result/'quick_review',images[0].name)
    return 0


if __name__=='__main__':raise SystemExit(main())


