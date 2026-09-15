"""Visible-path density analysis; also called per image by the full batch runner.

Automatic ranges describe sensitivity to segmentation and continuation settings,
not confidence bounds on the true fiber count. A manual reference is independent.
"""
import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy import ndimage as ndi
import fiber_analysis as fa
from sem_image import prepare,segment
from sem_metadata import check_sidecar
from sem_scale import analyze_tiff


def write_csv(path,rows,fields):
    with Path(path).open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)


def density_metrics(count_low,count_high,area_um2):
    if count_low<0 or count_high<count_low:raise ValueError('Invalid count range')
    if area_um2 is None:return None,None
    if not np.isfinite(area_um2) or area_um2<=0:raise ValueError('Invalid physical area')
    return count_low/area_um2,count_high/area_um2


def trace_paths(gray,threshold,pair_score,min_length):
    mask=segment(gray,threshold=threshold)
    dist=ndi.distance_transform_edt(mask)
    _,centers,branches=fa.skeleton_branches(mask)
    zones,zone_of,connectors=fa.build_zones(branches,centers,dist)
    fibers=fa.stitch_fibers(branches,zones,zone_of,connectors,pair_score_min=pair_score)
    paths=[]
    for f in fibers:
        points=f['pts']
        length=float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum())
        if length>=min_length:paths.append({'points':points,'length':length})
    paths.sort(key=lambda f:(float(f['points'][:,0].mean()),float(f['points'][:,1].mean())))
    return paths,mask


def overlay(gray,paths,path):
    im=Image.fromarray(np.uint8(np.clip(gray,0,1)*255)).convert('RGB')
    draw=ImageDraw.Draw(im)
    for i,f in enumerate(paths,1):
        points=f['points'];color=(255,200,40)
        if len(points)>1:draw.line([(float(p[1]),float(p[0])) for p in points],fill=color,width=2)
        middle=points[len(points)//2]
        draw.text((float(middle[1])+3,float(middle[0])+3),f'P{i}',fill=(255,255,255),stroke_width=2,stroke_fill=(0,0,0))
    im.save(path)


def analyze_density(image,outdir='output/density',max_dim=1600,crop_bottom=None,
                    pair_score=.55,min_visible_fraction=.05,threshold_delta=.03,
                    pair_delta=.15,nm_per_px=None,reference_count=None):
    image=Path(image)
    if not image.is_file():raise ValueError('Supply one original image file, not a folder')
    if max_dim<128 or not 0<=pair_score<=1 or not 0<min_visible_fraction<=1:
        raise ValueError('Invalid resolution, continuation score, or visible length fraction')
    if not all(np.isfinite(v) and v>=0 for v in (threshold_delta,pair_delta)):
        raise ValueError('Sensitivity deltas must be finite and nonnegative')
    if nm_per_px is not None and (not np.isfinite(nm_per_px) or nm_per_px<=0):
        raise ValueError('nm-per-px must be positive')
    if reference_count is not None:
        if len(reference_count)!=2 or any(v<0 for v in reference_count) or reference_count[1]<reference_count[0]:
            raise ValueError('Reference count must be a nonnegative ordered pair')
    metadata=check_sidecar(image)
    if metadata['status']!='not_available':
        expected=metadata['crop_bottom_px']
        if crop_bottom is not None and crop_bottom!=expected:raise ValueError('Crop conflicts with metadata')
        crop_bottom=expected
    gray,transform=prepare(image,max_dim,crop_bottom)
    nm_xy=([nm_per_px,nm_per_px] if nm_per_px is not None else metadata.get('nm_per_source_pixel_xy'))
    calibration='explicit source-pixel calibration' if nm_per_px is not None else 'JEOL sidecar consistency check'
    if nm_xy is None:
        nominal=analyze_tiff(image)['pixel_size_nm']
        nm_xy=[nominal,nominal] if nominal is not None else None
        calibration='nominal JEOL magnification/reference calibration' if nm_xy else 'unavailable'
    roi=transform['roi_xywh']
    area=roi[2]*roi[3]*nm_xy[0]*nm_xy[1]/1e6 if nm_xy else None
    _,seginfo=segment(gray,return_info=True)
    threshold=seginfo['threshold']
    min_length=min_visible_fraction*float(np.hypot(*gray.shape))
    # The central setting is listed first; count minima/maxima need not be
    # monotonic in thresholds or pairing score. Each setting gets its own IDs.
    thresholds=list(dict.fromkeys([threshold,max(0.,threshold-threshold_delta),min(1.,threshold+threshold_delta)]))
    scores=list(dict.fromkeys([pair_score,max(0.,pair_score-pair_delta),min(1.,pair_score+pair_delta)]))
    out=Path(outdir)/(image.stem+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True,exist_ok=False)
    settings=[];all_paths=[];candidates=[]
    for t in thresholds:
        for score in scores:
            paths,mask=trace_paths(gray,t,score,min_length)
            sid=f'S{len(settings)+1}'
            settings.append({'setting_id':sid,'threshold':t,'pair_score_min':score,
                'min_visible_length_analysis_px':min_length,'path_count':len(paths),
                'paths_per_um2':len(paths)/area if area else '',
                'foreground_coverage_pct':float(mask.mean()*100)})
            all_paths.append(paths)
            for i,f in enumerate(paths,1):
                pts=f['points'];mid=pts[len(pts)//2]
                candidates.append({'setting_id':sid,'path_id':f'P{i}','visible_length_analysis_px':f['length'],
                    'label_x_analysis_px':float(mid[1]),'label_y_analysis_px':float(mid[0]),
                    'touches_border':bool(((pts[:,0]<3)|(pts[:,0]>gray.shape[0]-4)|(pts[:,1]<3)|(pts[:,1]>gray.shape[1]-4)).any()),
                    'researcher_fiber_id':'','researcher_decision':'','notes':''})
            print(f"{sid}: {len(paths)} visible paths; threshold {t:.3f}, continuation {score:.2f}",flush=True)
    counts=[r['path_count'] for r in settings];lo,hi=min(counts),max(counts)
    dlo,dhi=density_metrics(lo,hi,area)
    central=settings[0]
    overlay(gray,all_paths[0],out/'count_overlay.png')
    for tag,index in [('low',int(np.argmin(counts))),('high',int(np.argmax(counts)))]:
        overlay(gray,all_paths[index],out/f'count_{tag}_overlay.png')
    report={'source':str(image.resolve()),'source_sha256':hashlib.sha256(image.read_bytes()).hexdigest(),
        'status':'researcher_review_required','method':'skeleton_paths_sensitivity_pilot_v1',
        'count_definition':'Estimated visible centerline paths after crossing stitching and minimum visible-length filtering. Includes retained border paths; may split or merge individual fibers.',
        'range_definition':'Minimum and maximum across listed settings; NOT a confidence interval or bound on true fiber count.',
        'geometry':transform,'metadata_check':metadata,'calibration':calibration,'nm_per_source_pixel_xy':nm_xy,
        'analyzed_area_um2':area,'automatic_path_count':central['path_count'],
        'automatic_paths_per_um2':central['path_count']/area if area else None,
        'automatic_paths_per_megapixel':central['path_count']/(roi[2]*roi[3])*1e6,
        'automatic_count_low':lo,'automatic_count_high':hi,'automatic_paths_per_um2_low':dlo,'automatic_paths_per_um2_high':dhi,
        'foreground_coverage_pct':central['foreground_coverage_pct'],
        'foreground_coverage_low_pct':min(s['foreground_coverage_pct'] for s in settings),
        'foreground_coverage_high_pct':max(s['foreground_coverage_pct'] for s in settings),
        'coverage_definition':'Threshold-dependent projected foreground area fraction; not 3D density or porosity.',
        'reference_count_low':reference_count[0] if reference_count else None,
        'reference_count_high':reference_count[1] if reference_count else None,
        'reference_definition':'Researcher-supplied count, not used to tune or compute automatic counts.',
        'settings':settings,'overlay_settings':{'count_overlay.png':'S1','count_low_overlay.png':settings[int(np.argmin(counts))]['setting_id'],'count_high_overlay.png':settings[int(np.argmax(counts))]['setting_id']}}
    if reference_count:
        reflo,refhi=density_metrics(*reference_count,area)
        report.update(reference_fibers_per_um2_low=reflo,reference_fibers_per_um2_high=refhi,
            automatic_range_overlaps_reference=bool(lo<=reference_count[1] and hi>=reference_count[0]))
    write_csv(out/'count_sensitivity.csv',settings,list(settings[0]))
    write_csv(out/'count_candidates.csv',candidates,['setting_id','path_id','visible_length_analysis_px','label_x_analysis_px','label_y_analysis_px','touches_border','researcher_fiber_id','researcher_decision','notes'])
    summary={k:v for k,v in report.items() if isinstance(v,(str,int,float,bool)) or v is None}
    write_csv(out/'density_summary.csv',[summary],list(summary))
    (out/'density_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (out/'READ_ME.txt').write_text(
        f"Automatic path range: {lo}-{hi}; central setting: {central['path_count']}.\n"
        f"Analyzed area: {area} um^2. Calibration: {calibration}.\n"
        f"Automatic path density: {dlo}-{dhi} paths/um^2.\n"
        f"Researcher reference count: {reference_count}. See separate reference density fields.\n"
        "Compare count_overlay.png against the source and your manual labels. P IDs are paths, not confirmed fibers.\n"
        "The candidate CSV has blank researcher IDs/decisions for reviewing duplicate or missed fibers.\n"
        "Changing tolerance can alter identities even when totals are similar. No statistical confidence interval is claimed.\n",
        encoding='utf-8')
    print('Density pilot saved: '+str(out.resolve()),flush=True)
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('image',help='one original TIFF; single-image pilot only')
    p.add_argument('-o','--outdir',default='output/density')
    p.add_argument('--pair-score',type=float,default=.55)
    p.add_argument('--pair-delta',type=float,default=.15)
    p.add_argument('--threshold-delta',type=float,default=.03)
    p.add_argument('--min-visible-fraction',type=float,default=.05,help='minimum path length / analysis-image diagonal')
    p.add_argument('--max-dim',type=int,default=1600)
    p.add_argument('--crop-bottom',type=int)
    p.add_argument('--nm-per-px',type=float)
    p.add_argument('--reference-count',type=int,nargs=2,metavar=('LOW','HIGH'))
    args=p.parse_args()
    try:analyze_density(**vars(args))
    except (ValueError,OSError) as exc:p.error(str(exc))

if __name__=='__main__':main()
