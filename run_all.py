"""Native-checked geometry, independent non-AI diameters, and visible-path counting/density."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import shutil
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import run_geometry as geometry
import run_density as density
import classical_diameter as classical

DENSITY_FIELDS=['automatic_path_count','automatic_count_low','automatic_count_high',
    'automatic_paths_per_um2','automatic_paths_per_um2_low','automatic_paths_per_um2_high',
    'automatic_paths_per_megapixel','analyzed_area_um2','foreground_coverage_pct',
    'foreground_coverage_low_pct','foreground_coverage_high_pct']
NON_AI_FIELDS=['non_ai_candidates','non_ai_ok','non_ai_check','method_agree','method_disagree','method_not_comparable']
FIELDS=['image','status','geometry_status','non_ai_status','density_status','density_review_status']+geometry.MEASUREMENT_FIELDS+geometry.REVIEW_FIELDS+DENSITY_FIELDS+NON_AI_FIELDS+[
    'geometry_output','non_ai_output','density_output','review_geometry','review_non_ai','review_segmentation','review_count','error']


def collect_inputs(inputs):
    files=[];seen=set()
    for value in inputs:
        path=Path(value)
        for image in geometry.input_images(path):
            if image.suffix.lower() not in ('.tif','.tiff'):
                raise ValueError('Expected an original TIFF: '+str(image))
            key=str(image.resolve()).casefold()
            if key not in seen:files.append(image.resolve());seen.add(key)
    if not files:raise ValueError('No TIFF files found; folders are scanned non-recursively')
    return files


def run_all(images,outdir='output/full_review',geometry_options=None,density_options=None):
    geometry_options=dict(geometry_options or {})
    density_options=dict(density_options or {})
    run=Path(outdir)/('batch_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    run.mkdir(parents=True,exist_ok=False)
    review=run/'quick_review';review.mkdir()
    versions={}
    for name in ['numpy','scipy','Pillow','scikit-image','networkx','torch','transformers']:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:pass
    source=Path(__file__).parent
    manifest={'pipeline':'native_geometry_classical_comparison_density_v2','model':'Depth Anything V2 Large',
        'revision':geometry.LARGE_REVISION,'python':sys.version,'platform':platform.platform(),
        'dependencies':versions,'inputs':[str(p.resolve()) for p in images],
        'geometry_options':geometry_options,'density_options':density_options,
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()
                         if p.is_file() and p.suffix in ('.py','.ps1')},
        'count_scope':'Estimated visible paths, not confirmed individual fibers. Sensitivity ranges are not confidence intervals.'}
    (run/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    rows=[]
    def save():
        density.write_csv(run/'batch_summary.csv',rows,FIELDS)
    for index,image in enumerate(images,1):
        print(f'[{index}/{len(images)}] {image.name}',flush=True)
        item=run/f'{index:03d}__{image.stem}';item.mkdir()
        row={'image':str(image.resolve()),'status':'running','geometry_status':'running',
             'non_ai_status':'pending','density_status':'pending','error':''}
        rows.append(row);save()
        errors=[]
        try:
            print('  Geometry and diameters (Large depth model)',flush=True)
            result=geometry.analyze_image(image,item/'geometry',**geometry_options)
            notes=json.loads((result/'preview_notes.json').read_text(encoding='utf-8'))
            row.update(geometry.measurement_counts(notes));row.update(notes['measurement_review'])
            row.update(geometry_status='completed',geometry_output=str(result.resolve()))
            target=review/f'{index:03d}__{image.name}__geometry.png'
            shutil.copy2(result/'source_geometry_preview.png',target)
            row['review_geometry']=str(target.resolve())
        except Exception as exc:
            if row['geometry_status']!='completed':row['geometry_status']='failed'
            errors.append(f'Geometry: {type(exc).__name__}: {exc}')
            print('  '+errors[-1],flush=True)
        row['non_ai_status']='running';row['error']='; '.join(errors);save()
        try:
            print('  Independent non-AI diameters and source segmentation',flush=True)
            result=classical.analyze_classical(image,item/'non_ai',crop_bottom=geometry_options.get('crop_bottom'),
                geometry_output=row.get('geometry_output'))
            report=json.loads((result/'non_ai_report.json').read_text(encoding='utf-8'))
            row.update({key:report.get(key) for key in NON_AI_FIELDS})
            row.update(non_ai_status='completed',non_ai_output=str(result.resolve()))
            for suffix,filename,key in [('non_ai','non_ai_overlay.png','review_non_ai'),
                                        ('segmentation','segmentation_overlay.png','review_segmentation')]:
                target=review/f'{index:03d}__{image.name}__{suffix}.png'
                shutil.copy2(result/filename,target);row[key]=str(target.resolve())
        except Exception as exc:
            if row['non_ai_status']!='completed':row['non_ai_status']='failed'
            errors.append(f'Non-AI: {type(exc).__name__}: {exc}')
            print('  '+errors[-1],flush=True)
        row['density_status']='running';row['error']='; '.join(errors);save()
        try:
            print('  Visible-path count and density sensitivity sweep',flush=True)
            result=density.analyze_density(image,item/'density',**density_options)
            report=json.loads((result/'density_report.json').read_text(encoding='utf-8'))
            row.update({key:report.get(key) for key in DENSITY_FIELDS})
            row.update(density_status='completed',density_review_status=report['status'],density_output=str(result.resolve()))
            target=review/f'{index:03d}__{image.name}__count.png'
            shutil.copy2(result/'count_overlay.png',target)
            row['review_count']=str(target.resolve())
            print(f"  Count: {report['automatic_path_count']} paths; sensitivity {report['automatic_count_low']}-{report['automatic_count_high']}; review required.",flush=True)
        except Exception as exc:
            if row['density_status']!='completed':row['density_status']='failed'
            errors.append(f'Density: {type(exc).__name__}: {exc}')
            print('  '+errors[-1],flush=True)
        row['error']='; '.join(errors)
        completed=sum(row[key]=='completed' for key in ('geometry_status','non_ai_status','density_status'))
        row['status']=('completed' if not errors else 'review_copy_failed') if completed==3 else ('partial' if completed else 'failed')
        (item/'image_summary.json').write_text(json.dumps(row,indent=2),encoding='utf-8')
        save()
    failures=sum(row['status']!='completed' for row in rows)
    below=sum(row.get('review_status') in ('no_ok_measurements','below_target') for row in rows)
    print(f'Finished: {len(rows)-failures} completed; {failures} partial/failed. {below} images below the diameter sampling target.',flush=True)
    print('Counts require identity review even when processing completed.',flush=True)
    print('Results: '+str(run.resolve()),flush=True)
    return 1 if failures else 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('inputs',nargs='*',default=['pictures/TIFF'],help='original TIFFs or folders (non-recursive)')
    p.add_argument('-o','--outdir',default='output/full_review')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--crop-bottom',type=int)
    p.add_argument('--nearest',type=float,default=15)
    p.add_argument('--sample-step',type=float,default=4.)
    p.add_argument('--crossbar-spacing',type=float,default=6.)
    p.add_argument('--target-measurements',type=int,default=100)
    p.add_argument('--pair-score',type=float,default=.55)
    p.add_argument('--pair-delta',type=float,default=.15)
    p.add_argument('--threshold-delta',type=float,default=.03)
    p.add_argument('--min-visible-fraction',type=float,default=.05)
    # Retain the previously supplied Large-model command; smaller models cannot silently substitute.
    p.add_argument('--depth-model',choices=['large'],default='large')
    p.add_argument('--depth-revision',choices=[geometry.LARGE_REVISION],default=geometry.LARGE_REVISION)
    args=p.parse_args()
    if args.threads<1 or args.target_measurements<1:p.error('threads and target-measurements must be positive')
    if not 0<args.nearest<100:p.error('nearest must be between 0 and 100')
    if not all(np.isfinite(v) and v>0 for v in (args.sample_step,args.crossbar_spacing)):p.error('spacing must be finite and positive')
    if args.crop_bottom is not None and args.crop_bottom<0:p.error('crop-bottom must be nonnegative')
    if not 0<=args.pair_score<=1 or not 0<args.min_visible_fraction<=1:p.error('invalid count score or visible fraction')
    if not all(np.isfinite(v) and v>=0 for v in (args.pair_delta,args.threshold_delta)):p.error('sensitivity deltas must be finite and nonnegative')
    try:images=collect_inputs(args.inputs)
    except ValueError as exc:p.error(str(exc))
    import torch
    torch.set_num_threads(args.threads)
    return run_all(images,args.outdir,
        dict(nearest=args.nearest,crop_bottom=args.crop_bottom,sample_step=args.sample_step,
             crossbar_spacing=args.crossbar_spacing,target_measurements=args.target_measurements),
        dict(crop_bottom=args.crop_bottom,pair_score=args.pair_score,pair_delta=args.pair_delta,
             threshold_delta=args.threshold_delta,min_visible_fraction=args.min_visible_fraction))


if __name__=='__main__':raise SystemExit(main())
