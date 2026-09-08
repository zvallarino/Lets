"""Large-model visual geometry review. No fiber counts or density calculations.

Crossbars show apparent boundaries in predicted depth, not validated physical
diameters. Inspect the matching source overlay. Straight prominent fibers only;
curved, hidden, weakly resolved, or ambiguous crossings are deliberately omitted.
"""
import argparse,json,csv
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from skimage.transform import hough_line,hough_line_peaks
from fiber_diameter import measure_at
from sem_image import prepare
from sem_metadata import check_sidecar
from top_fiber_depth import run_depth

LARGE_REVISION='7581137eff8d4e94f6e796d3baea0e9fa79b22d2'

def geometry_preview(gray,depth,nearest_percent=15):
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
    records=[]
    for a,t,rh in zip(acc,theta,rho):
        normal=np.array([np.sin(t),np.cos(t)])
        tangent=np.array([-normal[1],normal[0]])
        origin=rh*normal
        positions=[];crossbars=[]
        for s in np.arange(-2*max(H,W),2*max(H,W),12.):
            p=origin+s*tangent;r,c=np.round(p).astype(int)
            if not (8<=r<H-8 and 8<=c<W-8) or not mask[r,c]:
                positions.append(None);continue
            # Do not measure merged silhouettes around another candidate axis.
            near_crossing=any(abs(float(np.dot(p,np.array([np.sin(t2),np.cos(t2)])))-rh2)<max(65.,2*dist[r,c])
                              for t2,rh2 in zip(theta,rho) if abs(t2-t)>.05)
            if near_crossing:
                positions.append(None);continue
            m=measure_at(sm,mask,r,c,tangent,max(float(dist[r,c]),10))
            if m is None:
                positions.append(None);continue
            e1,e2=np.array(m['edge1_rc']),np.array(m['edge2_rc'])
            mid=(e1+e2)/2
            if np.linalg.norm(mid-p)>35:
                positions.append(None);continue
            positions.append(mid)
            if len(crossbars)==0 or np.linalg.norm(mid-crossbars[-1][0])>95:
                crossbars.append((mid,e1,e2))
        valid=[p for p in positions if p is not None]
        if len(valid)<12:continue
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
        records.append({'line_normal_angle':float(t),'line_offset_px':float(rh),'hough_support':int(a),'preview_crossbars':len(crossbars)})
    return im,source,{'purpose':'Visual review only; depth-derived crossbars are not validated physical diameters',
                      'depth_cutoff':cutoff,'nearest_percent':nearest_percent,'axes':records}

def input_images(path):
    path=Path(path)
    if path.is_file():return [path]
    if path.is_dir():
        return sorted((f for f in path.iterdir() if f.is_file() and f.suffix.lower() in ('.tif','.tiff')),key=lambda f:f.name.lower())
    raise ValueError('Input does not exist: '+str(path))


def analyze_image(image,outdir,nearest=15,crop_bottom=None):
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
    preview,source,notes=geometry_preview(gray,depth,nearest)
    from datetime import datetime
    out=Path(outdir)/(Path(image).stem+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True,exist_ok=False)
    preview.save(out/'geometry_preview.png');source.save(out/'source_geometry_preview.png')
    np.save(out/'large_depth.npy',depth)
    notes.update({'model':'Depth Anything V2 Large','revision':LARGE_REVISION,'source':str(Path(image).resolve()),'geometry':transform,'metadata_check':metadata_check})
    (out/'preview_notes.json').write_text(json.dumps(notes,indent=2),encoding='utf-8')
    print('Preview saved: '+str(out.resolve()),flush=True)
    return out


def run_batch(images,outdir,nearest=15,crop_bottom=None):
    from datetime import datetime
    run=Path(outdir)/('batch_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    run.mkdir(parents=True,exist_ok=False)
    rows=[]
    for index,image in enumerate(images,1):
        print(f'[{index}/{len(images)}] {image.name}',flush=True)
        try:
            result=analyze_image(image,run,nearest,crop_bottom)
            row={'image':str(image.resolve()),'status':'completed','output':str(result.resolve()),'error':''}
        except Exception as exc:
            row={'image':str(image.resolve()),'status':'failed','output':'','error':f'{type(exc).__name__}: {exc}'}
            print('Failed: '+row['error'],flush=True)
        rows.append(row)
        with (run/'batch_summary.csv').open('w',newline='',encoding='utf-8-sig') as handle:
            writer=csv.DictWriter(handle,fieldnames=['image','status','output','error'])
            writer.writeheader();writer.writerows(rows)
    failed=sum(row['status']=='failed' for row in rows)
    print(f'Finished: {len(rows)-failed} completed, {failed} failed. Results: {run.resolve()}',flush=True)
    return 1 if failed else 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('image',help='one image or a folder of TIFFs (non-recursive)')
    p.add_argument('-o','--outdir',default='output/geometry')
    p.add_argument('--nearest',type=float,default=15)
    p.add_argument('--crop-bottom',type=int)
    p.add_argument('--threads',type=int,default=4)
    args=p.parse_args()
    if args.threads<1 or not 0<args.nearest<100:p.error('Invalid threads or nearest percentage')
    if args.crop_bottom is not None and args.crop_bottom<0:p.error('crop-bottom must be nonnegative')
    try:images=input_images(args.image)
    except ValueError as exc:p.error(str(exc))
    if not images:p.error('No TIFF files found in the input folder')
    import torch
    torch.set_num_threads(args.threads)
    print('Running Depth Anything V2 Large. No count or density analysis.',flush=True)
    # run_depth caches the model pipeline, so a folder loads Large only once.
    if Path(args.image).is_dir():
        return run_batch(images,args.outdir,args.nearest,args.crop_bottom)
    analyze_image(images[0],args.outdir,args.nearest,args.crop_bottom)
    return 0


if __name__=='__main__':raise SystemExit(main())
