"""Native-pixel verification of proposed cross-sections; never accepts a rejected seed."""
from collections import Counter
import numpy as np
from PIL import Image,ImageDraw
from sem_image import load_gray
from source_geometry import SourceBoundaryCheck


def refine_sites(image,notes):
    gray=load_gray(image)
    x,y,w,h=notes['geometry']['roi_xywh'];gray=gray[y:y+h,x:x+w]
    sy=notes['geometry']['source_px_per_analysis_y'];sx=notes['geometry']['source_px_per_analysis_x']
    scale=np.array([sy,sx]);outcomes=Counter()
    checker=SourceBoundaryCheck(gray,diagnostics=False,interpolate=True,sample_scale=scale)
    for site in notes['sites']:
        if site['status']!='accepted':continue
        a,b=np.array(site['edge1_rc']),np.array(site['edge2_rc'])
        e1=a;e2=b
        normal=(e2-e1)/np.linalg.norm(e2-e1)
        tangent=np.array([-normal[1],normal[0]])
        proposal={'edge1_rc':e1,'edge2_rc':e2,'diameter_px':float(np.linalg.norm(e2-e1))}
        measured,reason=checker.check(proposal,tangent,keep_rejected=True)
        site['preview_edge1_rc']=a.tolist();site['preview_edge2_rc']=b.tolist()
        site['preview_width_analysis_px']=site['width_analysis_px']
        if measured is not None:
            a,b=np.array(measured['edge1_rc']),np.array(measured['edge2_rc'])
            width=measured['diameter_px']
            # A large source/preview disagreement must be reviewed, not silently corrected.
            if not reason and abs(width-proposal['diameter_px'])>max(2.,.20*min(width,proposal['diameter_px'])):
                reason='preview_width_disagreement'
            site.update(edge1_rc=a.tolist(),edge2_rc=b.tolist(),midpoint_rc=((a+b)/2).tolist(),
                        width_analysis_px=float(np.linalg.norm(b-a)),native_width_source_px=float(np.linalg.norm((b-a)*scale)))
        if reason:
            site['status']='rejected';site['reason']='native_'+reason
            notes['rejection_counts'][site['reason']]=notes['rejection_counts'].get(site['reason'],0)+1
            outcomes['rejected']+=1
        else:outcomes['verified']+=1
    notes['native_verification']={'method':'source_gradient_edges_at_original_resolution',
        'interpolated_gradient_peak':True,'smoothing_sigma_source_yx':scale.tolist(),'verified':outcomes['verified'],'rejected':outcomes['rejected'],
        'scope':'Only previously accepted proposals are rechecked; no rejected candidate is promoted.'}
    notes['geometry_method']='source_checked_v4_native'
    return notes


def render_final(gray,depth,notes,rows,crossbar_spacing):
    """Draw exactly the final OK population, with gaps across excluded sections."""
    im=Image.fromarray(np.uint8(np.clip((depth-depth.min())/np.ptp(depth),0,1)*255)).convert('RGB')
    source=Image.fromarray(np.uint8(np.clip(gray,0,1)*255)).convert('RGB')
    painters=[ImageDraw.Draw(im),ImageDraw.Draw(source)]
    accepted={r['sample_id'] for r in rows if r['review']=='OK'}
    for index,axis in enumerate(notes['axes'],1):
        samples=[(j,s) for j,s in enumerate(notes['sites'],1) if s['axis']==index and j in accepted]
        t=axis['line_normal_angle'];tangent=np.array([-np.cos(t),np.sin(t)])
        samples.sort(key=lambda js:float(np.dot(js[1]['midpoint_rc'],tangent)))
        previous=None;last_bar=None;bars=0
        for _,site in samples:
            mid=np.asarray(site['midpoint_rc']);a=np.array(site['edge1_rc']);b=np.array(site['edge2_rc'])
            if previous is not None and np.linalg.norm(mid-previous)<=1.8*notes['sample_step_analysis_px']:
                for painter in painters:painter.line([tuple(previous[::-1]),tuple(mid[::-1])],fill=(40,175,255),width=2)
            previous=mid
            if last_bar is None or np.linalg.norm(mid-last_bar)>=crossbar_spacing:
                for painter in painters:painter.line([tuple(a[::-1]),tuple(b[::-1])],fill=(255,50,50),width=2)
                last_bar=mid;bars+=1
        if samples:
            mid=samples[len(samples)//2][1]['midpoint_rc']
            for painter in painters:painter.text((mid[1]+6,mid[0]+4),f'F{index}',fill=(255,230,40),stroke_width=1,stroke_fill=(0,0,0))
        axis['preview_crossbars']=bars;axis['accepted_sites']=len(samples)
    return im,source
