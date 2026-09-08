"""Read JEOL text sidecars as data and cross-check against their TIFFs."""
import hashlib,math,re
from pathlib import Path
import tifffile


def read_sidecar(path):
    raw=Path(path).read_bytes()
    try:text=raw.decode('utf-8-sig');encoding='utf-8'
    except UnicodeDecodeError:text=raw.decode('cp932');encoding='cp932'
    fields={}
    for line in text.splitlines():
        if line.startswith('$'):
            parts=line[1:].split(maxsplit=1)
            if parts[0] in fields:raise ValueError('Duplicate metadata field: '+parts[0])
            fields[parts[0]]=parts[1].strip() if len(parts)>1 else ''
    return fields,encoding,hashlib.sha256(raw).hexdigest()


def quantities_um(text):
    pattern=r'([0-9]+(?:\.[0-9]+)?)\s*(nm|[µμu]m|mm)'
    values=[float(n)*{'nm':.001,'um':1,'µm':1,'μm':1,'mm':1000}[unit]
            for n,unit in re.findall(pattern,text)]
    if not values or any(not math.isfinite(v) or v<=0 for v in values):
        raise ValueError('Missing or invalid physical units: '+text)
    return values


def check_sidecar(image):
    image=Path(image)
    sidecars=[p for p in image.parent.iterdir() if p.is_file() and p.suffix.lower()=='.txt' and p.stem.lower()==image.stem.lower()]
    if not sidecars:return {'status':'not_available','warnings':['No matching text sidecar; automatic crop will be used.']}
    if len(sidecars)!=1:raise ValueError('Multiple matching text sidecars')
    path=sidecars[0];m,encoding,sha=read_sidecar(path)
    required=['CM_IMAGE_SIZE','CM_LABEL','CM_FIELD_OF_VIEW','CM_MAG','SM_PNU_HEIGHT','SM_MICRON_MARKER','SM_MICRON_BAR']
    missing=[key for key in required if not m.get(key)]
    if missing:raise ValueError('Missing metadata fields: '+', '.join(missing))
    dimensions=[int(v) for v in m['CM_IMAGE_SIZE'].split()]
    if len(dimensions)!=2 or min(dimensions)<=0:raise ValueError('Invalid CM_IMAGE_SIZE')
    w,h=dimensions;banner=int(m['SM_PNU_HEIGHT'])
    mag=float(m['CM_MAG']);bar=float(m['SM_MICRON_BAR'])
    if banner<0 or not all(math.isfinite(v) and v>0 for v in (mag,bar)):
        raise ValueError('Invalid magnification, banner, or bar length')
    fov=quantities_um(m['CM_FIELD_OF_VIEW']);marker=quantities_um(m['SM_MICRON_MARKER'])
    if len(fov)!=2 or len(marker)!=1:raise ValueError('Unexpected field-of-view or marker dimensions')
    with tifffile.TiffFile(image) as tf:
        page=tf.pages[0];actual=[page.imagewidth,page.imagelength]
        def value(name):
            tag=page.tags.get(name)
            return str(tag.value).strip('\x00 ') if tag else None
        label=value('ImageDescription');instrument=value('Model')
    if actual==[w,h+banner]:crop=banner
    elif actual==[w,h]:crop=0
    else:raise ValueError(f'TIFF dimensions {actual} conflict with text image {dimensions} and banner {banner}')
    warnings=[]
    if label and label!=m['CM_LABEL']:raise ValueError('TIFF embedded label differs from sidecar label')
    if instrument and m.get('CM_INSTRUMENT') and instrument!=m['CM_INSTRUMENT']:
        raise ValueError('TIFF instrument differs from sidecar instrument')
    file_label=re.sub(r'_\d+$','',image.stem)
    if file_label!=m['CM_LABEL']:
        warnings.append(f'Filename label {file_label} differs from TIFF/text label {m["CM_LABEL"]}; filename preserved.')
    nm_x=fov[0]*1000/w;nm_y=fov[1]*1000/h;nm_bar=marker[0]*1000/bar
    if not math.isclose(nm_x,nm_y,rel_tol=.005) or not math.isclose(nm_x,nm_bar,rel_tol=.005):
        raise ValueError('Field-of-view and scale-bar pixel calibrations disagree by more than 0.5%')
    standard=m.get('SM_STANDARD_FIELD_SIZE')
    if standard:
        reference=[float(x) for x in standard.split()]
        if len(reference)!=2 or any(not math.isclose(reference[i]*1000/mag,fov[i],rel_tol=.005) for i in range(2)):
            raise ValueError('Magnification/reference field conflicts with field of view')
    return {'status':'warning' if warnings else 'passed','sidecar':str(path.resolve()),'sidecar_sha256':sha,
            'encoding':encoding,'warnings':warnings,'filename_label':file_label,'metadata_label':m['CM_LABEL'],
            'embedded_label':label,'image_size_xy':dimensions,'tiff_size_xy':actual,'crop_bottom_px':crop,
            'magnification':mag,'field_of_view_um_xy':fov,'nm_per_source_pixel_xy':[nm_x,nm_y],
            'nm_per_pixel_from_bar':nm_bar,'scale_bar_pixels':bar,'metadata':m,
            'note':'Consistency check of acquisition metadata; not independent instrument calibration.'}
