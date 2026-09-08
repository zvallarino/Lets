"""Compare manual reference readings with accepted automated diameters.

python compare_reference.py output/review/run_... manual_reference.csv -o comparison
Reference columns: image_sha256,sample_id,analyst,session,manual_diameter_nm
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from run_pipeline import write_csv


def compare(run,reference,outdir,tolerance_pct=10):
    if not np.isfinite(tolerance_pct) or tolerance_pct<=0:
        raise ValueError('Tolerance must be positive and finite')
    automated={}
    for report in Path(run).glob('*/report.json'):
        meta=json.loads(report.read_text(encoding='utf-8'))
        with (report.parent/'diameters.csv').open(encoding='utf-8-sig',newline='') as f:
            for row in csv.DictReader(f):
                if row['status']=='accepted' and row['diameter_nm']:
                    automated[(meta['input_sha256'],row['sample_id'])]=float(row['diameter_nm'])
    paired=[];seen=set()
    with Path(reference).open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        required={'image_sha256','sample_id','analyst','session','manual_diameter_nm'}
        if not required.issubset(reader.fieldnames or []):raise ValueError('Missing reference columns')
        for line,row in enumerate(reader,2):
            key=(row['image_sha256'],row['sample_id'])
            identity=(*key,row['analyst'].strip(),row['session'].strip())
            if identity in seen:raise ValueError(f'Duplicate reference reading at line {line}')
            seen.add(identity)
            if not identity[-1] or not identity[-2]:raise ValueError(f'Missing analyst/session at line {line}')
            if key not in automated:raise ValueError(f'No accepted calibrated automated match at line {line}')
            manual=float(row['manual_diameter_nm'])
            if not np.isfinite(manual) or manual<=0:raise ValueError(f'Invalid manual diameter at line {line}')
            auto=automated[key];error=auto-manual;relative=100*error/manual
            paired.append({**row,'automated_diameter_nm':auto,'error_nm':error,'relative_error_pct':relative,
                           'within_tolerance':abs(relative)<=tolerance_pct})
    if not paired:raise ValueError('No reference readings supplied')
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'paired.csv',paired,list(paired[0]))
    groups=[]
    for analyst in sorted({r['analyst'] for r in paired}):
        rows=[r for r in paired if r['analyst']==analyst]
        errors=np.array([r['error_nm'] for r in rows])
        groups.append({'analyst':analyst,'readings':len(rows),'images':len({r['image_sha256'] for r in rows}),
                       'mean_bias_nm':float(errors.mean()),'mean_absolute_error_nm':float(np.abs(errors).mean()),
                       'within_tolerance_pct':100*sum(r['within_tolerance'] for r in rows)/len(rows)})
    summary={'tolerance_pct':tolerance_pct,'paired_readings':len(paired),
             'unmatched_automated_sites':len(set(automated)-{(r['image_sha256'],r['sample_id']) for r in paired}),
             'by_analyst':groups,
             'limitation':'Descriptive paired-site comparison only. Repeated readings are correlated; this does not establish 99% image reliability or detection accuracy.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run');p.add_argument('reference');p.add_argument('-o','--outdir',default='comparison')
    p.add_argument('--tolerance-pct',type=float,default=10)
    a=p.parse_args();print(json.dumps(compare(a.run,a.reference,a.outdir,a.tolerance_pct),indent=2))
