import csv,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from run_geometry import export_measurements
from run_density import density_metrics,trace_paths


class MeasurementExportTests(unittest.TestCase):
    def notes(self):
        sites=[]
        for i,w in enumerate([2.2,4.5,2.2]):
            sites.append({'axis':1,'edge1_rc':[i*10.,0.],'edge2_rc':[i*10.,w],
                          'width_analysis_px':w,'status':'accepted'})
        return {'source':'unused.tif','geometry':{'source_px_per_analysis_x':1.,'source_px_per_analysis_y':1.},
                'metadata_check':{'nm_per_source_pixel_xy':[1000.,1000.]},
                'sites':sites,'axes':[{'line_normal_angle':0.}]}

    def test_three_columns_and_only_outlier_flagged(self):
        with tempfile.TemporaryDirectory() as d:
            notes=self.notes();export_measurements(d,notes)
            with (Path(d)/'diameters.csv').open(encoding='utf-8-sig',newline='') as f:
                reader=csv.DictReader(f);rows=list(reader)
                self.assertEqual(reader.fieldnames,['fiber_id','diameter_um','review'])
            by_width={float(r['diameter_um']):r['review'] for r in rows}
            self.assertEqual(by_width,{2.2:'OK',4.5:'CHECK'})
            self.assertTrue((Path(d)/'diameters_audit.csv').exists())

    def test_scaling_and_rejected_numeric_value_retained(self):
        with tempfile.TemporaryDirectory() as d:
            notes=self.notes();notes['geometry']['source_px_per_analysis_x']=2.
            notes['sites'][1].update(status='rejected',reason='track_width_outlier')
            rows=export_measurements(d,notes)
            flagged=next(r for r in rows if r['sample_id']==2)
            self.assertAlmostEqual(flagged['diameter'],9.)
            self.assertEqual(flagged['review'],'CHECK')
            self.assertIn('track_width_outlier',flagged['reason'])

    def test_missing_calibration_uses_explicit_pixel_header(self):
        with tempfile.TemporaryDirectory() as d:
            notes=self.notes();notes['metadata_check']={}
            with patch('sem_scale.analyze_tiff',return_value={'pixel_size_nm':None}):export_measurements(d,notes)
            self.assertIn('diameter_source_px',(Path(d)/'diameters.csv').read_text(encoding='utf-8-sig').splitlines()[0])


class DensityTests(unittest.TestCase):
    def test_physical_density_and_missing_calibration(self):
        lo,hi=density_metrics(20,21,120.9675)
        self.assertAlmostEqual(lo,20/120.9675)
        self.assertAlmostEqual(hi,21/120.9675)
        self.assertEqual(density_metrics(20,21,None),(None,None))
        with self.assertRaises(ValueError):density_metrics(21,20,1.)

    def test_isolated_fibers_counted_once_and_blank_zero(self):
        image=np.zeros((200,200));image[:,40:55]=.8;image[:,130:145]=.8
        paths,_=trace_paths(image,.4,.55,50)
        self.assertEqual(len(paths),2)
        blank,_=trace_paths(np.zeros((200,200)),.4,.55,50)
        self.assertEqual(len(blank),0)

if __name__=='__main__':unittest.main()
