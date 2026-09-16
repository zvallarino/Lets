import csv,json,tempfile,unittest
from pathlib import Path
import numpy as np
from PIL import Image
from classical_diameter import MaskDiameter,scaled_width,compare_geometry
from native_geometry import refine_sites,render_final

class ClassicalTests(unittest.TestCase):
    def test_known_widths_including_odd_pixels(self):
        for width in [10,15,20,31,50]:
            gray=np.full((240,200),.1);gray[:,70:70+width]=.8
            rows=MaskDiameter(gray).sample();good=[r['width_analysis_px'] for r in rows if r['review']=='OK']
            self.assertGreater(len(good),5)
            self.assertAlmostEqual(float(np.median(good)),width,delta=.25)
    def test_rotated_widths(self):
        rr,cc=np.indices((300,300))
        for angle in [17,35,63]:
            theta=np.deg2rad(angle);normal=np.array([np.sin(theta),np.cos(theta)])
            gray=np.where(abs((rr-150)*normal[0]+(cc-150)*normal[1])<15.5,.8,.1)
            measured,reason=MaskDiameter(gray).measure([150.,150.])
            self.assertIsNone(reason)
            self.assertAlmostEqual(measured['width_analysis_px'],31,delta=1.)
    def test_blank_thin_and_low_contrast_rejected(self):
        self.assertEqual(MaskDiameter(np.zeros((128,128))).sample(),[])
        for width,intensity in [(6,.8),(25,.11)]:
            gray=np.full((240,200),.1);gray[:,70:70+width]=intensity
            self.assertFalse(any(r['review']=='OK' for r in MaskDiameter(gray).sample()))
    def test_adjacent_and_crossing_fibers(self):
        gray=np.full((300,250),.1);gray[:,80:105]=.8;gray[:,112:137]=.8
        check=MaskDiameter(gray)
        for c in [92.,124.]:
            m,reason=check.measure([150.,c]);self.assertIsNone(reason);self.assertAlmostEqual(m['width_analysis_px'],25,delta=1.)
        self.assertIsNotNone(check.measure([150.,108.])[1])
        gray=np.full((300,250),.1);gray[:,100:130]=.8;gray[135:165,:]=.8
        check=MaskDiameter(gray)
        self.assertIsNotNone(check.measure([150.,115.])[1])
        self.assertIsNone(check.measure([60.,115.])[1])
    def test_rims_and_internal_seam(self):
        gray=np.full((300,250),.1);gray[:,90:150]=.45;gray[:,90:94]=.85;gray[:,146:150]=.85;gray[:,117:123]=.85
        m,reason=MaskDiameter(gray).measure([150.,120.])
        self.assertIsNotNone(reason)
        gray=np.full((300,250),.1);gray[:,100:130]=.4;gray[:,100:104]=.9;gray[:,126:130]=.9
        m,reason=MaskDiameter(gray).measure([150.,115.]);self.assertIsNone(reason)
        self.assertAlmostEqual(m['width_analysis_px'],30,delta=2.)
    def test_comparison_distinguishes_disagreement_and_missing_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);geom=root/'geometry';geom.mkdir()
            gray=np.full((300,200),.1);gray[:,70:100]=.8
            transform={'roi_xywh':[0,0,200,300],'source_px_per_analysis_x':1.,'source_px_per_analysis_y':1.}
            notes={'geometry':transform,'sites':[{'edge1_rc':[150,69.5],'edge2_rc':[150,99.5]},
                {'edge1_rc':[160,71.5],'edge2_rc':[160,97.5]}, {'edge1_rc':[170,20.],'edge2_rc':[170,30.]}]}
            (geom/'preview_notes.json').write_text(json.dumps(notes))
            (geom/'diameters_audit.csv').write_text('sample_id,review\n1,OK\n2,OK\n3,OK\n')
            result=compare_geometry(MaskDiameter(gray),transform,geom,root)
            self.assertEqual(result,{'method_agree':1,'method_disagree':1,'method_not_comparable':1})
            with self.assertRaises(ValueError):compare_geometry(MaskDiameter(gray),transform,geom,root,source=root/'wrong.tif')
            with (root/'cross_checked_measurements.csv').open(encoding='utf-8-sig') as handle:
                self.assertEqual(len(list(csv.DictReader(handle))),1)

    def test_scaling_uses_both_axes(self):
        transform={'source_px_per_analysis_x':2.,'source_px_per_analysis_y':3.}
        self.assertAlmostEqual(scaled_width([0,0],[4,3],transform,[1000.,2000.]),np.hypot(24,6))

class NativeTests(unittest.TestCase):
    def test_refine_original_pixels_and_never_promote_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            gray=np.full((300,300),25,np.uint8);gray[:,100:131]=204
            image=Path(tmp)/'fiber.tif';Image.fromarray(gray).save(image)
            a=np.array([150.,101.]);b=np.array([150.,129.])
            site={'axis':1,'edge1_rc':((a+.5)/3-.5).tolist(),'edge2_rc':((b+.5)/3-.5).tolist(),
                  'width_analysis_px':28/3,'status':'accepted'}
            rejected=dict(site,status='rejected',reason='prior_rejection')
            notes={'geometry':{'roi_xywh':[0,0,300,300],'source_px_per_analysis_x':3,'source_px_per_analysis_y':3},
                   'sites':[site,rejected],'rejection_counts':{}}
            refine_sites(image,notes)
            self.assertEqual(site['status'],'accepted')
            self.assertAlmostEqual(site['native_width_source_px'],31,delta=.3)
            self.assertEqual(rejected['status'],'rejected')
            self.assertNotIn('native_width_source_px',rejected)
    def test_overlay_draws_final_ok_only(self):
        gray=np.zeros((100,100));depth=gray.copy();depth[:,40:60]=1
        sites=[dict(axis=1,midpoint_rc=[r,50],edge1_rc=[r,40],edge2_rc=[r,60]) for r in (30,34,38)]
        notes={'sites':sites,'axes':[{'line_normal_angle':0}],'sample_step_analysis_px':4}
        rows=[{'sample_id':1,'review':'OK'},{'sample_id':2,'review':'CHECK'},{'sample_id':3,'review':'OK'}]
        _,image=render_final(gray,depth,notes,rows,6)
        self.assertEqual(notes['axes'][0]['preview_crossbars'],2)
        self.assertEqual(image.getpixel((45,34)),(0,0,0))
        self.assertEqual(image.getpixel((50,34)),(0,0,0))

if __name__=='__main__':unittest.main()
