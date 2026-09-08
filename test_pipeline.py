import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import fiber_analysis as fa
import fiber_diameter as fd
from sem_image import prepare,load_gray,segment,detect_databar_top,crop_edges
from sem_scale import pixel_size_nm_from_mag
from run_pipeline import Config,analyze_one,original_points


class RegressionTests(unittest.TestCase):
    def test_scale_rejects_invalid(self):
        for v in [0,-1,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):pixel_size_nm_from_mag(v,5120)
        self.assertAlmostEqual(pixel_size_nm_from_mag(10000,5120),2.48046875)

    def test_sixteen_bit_not_clipped(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'16.tif'
            a=np.tile(np.linspace(1000,50000,128,dtype=np.uint16),(64,1))
            Image.fromarray(a).save(path)
            gray=load_gray(path)
            self.assertLess(gray[:,20].mean(),.3)
            self.assertGreater(gray[:,-20].mean(),.7)

    def test_dark_specimen_row_not_banner(self):
        a=np.full((400,500),.4);a[270:310]=.1
        self.assertEqual(detect_databar_top(a),400)
        a[375:]=0;a[385:388,100:150]=1
        self.assertEqual(detect_databar_top(a),375)

    def test_crop_validation(self):
        for kw in [{'bottom':500},{'top':-1}]:
            with self.assertRaises(ValueError):crop_edges(np.ones((50,50)),**kw)

    def test_colored_overlay_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'overlay.png';a=np.zeros((50,50,3),np.uint8);a[...,0]=230
            Image.fromarray(a).save(p)
            with self.assertRaises(ValueError):load_gray(p)

    def test_empty_frame(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'blank.tif';Image.fromarray(np.zeros((128,128),np.uint8)).save(p)
            r=analyze_one(p,Path(d)/'results',Config(nm_per_px=2))
            self.assertEqual(r['estimated_visible_paths'],0)
            self.assertEqual(r['accepted_diameter_samples'],0)
            self.assertEqual(r['coverage_pct'],0)
            fa.render(np.zeros((32,32)),np.zeros((32,32),bool),[],[],np.zeros((32,32)),[],Path(d)/'blank.png')

    def test_diameter_does_not_jump_neighbor(self):
        a=np.full((160,160),.05);a[65:85]=.5;a[100:135]=.95
        sm=ndi.gaussian_filter(a,1);mask=a>.2
        m=fd.measure_at(sm,mask,75,80,np.array([0.,1.]),10)
        self.assertIsNotNone(m)
        self.assertAlmostEqual(m['diameter_px'],20,delta=1)

    def test_rotated_diameters(self):
        rr,cc=np.mgrid[:256,:256]
        for angle in [0,25,45,70,90]:
            theta=np.deg2rad(angle);t=np.array([np.sin(theta),np.cos(theta)])
            normal=np.array([-t[1],t[0]])
            distance=(rr-128)*normal[0]+(cc-128)*normal[1]
            mask=np.abs(distance)<10
            img=ndi.gaussian_filter(np.where(mask,.8,.05),1)
            m=fd.measure_at(img,mask,128,128,t,10)
            self.assertIsNotNone(m,msg=str(angle))
            self.assertAlmostEqual(m['diameter_px'],20,delta=1.5)

    def test_uniform_crossing_abstains(self):
        coords=np.arange(20,110,dtype=float)
        fibers=[{'pts':np.column_stack([np.full_like(coords,64),coords])},
                {'pts':np.column_stack([coords,np.full_like(coords,64)])}]
        img=np.full((128,128),.5);dist=np.full_like(img,5)
        top,_=fa.infer_z(img,img,fibers,0,1,np.array([64,64]),dist,[])
        self.assertIsNone(top)

    def test_coordinate_transform(self):
        g={'source_px_per_analysis_x':3.1,'source_px_per_analysis_y':3.2}
        pts=original_points([[0,0],[1,1]],g)
        np.testing.assert_allclose(pts[1]-pts[0],[3.2,3.1])

    def test_isolated_fiber_count_and_density(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'straight.tif';a=np.full((256,256),12,np.uint8)
            a[55:75,10:246]=210;a[170:190,10:246]=210
            Image.fromarray(a).save(p)
            r=analyze_one(p,Path(d)/'results',Config(nm_per_px=10,crop_bottom=0,max_dim=256))
            self.assertEqual(r['estimated_visible_paths'],2)
            self.assertAlmostEqual(r['paths_per_um2'],2/(256*256*.01**2))
            self.assertGreater(r['accepted_diameter_samples'],0)
            self.assertAlmostEqual(r['sample_median_diameter_nm'],200,delta=15)

    def test_dark_core_remains_part_of_fiber(self):
        a=np.full((256,256),.05)
        a[90:130,10:246]=.22
        a[90:94,10:246]=.9;a[126:130,10:246]=.9
        mask=segment(a)
        self.assertTrue(mask[110,128])

    def test_first_branch_is_oriented_toward_link(self):
        # First segment's stored start is at the crossing. It must reverse.
        a={'pix':np.column_stack([np.full(40,50.),np.arange(49,9,-1.)]),'j0':1,'j1':None}
        b={'pix':np.column_stack([np.full(40,50.),np.arange(51,91.)]),'j0':1,'j1':None}
        fibers=fa.stitch_fibers([a,b],{1:{}},{1:1},set())
        self.assertEqual(len(fibers),1)
        pts=fibers[0]['pts']
        self.assertLess(np.max(np.linalg.norm(np.diff(pts,axis=0),axis=1)),2)

    def test_zone_merging_is_bounded(self):
        centers={i:np.array([20.,float(i*2)]) for i in range(1,30)}
        branches=[{'pix':np.array([[20.,i*2+1.]]),'j0':i,'j1':i+1} for i in range(1,29)]
        zones,_,_=fa.build_zones(branches,centers,np.full((64,64),2.))
        self.assertGreater(len(zones),1)
        self.assertTrue(all(len(z['junctions'])<=16 for z in zones.values()))

    def test_reference_comparison_validates_duplicates(self):
        import json,csv
        from compare_reference import compare
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);run=root/'run';site=run/'site';site.mkdir(parents=True)
            (site/'report.json').write_text(json.dumps({'input_sha256':'abc'}))
            (site/'diameters.csv').write_text('sample_id,status,diameter_nm\n1,accepted,110\n')
            ref=root/'manual.csv'
            header='image_sha256,sample_id,analyst,session,manual_diameter_nm\n'
            ref.write_text(header+'abc,1,A,day1,100\n')
            result=compare(run,ref,root/'out')
            self.assertEqual(result['by_analyst'][0]['mean_bias_nm'],10)
            ref.write_text(header+'abc,1,A,day1,100\nabc,1,A,day1,100\n')
            with self.assertRaises(ValueError):compare(run,ref,root/'out')


if __name__=='__main__':unittest.main(verbosity=2)
