import unittest
import numpy as np
from source_geometry import SourceBoundaryCheck, filter_track


def proposal(row, left, right):
    return {'edge1_rc':(row,left),'edge2_rc':(row,right),'diameter_px':right-left}


class SourceGeometryTests(unittest.TestCase):
    def test_isolated_fiber_retains_known_width(self):
        gray=np.full((180,180),.1);gray[:,70:100]=.8
        m,reason=SourceBoundaryCheck(gray).check(proposal(90,69.5,99.5),np.array([1.,0.]))
        self.assertIsNone(reason)
        self.assertAlmostEqual(m['diameter_px'],30,delta=2)

    def test_merged_depth_over_two_fibers_does_not_span_gap(self):
        gray=np.full((180,180),.1);gray[:,50:75]=.8;gray[:,83:108]=.8
        m,reason=SourceBoundaryCheck(gray).check(proposal(90,49.5,107.5),np.array([1.,0.]))
        self.assertIsNone(m)
        self.assertIsNotNone(reason)

    def test_neighbor_does_not_enlarge_source_width(self):
        gray=np.full((180,180),.1);gray[:,50:75]=.7;gray[:,83:108]=.95
        m,reason=SourceBoundaryCheck(gray).check(proposal(90,49.5,74.5),np.array([1.,0.]))
        self.assertIsNone(reason)
        self.assertAlmostEqual(m['diameter_px'],25,delta=2)

    def test_crossing_and_approach_are_omitted_but_clear_section_retained(self):
        gray=np.full((240,240),.1);gray[:,100:130]=.8;gray[110:135,:]=.8
        check=SourceBoundaryCheck(gray)
        for row in (100,120,140):
            m,reason=check.check(proposal(row,99.5,129.5),np.array([1.,0.]))
            self.assertIsNone(m)
        m,reason=check.check(proposal(60,99.5,129.5),np.array([1.,0.]))
        self.assertIsNone(reason)
        self.assertAlmostEqual(m['diameter_px'],30,delta=2)

    def test_bright_rims_do_not_become_two_thin_fibers(self):
        gray=np.full((180,180),.1)
        gray[:,70:100]=.4
        gray[:,70:74]=.9;gray[:,96:100]=.9
        m,reason=SourceBoundaryCheck(gray).check(proposal(90,69.5,99.5),np.array([1.,0.]))
        self.assertIsNone(reason)
        self.assertAlmostEqual(m['diameter_px'],30,delta=2)

    def test_preview_retains_straight_fiber_and_rejects_merged_pair(self):
        from run_geometry import geometry_preview
        gray=np.full((400,240),.1);gray[:,100:130]=.8
        depth=np.zeros_like(gray);depth[:,100:130]=1
        _,_,notes=geometry_preview(gray,depth,5)
        accepted=[s for s in notes['sites'] if s['status']=='accepted']
        self.assertGreater(len(accepted),10)
        self.assertTrue(all(abs(s['width_analysis_px']-30)<2 for s in accepted))
        gray=np.full_like(gray,.1);gray[:,85:108]=.8;gray[:,122:145]=.8
        depth=np.zeros_like(gray);depth[:,85:145]=1
        _,_,notes=geometry_preview(gray,depth,10)
        self.assertFalse(any(s['status']=='accepted' for s in notes['sites']))

    def test_persistent_bright_seam_rejects_merged_touching_fibers(self):
        gray=np.full((240,240),.1);gray[:,90:150]=.45
        gray[:,90:94]=.85;gray[:,146:150]=.85
        gray[:,117:123]=.85
        check=SourceBoundaryCheck(gray)
        m,reason=check.check(proposal(120,89.5,149.5),np.array([1.,0.]))
        self.assertIsNone(m)
        self.assertEqual(reason,'persistent_internal_seam')

    def test_crossing_highlight_is_not_a_persistent_longitudinal_seam(self):
        gray=np.full((240,240),.1);gray[:,90:150]=.45
        gray[117:123,117:123]=.85
        check=SourceBoundaryCheck(gray)
        self.assertFalse(check.has_internal_seam(np.array([120.,89.5]),np.array([120.,149.5]),np.array([1.,0.])))

    def test_one_third_widening_is_rejected_but_gradual_change_retained(self):
        samples=[{'mid':np.array([i*6.,0.]),'width':30.} for i in range(30)]
        samples[15]['width']=40.
        result=filter_track(samples,np.array([1.,0.]))
        self.assertEqual(result[15]['rejection'],'track_width_outlier')
        gradual=[{'mid':np.array([i*6.,0.]),'width':30.+i*.1} for i in range(30)]
        self.assertFalse(any(s.get('rejection') for s in filter_track(gradual,np.array([1.,0.]))))

    def test_track_rejects_widening_and_lateral_jump(self):
        samples=[{'mid':np.array([i*12.,0.]),'width':20.} for i in range(10)]
        samples[4]['width']=40.
        samples[7]['mid'][1]=12.
        result=filter_track(samples,np.array([1.,0.]))
        self.assertEqual(result[4]['rejection'],'track_width_outlier')
        self.assertEqual(result[7]['rejection'],'track_center_jump')
        self.assertNotIn('rejection',result[0])

    def test_missing_section_remains_gap(self):
        samples=[{'mid':np.array([0.,0.]),'width':20.},None,
                 {'mid':np.array([24.,0.]),'width':20.}]
        self.assertIsNone(filter_track(samples,np.array([1.,0.]))[1])

if __name__=='__main__':unittest.main()
