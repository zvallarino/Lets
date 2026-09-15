import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
from run_geometry import geometry_preview,run_batch


class GeometryOutputTests(unittest.TestCase):
    def test_denser_sampling_adds_valid_widths(self):
        gray=np.full((400,240),.1);gray[:,100:130]=.8
        depth=np.zeros_like(gray);depth[:,100:130]=1
        _,_,old=geometry_preview(gray,depth,5,12,95)
        _,_,new=geometry_preview(gray,depth,5)
        self.assertGreater(sum(a['preview_crossbars'] for a in new['axes']),sum(a['preview_crossbars'] for a in old['axes']))
        self.assertGreater(sum(a['accepted_sites'] for a in new['axes']),sum(a['accepted_sites'] for a in old['axes']))
        self.assertTrue(all(abs(s['width_analysis_px']-30)<2 for s in new['sites'] if s['status']=='accepted'))

    def test_batch_copies_only_source_and_continues_after_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            inputs=[root/'a.tif',root/'bad.tif',root/'a.tiff']
            def analyze(image,run,*args):
                if image.name=='bad.tif':raise ValueError('bad input')
                result=run/image.name;result.mkdir()
                for filename in ('source_geometry_preview.png','geometry_preview.png'):
                    Image.new('RGB',(8,8),(20,30,40)).save(result/filename)
                (result/'preview_notes.json').write_text('{}')
                return result
            with patch('run_geometry.analyze_image',side_effect=analyze):
                self.assertEqual(run_batch(inputs,root/'out'),1)
            run=next((root/'out').iterdir())
            copies=list((run/'quick_review').glob('*.png'))
            self.assertEqual(len(copies),2)
            with (run/'batch_summary.csv').open(encoding='utf-8-sig',newline='') as f:
                rows=list(csv.DictReader(f))
            self.assertEqual([r['status'] for r in rows],['completed','failed','completed'])
            for row in (rows[0],rows[2]):
                result=Path(row['output'])
                self.assertTrue((result/'preview_notes.json').exists())
                self.assertEqual(Path(row['review_source']).read_bytes(),(result/'source_geometry_preview.png').read_bytes())
                self.assertTrue((result/'geometry_preview.png').exists())
                self.assertNotIn('review_depth',row)
            self.assertFalse(list((run/'quick_review').glob('*__depth.png')))

    def test_invalid_spacing(self):
        for value in (0,-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):
                geometry_preview(np.zeros((8,8)),np.zeros((8,8)),sample_step=value)

if __name__=='__main__':unittest.main()
