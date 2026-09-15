import csv,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image
import run_all as full
import run_pipeline

class FullPipelineTests(unittest.TestCase):
    def test_current_cli_dispatch(self):
        with patch('run_all.main',return_value=7) as current:
            self.assertEqual(run_pipeline.main(),7)
            current.assert_called_once_with()

    def test_inputs_are_tiffs_deduplicated_nonrecursive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'one.TIFF').touch();(root/'metadata.txt').touch()
            sub=root/'nested';sub.mkdir();(sub/'two.tif').touch()
            self.assertEqual(full.collect_inputs([root,root/'one.TIFF']),[(root/'one.TIFF').resolve()])
            with self.assertRaises(ValueError):full.collect_inputs([root/'metadata.txt'])

    def test_stages_and_failures_preserve_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            def geom(image,out,**options):
                if image.stem in ('geometrybad','bothbad'):raise ValueError('geometry failed')
                out.mkdir(parents=True)
                notes={'diameter_export':{'rows':12,'check_rows':2},'axes':[{'preview_crossbars':6}],
                    'measurement_review':{'review_status':'below_target','target_ok_measurements':100,'ok_fraction':10/12,'top_check_reasons':'local_width_change: 2'}}
                (out/'preview_notes.json').write_text(json.dumps(notes))
                Image.new('RGB',(8,8)).save(out/'source_geometry_preview.png')
                return out
            def dens(image,out,**options):
                if image.stem in ('densitybad','bothbad'):raise ValueError('density failed')
                out.mkdir(parents=True)
                report={'status':'researcher_review_required','automatic_path_count':4,'automatic_count_low':3,'automatic_count_high':6,
                    'automatic_paths_per_um2':None,'analyzed_area_um2':None}
                (out/'density_report.json').write_text(json.dumps(report))
                Image.new('RGB',(8,8)).save(out/'count_overlay.png')
                return out
            def non_ai(image,out,**options):
                if image.stem=='bothbad':raise ValueError('non-AI failed')
                out.mkdir(parents=True)
                (out/'non_ai_report.json').write_text(json.dumps({'non_ai_candidates':8,'non_ai_ok':5,'non_ai_check':3}))
                for filename in ('non_ai_overlay.png','segmentation_overlay.png'):
                    Image.new('RGB',(8,8)).save(out/filename)
                return out
            images=[root/(name+'.tif') for name in ('good','geometrybad','densitybad','bothbad','last')]
            with patch('run_all.geometry.analyze_image',side_effect=geom),patch('run_all.density.analyze_density',side_effect=dens),patch('run_all.classical.analyze_classical',side_effect=non_ai):
                self.assertEqual(full.run_all(images,root/'out'),1)
            batch=next((root/'out').iterdir())
            with (batch/'batch_summary.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
            self.assertEqual([r['status'] for r in rows],['completed','partial','partial','failed','completed'])
            self.assertEqual(rows[0]['measurements_ok'],'10')
            self.assertEqual(rows[0]['automatic_path_count'],'4')
            self.assertEqual(rows[0]['automatic_paths_per_um2'],'')
            self.assertEqual(rows[1]['measurements_ok'],'')
            self.assertEqual(rows[2]['automatic_path_count'],'')
            self.assertEqual(len(list((batch/'quick_review').glob('*.png'))),14)
            self.assertTrue((batch/'manifest.json').exists())

if __name__=='__main__':unittest.main()
