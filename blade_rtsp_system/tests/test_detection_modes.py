import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.classic_detector import ClassicDetector
from src import blade_locator
spec = importlib.util.spec_from_file_location('blade_main', ROOT/'main.py')
main = importlib.util.module_from_spec(spec)
spec.loader.exec_module(main)


def frame(x=None):
    f = np.zeros((120,160,3),np.uint8)
    if x is not None:
        cv2.rectangle(f,(x,25),(x+14,110),(220,220,220),-1)
    return f


class DetectionTests(unittest.TestCase):
    def test_classic_localizes_current_blade_not_previous_ghost(self):
        d=ClassicDetector({'warmup_frames':2,'min_component_area':40})
        self.assertFalse(d.predict(frame())[0]['blade_detected'])
        d.predict(frame())
        a,_=d.predict(frame(30));b,_=d.predict(frame(80))
        self.assertTrue(a['blade_detected']);self.assertEqual(b['blade_count'],1)
        self.assertGreater(b['blade_instances'][0]['bbox'][0],70)
        loc=blade_locator.locate_blades(b['blade_instances'],roi_offset=(10,20),frame_width=160,frame_height=120)
        self.assertGreater(loc[0]['bbox'][0],80)
        self.assertFalse(d.predict(frame())[0]['blade_detected'])
        self.assertFalse(b['diagnostics']['measurement_valid'])
        self.assertIsNone(b['diagnostics']['cpx_euclidean_px'])

    def test_static_global_change_and_bad_config(self):
        d=ClassicDetector({'warmup_frames':1})
        d.predict(frame());self.assertFalse(d.predict(frame())[0]['blade_detected'])
        r,_=d.predict(np.full((120,160,3),255,np.uint8))
        self.assertEqual(r['diagnostics']['status'],'FRAME_QUALITY_LOW')
        with self.assertRaises(ValueError):ClassicDetector({'trigger_threshold':40,'localize_threshold':30})

    def test_main_both_modes_roi_and_outputs(self):
        class Reader:
            cap=None;max_reconnect_attempts=0;reconnect_count=0
            def __init__(self,*a,**kw):self.frame_id=0
            def _masked_source(self):return 'synthetic'
            def open(self):return True
            def read(self):
                time.sleep(.005);self.frame_id+=1
                return True,frame(None if self.frame_id<4 else 40+(self.frame_id%4)*10),time.time()
            def release(self):pass
        class Yolo:
            def __init__(self,**kw):pass
            def load(self):pass
            def predict(self,f):return None,.1
        for mode in ('classic','yolo'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp);(p/'model.pt').touch()
                cfg={'model':{'path':str(p/'model.pt')},'classic':{'warmup_frames':2,'min_component_area':30},
                     'processing':{'resize':[160,120],'use_roi':True,'roi':{'x1':10,'y1':10,'x2':150,'y2':120}},
                     'output':{'show_window':False,'csv':str(p/'result.csv'),'jsonl':str(p/'result.jsonl'),'logs':str(p/'logs')}}
                if mode=='classic': del cfg['model']  # No model configuration required.
                (p/'config.yaml').write_text(yaml.safe_dump(cfg))
                fake=types.ModuleType('src.yolo_detector');fake.YOLODetector=Yolo
                with patch.object(main,'RTSPReader',Reader), patch.dict(sys.modules,{'src.yolo_detector':fake}), patch.object(sys,'argv',['main.py','--config',str(p/'config.yaml'),'--mode',mode,'--max-duration','.2']):
                    self.assertEqual(main.main(),0)
                rows=[json.loads(s) for s in (p/'result.jsonl').read_text().splitlines()]
                self.assertTrue(rows);self.assertTrue(all(r['detector_mode']==mode for r in rows))
                if mode=='classic':self.assertTrue(any(r['blade_detected'] for r in rows))
                self.assertIn('detector_mode',(p/'result.csv').read_text().splitlines()[0])


if __name__=='__main__':unittest.main()
