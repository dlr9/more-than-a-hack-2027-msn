"""Regression tests for V5 Event-Triggered Multimodal Cascade possession model."""
import json
from pathlib import Path
import sys, unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fusion_v5 import DEFAULT, latch_outside, detect_anomalies, apply_event_refinement

class V5Tests(unittest.TestCase):
    def test_default_config_schema(self):
        self.assertIn('out_latch', DEFAULT)
        self.assertIn('confirm_frames', DEFAULT)
        self.assertIn('event_refine', DEFAULT)
        self.assertTrue(DEFAULT['out_latch'])
        self.assertEqual(DEFAULT['confirm_frames'], 10)

    def test_latch_outside_logic(self):
        # 10 frames outside pitch (x > 105.3)
        ball = np.zeros((30, 2))
        ball[:10, 0] = 110.0  # outside
        ball[10:, 0] = 50.0   # inside
        
        # Players at (50, 0), within 3m
        players = np.zeros((30, 22, 2))
        players[:, 0, 0] = 50.0
        
        held = latch_outside(ball, players, confirm=5)
        # Outside frames must be held
        self.assertTrue(held[0])
        self.assertTrue(held[9])
        
    def test_anomaly_detection(self):
        ball = np.zeros((100, 2))
        ball[20:80] = np.nan
        anomalies = detect_anomalies(ball, threshold=50)
        self.assertEqual(len(anomalies), 1)
        self.assertEqual(anomalies[0], (20, 80))

    def test_event_refinement(self):
        labels = np.full(20000, 'X', dtype=object)
        refined = apply_event_refinement(labels)
        self.assertEqual(refined[15365], 'A')
        self.assertEqual(refined[15766], 'H')
        self.assertEqual(refined[15799], 'A')
        self.assertEqual(refined[15889], 'D')
        self.assertEqual(refined[16399], 'D')

if __name__ == '__main__':
    unittest.main()
