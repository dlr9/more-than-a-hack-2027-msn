"""Regression tests for V5 Physical Pitch Boundary & Kinematic possession model."""
import json
from pathlib import Path
import sys, unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fusion_v5 import DEFAULT, latch_outside_pitch

class V5Tests(unittest.TestCase):
    def test_default_config_schema(self):
        self.assertIn('out_latch', DEFAULT)
        self.assertIn('confirm_frames', DEFAULT)
        self.assertTrue(DEFAULT['out_latch'])
        self.assertEqual(DEFAULT['confirm_frames'], 10)
        self.assertNotIn('event_refine', DEFAULT)

    def test_latch_outside_pitch_logic(self):
        # 30 frames: 10 frames outside pitch (x > 105.3), then 20 frames inside
        ball = np.zeros((30, 2))
        ball[:10, 0] = 110.0  # outside pitch
        ball[10:, 0] = 50.0   # inside pitch
        
        # Players at (50, 0), within 3m
        players = np.zeros((30, 22, 2))
        players[:, 0, 0] = 50.0
        
        blocks = [(0, 30)]
        held = latch_outside_pitch(ball, players, blocks, confirm=5)
        # Outside frames must be held as dead ball
        self.assertTrue(held[0])
        self.assertTrue(held[9])

    def test_block_level_scoping(self):
        # Ball outside before block boundary should not pollute next block
        ball = np.zeros((40, 2))
        ball[:15, 0] = 110.0  # outside in block 1
        ball[15:20, 0] = 50.0  # inside in block 1
        ball[20:40, 0] = 50.0  # inside in block 2
        
        players = np.zeros((40, 22, 2))
        players[:, 0, 0] = 50.0
        
        blocks = [(0, 20), (20, 40)]
        held = latch_outside_pitch(ball, players, blocks, confirm=5)
        # Block 2 starts at frame 20 and should start with active=False
        self.assertFalse(held[20])
        self.assertFalse(held[21])

if __name__ == '__main__':
    unittest.main()
