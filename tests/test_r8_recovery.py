import tempfile,unittest
from pathlib import Path
import numpy as np
from scripts.train_r8_boundaries import save_or_verify

class RecoveryTests(unittest.TestCase):
    def test_exact_orphan_replay_retains_bytes(self):
        with tempfile.TemporaryDirectory()as d:
            p=Path(d)/'prediction.npz';a=np.array([.2,.8])
            save_or_verify(p,p_alive=a);before=p.read_bytes()
            save_or_verify(p,p_alive=a.copy())
            self.assertEqual(p.read_bytes(),before)

    def test_different_orphan_is_not_overwritten(self):
        with tempfile.TemporaryDirectory()as d:
            p=Path(d)/'prediction.npz';save_or_verify(p,p_alive=np.array([.2]));before=p.read_bytes()
            with self.assertRaises(AssertionError):save_or_verify(p,p_alive=np.array([.3]))
            self.assertEqual(p.read_bytes(),before)

if __name__=='__main__':unittest.main()
