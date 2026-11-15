import unittest
import numpy as np
from scripts.evaluate_r9_heads import mix_heads
class HeadAttributionTests(unittest.TestCase):
    def test_exact_heads_no_blending(self):
        m=dict(p_alive=np.array([.9,.8]),p_home=np.array([.2,.3]));c=dict(p_alive=np.array([.6,.7]),p_home=np.array([.4,.5]))
        for a in ['M2','C1']:
            for h in ['M2','C1']:
                aa,hh=mix_heads(m,c,a,h)
                self.assertIs(aa,{'M2':m,'C1':c}[a]['p_alive']);self.assertIs(hh,{'M2':m,'C1':c}[h]['p_home'])
    def test_rejects_unregistered_cell(self):
        with self.assertRaises(ValueError):mix_heads({}, {},'new','M2')
