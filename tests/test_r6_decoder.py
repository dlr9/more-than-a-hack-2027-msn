import unittest
import numpy as np
from solution.decoder import viterbi_decode
class MissingStabilityTests(unittest.TestCase):
    def test_identity_preserves_frozen_decoder(self):
        rng=np.random.default_rng(2027);p=rng.dirichlet([2,2,1],200)
        base=viterbi_decode(p,[(0,89),(100,199)],3,1)
        same=viterbi_decode(p,[(0,89),(100,199)],3,1,dead_scale=np.ones(200))
        np.testing.assert_array_equal(base,same)
    def test_missing_does_not_force_alive_or_dead(self):
        for state in [0,1,2]:
            p=np.full((80,3),.01);p[:,state]=.98
            y=viterbi_decode(p,[(0,79)],3,1,dead_scale=np.full(80,4.))
            self.assertTrue((y=='HAD'[state]).all())
    def test_invalid_scale_rejected(self):
        for scale in [np.ones(3),np.array([1,np.nan]),np.array([1,0])]:
            with self.assertRaises(ValueError):viterbi_decode(np.full((2,3),1/3),[(0,1)],3,1,dead_scale=scale)
