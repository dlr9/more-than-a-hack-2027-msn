import unittest
import numpy as np
from solution.boundary_sampling import select

class BoundarySamplingTests(unittest.TestCase):
    def test_budget_reproducibility_and_eligibility(self):
        frame=np.arange(1000);period=np.ones(1000);truth=np.repeat([0,1,2,1],250);eligible=np.ones(1000,bool);eligible[::3]=False
        x,u=select(frame,period,truth,eligible,fps=25,seed=8203)
        self.assertEqual(x.sum(),u.sum());self.assertFalse((x&~eligible).any())
        np.testing.assert_array_equal(x,select(frame,period,truth,eligible,fps=25,seed=8203)[0])
        near=np.zeros(1000,bool)
        for t in [250,500,750]:near[t-12:t+13]=True
        self.assertGreater((near&x).sum(),(near&u).sum())

    def test_constant_labels_fall_back_without_duplicate_samples(self):
        x,u=select(np.arange(100),np.ones(100),np.ones(100),np.ones(100,bool),fps=25,seed=8203)
        self.assertEqual(x.sum(),u.sum())

    def test_no_outside_samples(self):
        period=np.r_[np.zeros(20),np.ones(80)]
        x,u=select(np.arange(100),period,np.ones(100),np.ones(100,bool),fps=25,seed=8203)
        self.assertFalse(x[:20].any())

if __name__=='__main__':unittest.main()
