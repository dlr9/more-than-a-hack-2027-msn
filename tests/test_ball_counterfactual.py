import unittest
import numpy as np
from solution.ball_counterfactual import intervene

class CounterfactualTests(unittest.TestCase):
    def test_determinism_and_label_free_schedule(self):
        period=np.repeat([1,2],1500);ball=np.column_stack([np.arange(3000)*.01,np.zeros(3000)])
        a,info=intervene(ball,period,25.,mode='combined')
        b,other=intervene(ball+10,period,25.,mode='combined')
        np.testing.assert_array_equal(info['freeze_mask'],other['freeze_mask'])
        np.testing.assert_array_equal(info['missing_mask'],other['missing_mask'])
        np.testing.assert_allclose(a+10,b,equal_nan=True,atol=1e-10)
        np.testing.assert_array_equal(a,intervene(ball,period,25.,mode='combined')[0])
    def test_noise_breaks_freeze_without_truth(self):
        period=np.ones(3000);ball=np.zeros((3000,2));ball[300]=np.nan
        a,_=intervene(ball,period,25.,mode='noise')
        self.assertTrue(np.isnan(a[300]).all());self.assertGreater(np.std(a[:,0][np.isfinite(a[:,0])]),.05)
        self.assertLess(np.nanstd(a[:,0]),.5)
    def test_period_noise_independent(self):
        period=np.repeat([1,2],1500);ball=np.zeros((3000,2))
        a,_=intervene(ball,period,25.,mode='noise');ball[:1500]+=100
        b,_=intervene(ball,period,25.,mode='noise');np.testing.assert_array_equal(a[1500:],b[1500:])
    def test_freeze_changes_only_scheduled_observed_frames(self):
        period=np.ones(3000);ball=np.column_stack([np.arange(3000)*.01,np.zeros(3000)]);ball[100]=np.nan
        a,info=intervene(ball,period,25.,mode='freeze')
        np.testing.assert_array_equal(a[~info['freeze_mask']],ball[~info['freeze_mask']]);self.assertTrue(np.isnan(a[100]).all())
        self.assertGreater(np.sum(np.diff(a[:,0])==0),10)

if __name__=='__main__':unittest.main()
