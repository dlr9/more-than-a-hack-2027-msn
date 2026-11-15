import unittest
import numpy as np
from solution.observation_memory import carry_priors,blackout_mask

class MemoryTests(unittest.TestCase):
    def fixture(self,n=500):
        names=['ball_visible','n_home_positions','n_away_positions','ball_speed_missing','home_d1_m','away_d1_m']
        x=np.tile([1,11,11,0,1,5],(n,1)).astype(float)
        return dict(X=x,feature_names=np.asarray(names),fps=10.,period=np.ones(n),frame=np.arange(n))
    def test_no_change_when_observed(self):
        b=self.fixture();a=np.full(500,.99);h=a.copy();out=carry_priors(b,a,h,team_memory=True)
        np.testing.assert_array_equal(a,out[0]);np.testing.assert_array_equal(h,out[1])
    def test_expiration_and_same_alive(self):
        b=self.fixture();b['X'][10:,:3]=0;b['X'][10:,3]=1;a=np.full(500,.4);a[:10]=.99;h=np.full(500,.45);h[:10]=.95
        aa,hh,_=carry_priors(b,a,h,team_memory=True);control=carry_priors(b,a,h,team_memory=False)
        np.testing.assert_array_equal(aa,control[0]);self.assertGreater(hh[50],.5);self.assertEqual(aa[400],a[400]);self.assertEqual(hh[400],h[400])
    def test_dead_clears_team_and_half_resets(self):
        b=self.fixture();a=np.full(500,.99);h=np.full(500,.95);a[9]=.01;b['X'][10:,:3]=0;b['X'][10:,3]=1
        aa,hh,info=carry_priors(b,a,h,team_memory=True);self.assertLess(aa[50],.5);self.assertFalse(info['home_prior_used'].any())
        b['period'][20:]=2;aa,_,info=carry_priors(b,a,h,team_memory=True);self.assertFalse(info['alive_prior_used'][20:].any())
    def test_ball_only_missing_does_not_trigger(self):
        b=self.fixture();b['X'][10:,0]=0;a=np.full(500,.8);h=np.full(500,.6)
        aa,hh,info=carry_priors(b,a,h,team_memory=True);np.testing.assert_array_equal(a,aa);self.assertFalse(info['alive_prior_used'].any())
    def test_gap_resets_and_mask_is_deterministic(self):
        b=self.fixture();a=np.full(500,.99);h=a.copy();b['X'][10:,:3]=0;b['frame'][10:]+=2
        self.assertFalse(carry_priors(b,a,h,team_memory=True)[2]['alive_prior_used'].any())
        p=np.repeat([1,2],30000);np.testing.assert_array_equal(blackout_mask(p,25),blackout_mask(p,25));self.assertTrue(blackout_mask(p,25).any())

if __name__=='__main__':unittest.main()
