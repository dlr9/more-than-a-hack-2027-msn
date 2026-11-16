import unittest
import numpy as np
from r10.acquisition import geometry_evidence,acquisition_strength,rank_proposals,equal_budget,negative_controls,interval_mean

class AcquisitionTests(unittest.TestCase):
    def raw(self,n=100):
        players=np.zeros((n,16,2));players[:,8:,0]=10
        return dict(frame=np.arange(n),period=np.ones(n),ball_xy=np.zeros((n,2)),players_xy=players,player_team=np.r_[np.zeros(8,int),np.ones(8,int)],player_track_id=np.arange(16))
    def test_geometry_team0_home_and_swap(self):
        r=self.raw();e,v=geometry_evidence(r);self.assertTrue((e[:,0]>e[:,1]).all());self.assertTrue(v.all())
        np.testing.assert_allclose(geometry_evidence({**r,'player_team':1-r['player_team']})[0],e[:,::-1])
    def test_slots_and_id_renaming(self):
        r=self.raw();ix=np.arange(16)[::-1];s={**r,'players_xy':r['players_xy'][:,ix],'player_team':r['player_team'][ix],'player_track_id':np.arange(16)+98}
        np.testing.assert_array_equal(geometry_evidence(r)[0],geometry_evidence(s)[0])
    def test_zero_observations_and_truth_rejected(self):
        r=self.raw();r['ball_xy'][:]=np.nan;e,v=geometry_evidence(r);self.assertFalse(e.any());self.assertFalse(v.any())
        with self.assertRaises(ValueError):geometry_evidence({**r,'y_home':np.zeros(100)})
    def test_interval_edges(self):
        x=np.ones((5,2));np.testing.assert_array_equal(interval_mean(x,-2,0)[:,0],[0,0,1,1,1]);np.testing.assert_array_equal(interval_mean(x,0,2)[:,0],[1,1,1,1,0])
    def test_future_confirmation_and_no_cross_half_gap(self):
        n=200;e=np.zeros((n,2));e[:100,0]=1;e[100:,1]=1;f=np.arange(n);p=np.ones(n)
        s=acquisition_strength(e,f,p,10);self.assertGreater(s[:,1].max(),0);self.assertEqual(s[:,0].max(),0)
        p[100:]=2;self.assertFalse(acquisition_strength(e,f,p,10).any())
        p[:]=1;f[100:]+=10;self.assertFalse(acquisition_strength(e,f,p,10).any())
    def test_stationary_ownership_has_no_acquisition(self):
        e=np.tile([.8,.1],(200,1));self.assertFalse(acquisition_strength(e,np.arange(200),np.ones(200),10).any())
    def test_same_budget_and_controls(self):
        r=self.raw(300);s=np.zeros((300,2));s[::10,0]=.5;s[5::10,1]=.7
        p=rank_proposals(s,r['frame'],r['period'],10,limit=12);self.assertEqual(len(p),12)
        d,b=equal_budget({'one':p,'two':p[:6]},r['frame'],r['period']);self.assertEqual(b,[6]);self.assertEqual(len(d['one']),6)
        sh,ti=negative_controls(p,r['frame'],r['period'],10)
        self.assertEqual(sorted(x['team'] for x in sh),sorted(x['team'] for x in p));self.assertEqual(len(ti),len(p));self.assertEqual([x['index'] for x in sh],[x['index'] for x in p])

if __name__=='__main__':unittest.main()
