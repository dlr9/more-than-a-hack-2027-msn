import unittest
import numpy as np
from solution.control_events import control_episodes,relocate_boundaries

class ControlEventTests(unittest.TestCase):
    def raw(self,n=40):
        ball=np.column_stack([np.arange(n)*.04,np.zeros(n)]);players=np.zeros((n,16,2))
        players[:,:8]=ball[:,None]+[8,0];players[:,8:]=ball[:,None]+[0,8];players[:,0]=ball+[.5,0]
        return dict(frame=np.arange(n),period=np.ones(n),ball_xy=ball,players_xy=players,player_team=np.r_[np.zeros(8),np.ones(8)],player_track_id=np.arange(16))
    def test_relative_motion_control(self):
        e,s=control_episodes(self.raw(),25);self.assertEqual(len(e),1);self.assertEqual(e[0]['team'],'H');self.assertEqual(sum(s==0),39)
    def test_label_leak_rejected(self):
        r=self.raw();r['truth']=np.ones(40)
        with self.assertRaises(ValueError):control_episodes(r,25)
    def test_missing_and_halves_reset(self):
        r=self.raw();r['ball_xy'][15:25]=np.nan;r['period'][30:]=2
        e,s=control_episodes(r,25);self.assertTrue(np.all(s[15:26]==-1));self.assertTrue(all(not(a['start']<30<a['end']) for a in e))
    def test_flying_ball_not_control(self):
        r=self.raw();r['players_xy'][:]=r['players_xy'][0];r['ball_xy'][:,0]=np.arange(40)*.3
        e,_=control_episodes(r,25);self.assertEqual(e,[])
    def test_identity_switch_rejects_local_velocity(self):
        r=self.raw();r['player_track_id']=np.tile(np.arange(16),(40,1));r['player_track_id'][20:,0]=99
        e,s=control_episodes(r,25);self.assertEqual(s[20],-1)
    def test_no_events_is_identity_and_D_invariant(self):
        p=np.array(list('HHHHAAAADDHHAAA'));r,t=relocate_boundaries(p,np.ones(len(p))*.5,np.arange(len(p)),np.ones(len(p)),10,[])
        np.testing.assert_array_equal(r,p);self.assertEqual(t['changed_frames'],0)
    def test_local_boundary_uses_event_preserves_short_D(self):
        p=np.array(list('HHHHHHAAAADAHHH'));h=np.ones(len(p))*.5
        q,t=relocate_boundaries(p,h,np.arange(len(p)),np.ones(len(p)),10,[dict(start=4,team='A')])
        self.assertEqual(t['moved_boundaries'],1);np.testing.assert_array_equal(q=='D',p=='D');self.assertEqual(''.join(q[:10]),'HHHHAAAAAA')

    def test_provider_neutral_home_zero_contract(self):
        from solution.provider_neutral import provider_neutral_features_v3,FeatureContract
        raw=self.raw();b=provider_neutral_features_v3(raw,FeatureContract(provider='synthetic',observation_view='test',fps=25))
        names=b['feature_names'].tolist();e,_=control_episodes(raw,25)
        self.assertLess(b['X'][5,names.index('home_d1_m')],b['X'][5,names.index('away_d1_m')])
        self.assertEqual(e[0]['team'],'H')
        raw['player_team']=1-raw['player_team'];e,_=control_episodes(raw,25);self.assertEqual(e[0]['team'],'A')
    def test_zero_slots(self):
        r=self.raw();r['players_xy']=r['players_xy'][:,:0];r['player_team']=np.array([]);r['player_track_id']=np.array([])
        e,s=control_episodes(r,25);self.assertEqual(e,[]);self.assertTrue(np.all(s==-1))
