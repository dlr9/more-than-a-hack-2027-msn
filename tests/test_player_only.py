import unittest
import numpy as np
from solution.player_only_features import build_player_features,player_inputs
from solution.decoder import emission_probabilities,viterbi_decode
from solution.quality_experts import gate_mask,fuse

class PlayerOnlyTests(unittest.TestCase):
    def raw(self,n=80):
        rng=np.random.default_rng(1)
        xy=rng.normal(size=(1,20,2))+np.arange(n)[:,None,None]*np.array([.2,0])[None,None,:]
        return dict(frame=np.arange(n),period=np.ones(n,int),players_xy=xy,
                    player_team=np.repeat([0,1],10),player_track_id=np.arange(20))
    def build(self,r,**kwargs):return build_player_features(r,fps=10.,**kwargs)
    def col(self,b,name):return b['X'][:,b['feature_names'].tolist().index(name)]
    def test_ball_and_labels_never_read(self):
        r=self.raw();a=self.build(r)
        r.update(ball_xy=object(),ball_status=object(),truth=object(),provider_possession=object())
        np.testing.assert_array_equal(a['X'],self.build(player_inputs(r))['X'])
        with self.assertRaises(ValueError):self.build(r)
    def test_slot_permutation(self):
        r=self.raw();a=self.build(r);order=np.random.default_rng(2).permutation(20)
        for k in ['player_team','player_track_id']:r[k]=r[k][order]
        r['players_xy']=r['players_xy'][:,order]
        np.testing.assert_allclose(a['X'],self.build(r)['X'],atol=1e-7,equal_nan=True)
    def test_units_and_speed(self):
        r=self.raw();a=self.build(r)
        np.testing.assert_allclose(self.col(a,'home_speed_p50_mps')[1:],2.)
        np.testing.assert_allclose(self.col(a,'home_accel_p50_mps2')[2:],0.,atol=1e-10)
        r['players_xy']=r['players_xy']/[105,68]+.5
        np.testing.assert_allclose(a['X'],self.build(r,units='normalized')['X'],atol=1e-6,equal_nan=True)
    def test_id_switch_resets_speed(self):
        r=self.raw();r['player_track_id']=np.tile(r['player_track_id'],(80,1));r['player_track_id'][40:]+=100
        r['players_xy'][40:]+=100
        b=self.build(r);self.assertTrue(np.isnan(self.col(b,'activity_mean_mps')[40]))
        self.assertAlmostEqual(self.col(b,'activity_mean_mps')[41],2.)
    def test_half_and_gap_reset(self):
        r=self.raw();r['period'][40:]=2;r['frame'][60:]+=4;r['players_xy'][40:]+=100
        b=self.build(r)
        for i in [40,60]:self.assertTrue(np.isnan(self.col(b,'activity_5s_mean_mps')[i]))
    def test_referee_excluded(self):
        r=self.raw();a=self.build(r);r['players_xy']=np.concatenate([r['players_xy'],np.full((80,1,2),1e9)],axis=1)
        r['player_team']=np.r_[r['player_team'],-1];r['player_track_id']=np.r_[r['player_track_id'],100]
        np.testing.assert_allclose(a['X'],self.build(r)['X'],atol=1e-7,equal_nan=True)
    def test_low_visibility_and_missing_reset(self):
        r=self.raw();r['players_xy'][:,5:]=np.nan;r['players_xy'][30:40]=np.nan
        b=self.build(r);self.assertEqual(self.col(b,'home_count')[20],5)
        self.assertTrue(np.isnan(self.col(b,'activity_mean_mps')[40]));self.assertFalse(np.isinf(b['X']).any())
    def test_team_reversal(self):
        r=self.raw();a=self.build(r);r['player_team']=1-r['player_team'];b=self.build(r)
        for name in a['feature_names']:
            swapped=name.replace('home_','away_',1)if name.startswith('home_')else name.replace('away_','home_',1)if name.startswith('away_')else name
            np.testing.assert_allclose(self.col(a,name),self.col(b,swapped),atol=1e-7,equal_nan=True)
    def test_team_change_resets(self):
        r=self.raw();r['player_team']=np.tile(r['player_team'],(80,1));r['player_team'][40:]=1-r['player_team'][40:]
        self.assertTrue(np.isnan(self.col(self.build(r),'activity_mean_mps')[40]))
    def test_decoder_fixed_penalty_cannot_repair_emissions(self):
        a=np.full(225,.99);a[50:175]=.1;h=np.full(225,.99);scale=np.ones(225);scale[50:176]=4
        p=viterbi_decode(emission_probabilities(a,h),[(0,224)],7.5,2.5,dead_scale=scale)
        self.assertEqual(np.sum(p[50:175]=='D'),125)
        a[50:175]=.95
        p=viterbi_decode(emission_probabilities(a,h),[(0,224)],7.5,2.5,dead_scale=scale)
        self.assertTrue(np.all(p=='H'))
    def test_gate_uses_quality_and_resets(self):
        players=self.build(self.raw());n=80
        base=dict(X=np.zeros((n,1)),feature_names=np.asarray(['ball_visible']),frame=np.arange(n),period=np.ones(n),fps=np.asarray(10.))
        mask=gate_mask(base,players);self.assertFalse(mask[3]);self.assertTrue(mask[5])
        base['period'][40:]=2;players['period'][40:]=2;self.assertFalse(gate_mask(base,players)[40])
        players['X'][:,players['feature_names'].tolist().index('home_count')]=0
        a,use=fuse(base,players,np.full(n,.2),np.full(n,.9));self.assertFalse(use.any());np.testing.assert_array_equal(a,.2)
    def test_gate_rejects_alignment_error(self):
        players=self.build(self.raw());base=dict(X=np.zeros((80,1)),feature_names=np.asarray(['ball_visible']),frame=np.arange(80)+1,period=np.ones(80),fps=np.asarray(10.))
        with self.assertRaises(ValueError):gate_mask(base,players)

if __name__=='__main__':unittest.main()
