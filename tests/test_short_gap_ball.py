import unittest
import numpy as np
from solution.short_gap_ball import reconstruct
from solution.provider_neutral import provider_neutral_features_v3, FeatureContract


def fixture():
    n=80;ball=np.column_stack([np.arange(n)*.04,np.zeros(n)])
    offset=np.array([[.5,0]]+[[10+i,5]for i in range(7)]+[[-10-i,-5]for i in range(8)])
    players=ball[:,None,:]+offset[None,:,:]
    return dict(frame=np.arange(n),period=np.ones(n),ball_xy=ball,players_xy=players,
                player_team=np.repeat([0,1],8),player_track_id=np.arange(16))


class ShortGapTests(unittest.TestCase):
    def test_reconstruct_and_recompute_full_kinematics(self):
        raw=fixture();truth=raw['ball_xy'].copy();raw['ball_xy'][20:30]=np.nan
        ball,m=reconstruct(raw,fps=25)
        np.testing.assert_allclose(ball,truth);self.assertEqual(m['imputed'].sum(),10)
        self.assertTrue(np.isnan(raw['ball_xy'][20:30]).all())
        contract=FeatureContract('synthetic','r8_interpolated',25)
        b=provider_neutral_features_v3({**raw,'ball_xy':ball},contract)
        speed=b['X'][:,b['feature_names'].tolist().index('ball_speed_mps')]
        np.testing.assert_allclose(speed[19:32],1)

    def test_long_jump_gap_half_and_players_rejected(self):
        for case in ['long','jump','gap','half','players','turnover','identity']:
            r=fixture();r['ball_xy'][20:30]=np.nan
            if case=='long':r['ball_xy'][20:50]=np.nan
            if case=='jump':r['ball_xy'][30:]+=20
            if case=='gap':r['frame'][25:]+=1
            if case=='half':r['period'][25:]=2
            if case=='players':r['players_xy'][25]=np.nan
            if case=='turnover':r['player_team']=np.tile(r['player_team'],(80,1));r['player_team'][30,0]=1
            if case=='identity':r['player_track_id']=np.tile(r['player_track_id'],(80,1));r['player_track_id'][25,0]=99
            self.assertFalse(reconstruct(r,fps=25)[1]['imputed'].any(),case)

    def test_complete_observations_unchanged_and_labels_rejected(self):
        r=fixture();ball,m=reconstruct(r,fps=25)
        np.testing.assert_array_equal(ball,r['ball_xy']);self.assertFalse(m['imputed'].any())
        with self.assertRaises(ValueError):reconstruct({**r,'y_alive':np.ones(80)},fps=25)


if __name__=='__main__':unittest.main()
