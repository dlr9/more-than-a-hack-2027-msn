import itertools
import unittest
from unittest.mock import patch
import numpy as np
from fusion_v2 import DEFAULT, decode, observations, predict, rolling

class TemporalTests(unittest.TestCase):
    def test_viterbi_matches_exhaustive_paths(self):
        score=np.array([[2.,0,0],[0,3,0],[0,0,4],[0,0,2]])
        cfg=dict(DEFAULT,team_penalty=1.,dead_penalty=2.)
        def value(path):
            cost=sum(score[i,s] for i,s in enumerate(path))
            for a,b in zip(path,path[1:]):
                if a!=b: cost-=2 if 2 in (a,b) else 1
            return cost
        self.assertAlmostEqual(value(decode(score,cfg)),max(map(value,itertools.product(range(3),repeat=4))))

    def test_dead_restart_changes_team(self):
        score=np.repeat(np.eye(3)[[0,2,1]]*4,100,axis=0)
        for decoder in ['state','viterbi']:
            result=decode(score,dict(DEFAULT,decoder=decoder))
            self.assertEqual(result[50],0)
            self.assertEqual(result[150],2)
            self.assertEqual(result[250],1)

    def test_missing_speed_not_stationary(self):
        self.assertTrue(np.isnan(rolling(np.full(10,np.nan),2)).all())
        np.testing.assert_allclose(rolling([2,np.nan,2],1),2)

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError): decode(np.array([[np.nan,0,0]]),DEFAULT)

    def test_short_gap_repair_but_not_long_gap(self):
        n=100
        f=dict(ball=np.tile([20.,20.],(n,1)),players=np.tile([[20.,20.],[40.,20.]],(n,1,1)),teams=np.array([0,1]),mean_speed=np.ones(n),moving=np.ones(n),tags=np.zeros((n,3)))
        f['ball'][10:15]=np.nan
        f['ball'][30:90]=np.nan
        with patch('fusion_v2.BLOCKS',[(0,100)]):
            repaired=observations(f,dict(DEFAULT,window=0))
            raw=observations(f,dict(DEFAULT,gap=0,window=0))
            self.assertGreater(repaired[12,0],raw[12,0])
            self.assertEqual(repaired[60,0],raw[60,0])

    def test_switch_penalty_suppresses_single_frame_noise(self):
        score=np.tile([1.,0.,0.],(100,1))
        score[50]=[0.,3.,0.]
        self.assertTrue(np.all(decode(score,DEFAULT)==0))

    def test_missing_observations_and_half_boundaries(self):
        n=100
        f=dict(ball=np.tile([20.,20.],(n,1)),players=np.full((n,2,2),np.nan),teams=np.array([0,1]),mean_speed=np.full(n,np.nan),moving=np.full(n,np.nan),tags=np.zeros((n,3)))
        f['ball'][10:20]=np.nan
        f['ball'][30:90]=np.nan
        with patch('fusion_v2.BLOCKS',[(0,40),(60,100)]):
            self.assertTrue(np.isfinite(observations(f,DEFAULT)).all())
            labels=predict(f,DEFAULT)
            self.assertTrue(np.all(labels[40:60]=='X'))
            self.assertFalse(np.any(labels[:40]=='X'))

if __name__ == '__main__': unittest.main()
