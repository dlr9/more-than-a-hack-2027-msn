import itertools
import unittest
from unittest.mock import patch
import numpy as np
from fusion_v3 import DEFAULT, prepare, observations, decode, predict
from optimize_v3 import verify_reuse, candidates, masked


def fixture(n=100):
    return dict(ball=np.tile([20.,20.],(n,1)),players=np.tile([[20.,20.],[30.,20.]],(n,1,1)),teams=np.array([0,1]),mean_speed=np.full(n,.2),moving=np.zeros(n),tags=np.zeros((n,3)),slow=np.ones(n),speed_q25=np.full(n,.1),speed_q75=np.full(n,.3))

class V3Tests(unittest.TestCase):
    def test_dynamic_viterbi_exhaustive(self):
        obs=np.array([[2.,0,0],[0,0,4],[0,3,0],[0,3,0]])
        stop=np.array([0.,1.,0.,0.]); restart=np.array([0.,0.,1.,0.])
        c=dict(DEFAULT,team_penalty=2.,enter_penalty=3.,exit_penalty=4.,dynamic=.5)
        def score(path):
            value=sum(obs[i,s] for i,s in enumerate(path))
            for i,(a,b) in enumerate(zip(path,path[1:]),1):
                if a==b: continue
                if b==2: value-=3*(1-.5*stop[i])
                elif a==2: value-=4*(1-.5*restart[i])
                else: value-=2
            return value
        result=decode(obs,stop,restart,c)
        self.assertAlmostEqual(score(result),max(map(score,itertools.product(range(3),repeat=4))))

    def test_visible_stationary_adds_dead_evidence(self):
        f=fixture()
        with patch('fusion_v3.BLOCKS',[(0,100)]):
            p=prepare(f)
            on=observations(f,p,DEFAULT)[0]
            off=observations(f,p,dict(DEFAULT,visible_weight=0))[0]
            self.assertGreater(on[50,2],off[50,2])

    def test_distant_players_reduce_control(self):
        f=fixture(); f['players']+=50
        with patch('fusion_v3.BLOCKS',[(0,100)]):
            p=prepare(f)
            on=observations(f,p,DEFAULT)[0]
            off=observations(f,p,dict(DEFAULT,confidence=0))[0]
            self.assertLess(abs(on[50,0]),abs(off[50,0]))

    def test_missing_team_neutralizes_control(self):
        f=fixture(); f['players'][:,1]=np.nan
        with patch('fusion_v3.BLOCKS',[(0,100)]):
            p=prepare(f)
            self.assertTrue(np.all(p['confidence']==0))
            self.assertTrue(np.isfinite(observations(f,p,DEFAULT)[0]).all())

    def test_gaps_jumps_and_half_reset(self):
        f=fixture(); f['ball'][10:15]=np.nan; f['ball'][30:70]=np.nan
        f['ball'][80]=[10000,10000]
        with patch('fusion_v3.BLOCKS',[(0,45),(55,100)]):
            p=prepare(f,25)
            self.assertTrue(p['repaired'][12]); self.assertFalse(p['repaired'][40])
            self.assertTrue(np.isnan(p['speed'][55]))
            self.assertTrue(np.isnan(p['speed'][80]))
            labels=predict(f,DEFAULT,p)
            self.assertTrue(np.all(labels[45:55]=='X'))
            self.assertFalse(np.any(labels[:45]=='X'))

    def test_restart_to_other_team(self):
        obs=np.repeat(np.eye(3)[[0,2,1]]*5,80,axis=0)
        result=decode(obs,np.zeros(240),np.ones(240),DEFAULT)
        self.assertEqual(result[40],0); self.assertEqual(result[120],2); self.assertEqual(result[200],1)

    def test_invalid_config(self):
        with self.assertRaises(ValueError): predict(fixture(),dict(DEFAULT,distance=float('nan')))
        with self.assertRaises(ValueError): predict(fixture(),dict(DEFAULT,gap=-1))

    def test_reuse_rejects_stale_fingerprint(self):
        verify_reuse('same','same')
        with self.assertRaises(ValueError): verify_reuse('old-data','new-data')

    def test_search_fixed_and_bounded(self):
        self.assertEqual(candidates(),candidates()); self.assertEqual(len(candidates()),120)

    def test_scoring_masks_do_not_change_predictions(self):
        f=fixture()
        with patch('fusion_v3.BLOCKS',[(0,100)]):
            labels=predict(f,DEFAULT)
            copied=labels.copy()
            masked(np.zeros(100,dtype=np.int8),25,75)
            np.testing.assert_array_equal(labels,copied)

if __name__=='__main__': unittest.main()
