import unittest
import numpy as np
from r12.dense_control import features,flatten,training_targets,fixed_dead_decode

class DenseTests(unittest.TestCase):
 def raw(self):
  n=100;t=np.arange(n)/25;ball=np.c_[t,np.zeros(n)];players=np.stack([ball+[1,0],ball+[5,0]],axis=1)
  return dict(ball_xy=ball,players_xy=players,player_team=np.array([0,1]),player_track_id=np.array([12,34]),frame=np.arange(n),period=np.ones(n))
 def test_swap_features(self):
  r=self.raw();x=features(r);r['player_team']=1-r['player_team'];y=features(r)
  np.testing.assert_allclose(y,x[:,::-1]);np.testing.assert_allclose(flatten(y),flatten(x,True))
 def test_missing_ball_finite(self):
  r=self.raw();r['ball_xy'][:]=np.nan;x=features(r)
  self.assertTrue(np.isfinite(x).all());self.assertEqual(x.shape,(100,2,24))
 def test_non_event_supervision(self):
  n=200;g=np.full(n,'H');ev=[(1,'home_1','pass',30,'home_2',50)]
  y,keep,a=training_targets(g,np.arange(n),np.ones(n),ev,np.arange(n),'human')
  self.assertEqual(a['accepted_event_anchors'],1);self.assertEqual(y[50],0);self.assertEqual(y[70],-1);self.assertTrue(keep[150] and y[150]==2)
 def test_frozen_dead_mask(self):
  base=np.array(list('HHHDDDAAA'));x=fixed_dead_decode(np.full(9,.99),base,np.arange(9),np.ones(9))
  np.testing.assert_array_equal(x=='D',base=='D')
