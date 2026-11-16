import unittest
import numpy as np
from r11p5.timeline import targets,decode,features,PAIRS
class TimelineTests(unittest.TestCase):
 def test_negatives_and_restart(self):
  y=np.array(list('H'*100+'A'*100+'D'*40+'A'*100));h,t,c=targets(y,np.arange(len(y)),np.ones(len(y)),25)
  self.assertEqual(c,dict(direct=1,enter_D=1,exit_D=1,same_team_restart=1));self.assertEqual(h[100],1);self.assertEqual(h[50],0);self.assertEqual(h[200],-1)
 def test_decode(self):
  n=200;b=np.array(['H']*n);h=np.zeros(n);h[100]=.99;p=np.r_[np.zeros(100),np.ones(100)]
  out,used=decode(b,h,p,np.arange(n),np.ones(n),25);self.assertEqual(used,[100]);self.assertTrue((out[100:]=='A').all())
  out,_=decode(b,h,np.zeros(n),np.arange(n),np.ones(n),25);np.testing.assert_array_equal(out,b)
 def test_dead_short_preserved(self):
  b=np.array(list('H'*10+'D'*30+'A'*60));out,_=decode(b,np.ones(100),np.ones(100),np.arange(100),np.ones(100),25);np.testing.assert_array_equal(out,b)
 def test_swap(self):
  names=[s+'_'+k for s in ('home','away') for k in PAIRS]+['n_home_positions','n_away_positions','ball_visible'];x=np.random.default_rng(2).normal(size=(100,len(names)))
  b=dict(X=x,feature_names=np.array(names),frame=np.arange(100),period=np.ones(100),fps=25.)
  z=x.copy()
  for name in names:
   other=name.replace('home','away') if 'home' in name else name.replace('away','home')
   z[:,names.index(name)]=x[:,names.index(other)]
  a,s=features(b);aa,ss=features(dict(b,X=z));np.testing.assert_array_equal(a,aa);np.testing.assert_array_equal(s,-ss)
 def test_reject_labels(self):
  with self.assertRaises(ValueError):features({'y_home':np.ones(3)})
