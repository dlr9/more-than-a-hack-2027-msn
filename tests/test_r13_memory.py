import unittest
import numpy as np
from r12.control_memory import decode,transition_cost

class MemoryTests(unittest.TestCase):
 def test_no_dead_shortcut(self):
  for s in (-1,0,1):
   c=transition_cost(s)
   for old in range(4):
    for new in range(4):
     if old//2!=new//2:self.assertGreaterEqual(c[old,4+old//2]+c[4+old//2,new],c[old,new])
   self.assertTrue(np.isinf(c[4,5]) and np.isinf(c[5,4]))
 def test_real_opposite_restart_possible(self):
  a=np.r_[np.full(50,.999),np.full(100,.001),np.full(50,.999)]
  h=np.r_[np.full(100,.999),np.full(100,.001)];s=np.r_[np.zeros(100,int),np.ones(100,int)]
  x,_,_=decode(a,h,s,np.arange(200),np.ones(200),25)
  self.assertEqual((x[25],x[100],x[175]),('H','D','A'))
 def test_subsecond_not_forbidden(self):
  s=np.r_[np.zeros(10,int),np.ones(10,int),np.zeros(10,int)];h=np.where(s==0,1-1e-9,1e-9)
  x,_,_=decode(np.full(30,1-1e-9),h,s,np.arange(30),np.ones(30),25)
  self.assertTrue((x[10:20]=='A').all())
 def test_half_resets_memory(self):
  s=np.r_[np.zeros(100,int),np.ones(100,int)];h=np.where(s==0,.999,.001)
  x,_,_=decode(np.full(200,.999),h,s,np.arange(200),np.r_[np.ones(100),np.full(100,2)],25)
  self.assertTrue((x[:100]=='H').all() and (x[100:]=='A').all())
