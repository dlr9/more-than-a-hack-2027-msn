import unittest
import numpy as np
from r12.control_global import decode
class GlobalTests(unittest.TestCase):
 def test_swap(self):
  n=200;a=np.full(n,.99);h=np.r_[np.full(100,.99),np.full(100,.01)];s=np.r_[np.zeros(100,int),np.ones(100,int)];f=np.arange(n);p=np.ones(n)
  x,_,_=decode(a,h,s,f,p,25);y,_,_=decode(a,1-h,1-s,f,p,25)
  np.testing.assert_array_equal(y,np.where(x=='H','A',np.where(x=='A','H','D')))
 def test_restart(self):
  a=np.r_[np.full(100,.999),np.full(100,.001),np.full(100,.999)];h=np.full(300,.99);s=np.zeros(300,int)
  x,_,_=decode(a,h,s,np.arange(300),np.ones(300),25);self.assertEqual(x[50],'H');self.assertEqual(x[150],'D');self.assertEqual(x[250],'H')
 def test_future_recovers(self):
  n=500;s=np.zeros(n,int);s[100:105]=1
  x,_,_=decode(np.full(n,.99),np.full(n,.99),s,np.arange(n),np.ones(n),25)
  self.assertEqual(x[-1],'H')
 def test_short_allowed(self):
  n=30;s=np.r_[np.zeros(10,int),np.ones(10,int),np.zeros(10,int)];h=np.where(s==0,1.-1e-9,1e-9)
  x,_,_=decode(np.full(n,1.-1e-9),h,s,np.arange(n),np.ones(n),25)
  self.assertTrue((x[10:20]=='A').all())
 def test_reject(self):
  with self.assertRaises(ValueError):decode([2],[.5],[-1],[0],[1],25)
