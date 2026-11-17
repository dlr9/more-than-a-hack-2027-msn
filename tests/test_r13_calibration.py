import unittest
import numpy as np
from r12.control_calibration import correct,pool,eligible_prior

class CalibrationTests(unittest.TestCase):
 def test_neutral_identity(self):
  h=np.linspace(.01,.99,100);p=np.ones((100,3))/3
  np.testing.assert_allclose(pool(h,p),h,rtol=1e-14)
 def test_prior_recovers_uninformative(self):
  prior=np.array([.05,.05,.9]);np.testing.assert_allclose(correct([[1/3]*3],prior)[0],prior)
 def test_swap(self):
  p=np.array([[.8,.1,.1],[.1,.4,.5]]);h=np.array([.4,.6])
  np.testing.assert_allclose(pool(1-h,p[:,[1,0,2]]),1-pool(h,p))
 def test_unsampled_prior(self):
  prior=eligible_prior(dict(counts=[10,10,16],excluded=0),100)
  np.testing.assert_allclose(prior,[.1,.1,.8])
