import unittest
import numpy as np
from r11.event_learning import temporal_inputs,supervised_rows

class EventLearningTests(unittest.TestCase):
    def batch(self):return dict(X=np.arange(12,dtype=float).reshape(6,2),feature_names=np.array(['n_home_positions','n_away_positions']),frame=np.arange(6),period=np.array([1,1,1,2,2,2]),fps=np.array(1.))
    def test_temporal_contract_and_halves(self):
        b=self.batch();x=temporal_inputs(b,[-1,0,1]);self.assertEqual(x.shape,(6,6));self.assertTrue(np.isnan(x[3,:2]).all());self.assertTrue(np.isnan(x[2,-2:]).all());np.testing.assert_equal(x[1,:2],b['X'][0])
    def test_frame_gap(self):
        b=self.batch();b['period'][:]=1;b['frame'][3:]+=4;x=temporal_inputs(b,[-1,1]);self.assertTrue(np.isnan(x[3,:2]).all())
    def test_label_metadata_forbidden(self):
        with self.assertRaises(ValueError):temporal_inputs({**self.batch(),'event_kind':np.zeros(6)},[0])
    def test_unknown_never_negative_and_zero_teams_excluded(self):
        b=self.batch();y=np.array([1,0,-1,1,0,-1]);self.assertEqual(supervised_rows(y,b).tolist(),[False,True,False,True,True,False])
if __name__=='__main__':unittest.main()
