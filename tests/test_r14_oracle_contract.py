import unittest
import numpy as np
from research_only.oracle_bottleneck import ExternalTruth,interventions
from solution.observation_contract import ObservationContract,track_velocity,audit_speed,validate_timeline

def contract(**kw):
 d=dict(source='synthetic',fps=25,frame_base=0,pitch_m=(100.,50.),coord_units='normalized',speed_units_raw='km/h',speed_units_decoded='m/s',homography_direction='pitch_normalized_to_pixel',field_orientation='fixed_pitch_axes',team_map=('H','A'),identity_type='tracklet',source_license='synthetic');d.update(kw);return ObservationContract(**d)
class ContractOracleTests(unittest.TestCase):
 def test_oracle_factorial(self):
  g=np.array(list('HHADDDAHHAA'));b=np.array(list('HHHAAADDDAA'));h=np.linspace(.1,.9,len(g));o=interventions(b,h,ExternalTruth(g,'synthetic'),[(0,len(g))],25)
  self.assertEqual(len(o),12);np.testing.assert_array_equal(o['baseline'],b);np.testing.assert_array_equal(o['A_dead+B_times+C_owner'],g)
  np.testing.assert_array_equal(o['A_dead']=='D',g=='D');np.testing.assert_array_equal(o['C_owner']=='D',b=='D')
 def test_research_truth_scope(self):
  with self.assertRaises(ValueError):ExternalTruth(np.array(list('HAD')),'target')
 def test_boundaries_and_id_reset(self):
  x=np.zeros((5,1,2));x[:,0,0]=np.arange(5)*.002;ids=np.array([[1],[1],[2],[2],[2]])
  r=track_velocity(x,[0,1,2,4,5],[1]*5,ids,contract())
  self.assertAlmostEqual(r['velocity_mps'][1,0,0],5);self.assertTrue(np.isnan(r['velocity_mps'][2:4]).all());self.assertEqual(r['tracklet_age'][:,0].tolist(),[1,2,1,1,2])
 def test_no_slot_identity_guess(self):
  r=track_velocity(np.zeros((3,1,2)),[0,1,2],[1,1,1],np.ones((3,1)),contract(identity_type='dynamic_slot_unknown'))
  self.assertTrue(np.isnan(r['velocity_mps']).all())
 def test_unit_conversion_not_inferred(self):
  r=audit_speed([18.],[5.],[5.],contract(),'synthetic explicit metadata');self.assertTrue(r['usable'])
  self.assertFalse(audit_speed([18.],[18.],[5.],contract(),'synthetic')['usable'])
  self.assertFalse(audit_speed([18.],[5.],[5.],contract(),'')['usable'])
 def test_time_mismatch(self):
  with self.assertRaises(ValueError):validate_timeline([0,1],[1,1],25,[0,.1])
 def test_missing_is_not_zero(self):
  x=np.zeros((3,1,2));x[1]=np.nan
  r=track_velocity(x,[0,1,2],[1,1,1],np.ones((3,1)),contract());self.assertTrue(np.isnan(r['velocity_mps']).all())
if __name__=='__main__':unittest.main()
