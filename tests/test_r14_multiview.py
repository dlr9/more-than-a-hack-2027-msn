import unittest
from dataclasses import replace
import numpy as np
from scripts.research_r14_multiview import make_view
from multiview.align import align
from multiview.fuse import fuse
from multiview.fingerprint import fingerprint
from multiview.camera import camera_proxies,project
from solution.interception import interception,MotionBounds
from solution.group_reactions import group_reactions
from solution.multimodal_evidence import EvidencePacket,EditProposal,apply_edits

class MultiviewTests(unittest.TestCase):
 def test_self_align_and_team_permutation(self):
  a=make_view(100);r=align(a,a);np.testing.assert_array_equal(r['index'],np.arange(100))
  q=np.arange(12)[::-1];np.testing.assert_allclose(fingerprint(a.players,a.teams),fingerprint(a.players[:,q],1-a.teams[:,q]),atol=1e-14)
 def test_replay_and_no_anchor_refusal(self):
  a=make_view(40);b=replace(a,replay=np.ones(40,bool));self.assertTrue((align(a,b)['index']==-1).all())
  b=replace(a,ball=np.full_like(a.ball,np.nan),ball_observed=np.zeros(40,bool),clock_s=None)
  self.assertTrue((align(b,b)['index']==-1).all())
 def test_unknown_height_no_fill(self):
  a=make_view(40);b=replace(a,ball_grounded=None);a=replace(a,ball_observed=np.zeros(40,bool),ball=np.full_like(a.ball,np.nan));r=align(a,b)
  self.assertEqual(fuse(a,b,r,enabled=True,validated=True)['coverage_gain'],0)
 def test_exact_off_and_conflict(self):
  a=make_view(40);b=replace(a,ball=a.ball+.1);r=align(a,b)
  self.assertIs(fuse(a,b,r)['ball'],a.ball);self.assertIs(fuse(a,b,r,enabled=True)['ball'],a.ball)
  out=fuse(a,b,r,enabled=True,validated=True);np.testing.assert_array_equal(out['ball'],a.ball);self.assertGreater(out['conflict'].sum(),0)
 def test_ambiguous_static_shapes(self):
  a=make_view(40);a=replace(a,players=np.repeat(a.players[:1],40,axis=0));self.assertTrue((align(a,a)['index']==-1).all())
 def test_homography_horizon(self):
  with self.assertRaises(ValueError):project(np.zeros((3,3)),np.array([[.5,.5]]))
 def test_camera_cut_reset(self):
  H=np.repeat(np.diag([1920.,1080.,1.])[None],4,axis=0);H[2:,0,2]=800
  r=camera_proxies(H,np.ones(4,bool),np.zeros(4),np.arange(4),np.ones(4),np.zeros(4),25,1920,1080)
  self.assertTrue(r['cut_suspected'][2]);self.assertTrue(np.isnan(r['camera_pan_proxy_pxps'][2]).all())
 def test_ttc_is_not_control_and_unknown(self):
  p=np.array([[0.,0.],[10.,0.]]);v=np.zeros_like(p);b=MotionBounds(8,4,0,'synthetic')
  r=interception(p,v,[0,1],[1,0],[0,0],b,grounded=True);self.assertLess(r['arrival_s'][0],r['arrival_s'][1]);self.assertIsNone(r['acquisition_probability'])
  self.assertEqual(interception(p,v,[0,1],[np.nan,0],[0,0],b,grounded=True)['status'],'UNKNOWN')
 def test_group_swap_permutation_and_reverse_velocity(self):
  a=make_view(10);v=np.full_like(a.players,2);q=np.arange(12)[::-1]
  r=group_reactions(a.players,v,a.teams)['features'];s=group_reactions(a.players[:,q],v[:,q],1-a.teams[:,q])['features'];np.testing.assert_allclose(r,s[:,::-1])
  back=group_reactions(a.players,-v,a.teams)['features'];np.testing.assert_allclose(back[:,:,:2],-r[:,:,:2])
 def test_bounded_edits_and_all_off_modes(self):
  b=np.array(list('H'*100));period=np.ones(100,int);e=EvidencePacket(40,1,'synthetic','observed',.99,.01,'synthetic',1.,'test');q=EditProposal(38,43,'A',e)
  for kwargs in [dict(),dict(new_feature=True,trust=0,model_off=False),dict(new_feature=True,trust=1,model_off=True)]:self.assertIs(apply_edits(b,period,[q],**kwargs),b)
  p=apply_edits(b,period,[q],new_feature=True,trust=1,model_off=False,calibrated_gate=lambda x:True,max_changed_fraction=.1);self.assertEqual(np.sum(p!=b),5);self.assertTrue((p[43:]=='H').all())
  q=EditProposal(38,43,'D',e);np.testing.assert_array_equal(apply_edits(b,period,[q],new_feature=True,trust=1,model_off=False,calibrated_gate=lambda x:True,max_changed_fraction=.1),b)
if __name__=='__main__':unittest.main()
