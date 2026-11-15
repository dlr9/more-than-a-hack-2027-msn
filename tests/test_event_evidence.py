import unittest
from solution.event_evidence import restart_proposals
class EventEvidenceTests(unittest.TestCase):
    def row(self,**kw):
        r=dict(code='SET PIECES',start_s=2.02,end_s=3,tags={'Type':'Throw-in','Half':'1st Half','Team':'N/A'});r.update(kw);return r
    def call(self,rows):return restart_proposals(rows,fps=25,n_frames=300,half_blocks={1:(0,120),2:(150,300)},team_names={'blue':'H','red':'A'})
    def test_rounding_case_unknown_team_and_duplicates(self):
        r=self.row();r['tags']={'type':'throw-IN','HALF':'1ST HALF'};out=self.call([r,r])
        self.assertEqual(len(out['proposals']),1);e=out['proposals'][0];self.assertEqual(e['frame'],round(2.02*25));self.assertIsNone(e['team']);self.assertFalse(out['full_phase_labels_produced'])
    def test_bounds_wrong_half_and_nonfinite(self):
        for r in [self.row(start_s=-1),self.row(end_s=8),self.row(start_s=float('nan')),self.row(tags={'Type':'Throw-in','Half':'2nd Half'})]:
            out=self.call([r]);self.assertEqual(out['proposals'],[]);self.assertTrue(out['rejected'])
    def test_overlapping_phase_never_becomes_possession(self):
        out=self.call([self.row(),self.row(code='BUILD UP',start_s=0,end_s=10)])
        self.assertEqual(len(out['proposals']),1);self.assertLess(out['proposals'][0]['window'][1]-out['proposals'][0]['window'][0],30)
    def test_conflicting_teams_rejected(self):
        a=self.row(tags={'Type':'Corner Kick','Half':'1st Half','Team':'blue'});b=self.row(tags={'Type':'Corner Kick','Half':'1st Half','Team':'red'})
        self.assertEqual(self.call([a,b])['proposals'],[])
    def test_wrong_direction_and_unknown_team(self):
        for tag in [{'Direction of ball entry':'backwards'},{'Team':'green'}]:
            r=self.row();r['tags'].update(tag);self.assertTrue(self.call([r])['rejected'])
    def test_case_collision_and_invalid_half_contract(self):
        r=self.row();r['tags']['type']='Free Kick';self.assertTrue(self.call([r])['rejected'])
        with self.assertRaises(ValueError):restart_proposals([],fps=25,n_frames=200,half_blocks={1:(0,110),2:(100,200)},team_names={})
