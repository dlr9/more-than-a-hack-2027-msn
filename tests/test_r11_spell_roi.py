import unittest
from r11.spell_roi import score,audit_edit

class ScoreContractTests(unittest.TestCase):
    def test_identical_with_two_teams(self):
        truth=list('HHHAAADDDHHHAAD')
        r=score(truth,truth)
        self.assertEqual(r.spell_f1_mean,1.)
        self.assertEqual(r.n_gt,4)
    def test_same_team_restart_is_two_distinct_spells(self):
        truth=list('HHHDDDHHH')
        self.assertEqual(score(truth,truth).n_gt,2)
        self.assertLess(score(truth,list('HHHHHHHHH')).spell_f1_mean,1.)
    def test_x_prediction_forward_fills_inside_possession(self):
        truth=list('HHHHAAAA')
        p=list('HXXXAAAA')
        self.assertEqual(score(truth,p).spell_f1_mean,1.)
    def test_unknown_initial_backfills_not_new_segment(self):
        truth=list('HHHAA')
        p=list('XXHAA')
        self.assertEqual(score(truth,p).spell_f1_mean,1.)
    def test_two_graded_halves_never_merge(self):
        truth=list('HHHXXXXHHH')
        self.assertEqual(score(truth,truth).n_gt,2)
    def test_single_flip_oversegmenting_lowers_f1(self):
        truth=list('H'*30)
        p=list('H'*30);p[15]='A'
        self.assertLess(score(truth,p).spell_f1_mean,score(truth,truth).spell_f1_mean)
    def test_edit_can_improve_exact_spell_iou(self):
        truth=list('H'*20+'A'*20)
        pred=list('H'*40)
        d=audit_edit(truth,pred,20,40,'A')
        self.assertGreater(d['delta_spellF1'],0.)
        self.assertEqual(d['delta_pred_spells'],1)
    def test_overlong_merge_can_hurt_when_frame_classification_good(self):
        gt=list('H'*25+'A'*5+'H'*25)
        pred=list('H'*55)
        self.assertLess(score(gt,pred).spell_f1_mean,1.)
    def test_wrong_edit_cannot_claim_improvement(self):
        gt=list('H'*20+'A'*20)
        d=audit_edit(gt,gt,5,25,'A')
        self.assertLess(d['delta_spellF1'],0.)
    def test_type_safety(self):
        with self.assertRaises(ValueError):score(['H'],['K'])
        with self.assertRaises(ValueError):audit_edit(['H'],['H'],0,2,'A')

if __name__=='__main__':unittest.main()
