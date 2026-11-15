import unittest
import numpy as np
from scripts.diagnose_r9_spell_loss import budget, repairs, snap_boundaries
from evaluation import possession_metrics as pm

class SpellBudgetTests(unittest.TestCase):
    def calc(self,g,p):
        gt=pm.label_codes(list(g));pr=pm.label_codes(list(p));n=len(g)
        return budget(gt,pr,np.ones(n)*.8,[(0,n)],10,np.ones(n,bool),np.ones(n)*22,np.ones(n,bool))
    def test_false_cut_changes_spell_matching(self):
        r=self.calc('HHHHHHHHHH','HHHHDDHHHH')
        self.assertEqual(r['false_D_cut_count'],1)
        self.assertEqual(r['oracle']['false_D']['mean_f1'],1)
        self.assertGreater(r['oracle']['false_D']['delta_spell'],0)
    def test_missed_dead_joins_spells(self):
        r=self.calc('HHHDDHHHHH','HHHHHHHHHH')
        self.assertEqual(r['true_D_joined_by_one_pred_spell'],1)
        self.assertEqual(r['complete_missed_D'],1)
        self.assertEqual(r['oracle']['missed_D']['mean_f1'],1)
    def test_oracles_not_additive_and_no_input_mutation(self):
        g=pm.label_codes(list('HHHAAA'));p=pm.label_codes(list('DDDDDD'));old=p.copy()
        r,_=repairs(g,p,np.ones(6)*.8,[(0,6)],10,np.ones(6,bool))
        np.testing.assert_array_equal(p,old)
        np.testing.assert_array_equal(r['both'],g)
        np.testing.assert_array_equal(r['team'],p)
        self.assertNotEqual(self.calc('HHHAAA','DDDDDD')['D_HA_interaction'],0)
    def test_timing_keeps_types_and_half_boundaries(self):
        g=pm.label_codes(list('HHHAAAHHHAAA'));p=pm.label_codes(list('HHHHAAHHAAAA'))
        q,n=snap_boundaries(g,p,[(0,6),(6,12)],2)
        self.assertEqual(n,2)
        np.testing.assert_array_equal(q,g)
    def test_short_spells_and_wrong_switches(self):
        r=self.calc('HHHAHHHHHH','HHHHHHHHHH')
        self.assertEqual(r['short_spells']['lt0p5']['no_same_team_overlap'],1)
        r=self.calc('HHHHHHHHHH','HHHAAHHHHH')
        self.assertEqual(r['HA_switches_inside_constant_GT_spell'],2)
    def test_timing_never_collapses_short_segment(self):
        g=pm.label_codes(list('HHAAHHAAAA'));p=pm.label_codes(list('HHHAAHAAAA'))
        q,_=snap_boundaries(g,p,[(0,10)],3)
        self.assertEqual([s[2] for s in pm.segments(q,[(0,10)])],[s[2] for s in pm.segments(p,[(0,10)])])
