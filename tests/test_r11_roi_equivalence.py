import unittest
import numpy as np
from r11.spell_roi import score,_scored_runs,audit_edit
from evaluation import possession_metrics as pm

class ROIEquivalence(unittest.TestCase):
    def compare(self,g,p,blocks=None):
        runs=_scored_runs(list(g))
        if blocks is not None:runs=[(max(a,c),min(b,d)) for a,b in blocks for c,d in runs if max(a,c)<min(b,d)]
        official=pm.spell_scores(pm.label_codes(g),pm.fill_abstentions(pm.label_codes(p),runs),runs)
        r=score(g,p,blocks=blocks)
        self.assertEqual(r.n_gt,official['n_gt']);self.assertEqual(r.n_pred,official['n_pred'])
        self.assertAlmostEqual(r.spell_f1_mean,official['mean_f1'],places=15)
        for j,t in enumerate(pm.SPELL_IOUS):
            self.assertEqual(r.matched_by_iou[j],official['per_iou'][t]['n_matched'])
            self.assertEqual(r.f1_by_iou[j],official['per_iou'][t]['f1'])
    def test_random_multihalf_and_unknown(self):
        rng=np.random.default_rng(111601)
        for i in range(1000):
            n=int(rng.integers(4,240));g=rng.choice(list('HADX'),n);p=rng.choice(list('HADX'),n);cut=int(rng.integers(1,n))
            self.compare(g,p,[(0,cut),(cut,n)])
    def test_long_spells_random_edits(self):
        rng=np.random.default_rng(111602)
        for i in range(200):
            g=np.repeat(rng.choice(list('HAD'),12),rng.integers(1,20,12));p=g.copy();ix=rng.integers(0,len(g),20);p[ix]=rng.choice(list('HADX'),20)
            self.compare(g,p)
    def test_same_team_adjacent_halves(self):
        self.compare(list('HHHH'),list('HHHH'),[(0,2),(2,4)])
        self.assertEqual(score('HHHH','HHHH',blocks=[(0,2),(2,4)]).n_gt,2)
    def test_empty_all_dead_all_unknown(self):
        for g,p in [([],[]),('DDDD','DDDD'),('XXXX','HHHH'),('HHHH','XXXX'),('HDDH','HXXH')]:self.compare(g,p)
    def test_invalid_blocks(self):
        for b in [[(0,3),(2,4)],[(0,5)],[(2,1)]]:
            with self.assertRaises(ValueError):score('HHHH','HHHH',blocks=b)
    def test_edit_respects_halves(self):
        r=audit_edit('HHAA','HHHH',2,4,'A',blocks=[(0,2),(2,4)])
        self.assertEqual(r['edited']['spell_f1_mean'],1.)
if __name__=='__main__':unittest.main()
