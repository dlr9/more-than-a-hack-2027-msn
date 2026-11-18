import unittest
import numpy as np
from research_only.oracle_bottleneck import ExternalTruth, interventions
from r11.spell_roi import audit_edit

class OracleTests(unittest.TestCase):
    def setUp(self):
        self.g=np.array(list('HHADDDAHHAA'))
        self.b=np.array(list('HHHAAADDDAA'))
        self.h=np.linspace(.1,.9,len(self.g))
    def run_arms(self, blocks=None, fps=25):
        return interventions(self.b,self.h,ExternalTruth(self.g,'synthetic'),
                             [(0,len(self.g))] if blocks is None else blocks,fps)
    def test_factorial_and_inputs_unchanged(self):
        original=self.b.copy(); arms=self.run_arms()
        self.assertEqual(len(arms),12)
        np.testing.assert_array_equal(arms['A_dead+B_times+C_owner'],self.g)
        np.testing.assert_array_equal(arms['baseline'],original)
        np.testing.assert_array_equal(self.b,original)
        np.testing.assert_array_equal(arms['A_dead']=='D',self.g=='D')
        np.testing.assert_array_equal(arms['C_owner']=='D',self.b=='D')
    def test_generator_blocks_and_half_boundaries(self):
        a=self.run_arms([(0,5),(5,11)])
        b=self.run_arms(iter([(0,5),(5,11)]))
        for k in a:np.testing.assert_array_equal(a[k],b[k])
    def test_invalid_contract(self):
        for blocks in [[(1,11)],[(0,5)],[(0,7),(6,11)],[(0,5.5),(5.5,11)]]:
            with self.assertRaises(ValueError):self.run_arms(blocks)
        for fps in [0,-1,np.inf,np.nan]:
            with self.assertRaises(ValueError):self.run_arms(fps=fps)
        with self.assertRaises(ValueError):ExternalTruth(self.g,'target')
        with self.assertRaises(ValueError):ExternalTruth(np.array(['X']),'synthetic')
    def test_truth_copies_input(self):
        truth=ExternalTruth(self.g,'synthetic');self.g[0]='A'
        self.assertEqual(truth.labels[0],'H')
        with self.assertRaises(ValueError):truth.labels[0]='A'
    def test_random_abc_recovery(self):
        rng=np.random.default_rng(401)
        for _ in range(100):
            g=rng.choice(list('HAD'),50);b=rng.choice(list('HAD'),50)
            out=interventions(b,rng.random(50),ExternalTruth(g,'synthetic'),[(0,23),(23,50)],25)
            np.testing.assert_array_equal(out['A_dead+B_times+C_owner'],g)
    def test_gained_and_lost_not_only_net(self):
        # Same net TP can hide replacement of one true match with another.
        r=audit_edit('HHAA','HHHH',0,4,'A')
        c=r['matches_by_iou'][.5]
        self.assertEqual(c['lost'],[dict(start=0,end=2,team='H')])
        self.assertEqual(c['gained'],[dict(start=2,end=4,team='A')])
        self.assertEqual(r['delta_matched_iou_05'],0)
    def test_short_interventions_respect_halves_and_strict_duration(self):
        g=np.array(list('HHHH')); b=np.array(list('AAAA'))
        out=interventions(b,np.full(4,.2),ExternalTruth(g,'synthetic'),[(0,2),(2,4)],3)
        np.testing.assert_array_equal(out['D_short_lt1'],g)
        whole=interventions(b,np.full(4,.2),ExternalTruth(g,'synthetic'),[(0,4)],4)
        np.testing.assert_array_equal(whole['D_short_lt1'],b)
    def test_random_edit_gain_loss_counts(self):
        rng=np.random.default_rng(402)
        for _ in range(100):
            g=rng.choice(list('HADX'),30);p=rng.choice(list('HADX'),30)
            lo=int(rng.integers(0,29));hi=int(rng.integers(lo+1,31))
            r=audit_edit(g,p,lo,hi,'H',blocks=[(0,15),(15,30)])
            for t,c in r['matches_by_iou'].items():
                self.assertEqual(len(c['gained'])-len(c['lost']),
                    r['edited']['matched_by_iou'][t]-r['baseline']['matched_by_iou'][t])
    def test_edit_indices(self):
        with self.assertRaises(ValueError):audit_edit('HH','HH',.5,2,'A')

if __name__=='__main__':unittest.main()
