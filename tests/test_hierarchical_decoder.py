import unittest
import numpy as np
from solution.hierarchical_decoder import decode, binary_decode, anchor_probabilities
from solution.decoder import emission_probabilities


class HierarchicalTests(unittest.TestCase):
    def test_factorization_removes_team_uncertainty_dead_bias(self):
        q, r = np.full(100, .6), np.full(100, .51)
        self.assertTrue(np.all(emission_probabilities(q, r).argmax(1) == 2))
        self.assertTrue(np.all(decode(q, r, np.arange(100), np.ones(100), 25) == 'H'))

    def test_actual_low_alive_cannot_be_fixed_by_factorization(self):
        self.assertTrue(np.all(decode(np.full(195, .4272), np.full(195, .4626), np.arange(195), np.ones(195), 10) == 'D'))

    def test_dead_resets_home_and_gap_resets_alive(self):
        q = np.r_[np.full(20,.99), np.full(20,.01), np.full(20,.99)]
        r = np.r_[np.full(40,.99), np.full(20,.01)]
        out = decode(q,r,np.arange(60),np.ones(60),25)
        np.testing.assert_array_equal(out, np.array(list('H'*20+'D'*20+'A'*20)))
        out = decode([.99,.4], [.9,.9], [0,10], [1,1], 25)
        self.assertEqual(''.join(out),'HD')

    def test_time_scaling_and_outside(self):
        for fps in [10,25,50]:
            q=np.repeat([.99,.01,.99],fps)
            self.assertEqual(int(binary_decode(q,fps,.1).sum()),2*fps)
        self.assertEqual(decode([.99],[.9],[0],[0],25)[0],'X')

    def fixture(self, gap=10):
        n=40+gap; visible=np.ones(n,dtype=bool); visible[20:20+gap]=False
        q=np.full(n,.99);q[~visible]=.55
        return dict(alive=q,home=np.full(n,.95),frame=np.arange(n),period=np.ones(n),fps=10,
            visible=visible,home_distance=np.ones(n),away_distance=np.full(n,5.),
            ball_speed=np.ones(n),player_count=np.full(n,20))

    def test_bilateral_bounded_anchors(self):
        x=self.fixture();q,r,m=anchor_probabilities(**x)
        self.assertEqual(m['used'].sum(),10)
        self.assertTrue(np.all(q[20:30]>.55))
        self.assertFalse(anchor_probabilities(**self.fixture(31))[2]['used'].any())

    def test_reject_dead_half_gap_conflicting_or_absent_anchor(self):
        for change in ['dead','half','gap','conflict','absent']:
            x=self.fixture()
            if change=='dead':x['alive'][25]=.01
            if change=='half':x['period'][25:]=2
            if change=='gap':x['frame'][25:]+=10
            if change=='conflict':x['home'][30:]=.05;x['home_distance'][30:]=5;x['away_distance'][30:]=1
            if change=='absent':x['player_count'][:20]=0
            self.assertFalse(anchor_probabilities(**x)[2]['used'].any(),change)

    def test_no_observations_are_explicitly_unsupported(self):
        x=self.fixture();x['visible'][:]=False;x['player_count'][:]=0
        q,r,m=anchor_probabilities(**x)
        self.assertTrue(m['unsupported'].all());self.assertFalse(m['used'].any())
        np.testing.assert_array_equal(q,x['alive'])

    def test_first_reobserved_ball_uses_observed_forward_speed(self):
        x=self.fixture();x['ball_speed'][20:31]=np.nan
        self.assertEqual(anchor_probabilities(**x)[2]['used'].sum(),10)
        # A forward difference across a frame gap cannot establish velocity.
        x['frame'][31:]+=1
        self.assertFalse(anchor_probabilities(**x)[2]['used'].any())

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):decode([np.nan],[.5],[1],[1],25)
        with self.assertRaises(ValueError):binary_decode([.5],0,.1)
        with self.assertRaises(TypeError):anchor_probabilities(**self.fixture(), y_alive=[1])


if __name__=='__main__':unittest.main()
