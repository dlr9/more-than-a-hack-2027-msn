"""Synthetic regression tests for censored truth, physical time and candidate routing."""
import tempfile
import unittest
from pathlib import Path
import joblib
import numpy as np
from solution.candidate import Candidate, unweight_probability
from solution.evidence import (seconds_to_frames, score_sequence, period_reference,
                               visibility_strata, error_attribution, labels_from_batch)

class ConstantHead:
    def __init__(self, probability): self.probability = probability
    def predict_proba(self, x):
        p = np.full(len(x), self.probability)
        return np.column_stack([1-p, p])

class EvidenceTests(unittest.TestCase):
    def test_legacy_config_physical_time(self):
        from solution.config import timing_frames
        config={'short_team_run_frames':12}
        self.assertEqual(timing_frames(config,'short_team_run_frames',25),12)
        self.assertEqual(timing_frames(config,'short_team_run_frames',10),5)
        config['short_team_run_s']=1.
        self.assertEqual(timing_frames(config,'short_team_run_frames',10),10)

    def test_seconds_and_invalid_time(self):
        self.assertEqual(seconds_to_frames(.48,25),12)
        self.assertEqual(seconds_to_frames(.48,10),5)
        for seconds,fps in [(1,0),(-1,25),(1,np.nan)]:
            with self.assertRaises(ValueError): seconds_to_frames(seconds,fps)

    def test_partial_islands_cannot_certify_complete_spells(self):
        y=np.array(list('HHXXAA'));p=np.array(list('HHHHAA'));period=np.ones(6)
        result=score_sequence(y,p,period,10)
        self.assertEqual(result['n_label_islands'],2)
        self.assertIsNone(result['full_spell_f1'])
        self.assertIsNotNone(result['censored_island_spell_diagnostic'])
        with self.assertRaises(ValueError):
            score_sequence(y,p,period,10,semantics='independent_complete')
        # Complete phase proxies still cannot certify official spell quality.
        self.assertIsNone(score_sequence(p,p,period,10)['full_spell_f1'])

    def test_physical_transition_latency_and_duration(self):
        for fps in [10,25]:
            y=np.array(list('H'*(2*fps)+'A'*(2*fps)))
            p=np.array(list('H'*int(2.2*fps)+'A'*(4*fps-int(2.2*fps))))
            r=score_sequence(y,p,np.ones(len(y)),fps,semantics='synthetic_complete')
            self.assertEqual(r['full_spell_status'],'synthetic_test_only')
            self.assertAlmostEqual(r['partial_proxy_transitions'][0]['mean_latency_s'],.2)
            self.assertAlmostEqual(r['prediction_geometry']['mean_team_spell_s'],2)

    def test_unknown_windows_and_zero_support(self):
        y=np.array(list('HHHHHXA'*10));p=np.full(len(y),'H')
        r=score_sequence(y,p,np.ones(len(y)),10)
        for t in r['partial_proxy_transitions']: self.assertIsNone(t['f1'])

    def test_period_reference_never_smooths_across_halves(self):
        b=dict(frame=np.arange(8),fps=10,period=np.array([1]*4+[2]*4),nearest_codes=np.array([0]*4+[1]*4))
        self.assertEqual(''.join(period_reference(b)),'HHHHAAAA')

    def test_missing_gap_durations_reset_at_period_boundary(self):
        b=dict(X=np.array([[0,9,9]]*8),feature_names=np.array(['ball_visible','home_d1_m','away_d1_m']),fps=10,period=np.array([1]*4+[2]*4),nearest_codes=np.zeros(8))
        s=visibility_strata(b)
        self.assertTrue(s['missing_short'].all())
        self.assertFalse(s['missing_medium'].any())

    def test_prior_inverse_and_guard(self):
        np.testing.assert_allclose(unweight_probability(np.array([0,.5,1]),.8),[0,.8,1])
        with self.assertRaises(ValueError): unweight_probability([.5],0)

    def test_saved_candidate_gates_expert_and_checks_schema(self):
        b=dict(X=np.array([[1],[0],[1]]),feature_names=np.array(['ball_visible']),schema_version=2,period=np.array([1,1,0]),fps=10)
        model=Candidate(ConstantHead(.9),ConstantHead(.8),['ball_visible'],{'schema_version':2},ConstantHead(.1))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'candidate.joblib';joblib.dump(model,path);restored=joblib.load(path)
            a,h=restored.probabilities(b)
            np.testing.assert_allclose(h,[.8,.1,.8])
            self.assertEqual(restored.predict(b)[-1],'X')
        with self.assertRaises(ValueError): model.probabilities({**b,'schema_version':1})
        with self.assertRaises(ValueError): model.probabilities({**b,'feature_names':np.array(['wrong'])})

    def test_head_and_decoder_error_attribution(self):
        r=error_attribution(np.array(list('HAX')),np.array([.1,.9,.5]),np.array([.9,.1,.5]),np.array(list('HDA')))
        self.assertEqual(r['home_head_wrong'],0)
        self.assertEqual(r['joint_argmax_dead'],1)
        self.assertEqual(r['decoder_fixed_raw_wrong'],1)
        self.assertEqual(r['decoder_harmed_raw_correct'],1)

    def test_labels_outside_period_remain_unknown(self):
        b=dict(y_alive=np.array([0,1]),y_home=np.array([-1,1]),period=np.array([0,1]))
        self.assertEqual(''.join(labels_from_batch(b)),'XH')

if __name__=='__main__': unittest.main()
