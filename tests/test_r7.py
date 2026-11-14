import json
import unittest
from unittest.mock import patch
import numpy as np
from solution.r7_features import project, blind_intervals, DIRECT, GEOMETRY_EXTRA
from solution.provider_neutral import FeatureContract, provider_neutral_features_v3
from solution.r7_metrics import extended
from scripts.research_r7 import checked_path, model_path

class R7Tests(unittest.TestCase):
    def batch(self):
        n=240
        raw=dict(frame=np.arange(n),period=np.repeat([1,2],n//2),ball_xy=np.column_stack([np.arange(n)*.01,np.zeros(n)]),
                 players_xy=np.broadcast_to(np.array([[0.,0.],[2.,0.]]),(n,2,2)).copy(),player_team=np.array([0,1]))
        b=provider_neutral_features_v3(raw,FeatureContract('IDSSE_DFL','available_unknown',25.))
        return raw,b
    def test_schema_and_direct_feature_intervention(self):
        _,b=self.batch();before=b['X'].copy();p=project(b,'no_ball_motion')
        self.assertEqual(int(p['schema_version']),4);self.assertEqual(p['X'].shape[1],39)
        self.assertFalse(DIRECT.intersection(p['feature_names']))
        changed={**b,'X':b['X'].copy()}
        for k in DIRECT:changed['X'][:,b['feature_names'].tolist().index(k)]=999
        np.testing.assert_equal(p['X'],project(changed,'no_ball_motion')['X'])
        np.testing.assert_equal(before,b['X'])
        self.assertEqual(project(b,'geometry_context')['X'].shape[1],31)
        with self.assertRaises(ValueError):project(p,'no_ball_motion')
    def test_mask_reproducible_period_scoped_and_label_free(self):
        p=np.repeat([1,2],2000)
        a,windows=blind_intervals(p,25.,scheme='short',seed=7101)
        b,_=blind_intervals(p,25.,scheme='short',seed=7101)
        np.testing.assert_equal(a,b)
        self.assertTrue(a.any());self.assertTrue((~a).any())
        for lo,hi,period in windows:self.assertTrue(np.all(p[lo:hi]==period))
        c,_=blind_intervals(p,25.,scheme='short',seed=7102)
        self.assertFalse(np.array_equal(a,c))
    def test_raw_mask_recomputes_motion_and_distances(self):
        raw,b=self.batch();mask=np.zeros(240,dtype=bool);mask[30:70]=True
        c=provider_neutral_features_v3(raw,FeatureContract('IDSSE_DFL','available_unknown',25.),ball_dropout_mask=mask)
        names=c['feature_names'].tolist()
        for name in ['ball_x','ball_speed_mps','home_d1_m','ball_displacement_0p2s_m']:
            self.assertTrue(np.isnan(c['X'][50,names.index(name)]))
        self.assertTrue(np.isnan(c['X'][70,names.index('ball_speed_mps')]))
        self.assertTrue(np.isfinite(raw['ball_xy']).all())
    def test_unknown_and_target_sources_rejected_before_read(self):
        for m in ['J03WMX','BAR-ATH','../../evaluation/gt']:
            with self.assertRaises(ValueError):checked_path(m,'natural')
        with self.assertRaises(ValueError):checked_path('J03WPY','../gt.csv')
        self.assertTrue(str(model_path('J03WPY','M2')).endswith('J03WPY-M2_dropout.joblib'))
    def test_dead_misses_and_short_spell_denominators(self):
        _,b=self.batch();truth=np.array(list('H'*30+'D'*20+'A'*70+'H'*120))
        b.update(y_alive=(truth!='D').astype(int),y_home=np.where(truth=='D',-1,(truth=='H').astype(int)))
        pred=truth.copy();pred[30:50]='H';pred[10:15]='D'
        a=np.where(pred=='D',.05,.95);h=np.where(pred=='H',.95,.05)
        score=extended(b,a,h,pred)
        self.assertEqual(score['dead_errors']['missed_segments'],1)
        self.assertEqual(score['dead_errors']['false_cuts'],1)
        self.assertEqual(score['dead_errors']['missed_dead_frames'],20)
        self.assertEqual(score['dead_errors']['gt_dead_segments'],1)
        self.assertEqual(score['boundaries_pair_1s']['enter_dead']['n_matched'],1)
        b['y_alive'][5]=-1
        with self.assertRaises(ValueError):extended(b,a,h,pred)

if __name__=='__main__':unittest.main()
