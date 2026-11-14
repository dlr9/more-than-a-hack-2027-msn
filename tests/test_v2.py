import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from solution.atd import read_atd, team_metadata
from solution.external import prepare_skillcorner_match
from solution.modeling import _truth_and_blocks, train_from_split
from solution.decoder import emission_probabilities, viterbi_decode
from solution.calibration import BinaryCalibration, probability_metrics
from solution.features import add_temporal_features
from solution.quality import dropout_windows
from solution.submission import write_submission

class V2Tests(unittest.TestCase):
    def read(self, text, n=2, **kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'synthetic.txt';p.write_text(text)
            return read_atd(p,np.array([0,1]),n,(105,68),10,[1,2],[.2],**kwargs)

    def test_reject_duplicate_missing_unordered_and_truncated(self):
        row=':0,0,0;1,1,0:0,0,0\n'
        for ids in [(1,1),(1,3),(2,1),(1,)]:
            with self.subTest(ids=ids),self.assertRaises(ValueError):
                self.read(''.join(str(i)+row for i in ids))

    def test_reject_bad_numeric_shape_and_infinity(self):
        for row in ['broken','1:0,0,0:0,0,0','1:0,0,0;1,1,0:inf,0,0','1:0,0,0;1,1,0:oops,0,0']:
            with self.subTest(row=row),self.assertRaises(ValueError):self.read(row,1)

    def test_coordinate_only_is_explicit(self):
        text='1:0,0;1,1:0,0\n'
        with self.assertRaises(ValueError):self.read(text,1)
        self.assertEqual(self.read(text,1,coordinates_only=True).nearest_codes.tolist(),[0])

    def test_unknown_metadata_team_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'meta.xml';p.write_text("<Root><Team id='h'/><Team id='a'/><Player teamId='x'/></Root>")
            with self.assertRaises(ValueError):team_metadata(p)

    def test_half_boundary_resets_derived_speed(self):
        b=self.read('1:0,0,0;1,1,0:0,0,NaN\n2:0,0,0;1,1,0:1,1,NaN\n',blocks=[(0,0),(1,1)])
        self.assertTrue(np.isnan(b.values['ball_speed_mps']).all())

    def test_decode_blocks_independent_of_label_coverage(self):
        b=dict(y_alive=np.array([1,-1,-1,1]),y_home=np.array([1,-1,-1,0]),period=np.ones(4))
        truth,blocks=_truth_and_blocks(b)
        self.assertEqual(blocks,[(0,3)]);self.assertEqual(''.join(truth),'HXXA')

    def test_invalid_probability_and_overlapping_blocks(self):
        with self.assertRaises(ValueError):emission_probabilities(np.array([np.nan]),np.array([.5]))
        for probs,blocks in [(np.array([[.5,.5,.5]]),[(0,0)]),(np.ones((3,3))/3,[(0,1),(1,2)])]:
            with self.assertRaises(ValueError):viterbi_decode(probs,blocks,1,1)

    def test_event_control_unknown_matches_fixed(self):
        p=emission_probabilities(np.full(4,.99),np.array([.9,.9,.1,.1]))
        fixed=viterbi_decode(p,[(0,3)],3,1)
        np.testing.assert_array_equal(fixed,viterbi_decode(p,[(0,3)],3,1,control=np.full(4,-1)))

    def test_event_control_does_not_create_dead(self):
        p=emission_probabilities(np.full(4,.999),np.array([.9,.9,.1,.1]))
        result=viterbi_decode(p,[(0,3)],3,1,control=np.array([0,0,1,1]))
        self.assertNotIn('D',result)

    def test_calibration_match_overlap_rejected(self):
        with self.assertRaises(ValueError):BinaryCalibration().fit([.2,.8],[0,1],training_ids=['a'],calibration_ids=['a'])

    def test_calibration_single_class_fallback(self):
        c=BinaryCalibration().fit([.2,.8],[1,1],training_ids=['a'],calibration_ids=['b'])
        self.assertEqual(c.status,'unavailable_single_class');np.testing.assert_array_equal(c.predict([.2,.8]),[.2,.8])

    def test_reliability_metrics_known_answer(self):
        m=probability_metrics([0,1],[.25,.75]);self.assertAlmostEqual(m['brier'],.0625);self.assertAlmostEqual(m['ece'],.25)

    def test_submission_rejects_truncated_label_and_matrix(self):
        with tempfile.TemporaryDirectory() as tmp:
            for y in [['HOME'],[['H']]]:
                with self.assertRaises(ValueError):write_submission(Path(tmp)/'x.csv',y)

    def test_dropout_respects_period_and_is_reproducible(self):
        period=np.r_[np.ones(200),np.zeros(100),np.ones(200)*2]
        a=dropout_windows(period,10);np.testing.assert_array_equal(a,dropout_windows(period,10));self.assertFalse(a[200:300].any());self.assertTrue(a.any())

    def test_phase_absence_unknown_observed_view_and_raw_dropout(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'fake';folder.mkdir()
            meta=dict(home_team=dict(id=1),away_team=dict(id=2),players=[dict(id=10,team_id=1),dict(id=20,team_id=2)],match_periods=[dict(period=1,start_frame=0,end_frame=19)])
            (folder/'fake_match.json').write_text(json.dumps(meta))
            (folder/'fake_phases_of_play.csv').write_text('frame_start,frame_end,team_in_possession_id\n0,10,1\n')
            records=[dict(frame=i,ball_data=dict(x=0,y=0),possession=dict(group='home team'),player_data=[dict(player_id=10,x=0,y=0,is_detected=False),dict(player_id=20,x=5,y=0,is_detected=True)]) for i in range(20)]
            (folder/'fake_tracking_extrapolated.jsonl').write_text('\n'.join(map(json.dumps,records)))
            out=Path(tmp)/'p.npz';mask=np.zeros(20,dtype=bool);mask[3:6]=True
            prepare_skillcorner_match(folder,out,[1,2],[.2],boundary_exclusion_s=0,possession_stability_s=0,ball_dropout_mask=mask)
            with np.load(out) as b:
                self.assertTrue((b['y_alive'][11:]==-1).all());self.assertEqual(b['nearest_codes'][1],1)
                self.assertTrue((b['y_home'][3:6]==1).all());names=b['feature_names'].tolist()
                self.assertTrue(np.isnan(b['X'][3:6,names.index('away_d1_m')]).all())
                self.assertTrue((b['player_quality_counts'][:,1]==1).all())
                self.assertEqual(str(b['observation_view']),'observed')

    def test_cloud_target_entrypoint_fails_before_read(self):
        from argparse import Namespace
        from solution.run import generate
        with patch('solution.run.build_batch',side_effect=AssertionError('must not read')):
            with self.assertRaises(RuntimeError):generate(Namespace())

    def test_split_overlap_rejected_before_data_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'split.json';p.write_text(json.dumps(dict(train=['a'],validation=['a'],test=['b'])))
            with self.assertRaisesRegex(ValueError,'overlap'):train_from_split(tmp,p,tmp,{})

    def test_time_varying_channels_resolve_substitution_and_slot_order(self):
        from solution.atd import track_team_intervals
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'metadata.xml'
            p.write_text("""<Root><Team id='h'/><Team id='a'/><Player id='h1' teamId='h'/><Player id='a1' teamId='a'/><Player id='h2' teamId='h'/>
            <PlayerChannel id='c1' playerId='h1'/><PlayerChannel id='c2' playerId='a1'/><PlayerChannel id='c3' playerId='h2'/>
            <DataFormatSpecification startFrame='1' endFrame='1'><SplitRegister separator=';'><SplitRegister><PlayerChannelRef playerChannelId='c1'/></SplitRegister><SplitRegister><PlayerChannelRef playerChannelId='c2'/></SplitRegister></SplitRegister></DataFormatSpecification>
            <DataFormatSpecification startFrame='2' endFrame='2'><SplitRegister separator=';'><SplitRegister><PlayerChannelRef playerChannelId='c2'/></SplitRegister><SplitRegister><PlayerChannelRef playerChannelId='c3'/></SplitRegister></SplitRegister></DataFormatSpecification></Root>""")
            layout=track_team_intervals(p)
            self.assertEqual(layout[0][2].tolist(),[0,1]);self.assertEqual(layout[1][2].tolist(),[1,0])
            b=self.read('1:0,0,0;1,1,0:0,0,0\n2:0,0,0;1,1,0:0,0,0\n',track_layout=layout)
            self.assertEqual(b.nearest_codes.tolist(),[0,1])

    def test_displacement_never_bridges_missing_ball(self):
        values=dict(ball_x=np.array([0.,np.nan,1.]),ball_y=np.array([0.,np.nan,0.]),ball_visible=np.array([1.,0.,1.]),home_d1_m=np.ones(3),away_d1_m=np.ones(3))
        add_temporal_features(values,10,[.2],(105,68))
        self.assertTrue(np.isnan(values['ball_displacement_0p2s_m']).all())

    def test_public_path_guard_does_not_open_files(self):
        from scripts.check_public_tree import violations
        paths=['solution/run.py','docs/hack2027_solution_technical_report.html','data/tactical/a.txt','local_artifacts/external/x.npz']
        self.assertEqual(violations(paths),paths[1:])

    def test_reference_smoother_matches_official_on_synthetic_cases(self):
        from starter.baseline import smooth
        from solution.heuristic import majority_smooth
        rng=np.random.default_rng(7)
        for window in (0,1,25,50,125):
            for n in (1,7,301):
                codes=rng.choice([-1,0,1],n)
                np.testing.assert_array_equal(majority_smooth(codes,window),smooth(codes,window))

    def test_nearest_tie_preserves_input_slot_order(self):
        from solution.features import spatial_features
        _,nearest=spatial_features((.5,.5),0,[(1,.5,.5,0),(0,.5,.5,0)],(105,68),[1])
        self.assertEqual(nearest,1)

if __name__=='__main__':unittest.main()
