"""Only fictional synthetic coordinates/events; no remote or target fixtures."""
import unittest
from unittest.mock import patch
import numpy as np
from r10.adapters import adapt_tracking, player_velocity, predict_only
from r10.tactical_tag_evidence import ATTACK, tactical_evidence, soft_team_support
from solution.hierarchical_decoder import decode


class R10AdapterTests(unittest.TestCase):
    def tracking(self, **changes):
        args = dict(frame=np.arange(4), period=np.ones(4), fps=25,
            ball_xy=np.full((4, 2), .5), players_xy=np.tile([[.4, .5], [.6, .5]], (4, 1, 1)),
            player_team=np.array([7, 9]), player_track_id=np.array([11, 22]),
            team_map={7: 0, 9: 1}, coordinate_units='normalized_0_1', pitch_m=[100, 60])
        args.update(changes)
        return adapt_tracking(**args)

    def event(self, code='build up', team='Alpha', **extra):
        row = dict(code=code, start_s=.1, end_s=.8,
                   tags={'Team': team, 'Half': '1st half'}, source='synthetic')
        row.update(extra)
        return row

    def evidence(self, rows, fps=25):
        return tactical_evidence(rows, fps=fps, n_frames=100,
            half_blocks={1: [0, 40], 2: [60, 100]}, team_names={'Alpha': 'H', 'Beta': 'A'})

    def test_explicit_home0_away1_units_and_speed(self):
        b = self.tracking(reported_speed=np.full((4, 2), 36.), speed_units='km/h')
        np.testing.assert_array_equal(b['player_team'], np.tile([0, 1], (4, 1)))
        np.testing.assert_allclose(b['ball_xy'], 0)
        np.testing.assert_allclose(b['players_xy'][0], [[-10, 0], [10, 0]])
        np.testing.assert_allclose(b['reported_speed_mps'], 10)
        with self.assertRaises(ValueError): self.tracking(team_map={7: 1, 9: 1})
        with self.assertRaises(ValueError): self.tracking(reported_speed=np.ones((4, 2)))

    def test_slot_swap_renaming_and_no_cross_gap_velocity(self):
        b = self.tracking(coordinate_units='centered_metres')
        b['players_xy'] = np.array([[[0., 0], [10., 0]], [[11., 0], [1., 0]],
                                   [[12., 0], [2., 0]], [[13., 0], [3., 0]]])
        b['player_track_id'][1:] = [22, 11]
        b['player_team'][1:] = [1, 0]
        v = player_velocity(b)
        np.testing.assert_allclose(v[1:, :, 0], 25)
        renamed = {k: x.copy() if isinstance(x, np.ndarray) else x for k, x in b.items()}
        renamed['player_track_id'] += 100
        np.testing.assert_allclose(player_velocity(renamed), v, equal_nan=True)
        b['frame'][2:] += 3; b['period'][3] = 2
        self.assertTrue(np.isnan(player_velocity(b)[2:]).all())

    def test_fps_scaling_missing_ball_and_identity_change(self):
        b = self.tracking(fps=10, coordinate_units='centered_metres')
        b['players_xy'][1:] += 1
        np.testing.assert_allclose(player_velocity(b)[1], 10)
        b['player_track_id'][1] = [31, 32]
        self.assertTrue(np.isnan(player_velocity(b)[1]).all())
        ball = np.full((4, 2), np.nan); ball[0] = [.5, np.nan]
        b = self.tracking(ball_xy=ball)
        self.assertTrue(np.isnan(b['ball_xy']).all())
        with self.assertRaises(ValueError): self.tracking(player_track_id=[11, 11])
        with self.assertRaises(ValueError): self.tracking(frame=[0, 1, 1, 2])

    def test_every_attack_category_and_defense_reversal(self):
        for code in ATTACK:
            with self.subTest(code=code):
                self.assertEqual(self.evidence([self.event(code)])['intervals'][0]['team'], 'H')
        for code in ['defending in defensive third', 'defending in middle third', 'defending in attacking third']:
            for name, expected in [('Alpha', 'A'), ('Beta', 'H')]:
                with self.subTest(code=code, name=name):
                    self.assertEqual(self.evidence([self.event(code, name)])['intervals'][0]['team'], expected)

    def test_rounding_unknown_team_and_half_contract(self):
        e = self.evidence([self.event(team='N/A')])['intervals'][0]
        self.assertEqual((e['start_frame'], e['end_frame']), (round(.1*25), 20))
        self.assertIsNone(e['team']); self.assertTrue(e['unknown'])
        self.assertEqual(self.evidence([self.event()], fps=10)['intervals'][0]['start_frame'], 1)
        invalid = self.event(start_s=1.5, end_s=2.)
        self.assertEqual(len(self.evidence([invalid])['rejected']), 1)
        invalid = self.event(code='defending in unknown place')
        self.assertEqual(len(self.evidence([invalid])['rejected']), 1)
        invalid = self.event(team='unmapped')
        self.assertEqual(len(self.evidence([invalid])['rejected']), 1)

    def test_duplicates_conflicting_and_overlapping_tags_abstain(self):
        row = self.event(); e = self.evidence([row, row])
        self.assertEqual(len(e['intervals']), 1)
        e = self.evidence([row, self.event(team='Beta', start_s=.4, end_s=1.)])
        s = soft_team_support(e, frame=np.arange(25), period=np.ones(25))
        self.assertTrue(np.all(s['conflict'][10:20]))
        self.assertTrue(np.all(s['home_minus_away'][10:20] == 0))
        self.assertGreater(s['home_minus_away'][5], 0)
        self.assertLess(s['home_minus_away'][22], 0)
        duplicate_fields = self.event(); duplicate_fields['tags']['team'] = 'Alpha'
        self.assertEqual(len(self.evidence([duplicate_fields])['rejected']), 1)

    def test_set_piece_only_bounded_restart_not_dead_phase(self):
        row = self.event('set pieces', 'N/A')
        row['tags']['Type'] = 'corner kick'
        evidence = self.evidence([row, row])
        self.assertFalse(evidence['full_phase_labels_produced'])
        self.assertEqual(evidence['intervals'], [])
        self.assertEqual(len(evidence['restart_proposals']), 1)
        self.assertIsNone(evidence['restart_proposals'][0]['team'])
        self.assertTrue(np.all(soft_team_support(evidence, frame=np.arange(40), period=np.ones(40))['home_minus_away'] == 0))

    def test_predict_only_disabled_identity_no_io_and_nonmutation(self):
        n = 100; q = np.full(n, .9); h = np.linspace(.05, .95, n)
        frame = np.arange(n); period = np.r_[np.ones(40), np.zeros(20), np.full(40, 2)]
        expected = decode(q, h, frame, period, 25)
        original = h.copy()
        with patch('builtins.open', side_effect=AssertionError('no file IO')):
            for evidence in [None, self.evidence([self.event()])]:
                actual = predict_only(q, h, frame=frame, period=period, fps=25,
                    baseline_decoder=decode, tactical=evidence, tag_logit_weight=0)
                np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(h, original)
        self.assertTrue(np.all(actual[40:60] == 'X'))

    def test_team_swap_symmetry_and_alive_head_preservation(self):
        q = np.full(40, .8); h = np.linspace(.2, .8, 40)
        captured = []
        def decoder(a, b, f, p, fps):
            captured.append((a.copy(), b.copy()))
            return np.where(b > .5, 'H', 'A')
        args = dict(frame=np.arange(40), period=np.ones(40), fps=25,
                    baseline_decoder=decoder, tag_logit_weight=.7)
        first = predict_only(q, h, tactical=self.evidence([self.event()]), **args)
        second = predict_only(q, 1-h, tactical=self.evidence([self.event(team='Beta')]), **args)
        np.testing.assert_allclose(captured[0][0], q)
        np.testing.assert_allclose(captured[1][1], 1-captured[0][1])
        np.testing.assert_array_equal(second, np.where(first == 'H', 'A', 'H'))

    def test_invalid_probability_configuration_and_output(self):
        args = dict(frame=np.arange(2), period=np.ones(2), fps=25, baseline_decoder=decode)
        with self.assertRaises(ValueError): predict_only([1., np.nan], [.5, .5], **args)
        with self.assertRaises(ValueError): predict_only([1., 1.], [.5, .5], tag_logit_weight=-1, **args)
        with self.assertRaises(ValueError):
            predict_only([1., 1.], [.5, .5], **dict(args, frame=[0, np.nan]))
        args['baseline_decoder'] = lambda *a: ['invalid', 'H']
        with self.assertRaises(ValueError): predict_only([1., 1.], [.5, .5], **args)


if __name__ == '__main__': unittest.main()
