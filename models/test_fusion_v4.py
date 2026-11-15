"""Regression checks for conservative V4 decoding and pooled selection."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from fusion_v2 import decode as v2_decode
from fusion_v4 import BASE, DEFAULT, ROOT, predict, prepare
from optimize_v4 import candidates, refinement_candidates, select, publish_candidate


def trial(spell, macro, **config):
    return dict(config=dict(DEFAULT, **config),
                train0=dict(spell_f1_mean=spell, macro_f1=macro))


class V4Tests(unittest.TestCase):
    def test_fixed_candidates_include_exact_frozen_v2(self):
        first = candidates()
        self.assertEqual(first, candidates())
        self.assertEqual(len(first), 42)
        self.assertEqual(len({json.dumps(c, sort_keys=True) for c in first}), 42)
        self.assertIn(DEFAULT, first)
        frozen = json.loads((ROOT / 'models/fusion_v2_config.json').read_text())
        self.assertEqual(BASE, frozen)

    def test_refinement_retains_base_and_adds_only_declared_21_configs(self):
        base = candidates()
        refined = refinement_candidates()
        self.assertEqual(refined, refinement_candidates())
        self.assertEqual(len(refined), 63)
        base_keys = {json.dumps(c, sort_keys=True) for c in base}
        refined_keys = {json.dumps(c, sort_keys=True) for c in refined}
        self.assertEqual(len(refined_keys), 63)
        self.assertTrue(base_keys.issubset(refined_keys))
        additions = [c for c in refined if json.dumps(c, sort_keys=True) not in base_keys]
        self.assertEqual(len(additions), 21)
        expected = {(team, boost, support)
                    for team in (45., 50., 55.)
                    for boost in (0., .4, .8, 1.2)
                    for support in ('movement', 'stationary')
                    if boost or support == 'movement'}
        self.assertEqual({(c['team_penalty'], c['boost'], c['support']) for c in additions}, expected)
        self.assertTrue(all(c['dead_penalty'] == 30. for c in additions))

    def test_selection_uses_pooled_primary_metric_with_macro_floor(self):
        baseline = trial(.5, .7)
        rejected = trial(.95, .699, boost=1.2)
        winner = trial(.6, .71, boost=.4)
        frame_winner = trial(.59, .95, boost=.8)
        # Separate fold scores are deliberately misleading: selection must use
        # the requested pooled metric, not a fold mean or framewise preference.
        rejected['folds'] = [dict(spell_f1_mean=1., macro_f1=1.)]
        winner['folds'] = [dict(spell_f1_mean=.1, macro_f1=.1)]
        self.assertIs(select([baseline, rejected, winner, frame_winner],
                             baseline, 'train0'), winner)
        self.assertIsNone(select([rejected], baseline, 'train0'))

    def test_baseline_competes_and_ties_prefer_fewer_changes(self):
        baseline = trial(.5, .7)
        worse_spell = trial(.49, .9, boost=.4)
        self.assertIs(select([worse_spell, baseline], baseline, 'train0'), baseline)
        tied = trial(.5, .7, boost=.4)
        self.assertIs(select([tied, baseline], baseline, 'train0'), baseline)

    def test_zero_boost_matches_v2_and_preserves_prepared_observations(self):
        base = np.repeat(np.eye(3) * 5., 40, axis=0)
        original = base.copy()
        prepared = dict(base=base, movement=np.linspace(0, 1, len(base)),
                        stationary=np.ones(len(base)))
        blocks = [(0, 50), (70, 120)]
        expected = np.full(len(base), 'X')
        for lo, hi in blocks:
            expected[lo:hi] = np.array(['H', 'A', 'D'])[v2_decode(base[lo:hi], BASE)]
        with patch('fusion_v4.BLOCKS', blocks), patch('fusion_v4.decode', wraps=v2_decode) as decoder:
            actual = predict({}, DEFAULT, prepared)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(base, original)
        for call, (lo, hi) in zip(decoder.call_args_list, blocks):
            np.testing.assert_array_equal(call.args[0], original[lo:hi])
            self.assertEqual(call.args[1], BASE)

    def test_positive_boost_changes_only_dead_observation(self):
        base = np.tile([1., -.2, -.5], (20, 1))
        prepared = dict(base=base.copy(), movement=np.full(20, .75), stationary=np.zeros(20))
        with patch('fusion_v4.BLOCKS', [(0, 20)]), patch('fusion_v4.decode', wraps=v2_decode) as decoder:
            predict({}, dict(DEFAULT, boost=.4), prepared)
        observed = decoder.call_args.args[0]
        np.testing.assert_array_equal(observed[:, :2], base[:, :2])
        np.testing.assert_allclose(observed[:, 2], base[:, 2] + .3)
        np.testing.assert_array_equal(prepared['base'], base)

    def test_fully_missing_ball_or_motion_adds_no_visible_evidence(self):
        n = 30
        f = dict(slow=np.ones(n), mean_speed=np.full(n, .2))
        for visible, station in [(False, 0.), (True, np.nan)]:
            with self.subTest(visible=visible):
                motion = dict(present=np.full(n, visible), speed=np.full(n, np.nan),
                              station=np.full(n, station))
                with patch('fusion_v4.BLOCKS', [(0, n)]), \
                     patch('fusion_v4.motion_features', return_value=motion), \
                     patch('fusion_v4.v2_observations', return_value=np.zeros((n, 3))):
                    prepared = prepare(f)
                np.testing.assert_array_equal(prepared['movement'], np.zeros(n))
                np.testing.assert_array_equal(prepared['stationary'], np.zeros(n))

    def test_invalid_configuration_rejected_before_inference(self):
        cases = [dict(DEFAULT, boost=-.1), dict(DEFAULT, boost=np.nan),
                 dict(DEFAULT, team_penalty=np.inf), dict(DEFAULT, dead_penalty=-1.),
                 dict(DEFAULT, support='unknown'), dict(DEFAULT, extra=True)]
        missing = dict(DEFAULT)
        del missing['boost']
        cases.append(missing)
        for config in cases:
            with self.subTest(config=config), self.assertRaises(ValueError):
                predict({}, config)

    def test_unpromoted_experiments_preserve_deployed_artifacts(self):
        for accepted, promote in [(False, True), (True, False), (False, False)]:
            with self.subTest(accepted=accepted, promote=promote), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / 'models').mkdir()
                out = root / 'study'
                out.mkdir()
                results = root / 'evaluation/results'
                results.mkdir(parents=True)
                deployed = {
                    root / 'models/fusion_v4_config.json': b'{"boost":0.4,"deployed":true}\n',
                    root / 'models/default_model.json': b'{"model":"v4"}\n',
                    results / 'model_fusion_v4.csv': b'frame,label\n0,A\n1,H\n',
                }
                for path, content in deployed.items():
                    path.write_bytes(content)
                config = dict(DEFAULT, boost=.8)
                labels = np.array(['H', 'D', 'A'])
                publish_candidate(root, out, config, labels, accepted, promote)
                for path, content in deployed.items():
                    self.assertEqual(path.read_bytes(), content)
                self.assertEqual(json.loads((out / 'candidate_config.json').read_text()), config)
                self.assertEqual((out / 'candidate.csv').read_text().splitlines(),
                                 ['frame,label', '0,H', '1,D', '2,A'])

    def test_accepted_explicit_promotion_updates_deployed_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'models').mkdir()
            out = root / 'study'
            out.mkdir()
            config_path = root / 'models/fusion_v4_config.json'
            default_path = root / 'models/default_model.json'
            config_path.write_bytes(b'{"old_config":true}\n')
            default_path.write_bytes(b'{"model":"v2"}\n')
            config = dict(DEFAULT, boost=.8)
            labels = np.array(['H', 'D', 'A'])
            publish_candidate(root, out, config, labels, True, True)
            self.assertEqual(json.loads(config_path.read_text()), config)
            self.assertEqual(json.loads(default_path.read_text()), {'model': 'v4'})
            self.assertEqual((root / 'evaluation/results/model_fusion_v4.csv').read_text().splitlines(),
                             ['frame,label', '0,H', '1,D', '2,A'])
            self.assertEqual(json.loads((out / 'candidate_config.json').read_text()), config)
            self.assertEqual((out / 'candidate.csv').read_text().splitlines(),
                             ['frame,label', '0,H', '1,D', '2,A'])


if __name__ == '__main__':
    unittest.main()
