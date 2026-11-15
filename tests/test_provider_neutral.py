"""Synthetic contract, physical units and leakage regression tests; no match assets."""
from dataclasses import replace
import unittest

import numpy as np

from solution.candidate import Candidate
from solution.provider_neutral import FeatureContract, assert_compatible, provider_neutral_features_v3


def fixture(n=20):
    ball = np.column_stack([np.arange(n) * .1, np.zeros(n)])
    players = np.stack([ball + [1., 0.], ball + [0., 2.]], axis=1)
    return dict(frame=np.arange(n), period=np.ones(n, dtype=int), ball_xy=ball,
                players_xy=players, player_team=np.array([0, 1]))


def column(batch, name):
    return batch["X"][:, batch["feature_names"].tolist().index(name)]


class NeutralFeatureTests(unittest.TestCase):
    def setUp(self):
        self.contract = FeatureContract("synthetic", "available_unknown", 10., pitch=(100., 50.))

    def test_units_two_axes_and_position_derived_speed(self):
        raw = fixture()
        result = provider_neutral_features_v3(raw, self.contract)
        self.assertEqual(result["X"].shape, (20, 44))
        np.testing.assert_allclose(column(result, "home_d1_m"), 1.)
        np.testing.assert_allclose(column(result, "away_d1_m"), 2.)
        np.testing.assert_allclose(column(result, "ball_x"), .5 + np.arange(20) * .001)
        np.testing.assert_allclose(column(result, "ball_y"), .5)
        np.testing.assert_allclose(column(result, "ball_speed_mps")[1:], 1.)
        np.testing.assert_allclose(column(result, "home_local_speed_mean_mps")[1:], 1.)
        normal = {**raw, "ball_xy": raw["ball_xy"] / [100, 50] + .5,
                  "players_xy": raw["players_xy"] / [100, 50] + .5}
        other = provider_neutral_features_v3(normal, replace(self.contract, coordinate_units="normalized"))
        np.testing.assert_allclose(result["X"], other["X"], equal_nan=True)

    def test_label_event_speed_and_private_flag_fields_rejected(self):
        for name in ("BallStatus", "BallPossession", "y_home", "status", "events", "is_detected", "S", "Z"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "forbidden raw input"):
                provider_neutral_features_v3({**fixture(), name: np.zeros(20)}, self.contract)

    def test_raw_dropout_recomputes_all_dependent_features(self):
        raw = fixture()
        mask = np.zeros(20, dtype=bool)
        mask[5:10] = True
        result = provider_neutral_features_v3(raw, self.contract, ball_dropout_mask=mask)
        # Equivalent to genuinely absent raw coordinates, including windows.
        missing = {**raw, "ball_xy": raw["ball_xy"].copy()}
        missing["ball_xy"][mask] = np.nan
        other = provider_neutral_features_v3(missing, self.contract)
        np.testing.assert_allclose(result["X"], other["X"], equal_nan=True)
        self.assertTrue(np.isnan(column(result, "ball_speed_mps")[5:11]).all())
        self.assertTrue(np.isnan(column(result, "home_d1_m")[mask]).all())
        self.assertTrue((result["nearest_codes"][mask] == -1).all())
        self.assertLess(column(result, "visibility_rate_1p0s")[10], 1.)
        # Feature computation must not mutate the cached source or mask.
        self.assertTrue(np.isfinite(raw["ball_xy"]).all())

    def test_half_and_frame_gap_reset_temporal_context(self):
        raw = fixture()
        raw["ball_xy"][:10] = np.nan
        raw["period"][10:] = 2
        raw["frame"][10:] -= 10
        result = provider_neutral_features_v3(raw, self.contract)
        self.assertEqual(column(result, "visibility_rate_1p0s")[10], 1.)
        self.assertTrue(np.isnan(column(result, "ball_speed_mps")[10]))
        self.assertTrue(np.isnan(column(result, "ball_displacement_1p0s_m")[10:]).all())
        raw = fixture()
        raw["frame"][10:] += 5
        result = provider_neutral_features_v3(raw, self.contract)
        self.assertTrue(np.isnan(column(result, "ball_speed_mps")[10]))
        self.assertTrue(np.isnan(column(result, "ball_displacement_1p0s_m")[10:]).all())

    def test_tracklet_changes_never_create_speed_jumps(self):
        raw = fixture()
        ids = np.tile(["h1", "a1"], (20, 1))
        ids[10:, 0] = "h2"
        result = provider_neutral_features_v3({**raw, "player_track_id": ids}, self.contract)
        self.assertTrue(np.isnan(column(result, "home_local_speed_mean_mps")[10]))
        # IDs themselves are not numerical features, and renaming is invariant.
        ids[ids == "h1"] = "p1"
        other = provider_neutral_features_v3({**raw, "player_track_id": ids}, self.contract)
        np.testing.assert_allclose(result["X"], other["X"], equal_nan=True)
        unknown = np.tile([None, "a1"], (20, 1))
        other = provider_neutral_features_v3({**raw, "player_track_id": unknown}, self.contract)
        self.assertTrue(np.isnan(column(other, "home_local_speed_mean_mps")).all())

    def test_schema2_candidate_rejects_schema3_before_inference(self):
        result = provider_neutral_features_v3(fixture(), self.contract)
        old = Candidate(None, None, result["feature_names"].tolist(), {"schema_version": 2})
        with self.assertRaisesRegex(ValueError, "schema mismatch"):
            old.probabilities(result)

    def test_observation_and_provider_mixing_requires_explicit_waivers(self):
        observed = replace(self.contract, observation_view="observed")
        for purpose in ("train", "evaluate"):
            with self.assertRaisesRegex(ValueError, "observation_view"):
                assert_compatible([observed], self.contract, purpose=purpose)
        report = assert_compatible([observed], self.contract, purpose="evaluate", allow_cross_view=True)
        self.assertFalse(report["equivalence_established"])
        self.assertEqual(report["domain_shift_waivers"][0]["field"], "observation_view")
        other = replace(self.contract, provider="other", fps=25.)
        with self.assertRaisesRegex(ValueError, "provider"):
            assert_compatible([self.contract], other, purpose="train")
        with self.assertRaisesRegex(ValueError, "fps"):
            assert_compatible([self.contract], other, purpose="evaluate", allow_cross_provider=True)
        report = assert_compatible([self.contract], other, purpose="evaluate",
                                   allow_cross_provider=True, allow_cross_fps=True)
        self.assertEqual(len(report["domain_shift_waivers"]), 2)

    def test_invalid_units_timeline_and_orientation_fail_closed(self):
        for changes in ({"coordinate_units": "yards"}, {"orientation": "unknown"}, {"fps": 0}, {"schema_version": 2}):
            with self.assertRaises(ValueError):
                replace(self.contract, **changes)
        raw = fixture()
        raw["frame"][3] = raw["frame"][2]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            provider_neutral_features_v3(raw, self.contract)
        raw = fixture()
        raw["period"][:] = 0
        with self.assertRaisesRegex(ValueError, "in-period"):
            provider_neutral_features_v3(raw, self.contract)


if __name__ == "__main__":
    unittest.main()
