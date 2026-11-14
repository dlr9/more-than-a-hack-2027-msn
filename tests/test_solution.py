from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from solution.atd import read_atd, team_metadata
from solution.decoder import emission_probabilities, viterbi_decode
from solution.external import deterministic_match_split, prepare_skillcorner_match
from solution.features import AWAY, HOME, NOBODY, FeatureBatch
from solution.heuristic import (carry_missing_ball_prior, clean_short_team_runs,
                                find_restart, majority_smooth)
from solution.modeling import train_from_split
from solution.submission import write_submission


class SolutionTests(unittest.TestCase):
    def test_team_metadata_and_atd_frame_mapping(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            meta = root / "meta.xml"
            meta.write_text(
                "<Root><Team id='h'><Name>Home</Name></Team><Team id='a'><Name>Away</Name></Team>"
                "<Player teamId='h'/><Player teamId='a'/></Root>", encoding="utf-8"
            )
            raw = root / "raw.txt"
            raw.write_text(
                "1:0.0,0.0,0;1.0,1.0,0:0.01,0.0,1\n"
                "2:0.0,0.0,0;1.0,1.0,0:NaN,NaN,NaN\n",
                encoding="utf-8",
            )
            mapping, names = team_metadata(meta)
            self.assertEqual(names, ("Home", "Away"))
            batch = read_atd(raw, mapping, 2, (100.0, 50.0), 25, [1.0, 2.0], [0.2])
            self.assertEqual(batch.frame.tolist(), [0, 1])
            self.assertEqual(batch.nearest_codes.tolist(), [HOME, NOBODY])
            self.assertAlmostEqual(batch.values["home_d1_m"][0], 1.0)
            self.assertTrue(np.isnan(batch.values["home_d1_m"][1]))

    def test_majority_smoothing_matches_reference_rules(self):
        codes = np.array([HOME, HOME, NOBODY, AWAY, AWAY, NOBODY], dtype=np.int8)
        labels = majority_smooth(codes, 1)
        self.assertEqual(labels.tolist(), ["H", "H", "H", "A", "A", "A"])

    def test_missing_ball_is_not_forced_dead_by_restart_detector(self):
        speed = np.array([np.nan, np.nan, 2.1, 2.2, np.nan, 2.3])
        visible = np.array([0, 0, 1, 1, 0, 1])
        self.assertEqual(find_restart(speed, visible, 0, 6, 2.0, 5, 3), 2)

    def test_missing_ball_prior_stays_within_half(self):
        labels = np.asarray(list("HDDDADDD"))
        visible = np.asarray([1, 0, 0, 0, 1, 0, 0, 0])
        carried = carry_missing_ball_prior(labels, visible, [(0, 3), (4, 7)])
        self.assertEqual("".join(carried), "HHHHAAAA")

    def test_short_run_cleanup_does_not_cross_dead_or_half(self):
        labels = np.array(list("HHAAHHDDHHAAHH"), dtype="<U1")
        cleaned = clean_short_team_runs(labels, 3, [(0, 5), (8, 13)])
        self.assertEqual("".join(cleaned), "HHHHHHDDHHHHHH")

    def test_viterbi_resets_between_blocks(self):
        probabilities = emission_probabilities(
            np.array([0.99, 0.99, 0.99, 0.99, 0.99]),
            np.array([0.99, 0.99, 0.01, 0.01, 0.01]),
        )
        labels = viterbi_decode(probabilities, [(0, 1), (3, 4)], 100.0, 1.0)
        self.assertEqual(labels.tolist(), ["H", "H", "X", "A", "A"])

    def test_match_split_has_no_leakage_and_is_reproducible(self):
        ids = [str(value) for value in range(10)]
        first = deterministic_match_split(ids, 2027)
        second = deterministic_match_split(ids, 2027)
        self.assertEqual(first, second)
        sets = [set(first[name]) for name in ("train", "validation", "test")]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        self.assertEqual(set.union(*sets), set(ids))

    def test_skillcorner_player_id_maps_to_metadata_team(self):
        with tempfile.TemporaryDirectory() as folder:
            match = Path(folder) / "123"
            match.mkdir()
            metadata = {
                "home_team": {"id": 10},
                "away_team": {"id": 20},
                "match_periods": [{"period": 1, "start_frame": 0, "end_frame": 3}],
                "players": [
                    {"id": 101, "trackable_object": 1101, "team_id": 10},
                    {"id": 201, "trackable_object": 1201, "team_id": 20},
                ],
            }
            (match / "123_match.json").write_text(json.dumps(metadata), encoding="utf-8")
            (match / "123_phases_of_play.csv").write_text(
                "frame_start,frame_end\n0,3\n", encoding="utf-8"
            )
            records = []
            for frame in range(4):
                home_x, away_x = ((0.1, 2.0) if frame % 2 == 0 else (2.0, 0.1))
                records.append({
                    "frame": frame,
                    "ball_data": {"x": 0.0, "y": 0.0},
                    "possession": {"group": "home team" if frame % 2 == 0 else "away team"},
                    "player_data": [
                        {"player_id": 101, "x": home_x, "y": 0.0, "is_detected": True},
                        {"player_id": 201, "x": away_x, "y": 0.0, "is_detected": True},
                    ],
                })
            (match / "123_tracking_extrapolated.jsonl").write_text(
                "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
            )
            output = Path(folder) / "prepared.npz"
            prepare_skillcorner_match(
                match, output, [1.0, 2.0], [0.2], fps=10.0,
                pitch=(105.0, 68.0), boundary_exclusion_s=0.0,
                possession_stability_s=0.0, require_phase_agreement=False,
            )
            with np.load(output, allow_pickle=False) as prepared:
                self.assertEqual(prepared["nearest_codes"].tolist(), [HOME, AWAY, HOME, AWAY])
                self.assertEqual(prepared["raw_y_home"].tolist(), [1, 0, 1, 0])
                self.assertEqual(prepared["y_home"].tolist(), [-1, 0, 1, -1])
                self.assertEqual(prepared["y_alive"].tolist(), [-1, 1, 1, -1])
                self.assertEqual(prepared["eval_y_alive"].tolist(), [1, 1, 1, 0])
                names = prepared["feature_names"].astype(str).tolist()
                speed_column = names.index("home_local_speed_mean_mps")
                self.assertTrue(np.isfinite(prepared["X"][1, speed_column]))

    def test_submission_contract(self):
        with tempfile.TemporaryDirectory() as folder:
            path = write_submission(Path(folder) / "entry.csv", ["H", "A", "D", "X"])
            self.assertEqual(path.read_text().splitlines(),
                             ["frame,label", "0,H", "1,A", "2,D", "3,X"])

    def test_two_head_training_and_viterbi_selection_smoke(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            prepared = root / "prepared"
            prepared.mkdir()
            for match_index in range(3):
                n_frames = 240
                frame = np.arange(n_frames)
                phase = frame % 60
                y_alive = (phase >= 15).astype(np.int8)
                y_home = ((frame // 30 + match_index) % 2).astype(np.int8)
                y_home[y_alive == 0] = -1
                features = np.column_stack((
                    y_alive + 0.02 * np.sin(frame),
                    np.where(y_home < 0, 0.5, y_home) + 0.02 * np.cos(frame),
                ))
                np.savez_compressed(
                    prepared / f"m{match_index}.npz",
                    X=features,
                    feature_names=np.asarray(["alive_hint", "home_hint"]),
                    y_alive=y_alive,
                    y_home=y_home,
                    period=np.ones(n_frames, dtype=np.int8),
                    frame=frame,
                    fps=np.asarray(10.0),
                    match_id=np.asarray(f"m{match_index}"),
                    nearest_codes=np.zeros(n_frames, dtype=np.int8),
                )
            split_path = root / "split.json"
            split_path.write_text(json.dumps({
                "seed": 2027,
                "train": ["m0"],
                "validation": ["m1"],
                "test": ["m2"],
            }), encoding="utf-8")
            config = {
                "random_seed": 2027,
                "model_grid": {
                    "max_leaf_nodes": [15],
                    "learning_rate": [0.1],
                    "l2_regularization": [0.0],
                    "max_iter": 30,
                },
                "viterbi_grid": {
                    "team_switch_penalty": [1.0],
                    "dead_switch_penalty": [0.5],
                },
            }
            output = root / "models"
            report = train_from_split(prepared, split_path, output, config)
            self.assertIn("mean_macro_f1", report["selected_viterbi"])
            self.assertTrue((output / "alive.joblib").is_file())
            self.assertTrue((output / "home.joblib").is_file())
            self.assertTrue((output / "training_report.json").is_file())


if __name__ == "__main__":
    unittest.main()

