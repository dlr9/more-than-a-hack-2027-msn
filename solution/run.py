"""Generate reference, heuristic, or trained-model submissions.

Examples:
    python -m solution.run --mode reference --out evaluation/results/reference_smooth50.csv
    python -m solution.run --mode heuristic --out evaluation/results/heuristic_v1.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from games import GAME, GAMES, half_video_blocks, require, source_frames
from starter.load_smart_tagging import load as load_smart_tagging

from .atd import read_atd, team_metadata, track_team_intervals
from .config import load_config
from .decoder import emission_probabilities, viterbi_decode
from .heuristic import heuristic_labels, majority_smooth
from .submission import sha256, write_submission


def build_batch(data_dir: str | None, config: dict):
    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME, data_dir)
    team_map, team_names = team_metadata(meta_path)
    batch = read_atd(
        raw_path,
        team_map,
        source_frames("tactical", GAME),
        GAMES[GAME]["pitch"],
        GAMES[GAME]["fps"],
        [float(value) for value in config["distance_radii_m"]],
        [float(value) for value in config["temporal_windows_s"]],
        blocks=[(start, stop) for _, start, stop in half_video_blocks(GAME)],
        schema_version=int(config.get("feature_schema_version", 1)),
        track_layout=track_team_intervals(meta_path),
    )
    return batch, team_names


def _load_models(model_dir: str | Path):
    try:
        import joblib
    except ImportError as error:
        raise RuntimeError("model mode needs scikit-learn/joblib; install requirements-solution.txt") from error
    folder = Path(model_dir)
    return joblib.load(folder / "alive.joblib"), joblib.load(folder / "home.joblib")


def generate(args: argparse.Namespace) -> Path:
    if not getattr(args, "offline_team_ack", False):
        raise RuntimeError("Target inputs require a team-controlled offline environment; no cloud access")
    config = load_config(args.config)
    if args.mode in ("model", "final"):
        if not args.model_dir:
            raise ValueError("--model-dir is required")
        alive_model, home_model = _load_models(args.model_dir)
        metadata = getattr(alive_model, "mtah_metadata", {})
        other = getattr(home_model, "mtah_metadata", {})
        if metadata != other:
            raise ValueError("classifier pair metadata differs")
        config["feature_schema_version"] = int(metadata.get("schema_version", 1))
        if args.mode == "final":
            raise RuntimeError("No candidate has a validated full H/A/D gate; final mode is disabled")
    batch, team_names = build_batch(args.data_dir, config)
    blocks = [(start, stop) for _, start, stop in half_video_blocks(GAME)]

    if args.mode == "reference":
        labels = majority_smooth(batch.nearest_codes, int(config["reference_window_frames"]))
    elif args.mode == "heuristic":
        events = load_smart_tagging(data_dir=args.data_dir)
        labels = heuristic_labels(batch, events, team_names, blocks, config)
    else:
        if not args.model_dir:
            raise ValueError("--model-dir is required for model/final mode")
        try:
            import pandas as pd
        except ImportError as error:
            raise RuntimeError("model mode needs pandas; install requirements-solution.txt") from error
        required_names = sorted(set(alive_model.feature_names_in_) | set(home_model.feature_names_in_))
        matrix, _ = batch.matrix(required_names)
        frame = pd.DataFrame(matrix, columns=required_names)
        p_alive = alive_model.predict_proba(frame[list(alive_model.feature_names_in_)])[:, 1]
        p_home = home_model.predict_proba(frame[list(home_model.feature_names_in_)])[:, 1]
        probabilities = emission_probabilities(p_alive, p_home)
        metadata = getattr(alive_model, "mtah_metadata", {})
        labels = viterbi_decode(
            probabilities,
            blocks,
            float(metadata.get("team_switch_penalty", 3.0)),
            float(metadata.get("dead_switch_penalty", 1.0)),
        )

    path = write_submission(args.out, labels)
    counts = {label: int(np.sum(np.asarray(labels) == label)) for label in "HADX"}
    print(f"wrote {path}")
    print("labels " + " ".join(f"{label}={counts[label]}" for label in "HADX"))
    print(f"sha256 {sha256(path)}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=("reference", "heuristic", "model", "final"), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--data-dir")
    parser.add_argument("--config")
    parser.add_argument("--model-dir")
    parser.add_argument("--offline-team-ack", action="store_true", help="team confirms offline execution and data isolation")
    generate(parser.parse_args())


if __name__ == "__main__":
    main()

