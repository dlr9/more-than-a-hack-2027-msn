"""Prepare public SkillCorner and Metrica matches into a common NPZ protocol."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterator

import numpy as np

from .atd import read_atd, team_metadata
from .features import AWAY, HOME, FeatureBatch, add_temporal_features, spatial_features


def save_prepared(
    path: str | Path,
    batch: FeatureBatch,
    y_alive: np.ndarray,
    y_home: np.ndarray,
    period: np.ndarray,
    match_id: str,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    names = sorted(batch.values)
    matrix, _ = batch.matrix(names)
    np.savez_compressed(
        output,
        X=matrix,
        feature_names=np.asarray(names),
        y_alive=np.asarray(y_alive, dtype=np.int8),
        y_home=np.asarray(y_home, dtype=np.int8),
        period=np.asarray(period, dtype=np.int8),
        frame=batch.frame,
        fps=np.asarray(batch.fps),
        match_id=np.asarray(str(match_id)),
        nearest_codes=batch.nearest_codes,
    )
    return output


def deterministic_match_split(match_ids: list[str], seed: int = 2027) -> dict[str, list[str]]:
    ids = np.asarray(sorted({str(match_id) for match_id in match_ids}))
    if len(ids) < 3:
        raise ValueError("at least three matches are required for train/validation/test splitting")
    rng = np.random.default_rng(seed)
    shuffled = ids[rng.permutation(len(ids))].tolist()
    n_test = max(1, int(round(0.15 * len(shuffled))))
    n_validation = max(1, int(round(0.15 * len(shuffled))))
    if n_test + n_validation >= len(shuffled):
        n_validation = 1
        n_test = 1
    return {
        "train": sorted(shuffled[: len(shuffled) - n_validation - n_test]),
        "validation": sorted(shuffled[len(shuffled) - n_validation - n_test: len(shuffled) - n_test]),
        "test": sorted(shuffled[-n_test:]),
    }


def write_split(path: str | Path, split: dict[str, list[str]], seed: int) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"seed": seed, **split}
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def _jsonl_objects(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if isinstance(value, list):
                yield from value
            else:
                yield value


def _normalise_group(value, home_id: int, away_id: int) -> int:
    if isinstance(value, dict):
        value = value.get("id", value.get("name", value.get("group")))
    if value in (home_id, str(home_id)):
        return HOME
    if value in (away_id, str(away_id)):
        return AWAY
    text = str(value or "").strip().casefold()
    if "home" in text:
        return HOME
    if "away" in text:
        return AWAY
    return -1


def prepare_skillcorner_match(
    match_dir: str | Path,
    output: str | Path,
    radii: list[float],
    temporal_windows_s: list[float],
    fps: float = 10.0,
    pitch: tuple[float, float] = (105.0, 68.0),
    boundary_exclusion_s: float = 1.0,
) -> Path:
    folder = Path(match_dir)
    match_id = folder.name
    metadata = json.loads((folder / f"{match_id}_match.json").read_text(encoding="utf-8"))
    home_id, away_id = int(metadata["home_team"]["id"]), int(metadata["away_team"]["id"])
    object_team = {
        player.get("trackable_object"): (HOME if int(player["team_id"]) == home_id else AWAY)
        for player in metadata.get("players", [])
        if player.get("trackable_object") is not None
    }
    periods = metadata["match_periods"]
    n_frames = max(int(item["end_frame"]) for item in periods) + 1
    period = np.zeros(n_frames, dtype=np.int8)
    for item in periods:
        period[int(item["start_frame"]): int(item["end_frame"]) + 1] = int(item["period"])

    y_alive = np.full(n_frames, -1, dtype=np.int8)
    y_alive[period > 0] = 0
    phase_path = folder / f"{match_id}_phases_of_play.csv"
    boundaries: list[int] = []
    with phase_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            start, stop = int(row["frame_start"]), int(row["frame_end"])
            y_alive[max(0, start): min(n_frames, stop + 1)] = 1
            boundaries.extend((start, stop))
    exclusion = int(round(boundary_exclusion_s * fps))
    for boundary in boundaries:
        y_alive[max(0, boundary - exclusion): min(n_frames, boundary + exclusion + 1)] = -1

    arrays: dict[str, np.ndarray] = {}
    nearest_codes = np.full(n_frames, -1, dtype=np.int8)
    y_home = np.full(n_frames, -1, dtype=np.int8)
    frames = np.arange(n_frames, dtype=np.int32)
    ball_xy_history = np.full((n_frames, 2), np.nan)
    rows: dict[int, tuple[dict[str, float], int]] = {}
    tracking_path = folder / f"{match_id}_tracking_extrapolated.jsonl"
    for record in _jsonl_objects(tracking_path):
        frame = int(record["frame"])
        if not 0 <= frame < n_frames:
            continue
        ball = record.get("ball_data") or {}
        bx, by = ball.get("x"), ball.get("y")
        ball_xy = None if bx is None or by is None else ((float(bx) + pitch[0] / 2) / pitch[0],
                                                         (float(by) + pitch[1] / 2) / pitch[1])
        if ball_xy is not None:
            ball_xy_history[frame] = ball_xy
        players = []
        for player in record.get("player_data") or []:
            team = _normalise_group(player.get("group_name", player.get("group")), home_id, away_id)
            if team < 0:
                team = object_team.get(player.get("trackable_object"), -1)
            if team < 0 or player.get("x") is None or player.get("y") is None:
                continue
            x = (float(player["x"]) + pitch[0] / 2) / pitch[0]
            y = (float(player["y"]) + pitch[1] / 2) / pitch[1]
            players.append((team, x, y, float(player.get("speed", np.nan))))
        row, nearest = spatial_features(ball_xy, np.nan, players, pitch, radii)
        rows[frame] = row, nearest
        possession = record.get("possession") or {}
        possession_team = _normalise_group(
            possession.get("group", possession.get("team_id")), home_id, away_id
        )
        if possession_team >= 0:
            y_home[frame] = int(possession_team == HOME)

    valid_pairs = np.isfinite(ball_xy_history[1:, 0]) & np.isfinite(ball_xy_history[:-1, 0])
    speed = np.full(n_frames, np.nan)
    delta = ball_xy_history[1:] - ball_xy_history[:-1]
    movement = np.hypot(delta[:, 0] * pitch[0], delta[:, 1] * pitch[1]) * fps
    speed[1:][valid_pairs] = movement[valid_pairs]
    for frame, (row, nearest) in rows.items():
        row["ball_speed_mps"] = speed[frame]
        row["ball_speed_missing"] = float(not np.isfinite(speed[frame]))
        nearest_codes[frame] = nearest
        if not arrays:
            arrays = {name: np.full(n_frames, np.nan) for name in row}
        for name, value in row.items():
            arrays[name][frame] = value
    if not arrays:
        raise ValueError(f"no tracking frames read from {tracking_path}")
    add_temporal_features(arrays, fps, temporal_windows_s, pitch)
    batch = FeatureBatch(frames, fps, arrays, nearest_codes)
    return save_prepared(output, batch, y_alive, y_home, period, match_id)


def prepare_metrica_game3(
    meta_path: str | Path,
    raw_path: str | Path,
    events_path: str | Path,
    output: str | Path,
    radii: list[float],
    temporal_windows_s: list[float],
    fps: float = 25.0,
    pitch: tuple[float, float] = (105.0, 68.0),
) -> Path:
    team_map, team_names = team_metadata(meta_path)
    events_doc = json.loads(Path(events_path).read_text(encoding="utf-8"))
    events = sorted(events_doc.get("data", events_doc), key=lambda event: event["start"]["frame"])
    max_event_frame = max(int(event["end"]["frame"]) for event in events)
    with Path(raw_path).open("r", encoding="utf-8") as handle:
        line_count = sum(1 for _ in handle)
    n_frames = max(max_event_frame + 1, line_count)
    batch = read_atd(raw_path, team_map, n_frames, pitch, fps, radii, temporal_windows_s)
    y_alive = np.full(n_frames, -1, dtype=np.int8)
    y_home = np.full(n_frames, -1, dtype=np.int8)
    conflict = np.zeros(n_frames, dtype=bool)
    period = np.zeros(n_frames, dtype=np.int8)
    team_lookup = {team_names[0].casefold(): HOME, team_names[1].casefold(): AWAY,
                   "team a": HOME, "team b": AWAY}
    for event in events:
        start, stop = int(event["start"]["frame"]), int(event["end"]["frame"])
        event_type = (event.get("type", {}).get("name") or "").upper()
        event_period = int(event.get("period", 0))
        period[max(0, start): min(n_frames, stop + 1)] = event_period
        if event_type in {"PASS", "CARRY"}:
            low, high = start + 2, stop - 1
            team = team_lookup.get((event.get("team", {}).get("name") or "").casefold(), -1)
            if team >= 0 and low < high:
                desired_home = int(team == HOME)
                selected = slice(max(0, low), min(n_frames, high))
                existing = y_alive[selected] >= 0
                conflict[selected] |= existing & (
                    (y_alive[selected] != 1) | (y_home[selected] != desired_home)
                )
                y_alive[selected] = 1
                y_home[selected] = desired_home
    for index, event in enumerate(events):
        if (event.get("type", {}).get("name") or "").upper() != "BALL OUT":
            continue
        start = int(event["end"]["frame"]) + 1
        next_restart = next(
            (candidate for candidate in events[index + 1:]
             if (candidate.get("type", {}).get("name") or "").upper() == "SET PIECE"),
            None,
        )
        if next_restart is not None:
            stop = int(next_restart["start"]["frame"])
            if start < stop:
                selected = slice(max(0, start), min(n_frames, stop))
                conflict[selected] |= (y_alive[selected] >= 0) & (y_alive[selected] != 0)
                y_alive[selected] = 0
                y_home[selected] = -1
    y_alive[conflict] = -1
    y_home[conflict] = -1
    return save_prepared(output, batch, y_alive, y_home, period, "metrica-game-3")

