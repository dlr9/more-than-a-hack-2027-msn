"""Prepare public SkillCorner and Metrica matches into a common NPZ protocol."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterator

import numpy as np

from .atd import read_atd, team_metadata, track_team_intervals
from .features import AWAY, HOME, FeatureBatch, add_temporal_features, spatial_features


def save_prepared(
    path: str | Path,
    batch: FeatureBatch,
    y_alive: np.ndarray,
    y_home: np.ndarray,
    period: np.ndarray,
    match_id: str,
    extra_arrays: dict[str, np.ndarray] | None = None,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    names = sorted(batch.values)
    matrix, _ = batch.matrix(names)
    payload = dict(
        X=matrix,
        feature_names=np.asarray(names),
        y_alive=np.asarray(y_alive, dtype=np.int8),
        y_home=np.asarray(y_home, dtype=np.int8),
        period=np.asarray(period, dtype=np.int8),
        frame=batch.frame,
        fps=np.asarray(batch.fps),
        match_id=np.asarray(str(match_id)),
        nearest_codes=batch.nearest_codes,
        schema_version=np.asarray(batch.schema_version),
        observation_view=np.asarray(batch.observation_view),
    )
    if extra_arrays:
        payload.update({name: np.asarray(value) for name, value in extra_arrays.items()})
    np.savez_compressed(output, **payload)
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


def _stable_team_mask(labels: np.ndarray, radius: int) -> np.ndarray:
    """Frames whose centred window is fully known and belongs to one team."""
    labels = np.asarray(labels, dtype=np.int8)
    index = np.arange(len(labels))
    low = np.maximum(0, index - max(0, radius))
    high = np.minimum(len(labels), index + max(0, radius) + 1)
    home_sum = np.concatenate(([0], np.cumsum(labels == 1)))
    away_sum = np.concatenate(([0], np.cumsum(labels == 0)))
    home_count = home_sum[high] - home_sum[low]
    away_count = away_sum[high] - away_sum[low]
    width = high - low
    return (home_count == width) | (away_count == width)


def prepare_skillcorner_match(
    match_dir: str | Path,
    output: str | Path,
    radii: list[float],
    temporal_windows_s: list[float],
    fps: float = 10.0,
    pitch: tuple[float, float] = (105.0, 68.0),
    boundary_exclusion_s: float = 1.0,
    possession_stability_s: float = 1.0,
    require_phase_agreement: bool = True,
    label_policy: str = "conservative",
    player_view: str = "observed",
    ball_dropout_mask: np.ndarray | None = None,
) -> Path:
    if label_policy not in {"conservative", "legacy_phase_complement"}:
        raise ValueError("unknown label policy")
    if player_view not in {"observed", "all"}:
        raise ValueError("unknown player view")
    folder = Path(match_dir)
    match_id = folder.name
    metadata = json.loads((folder / f"{match_id}_match.json").read_text(encoding="utf-8"))
    pitch = (float(metadata.get("pitch_length", pitch[0])), float(metadata.get("pitch_width", pitch[1])))
    home_id, away_id = int(metadata["home_team"]["id"]), int(metadata["away_team"]["id"])
    # Current SkillCorner Open Data tracking rows identify players with ``player_id``. Older
    # exports and some customer examples use ``trackable_object`` instead, so retain both keys.
    player_team: dict[object, int] = {}
    for player in metadata.get("players", []):
        team_id = int(player["team_id"])
        if team_id not in (home_id, away_id):
            continue
        team = HOME if team_id == home_id else AWAY
        for key in ("id", "trackable_object"):
            identifier = player.get(key)
            if identifier is not None:
                player_team[identifier] = team
                player_team[str(identifier)] = team
    periods = metadata["match_periods"]
    n_frames = max(int(item["end_frame"]) for item in periods) + 1
    period = np.zeros(n_frames, dtype=np.int8)
    if ball_dropout_mask is not None and np.shape(ball_dropout_mask) != (n_frames,):
        raise ValueError("dropout mask length mismatch")
    for item in periods:
        period[int(item["start_frame"]): int(item["end_frame"]) + 1] = int(item["period"])

    raw_y_alive = np.full(n_frames, -1, dtype=np.int8)
    raw_y_alive[period > 0] = 0
    phase_y_home = np.full(n_frames, -1, dtype=np.int8)
    phase_team_conflict = np.zeros(n_frames, dtype=bool)
    phase_path = folder / f"{match_id}_phases_of_play.csv"
    boundaries: list[int] = []
    with phase_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            start, stop = int(row["frame_start"]), int(row["frame_end"])
            # SkillCorner duration is (frame_end - frame_start) / fps, so intervals are half-open.
            low, high = max(0, start), min(n_frames, stop)
            if low >= high:
                continue
            raw_y_alive[low:high] = 1
            phase_team = _normalise_group(row.get("team_in_possession_id"), home_id, away_id)
            if phase_team >= 0:
                existing = phase_y_home[low:high]
                phase_team_conflict[low:high] |= (existing >= 0) & (existing != int(phase_team == HOME))
                phase_y_home[low:high] = int(phase_team == HOME)
            boundaries.extend((start, stop))
    raw_y_alive[period == 0] = -1
    phase_y_home[period == 0] = -1
    phase_y_home[phase_team_conflict] = -1
    eval_y_alive = raw_y_alive.copy()
    eval_y_home = np.full(n_frames, -1, dtype=np.int8)
    eval_team_known = (raw_y_alive == 1) & (phase_y_home >= 0)
    eval_y_home[eval_team_known] = phase_y_home[eval_team_known]
    y_alive = raw_y_alive.copy()
    if label_policy == "conservative":
        y_alive[y_alive == 0] = -1  # Phase absence has no verified D semantics.
    exclusion = int(round(boundary_exclusion_s * fps))
    for boundary in boundaries:
        y_alive[max(0, boundary - exclusion): min(n_frames, boundary + exclusion + 1)] = -1

    arrays: dict[str, np.ndarray] = {}
    nearest_codes = np.full(n_frames, -1, dtype=np.int8)
    raw_y_home = np.full(n_frames, -1, dtype=np.int8)
    frames = np.arange(n_frames, dtype=np.int32)
    ball_xy_history = np.full((n_frames, 2), np.nan)
    rows: dict[int, tuple[dict[str, float], int]] = {}
    quality = np.zeros((n_frames, 3), dtype=np.int16)  # detected, extrapolated, unknown
    seen_frames = set()
    previous_player_position: dict[object, tuple[int, float, float]] = {}
    tracking_path = folder / f"{match_id}_tracking_extrapolated.jsonl"
    for record in _jsonl_objects(tracking_path):
        frame = int(record["frame"])
        if frame != len(seen_frames) or not 0 <= frame < n_frames:
            raise ValueError("duplicate, missing, unordered or out-of-range external frame")
        seen_frames.add(frame)
        ball = record.get("ball_data") or {}
        bx, by = ball.get("x"), ball.get("y")
        ball_xy = None if bx is None or by is None else ((float(bx) + pitch[0] / 2) / pitch[0],
                                                         (float(by) + pitch[1] / 2) / pitch[1])
        if ball_dropout_mask is not None and ball_dropout_mask[frame]:
            ball_xy = None
        if ball_xy is not None and not np.isfinite(ball_xy).all():
            ball_xy = None
        if ball_xy is not None:
            ball_xy_history[frame] = ball_xy
        players = []
        for player in record.get("player_data") or []:
            detected = player.get("is_detected")
            quality[frame, 0 if detected is True else 1 if detected is False else 2] += 1
            if player_view == "observed" and detected is not True:
                continue
            identifier = player.get("player_id", player.get("trackable_object"))
            team = _normalise_group(player.get("group_name", player.get("group")), home_id, away_id)
            if team < 0:
                team = player_team.get(identifier, player_team.get(str(identifier), -1))
            if team < 0 or player.get("x") is None or player.get("y") is None:
                continue
            player_x_m, player_y_m = float(player["x"]), float(player["y"])
            player_speed = float(player.get("speed", np.nan))
            if not np.isfinite(player_speed) and identifier is not None:
                previous = previous_player_position.get(identifier)
                if previous is not None and previous[0] == frame - 1 and period[frame] == period[frame - 1]:
                    player_speed = float(
                        np.hypot(player_x_m - previous[1], player_y_m - previous[2]) * fps
                    )
            if identifier is not None:
                previous_player_position[identifier] = (frame, player_x_m, player_y_m)
            x = (player_x_m + pitch[0] / 2) / pitch[0]
            y = (player_y_m + pitch[1] / 2) / pitch[1]
            players.append((team, x, y, player_speed))
        row, nearest = spatial_features(ball_xy, np.nan, players, pitch, radii)
        rows[frame] = row, nearest
        possession = record.get("possession") or {}
        possession_team = _normalise_group(
            possession.get("group", possession.get("team_id")), home_id, away_id
        )
        if possession_team >= 0:
            raw_y_home[frame] = int(possession_team == HOME)

    if len(seen_frames) != n_frames:
        raise ValueError("incomplete external tracking timeline")
    valid_pairs = np.isfinite(ball_xy_history[1:]).all(axis=1) & np.isfinite(ball_xy_history[:-1]).all(axis=1) & (period[1:] == period[:-1])
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
    blocks = [(int(item["start_frame"]), int(item["end_frame"])) for item in periods]
    add_temporal_features(arrays, fps, temporal_windows_s, pitch, blocks=blocks)
    for team in ("home", "away"):
        arrays[f"n_{team}_positions"] = arrays.pop(f"n_{team}_visible")
    batch = FeatureBatch(frames, fps, arrays, nearest_codes, schema_version=2, observation_view=player_view)
    stability_radius = int(round(possession_stability_s * fps))
    stable_home = _stable_team_mask(raw_y_home, stability_radius)
    home_eligible = (y_alive == 1) & (raw_y_home >= 0) & stable_home
    if require_phase_agreement:
        home_eligible &= phase_y_home == raw_y_home
    y_home = np.full(n_frames, -1, dtype=np.int8)
    y_home[home_eligible] = raw_y_home[home_eligible]
    return save_prepared(
        output, batch, y_alive, y_home, period, match_id,
        extra_arrays={
            "label_policy": np.asarray(label_policy),
            "label_source": np.where(y_alive == 1, "phase_presence_weak", np.where(y_alive == 0, "phase_complement_unverified", "unknown")),
            "label_mask": (y_alive >= 0),
            "label_confidence": np.where(y_alive >= 0, "weak", "unknown"),
            "player_quality_counts": quality,
            "ball_dropout_mask": np.zeros(n_frames, dtype=bool) if ball_dropout_mask is None else ball_dropout_mask,
            "raw_y_alive": raw_y_alive,
            "raw_y_home": raw_y_home,
            "phase_y_home": phase_y_home,
            "eval_y_alive": eval_y_alive,
            "eval_y_home": eval_y_home,
            "home_label_source": np.where(y_home >= 0, "stable_possession_phase_agreement" if require_phase_agreement else "stable_possession", "unknown"),
            "home_label_mask": y_home >= 0,
            "home_stable": stable_home.astype(np.int8),
            "phase_team_conflict": phase_team_conflict.astype(np.int8),
        },
    )


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
    n_frames = line_count
    if max_event_frame > n_frames:
        raise ValueError("events exceed tracking timeline")
    # Metrica event frame IDs and ATD IDs are both one-based; convert together.
    for event in events:
        for endpoint in ("start", "end"):
            event[endpoint]["frame"] = int(event[endpoint]["frame"]) - 1
    import xml.etree.ElementTree as ET
    metadata_root = ET.parse(meta_path).getroot()
    parameters = {node.findtext("Name"): node.findtext("Value") for node in metadata_root.iter("ProviderParameter")}
    period = np.zeros(n_frames, dtype=np.int8)
    blocks = []
    for number, name in enumerate(("first_half", "second_half"), 1):
        low, high = int(parameters[name + "_start"]) - 1, int(parameters[name + "_end"]) - 1
        if low < 0 or high >= n_frames or high < low:
            raise ValueError("invalid metadata period bounds")
        period[low:high + 1] = number
        blocks.append((low, high))
    batch = read_atd(raw_path, team_map, n_frames, pitch, fps, radii, temporal_windows_s,
                     blocks=blocks, schema_version=2, coordinates_only=True,
                     track_layout=track_team_intervals(meta_path))
    y_alive = np.full(n_frames, -1, dtype=np.int8)
    y_home = np.full(n_frames, -1, dtype=np.int8)
    conflict = np.zeros(n_frames, dtype=bool)
    team_lookup = {team_names[0].casefold(): HOME, team_names[1].casefold(): AWAY,
                   "team a": HOME, "team b": AWAY}
    for event in events:
        start, stop = int(event["start"]["frame"]), int(event["end"]["frame"])
        event_type = (event.get("type", {}).get("name") or "").upper()
        event_period = int(event.get("period", 0))
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
             if candidate.get("period") == event.get("period")
             and (candidate.get("type", {}).get("name") or "").upper() == "SET PIECE"),
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
    return save_prepared(
        output, batch, y_alive, y_home, period, "metrica-game-3",
        extra_arrays={"event_conflict": conflict.astype(np.int8),
                      "label_policy": np.asarray("partial_event_interiors"),
                      "full_had_ground_truth": np.asarray(False)},
    )

