"""Reference smoothing and the versioned heuristic candidate."""

from __future__ import annotations

import numpy as np

from .features import AWAY, HOME, NOBODY, FeatureBatch


def majority_smooth(codes: np.ndarray, window: int) -> np.ndarray:
    codes = np.asarray(codes)
    if window <= 0:
        return np.where(codes == HOME, "H", np.where(codes == AWAY, "A", "D"))
    home_cumulative = np.concatenate(([0], np.cumsum(codes == HOME)))
    away_cumulative = np.concatenate(([0], np.cumsum(codes == AWAY)))
    index = np.arange(len(codes))
    low = np.maximum(0, index - window)
    high = np.minimum(len(codes), index + window + 1)
    home = home_cumulative[high] - home_cumulative[low]
    away = away_cumulative[high] - away_cumulative[low]
    labels = np.full(len(codes), "D", dtype="<U1")
    seen = (home + away) > 0
    labels[seen] = np.where(home[seen] >= away[seen], "H", "A")
    return labels


def clean_short_team_runs(labels: np.ndarray, min_frames: int, blocks: list[tuple[int, int]]) -> np.ndarray:
    out = np.asarray(labels, dtype="<U1").copy()
    for block_start, block_stop in blocks:
        start = block_start
        while start <= block_stop:
            label = out[start]
            stop = start + 1
            while stop <= block_stop and out[stop] == label:
                stop += 1
            if (
                label in ("H", "A")
                and stop - start < min_frames
                and start > block_start
                and stop <= block_stop
                and out[start - 1] == out[stop]
                and out[start - 1] in ("H", "A")
            ):
                out[start:stop] = out[start - 1]
            start = stop
    return out


def carry_missing_ball_prior(
    labels: np.ndarray, ball_visible: np.ndarray, blocks: list[tuple[int, int]]
) -> np.ndarray:
    """Keep the last team as a weak prior; explicit dead-ball evidence is applied later."""
    out = np.asarray(labels, dtype="<U1").copy()
    visible = np.asarray(ball_visible) > 0
    for block_start, block_stop in blocks:
        previous = None
        for frame in range(block_start, block_stop + 1):
            if out[frame] in ("H", "A"):
                previous = out[frame]
            elif not visible[frame] and previous is not None:
                out[frame] = previous
    return out


def find_restart(
    ball_speed: np.ndarray,
    ball_visible: np.ndarray,
    start: int,
    stop: int,
    threshold: float,
    future_window: int,
    min_hits: int,
) -> int:
    start = max(0, start)
    stop = min(len(ball_speed), max(start, stop))
    active = np.isfinite(ball_speed) & (ball_speed >= threshold) & (ball_visible > 0)
    for frame in range(start, stop):
        high = min(stop, frame + future_window)
        if active[frame] and int(active[frame:high].sum()) >= min_hits:
            return frame
    return stop


def apply_set_piece_evidence(
    labels: np.ndarray,
    batch: FeatureBatch,
    events: list[dict],
    team_name_to_label: dict[str, str],
    config: dict,
) -> np.ndarray:
    out = labels.copy()
    speed = batch.values["ball_speed_mps"]
    visible = batch.values["ball_visible"]
    for event in events:
        if (event.get("code") or "").strip().lower() != "set pieces":
            continue
        start = max(0, int(event["start_frame"]))
        stop = min(len(out), int(event["end_frame"]))
        restart = find_restart(
            speed,
            visible,
            start,
            stop,
            float(config["restart_speed_mps"]),
            int(config["restart_future_window_frames"]),
            int(config["restart_future_min_hits"]),
        )
        # A broad set-piece tag alone does not establish a dead interval.
        team = (event.get("tags", {}).get("Team") or "").strip().casefold()
        label = team_name_to_label.get(team)
        if label and restart < stop:
            seed_stop = min(len(out), restart + int(config["restart_team_seed_frames"]))
            out[restart:seed_stop] = label
    return out


def heuristic_labels(
    batch: FeatureBatch,
    events: list[dict],
    team_names: tuple[str, str],
    blocks: list[tuple[int, int]],
    config: dict,
) -> np.ndarray:
    labels = majority_smooth(batch.nearest_codes, int(config["reference_window_frames"]))
    labels = carry_missing_ball_prior(labels, batch.values["ball_visible"], blocks)
    labels = clean_short_team_runs(labels, int(config["short_team_run_frames"]), blocks)
    mapping = {team_names[0].strip().casefold(): "H", team_names[1].strip().casefold(): "A"}
    labels = apply_set_piece_evidence(labels, batch, events, mapping, config)
    output = np.full(len(labels), "X", dtype="<U1")
    for start, stop in blocks:
        output[start:stop + 1] = labels[start:stop + 1]
    return output

