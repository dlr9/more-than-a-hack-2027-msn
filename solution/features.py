"""Provider-neutral spatial feature construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


HOME, AWAY, NOBODY = 0, 1, -1


@dataclass
class FeatureBatch:
    """Aligned arrays for one tracking timeline."""

    frame: np.ndarray
    fps: float
    values: dict[str, np.ndarray]
    nearest_codes: np.ndarray
    schema_version: int = 1
    observation_view: str = "legacy_all"
    quality: dict[str, np.ndarray] = field(default_factory=dict)

    def matrix(self, names: Iterable[str] | None = None) -> tuple[np.ndarray, list[str]]:
        selected = list(names) if names is not None else sorted(self.values)
        return np.column_stack([self.values[name] for name in selected]), selected


def _trailing_nanmean(values: np.ndarray, window: int) -> np.ndarray:
    valid = np.isfinite(values)
    cumulative = np.concatenate(([0.0], np.cumsum(np.where(valid, values, 0.0))))
    counts = np.concatenate(([0], np.cumsum(valid)))
    index = np.arange(len(values))
    low = np.maximum(0, index - window + 1)
    total = cumulative[index + 1] - cumulative[low]
    number = counts[index + 1] - counts[low]
    return np.divide(total, number, out=np.full(len(values), np.nan), where=number > 0)


def add_temporal_features(
    values: dict[str, np.ndarray], fps: float, windows_s: Iterable[float], pitch: tuple[float, float],
    blocks: list[tuple[int, int]] | None = None,
) -> None:
    if blocks is not None:
        windows_s = list(windows_s)
        for start, stop in blocks:
            local = {k: v[start:stop + 1].copy() for k, v in values.items()}
            add_temporal_features(local, fps, windows_s, pitch)
            for name, column in local.items():
                if name not in values:
                    values[name] = np.full(len(values["ball_x"]), np.nan)
                values[name][start:stop + 1] = column
        return
    visible = values["ball_visible"].astype(float)
    ball_x = values["ball_x"]
    ball_y = values["ball_y"]
    length, width = pitch
    for seconds in windows_s:
        window = max(1, int(round(seconds * fps)))
        suffix = str(seconds).replace(".", "p") + "s"
        values[f"visibility_rate_{suffix}"] = _trailing_nanmean(visible, window)
        values[f"home_d1_mean_{suffix}"] = _trailing_nanmean(values["home_d1_m"], window)
        values[f"away_d1_mean_{suffix}"] = _trailing_nanmean(values["away_d1_m"], window)
        displacement = np.full(len(ball_x), np.nan, dtype=float)
        if window < len(ball_x):
            valid = (
                np.isfinite(ball_x[window:])
                & np.isfinite(ball_y[window:])
                & np.isfinite(ball_x[:-window])
                & np.isfinite(ball_y[:-window])
            )
            known = np.isfinite(ball_x) & np.isfinite(ball_y)
            cumulative = np.concatenate(([0], np.cumsum(known)))
            valid &= cumulative[window + 1:] - cumulative[:-window - 1] == window + 1
            dx = (ball_x[window:] - ball_x[:-window]) * length
            dy = (ball_y[window:] - ball_y[:-window]) * width
            target = displacement[window:]
            target[valid] = np.hypot(dx[valid], dy[valid])
        values[f"ball_displacement_{suffix}_m"] = displacement


def spatial_features(
    ball_xy: tuple[float, float] | None,
    ball_speed: float,
    players: Iterable[tuple[int, float, float, float]],
    pitch: tuple[float, float],
    radii: Iterable[float],
) -> tuple[dict[str, float], int]:
    """Features for one frame from ``(team, x_norm, y_norm, speed)`` players."""
    length, width = pitch
    if ball_xy is not None and not np.isfinite(ball_xy).all():
        ball_xy = None
    if ball_xy is None:
        ball_speed = np.nan
    by_team: dict[int, list[tuple[float, float]]] = {HOME: [], AWAY: []}
    player_counts = {HOME: 0, AWAY: 0}
    nearest, nearest_distance = NOBODY, np.inf
    for team, x, y, speed in players:
        if team not in (HOME, AWAY) or not np.isfinite([x, y]).all():
            continue
        player_counts[team] += 1
        if ball_xy is not None:
            distance = float(np.hypot((x - ball_xy[0]) * length, (y - ball_xy[1]) * width))
            by_team[team].append((distance, speed))
            if distance < nearest_distance:
                nearest, nearest_distance = team, distance
    for entries in by_team.values():
        entries.sort(key=lambda item: item[0])

    out: dict[str, float] = {
        "ball_visible": float(ball_xy is not None),
        "ball_x": ball_xy[0] if ball_xy is not None else np.nan,
        "ball_y": ball_xy[1] if ball_xy is not None else np.nan,
        "ball_speed_mps": ball_speed,
        "ball_speed_missing": float(not np.isfinite(ball_speed)),
        "n_home_visible": float(player_counts[HOME]),
        "n_away_visible": float(player_counts[AWAY]),
    }
    labels = {HOME: "home", AWAY: "away"}
    for team in (HOME, AWAY):
        entries = by_team[team]
        distances = [item[0] for item in entries]
        for rank in range(3):
            out[f"{labels[team]}_d{rank + 1}_m"] = distances[rank] if rank < len(distances) else np.nan
        out[f"{labels[team]}_d3_mean_m"] = float(np.mean(distances[:3])) if distances else np.nan
        local_speeds = [speed for distance, speed in entries if distance <= 5 and np.isfinite(speed)]
        out[f"{labels[team]}_local_speed_mean_mps"] = (
            float(np.mean(local_speeds)) if local_speeds else np.nan
        )
        for radius in radii:
            key = str(radius).replace(".", "p")
            out[f"{labels[team]}_within_{key}m"] = float(sum(d <= radius for d in distances))

    home_d1, away_d1 = out["home_d1_m"], out["away_d1_m"]
    out["nearest_distance_margin_m"] = away_d1 - home_d1
    if ball_xy is None:
        out.update({"ball_to_x0_m": np.nan, "ball_to_x1_m": np.nan,
                    "ball_to_y0_m": np.nan, "ball_to_y1_m": np.nan})
    else:
        x, y = ball_xy
        out.update({"ball_to_x0_m": x * length, "ball_to_x1_m": (1 - x) * length,
                    "ball_to_y0_m": y * width, "ball_to_y1_m": (1 - y) * width})

    return out, nearest
