"""Streaming FIFA/EPTS ATD parsing for the tactical source."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .features import AWAY, HOME, NOBODY, FeatureBatch, add_temporal_features, spatial_features


def team_metadata(meta_path: str | Path) -> tuple[np.ndarray, tuple[str, str]]:
    root = ET.parse(meta_path).getroot()
    teams = list(root.iter("Team"))
    if len(teams) != 2:
        raise ValueError(f"expected two teams, found {len(teams)}")
    team_ids = [team.get("id") for team in teams]
    names = tuple(team.findtext("Name") or team_ids[i] or f"team-{i}" for i, team in enumerate(teams))
    slots = [player.get("teamId") for player in root.iter("Player")]
    if not slots:
        raise ValueError("ATD metadata contains no Player entries")
    mapping = np.array([HOME if team_id == team_ids[0] else AWAY for team_id in slots], dtype=np.int8)
    return mapping, (names[0], names[1])


def _triple(group: str) -> tuple[float, float, float] | None:
    if not group or group[0] == "N":
        return None
    fields = group.split(",")
    if len(fields) < 2:
        return None
    speed = float(fields[2]) if len(fields) > 2 and fields[2][0] != "N" else np.nan
    return float(fields[0]), float(fields[1]), speed


def read_atd(
    raw_path: str | Path,
    team_of_track: np.ndarray,
    n_frames: int,
    pitch: tuple[float, float],
    fps: float,
    radii: list[float],
    temporal_windows_s: list[float],
) -> FeatureBatch:
    arrays: dict[str, np.ndarray] = {}
    nearest_codes = np.full(n_frames, NOBODY, dtype=np.int8)
    frames = np.arange(n_frames, dtype=np.int32)

    with Path(raw_path).open("r") as handle:
        for line in handle:
            parts = line.rstrip("\r\n").split(":")
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1
            if not 0 <= frame < n_frames:
                raise ValueError(f"ATD frame outside 1..{n_frames}: {frame + 1}")
            ball = _triple(parts[2])
            ball_xy = (ball[0], ball[1]) if ball is not None else None
            ball_speed = ball[2] if ball is not None else np.nan
            players = []
            for slot, group in enumerate(parts[1].split(";")):
                parsed = _triple(group)
                if parsed is None:
                    continue
                if slot >= len(team_of_track):
                    raise ValueError(f"ATD has more track slots than metadata on frame {frame + 1}")
                players.append((int(team_of_track[slot]), parsed[0], parsed[1], parsed[2]))
            row, nearest = spatial_features(ball_xy, ball_speed, players, pitch, radii)
            nearest_codes[frame] = nearest
            if not arrays:
                arrays = {name: np.full(n_frames, np.nan, dtype=float) for name in row}
            for name, value in row.items():
                arrays[name][frame] = value

    if not arrays:
        raise ValueError(f"no readable ATD records in {raw_path}")
    # The export often provides a ball position without its scalar speed. Derive a conservative
    # frame-to-frame speed in those cases; retain ball_speed_missing as the observation-quality
    # flag so models can still distinguish supplied and reconstructed values.
    ball_x, ball_y = arrays["ball_x"], arrays["ball_y"]
    derived_speed = np.full(n_frames, np.nan)
    valid = (
        np.isfinite(ball_x[1:]) & np.isfinite(ball_y[1:])
        & np.isfinite(ball_x[:-1]) & np.isfinite(ball_y[:-1])
    )
    dx = (ball_x[1:] - ball_x[:-1]) * pitch[0]
    dy = (ball_y[1:] - ball_y[:-1]) * pitch[1]
    derived_speed[1:][valid] = np.hypot(dx[valid], dy[valid]) * fps
    missing_speed = ~np.isfinite(arrays["ball_speed_mps"])
    arrays["ball_speed_mps"][missing_speed] = derived_speed[missing_speed]
    add_temporal_features(arrays, fps, temporal_windows_s, pitch)
    return FeatureBatch(frame=frames, fps=fps, values=arrays, nearest_codes=nearest_codes)

