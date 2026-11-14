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
    if any(not value for value in team_ids) or len(set(team_ids)) != 2:
        raise ValueError("metadata requires two distinct nonempty team IDs")
    names = tuple(team.findtext("Name") or team_ids[i] or f"team-{i}" for i, team in enumerate(teams))
    slots = [player.get("teamId") for player in root.iter("Player")]
    if not slots:
        raise ValueError("ATD metadata contains no Player entries")
    if any(team_id not in team_ids for team_id in slots):
        raise ValueError("unknown team in track metadata")
    mapping = np.array([HOME if team_id == team_ids[0] else AWAY for team_id in slots], dtype=np.int8)
    return mapping, (names[0], names[1])


def _triple(group: str, coordinates_only: bool = False) -> tuple[float, float, float] | None:
    fields = group.split(",")
    if coordinates_only and len(fields) == 2:
        fields.append("NaN")
    if len(fields) != 3:
        raise ValueError("observation requires x,y,speed")
    try:
        x, y, speed = map(float, fields)
    except ValueError:
        raise ValueError("nonnumeric observation") from None
    if any(np.isinf(v) for v in (x, y, speed)):
        raise ValueError("infinite observation")
    if not np.isfinite(x) or not np.isfinite(y):
        return None
    return x, y, speed


def read_atd(
    raw_path: str | Path,
    team_of_track: np.ndarray,
    n_frames: int,
    pitch: tuple[float, float],
    fps: float,
    radii: list[float],
    temporal_windows_s: list[float],
    blocks: list[tuple[int, int]] | None = None,
    schema_version: int = 1,
    coordinates_only: bool = False,
    track_layout: list[tuple[int, int, np.ndarray]] | None = None,
) -> FeatureBatch:
    if schema_version not in (1, 2):
        raise ValueError("unsupported feature schema")
    if n_frames < 1 or fps <= 0 or not np.isfinite(fps):
        raise ValueError("invalid timeline")
    if not len(team_of_track) or not np.isin(team_of_track, [HOME, AWAY]).all():
        raise ValueError("invalid team mapping")
    arrays: dict[str, np.ndarray] = {}
    nearest_codes = np.full(n_frames, NOBODY, dtype=np.int8)
    frames = np.arange(n_frames, dtype=np.int32)

    count = 0
    with Path(raw_path).open("r", encoding="utf-8") as handle:
        for count, line in enumerate(handle, 1):
            parts = line.rstrip("\r\n").split(":")
            if len(parts) != 3:
                raise ValueError(f"line {count}: malformed ATD record")
            try:
                frame = int(parts[0]) - 1
            except ValueError:
                raise ValueError(f"line {count}: noninteger frame ID") from None
            if frame != count - 1 or frame >= n_frames:
                raise ValueError(f"line {count}: duplicate, missing, unordered or out-of-range frame")
            active_teams = team_of_track
            if track_layout:
                matches = [mapping for start, stop, mapping in track_layout if start <= frame <= stop]
                if len(matches) != 1:
                    raise ValueError(f"line {count}: no unique slot layout")
                active_teams = matches[0]
            groups = parts[1].split(";")
            if len(groups) != len(active_teams):
                raise ValueError(f"line {count}: slot count differs from metadata")
            try:
                ball = _triple(parts[2], coordinates_only)
                parsed_groups = [_triple(group, coordinates_only) for group in groups]
            except ValueError as error:
                raise ValueError(f"line {count}: {error}") from None
            ball_xy = (ball[0], ball[1]) if ball is not None else None
            ball_speed = ball[2] if ball is not None else np.nan
            players = []
            for slot, parsed in enumerate(parsed_groups):
                if parsed is None:
                    continue
                if slot >= len(active_teams):
                    raise ValueError(f"ATD has more track slots than metadata on frame {frame + 1}")
                players.append((int(active_teams[slot]), parsed[0], parsed[1], parsed[2]))
            row, nearest = spatial_features(ball_xy, ball_speed, players, pitch, radii)
            nearest_codes[frame] = nearest
            if not arrays:
                arrays = {name: np.full(n_frames, np.nan, dtype=float) for name in row}
            for name, value in row.items():
                arrays[name][frame] = value

    if count != n_frames:
        raise ValueError("incomplete ATD timeline")
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
    for start, _ in blocks or [(0, n_frames - 1)]:
        derived_speed[start] = np.nan
    missing_speed = ~np.isfinite(arrays["ball_speed_mps"])
    arrays["ball_speed_mps"][missing_speed] = derived_speed[missing_speed]
    add_temporal_features(arrays, fps, temporal_windows_s, pitch, blocks=blocks)
    if schema_version == 2:
        for side in ("home", "away"):
            arrays[f"n_{side}_positions"] = arrays.pop(f"n_{side}_visible")
    return FeatureBatch(frame=frames, fps=fps, values=arrays, nearest_codes=nearest_codes,
                        schema_version=schema_version, observation_view="available_unknown")


def track_team_intervals(meta_path: str | Path) -> list[tuple[int, int, np.ndarray]]:
    """Resolve time-varying EPTS slots through channel -> player -> team IDs."""
    root = ET.parse(meta_path).getroot()
    teams = [node.get('id') for node in root.iter('Team')]
    player_teams = {node.get('id'): node.get('teamId') for node in root.iter('Player')}
    channels = {node.get('id'): node.get('playerId') for node in root.iter('PlayerChannel')}
    intervals = []
    for spec in root.iter('DataFormatSpecification'):
        registers = [node for node in spec.findall('SplitRegister') if node.get('separator') == ';']
        if len(registers) != 1:
            raise ValueError('ambiguous player slot register')
        mapping=[]
        for slot in registers[0].findall('SplitRegister'):
            ids = {channels.get(ref.get('playerChannelId')) for ref in slot.iter('PlayerChannelRef')}
            if len(ids) != 1 or None in ids:
                raise ValueError('unknown or mixed player channels in slot')
            team = player_teams.get(next(iter(ids)))
            if team not in teams:
                raise ValueError('unknown channel team')
            mapping.append(teams.index(team))
        start,stop=int(spec.get('startFrame'))-1,int(spec.get('endFrame'))-1
        if start < 0 or stop < start or (intervals and start != intervals[-1][1]+1):
            raise ValueError('invalid or noncontiguous metadata slot intervals')
        intervals.append((start,stop,np.asarray(mapping,dtype=np.int8)))
    return intervals
