"""Schema 3 input boundary; labels and provider quality flags never enter features.

This preserves the 44 schema-2 feature names, not its model compatibility: speeds
are now consistently position-derived, so old model artifacts must reject it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Mapping, Sequence

import numpy as np

from .features import add_temporal_features, spatial_features

SCHEMA_VERSION = 3
FEATURE_DEFINITION = "provider_neutral_features_v3"
RADII = (1.0, 2.0, 3.0, 5.0, 10.0)
WINDOWS_S = (0.2, 0.5, 1.0)
RAW_FIELDS = frozenset(("frame", "period", "ball_xy", "players_xy", "player_team", "player_track_id"))


@dataclass(frozen=True)
class FeatureContract:
    provider: str
    observation_view: str
    fps: float
    pitch: tuple[float, float] = (105.0, 68.0)
    coordinate_units: str = "centered_m"
    orientation: str = "fixed_pitch_axes"
    schema_version: int = SCHEMA_VERSION
    feature_definition: str = FEATURE_DEFINITION

    def __post_init__(self):
        if not self.provider or not self.observation_view:
            raise ValueError("provider and observation_view must be explicit")
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps must be finite and positive")
        if len(self.pitch) != 2 or not np.isfinite(self.pitch).all() or min(self.pitch) <= 0:
            raise ValueError("pitch must contain positive length and width in metres")
        if self.coordinate_units not in {"centered_m", "normalized"}:
            raise ValueError("unsupported coordinate units")
        if self.orientation != "fixed_pitch_axes":
            raise ValueError("adapter must establish fixed pitch axes before feature construction")
        if self.schema_version != SCHEMA_VERSION or self.feature_definition != FEATURE_DEFINITION:
            raise ValueError("schema 3 feature contract required")


def assert_compatible(
    training_contracts: Sequence[FeatureContract],
    candidate_contract: FeatureContract,
    *,
    purpose: str,
    allow_cross_provider: bool = False,
    allow_cross_view: bool = False,
    allow_cross_fps: bool = False,
    allow_cross_pitch: bool = False,
) -> dict:
    """Fail closed by default, returning a serializable record of explicit waivers.

    Invoke before stacking training matches or evaluating an artifact. A waiver
    permits a domain-shift experiment; it does not certify domain equivalence.
    Input units may differ because both paths convert to the same physical units.
    """
    if purpose not in {"train", "evaluate"} or not training_contracts:
        raise ValueError("purpose and at least one training contract are required")
    waivers = []
    for source in training_contracts:
        for name, allowed in (("provider", allow_cross_provider), ("observation_view", allow_cross_view),
                              ("fps", allow_cross_fps), ("pitch", allow_cross_pitch)):
            left, right = getattr(source, name), getattr(candidate_contract, name)
            equal = np.array_equal(left, right)
            if not equal:
                if not allowed:
                    raise ValueError(f"{purpose}: incompatible {name}; explicit domain-shift waiver required")
                waivers.append({"field": name, "source": left, "destination": right})
        for name in ("schema_version", "feature_definition", "orientation"):
            if getattr(source, name) != getattr(candidate_contract, name):
                raise ValueError(f"incompatible {name}")
    return {"purpose": purpose, "domain_shift_waivers": waivers, "equivalence_established": not waivers}


def _normalise(xy, contract):
    xy = np.asarray(xy, dtype=float).copy()
    if np.isinf(xy).any():
        raise ValueError("coordinates must be finite or NaN, never infinity")
    if contract.coordinate_units == "centered_m":
        xy = xy / np.asarray(contract.pitch) + .5
    # An incomplete coordinate pair is unavailable, never partly imputed.
    xy[~np.isfinite(xy).all(axis=-1)] = np.nan
    return xy


def provider_neutral_features_v3(
    raw: Mapping[str, np.ndarray],
    contract: FeatureContract,
    *,
    ball_dropout_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Build an NPZ-compatible, label-free 44-feature batch from aligned positions.

    Players have shape (frames, slots, 2), teams are 0=H/1=A and have shape
    (slots,) or (frames, slots). Slots must represent the same tracklet within a
    period unless ``player_track_id`` is supplied with either of those shapes.
    IDs only guard speed differences; they are never encoded into model inputs.
    Frame numbers may reset between periods; gaps reset every temporal feature.
    """
    unexpected = set(raw) - RAW_FIELDS
    if unexpected:
        raise ValueError(f"forbidden raw input fields (labels/events/quality are separate): {sorted(unexpected)}")
    required = RAW_FIELDS - {"player_track_id"}
    if not required.issubset(raw):
        raise ValueError(f"missing raw position fields: {sorted(required - set(raw))}")
    frame, period = np.asarray(raw["frame"]), np.asarray(raw["period"])
    if frame.ndim != 1 or period.shape != frame.shape or not len(frame):
        raise ValueError("frame and period must be aligned nonempty vectors")
    n = len(frame)
    if (not np.isfinite(frame).all() or not np.isfinite(period).all()
            or (frame < 0).any() or (period < 0).any()
            or (frame != np.floor(frame)).any() or (period != np.floor(period)).any()):
        raise ValueError("frame and period must contain nonnegative integers")
    if not (period > 0).any():
        raise ValueError("at least one in-period tracking frame is required")
    same_period = period[1:] == period[:-1]
    if ((np.diff(frame) <= 0) & same_period).any():
        raise ValueError("duplicate or unordered frame inside period")
    ball = np.asarray(raw["ball_xy"])
    players = np.asarray(raw["players_xy"])
    if ball.shape != (n, 2) or players.ndim != 3 or players.shape[0] != n or players.shape[2] != 2:
        raise ValueError("ball_xy must be Nx2 and players_xy must be NxPx2")
    ball, players = _normalise(ball, contract), _normalise(players, contract)
    teams = np.asarray(raw["player_team"])
    if teams.shape not in {(players.shape[1],), players.shape[:2]}:
        raise ValueError("player_team must have P or NxP shape")
    teams = np.broadcast_to(teams, players.shape[:2])
    if not np.isin(teams, (-1, 0, 1)).all():
        raise ValueError("player_team must be -1=unknown, 0=home or 1=away")
    if ball_dropout_mask is not None:
        mask = np.asarray(ball_dropout_mask)
        if mask.shape != (n,) or mask.dtype != bool:
            raise ValueError("ball_dropout_mask must be a boolean N-vector")
        ball[mask] = np.nan  # Before speed, distances, nearest team and windows.
    ball[period == 0] = np.nan
    players[period == 0] = np.nan
    contiguous = same_period & (np.diff(frame) == 1) & (period[1:] > 0)
    cuts = np.r_[0, np.flatnonzero(~contiguous) + 1, n]
    blocks = [(int(a), int(b - 1)) for a, b in zip(cuts[:-1], cuts[1:]) if period[a] > 0]
    ball_speed = np.full(n, np.nan)
    ball_speed[1:] = np.linalg.norm(np.diff(ball, axis=0) * contract.pitch, axis=1) * contract.fps
    ball_speed[1:][~contiguous] = np.nan
    player_speed = np.full(players.shape[:2], np.nan)
    player_speed[1:] = np.linalg.norm(np.diff(players, axis=0) * contract.pitch, axis=2) * contract.fps
    valid_pairs = contiguous[:, None] & (teams[1:] == teams[:-1])
    if "player_track_id" in raw:
        ids = np.asarray(raw["player_track_id"])
        if ids.shape not in {(players.shape[1],), players.shape[:2]}:
            raise ValueError("player_track_id must have P or NxP shape")
        ids = np.broadcast_to(ids, players.shape[:2])
        valid_pairs &= ids[1:] == ids[:-1]
        # Unknown identifiers cannot establish that two positions are one tracklet.
        identified = (ids != None) & (ids != "")  # noqa: E711
        if np.issubdtype(ids.dtype, np.number):
            identified &= np.isfinite(ids) & (ids >= 0)
        valid_pairs &= identified[1:] & identified[:-1]
    player_speed[1:][~valid_pairs] = np.nan
    arrays = None
    nearest = np.full(n, -1, dtype=np.int8)
    for index in range(n):
        positions = ((int(team), xy[0], xy[1], speed) for team, xy, speed in
                     zip(teams[index], players[index], player_speed[index]))
        row, nearest[index] = spatial_features(tuple(ball[index]), ball_speed[index], positions,
                                               contract.pitch, RADII)
        if arrays is None:
            arrays = {name: np.full(n, np.nan) for name in row}
        for name, value in row.items():
            arrays[name][index] = value
    add_temporal_features(arrays, contract.fps, WINDOWS_S, contract.pitch, blocks=blocks)
    for team in ("home", "away"):
        arrays[f"n_{team}_positions"] = arrays.pop(f"n_{team}_visible")
    names = sorted(arrays)
    if len(names) != 44:
        raise RuntimeError("base feature definition changed; schema migration required")
    return {
        "X": np.column_stack([arrays[name] for name in names]), "feature_names": np.asarray(names),
        "frame": frame.astype(np.int64), "period": period.astype(np.int16),
        "fps": np.asarray(contract.fps), "nearest_codes": nearest,
        "schema_version": np.asarray(SCHEMA_VERSION),
        "provider": np.asarray(contract.provider), "observation_view": np.asarray(contract.observation_view),
        "feature_contract_json": np.asarray(json.dumps(asdict(contract), sort_keys=True)),
        "feature_definition": np.asarray(FEATURE_DEFINITION),
        "feature_units": np.asarray("positions_normalized;distances_m;speed_mps;windows_s"),
        "pitch": np.asarray(contract.pitch), "orientation": np.asarray(contract.orientation),
    }
