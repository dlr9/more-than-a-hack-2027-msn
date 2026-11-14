"""Three-state temporal decoding."""

from __future__ import annotations

import numpy as np


STATE_LABELS = np.array(["H", "A", "D"])


def emission_probabilities(p_alive: np.ndarray, p_home: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    if not (0 < eps < 0.5) or not np.isfinite(p_alive).all() or not np.isfinite(p_home).all():
        raise ValueError("probabilities must be finite")
    if np.ndim(p_alive) != 1 or np.ndim(p_home) != 1:
        raise ValueError("head probabilities must be vectors")
    p_alive = np.clip(np.asarray(p_alive, dtype=float), eps, 1 - eps)
    p_home = np.clip(np.asarray(p_home, dtype=float), eps, 1 - eps)
    if p_alive.shape != p_home.shape:
        raise ValueError("p_alive and p_home must have the same shape")
    probabilities = np.column_stack(
        (p_alive * p_home, p_alive * (1 - p_home), 1 - p_alive)
    )
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def _decode_block(probabilities: np.ndarray, team_penalty: float, dead_penalty: float, control: np.ndarray | None = None) -> np.ndarray:
    if len(probabilities) == 0:
        return np.empty(0, dtype=np.int8)
    cost = -np.log(np.clip(probabilities, 1e-12, 1.0))
    transitions = np.array(
        [
            [0.0, team_penalty, dead_penalty],
            [team_penalty, 0.0, dead_penalty],
            [dead_penalty, dead_penalty, 0.0],
        ]
    )
    scores = np.empty_like(cost)
    back = np.empty(cost.shape, dtype=np.int8)
    scores[0] = cost[0]
    back[0] = -1
    for frame in range(1, len(cost)):
        step = transitions
        if control is not None and control[frame] in (0, 1):
            # Local, observed control supports a team switch; it never asserts D.
            step = transitions.copy()
            destination = int(control[frame])
            step[1 - destination, destination] *= 0.25
        candidates = scores[frame - 1][:, None] + step
        back[frame] = np.argmin(candidates, axis=0)
        scores[frame] = cost[frame] + candidates[back[frame], np.arange(3)]
    states = np.empty(len(cost), dtype=np.int8)
    states[-1] = int(np.argmin(scores[-1]))
    for frame in range(len(cost) - 1, 0, -1):
        states[frame - 1] = back[frame, states[frame]]
    return states


def viterbi_decode(
    probabilities: np.ndarray,
    blocks: list[tuple[int, int]],
    team_switch_penalty: float,
    dead_switch_penalty: float,
    outside_label: str = "X",
    control: np.ndarray | None = None,
) -> np.ndarray:
    """Decode inclusive ``(start, stop)`` blocks independently."""
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != 3:
        raise ValueError("probabilities must have shape (n_frames, 3)")
    if (not np.isfinite(probabilities).all() or (probabilities < 0).any()
            or not np.allclose(probabilities.sum(axis=1), 1)):
        raise ValueError("invalid probability simplex")
    if not np.isfinite([team_switch_penalty, dead_switch_penalty]).all() or min(team_switch_penalty, dead_switch_penalty) < 0:
        raise ValueError("invalid transition penalty")
    if outside_label not in "HADX" or len(outside_label) != 1:
        raise ValueError("invalid outside label")
    if control is not None:
        control = np.asarray(control)
        if control.shape != (len(probabilities),) or not np.isin(control, [-1, 0, 1]).all():
            raise ValueError("invalid control evidence")
    labels = np.full(len(probabilities), outside_label, dtype="<U1")
    previous = -1
    for start, stop in blocks:
        if start <= previous or stop < start or stop >= len(labels):
            raise ValueError(f"invalid decode block {(start, stop)}")
        previous = stop
        states = _decode_block(
            probabilities[start:stop + 1], team_switch_penalty, dead_switch_penalty,
            None if control is None else control[start:stop + 1]
        )
        labels[start:stop + 1] = STATE_LABELS[states]
    return labels


def observed_control(batch: dict) -> np.ndarray:
    """Experimental proximity evidence, NOT independently verified control events."""
    names = batch['feature_names'].astype(str).tolist()
    x = batch['X']
    home, away = x[:,names.index('home_d1_m')], x[:,names.index('away_d1_m')]
    visible = x[:,names.index('ball_visible')] == 1
    raw = np.full(len(x), -1, dtype=np.int8)
    raw[visible & (home < 1.5) & (away-home > 2)] = 0
    raw[visible & (away < 1.5) & (home-away > 2)] = 1
    evidence = np.full(len(x), -1, dtype=np.int8)
    width = max(1, round(.3*float(batch['fps'])))
    for i in range(width-1, len(raw)):
        lo = i-width+1
        if raw[i]>=0 and np.all(raw[lo:i+1]==raw[i]) and np.all(batch['period'][lo:i+1]==batch['period'][i]):
            evidence[i]=raw[i]
    return evidence
