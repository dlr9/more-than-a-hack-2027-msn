"""Three-state temporal decoding."""

from __future__ import annotations

import numpy as np


STATE_LABELS = np.array(["H", "A", "D"])


def emission_probabilities(p_alive: np.ndarray, p_home: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    p_alive = np.clip(np.asarray(p_alive, dtype=float), eps, 1 - eps)
    p_home = np.clip(np.asarray(p_home, dtype=float), eps, 1 - eps)
    if p_alive.shape != p_home.shape:
        raise ValueError("p_alive and p_home must have the same shape")
    probabilities = np.column_stack(
        (p_alive * p_home, p_alive * (1 - p_home), 1 - p_alive)
    )
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def _decode_block(probabilities: np.ndarray, team_penalty: float, dead_penalty: float) -> np.ndarray:
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
        candidates = scores[frame - 1][:, None] + transitions
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
) -> np.ndarray:
    """Decode inclusive ``(start, stop)`` blocks independently."""
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != 3:
        raise ValueError("probabilities must have shape (n_frames, 3)")
    labels = np.full(len(probabilities), outside_label, dtype="<U1")
    for start, stop in blocks:
        if start < 0 or stop < start or stop >= len(labels):
            raise ValueError(f"invalid decode block {(start, stop)}")
        states = _decode_block(
            probabilities[start:stop + 1], team_switch_penalty, dead_switch_penalty
        )
        labels[start:stop + 1] = STATE_LABELS[states]
    return labels

