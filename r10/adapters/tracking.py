"""Explicit B-to-R9 geometry contract, implemented for synthetic/public inputs.

B source: models/fusion_v2.py::features(data_dir, cache=None), commit
 a54219dbff8948507b112234127816c133f7c08b (MIT; repository LICENSE).
Only unit/team semantics are reused. No source loader, constants, selected
parameters, default configuration, or match-specific geometry is imported.
"""
import numpy as np


def adapt_tracking(*, frame, period, fps, ball_xy, players_xy, player_team,
                   player_track_id, team_map, coordinate_units, pitch_m,
                   reported_speed=None, speed_units=None):
    """Return copies in centered metres, H=0/A=1, seconds-based speed.

    frame is an integer sampling index; gaps and half resets remain explicit.
    player_track_id is identity, never a persistent slot assumption.
    NaN means missing, never a zero-speed or dead-ball observation.
    """
    frame, period = np.asarray(frame), np.asarray(period)
    if (frame.ndim != 1 or period.shape != frame.shape or
            not np.isfinite(frame).all() or not np.isfinite(period).all() or
            np.any(frame != np.floor(frame)) or np.any(period != np.floor(period)) or
            np.any(frame < 0) or np.any(period < 0) or not np.isfinite(fps) or fps <= 0):
        raise ValueError('invalid timeline')
    if any(np.any(np.diff(frame[period == p]) <= 0) for p in set(period)):
        raise ValueError('duplicate or reversed frames within half')
    ball, players = np.array(ball_xy, float, copy=True), np.array(players_xy, float, copy=True)
    n = len(frame)
    if ball.shape != (n, 2) or players.ndim != 3 or players.shape[0] != n or players.shape[2] != 2:
        raise ValueError('geometry shape')
    if np.isinf(ball).any() or np.isinf(players).any():
        raise ValueError('infinite geometry')
    # A partial coordinate is not a valid position.
    ball[~np.isfinite(ball).all(axis=1)] = np.nan
    players[~np.isfinite(players).all(axis=2)] = np.nan
    pitch = np.asarray(pitch_m, float)
    if pitch.shape != (2,) or not np.isfinite(pitch).all() or np.any(pitch <= 0):
        raise ValueError('pitch dimensions')
    if coordinate_units == 'normalized_0_1':
        ball = (ball - .5) * pitch
        players = (players - .5) * pitch
    elif coordinate_units == 'corner_metres':
        ball -= pitch / 2
        players -= pitch / 2
    elif coordinate_units != 'centered_metres':
        raise ValueError('coordinate units must be explicit')
    shape = players.shape[:2]
    teams, ids = np.asarray(player_team), np.asarray(player_track_id)
    if teams.shape == (shape[1],): teams = np.broadcast_to(teams, shape)
    if ids.shape == (shape[1],): ids = np.broadcast_to(ids, shape)
    if teams.shape != shape or ids.shape != shape:
        raise ValueError('team/id shape')
    if set(team_map.values()) != {0, 1} or len(team_map) != 2:
        raise ValueError('explicit bijection to home0/away1 required')
    mapped = np.full(shape, -1, np.int8)
    for k, value in team_map.items(): mapped[teams == k] = value
    if np.any((mapped == -1) & np.isfinite(players).all(axis=2)):
        raise ValueError('visible player has unmapped team')
    if not np.issubdtype(ids.dtype, np.integer):
        raise ValueError('track identities must be integer, -1 for unknown')
    for row, pos in zip(ids, players):
        known = row[(row >= 0) & np.isfinite(pos).all(axis=1)]
        if len(known) != len(set(known.tolist())):
            raise ValueError('duplicate visible track identity')
    result = dict(frame=frame.astype(np.int64), period=period.astype(np.int64), fps=float(fps),
                  ball_xy=ball, players_xy=players, player_team=mapped,
                  player_track_id=ids.copy(), coordinate_units='centered_metres')
    if reported_speed is not None:
        speed = np.array(reported_speed, float, copy=True)
        if speed.shape != shape or np.isinf(speed).any() or np.any(speed < 0):
            raise ValueError('invalid reported speed')
        if speed_units == 'km/h': speed /= 3.6
        elif speed_units != 'm/s': raise ValueError('speed units must be explicit')
        speed[~np.isfinite(players).all(axis=2)] = np.nan
        result['reported_speed_mps'] = speed
    return result


def player_velocity(batch):
    """Identity-linked backward velocity; never bridge slots, gaps or halves."""
    p, ids, team = (batch[k] for k in ('players_xy', 'player_track_id', 'player_team'))
    out = np.full_like(p, np.nan, dtype=float)
    for i in range(1, len(p)):
        if batch['period'][i] <= 0 or batch['period'][i] != batch['period'][i-1] or batch['frame'][i] != batch['frame'][i-1] + 1:
            continue
        old = {int(v): j for j, v in enumerate(ids[i-1]) if v >= 0}
        for j, v in enumerate(ids[i]):
            k = old.get(int(v)) if v >= 0 else None
            if k is not None and team[i, j] == team[i-1, k]:
                out[i, j] = (p[i, j] - p[i-1, k]) * batch['fps']
    return out
