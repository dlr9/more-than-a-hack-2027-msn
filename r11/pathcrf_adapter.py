"""Public-other-match, identity-aware PathCRF input contract (no file IO).

Original tensor layout: hyunsungkim-ds/pathcrf dataset.py at 1b7c6f8,
MPL-2.0. This independently implemented adapter accepts observations only.
It rejects missing players/identity/keeper ambiguity rather than imputing them.
The raw model has no 25 Hz boundary or complete Alive/Dead output contract.
"""
import numpy as np
from scipy.signal import savgol_filter


class ContractError(ValueError):
    pass


def adapt_window(*, frame, period, players_xy, player_team, player_track_id,
                 fps, pitch_m, goalkeeper_ids):
    """Exactly ten seconds, sampled at 5 Hz, no ball or label input.

    Positions are centered metres. Each frame may permute its slots, but real
    identities and teams must remain consistent. 22 complete real trajectories
    are required; no track association or missing-player hallucination occurs.
    Keeper identities must be explicitly supplied, including their provenance
    at the caller. Returns left-first nodes and a mapping back to H=0/A=1.
    """
    f, half = np.asarray(frame), np.asarray(period)
    p = np.asarray(players_xy, float)
    team, ids = np.asarray(player_team), np.asarray(player_track_id)
    pitch = np.asarray(pitch_m, float)
    if not np.isfinite(fps) or fps not in (10., 25.):
        raise ContractError('only exact 10/25-to-5Hz sampling supported')
    n = round(10*fps)
    if f.shape != (n,) or half.shape != f.shape or not np.isfinite(f).all():
        raise ContractError('exactly ten seconds of frames required')
    if np.any(np.diff(f) != 1) or np.any(f != np.floor(f)) or np.any(f < 0):
        raise ContractError('frame gap or invalid index')
    if len(set(half.tolist())) != 1 or half[0] not in (1, 2):
        raise ContractError('window crosses half or invalid half')
    if p.ndim != 3 or p.shape[0] != n or p.shape[2] != 2 or np.isinf(p).any():
        raise ContractError('invalid positions')
    if team.ndim == 1: team = np.broadcast_to(team, p.shape[:2])
    if ids.ndim == 1: ids = np.broadcast_to(ids, p.shape[:2])
    if team.shape != p.shape[:2] or ids.shape != p.shape[:2]:
        raise ContractError('identity/team shape')
    if pitch.shape != (2,) or not np.isfinite(pitch).all() or np.any(pitch <= 0):
        raise ContractError('invalid pitch')
    if goalkeeper_ids is None or set(goalkeeper_ids) != {0, 1}:
        raise ContractError('goalkeeper identity unknown')
    # Only real visible players, not bench slots. Missingness at any time rejects.
    records = []
    for t in range(n):
        visible = np.isfinite(p[t]).all(axis=1)
        current = {}
        for j in np.flatnonzero(visible):
            key = str(ids[t, j])
            if key in ('', '-1', 'None', 'nan') or key in current or team[t, j] not in (0, 1):
                raise ContractError('unknown/duplicate identity or team')
            current[key] = (int(team[t, j]), p[t, j])
        if any(sum(v[0] == side for v in current.values()) != 11 for side in (0, 1)):
            raise ContractError('requires eleven observed players per team at every frame')
        records.append(current)
    ordered = sorted(records[0])
    canonical = np.empty((n, 22, 2)); identity_team = []
    for j, identity in enumerate(ordered):
        side = records[0][identity][0]; identity_team.append(side)
        for t, row in enumerate(records):
            if set(row) != set(ordered) or row[identity][0] != side:
                raise ContractError('identity set/team changed within window')
            canonical[t, j] = row[identity][1]
    identity_team = np.asarray(identity_team)
    keepers = [str(goalkeeper_ids[s]) for s in (0, 1)]
    if any(k not in ordered or identity_team[ordered.index(k)] != s for s, k in enumerate(keepers)):
        raise ContractError('keeper identity/team mismatch')
    gx = [canonical[:, ordered.index(k), 0].mean() for k in keepers]
    if abs(gx[0]-gx[1]) < pitch[0]/2:
        raise ContractError('keeper orientation not reliable')
    left = int(gx[1] < gx[0]); right = 1-left
    order = [j for side in (left, right) for j in range(22) if identity_team[j] == side]
    canonical = canonical[:, order]
    # Convert centered geometry to the upstream 105x68 corner coordinate system.
    scale = np.array([105., 68.])/pitch
    xy = (canonical + pitch/2)*scale
    velocity = savgol_filter(np.diff(xy, axis=0)*fps, 15, 2, axis=0)
    acceleration = savgol_filter(np.diff(velocity, axis=0)*fps, 9, 2, axis=0)
    v = np.concatenate([velocity[:1], velocity], axis=0)
    accel = np.zeros((n, 22)); accel[1:-1] = np.linalg.norm(acceleration, axis=2)
    tensor = np.zeros((n, 26, 8), np.float32)
    tensor[:, :11, 0] = -1; tensor[:, 11:22, 0] = 1
    for j, original in enumerate(order): tensor[:, j, 1] = ordered[original] in keepers
    tensor[:, :22, 2:4] = xy; tensor[:, :22, 4:6] = v
    tensor[:, :22, 6] = np.linalg.norm(v, axis=2); tensor[:, :22, 7] = accel
    tensor[:, 22:, 2:4] = [[0, 34], [105, 34], [52.5, 0], [52.5, 68]]
    sampled = np.arange(0, n, round(fps/5))
    return dict(nodes=tensor[sampled], frames=f[sampled].copy(), period=int(half[0]),
                fps=5., input_fps=float(fps), left_team=left,
                node_ids=[ordered[j] for j in order]+['out_left','out_right','out_bottom','out_top'],
                quality={'complete_players':True, 'identity_association':'provided_id',
                         'kinematic_contract':'SG15_velocity_SG9_accel_window_local_no_outlier_fill',
                         'boundary_resolution_seconds':.2, 'full_had_supported':False})
