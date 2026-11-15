"""R8 research-only factorized decoding and bounded observation anchors.

No label arguments, training, or production-runner integration. Transition costs
are seconds of unit negative log likelihood; emissions are integrated at 1/fps.
"""
import numpy as np


def continuous_blocks(frame, period):
    frame, period = np.asarray(frame), np.asarray(period)
    if frame.ndim != 1 or period.shape != frame.shape or not np.isfinite(frame).all():
        raise ValueError('invalid timeline')
    cuts = np.r_[0, np.flatnonzero((np.diff(frame) != 1) | (np.diff(period) != 0))+1, len(frame)]
    return [(a, b) for a, b in zip(cuts[:-1], cuts[1:]) if b > a and period[a] > 0]


def binary_decode(p, fps, switch_seconds):
    p = np.asarray(p, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('invalid head probability')
    if not np.isfinite([fps, switch_seconds]).all() or fps <= 0 or switch_seconds < 0:
        raise ValueError('invalid time scale')
    if not len(p):
        return np.zeros(0, dtype=bool)
    cost = -np.log(np.clip(np.column_stack((1-p, p)), 1e-12, 1))/fps
    back = np.zeros((len(p), 2), dtype=np.int8)
    score = cost[0].copy()
    for i in range(1, len(p)):
        b0 = int(score[1] + switch_seconds < score[0])
        b1 = int(score[1] <= score[0] + switch_seconds)
        back[i] = b0, b1
        score = cost[i] + [score[b0] + switch_seconds*b0,
                           score[b1] + switch_seconds*(1-b1)]
    states = np.empty(len(p), dtype=bool)
    states[-1] = score[1] < score[0]
    for i in range(len(p)-1, 0, -1):
        states[i-1] = back[i, int(states[i])]
    return states


def decode(alive, home, frame, period, fps, dead_seconds=.1, team_seconds=.3):
    alive, home = np.asarray(alive), np.asarray(home)
    if alive.shape != np.asarray(frame).shape or home.shape != alive.shape:
        raise ValueError('unaligned heads')
    # Validate even if all periods are outside play.
    for p in (alive, home):
        if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError('invalid head probability')
    output = np.full(len(alive), 'X', dtype='<U1')
    for a, b in continuous_blocks(frame, period):
        live = binary_decode(alive[a:b], fps, dead_seconds)
        output[a:b] = 'D'
        edges = np.flatnonzero(np.diff(np.r_[False, live, False]))
        for lo, hi in edges.reshape(-1, 2):
            h = binary_decode(home[a+lo:a+hi], fps, team_seconds)
            output[a+lo:a+hi] = np.where(h, 'H', 'A')
    return output


def anchor_probabilities(alive, home, *, frame, period, fps, visible,
                         home_distance, away_distance, ball_speed,
                         player_count, max_gap_seconds=2.):
    """Only missing runs bracketed by stable same-team control can use priors.

    A confident D anywhere in the gap vetoes it. Both endpoints must be adjacent
    to the gap and stable for .2s on their observed sides. No one-sided memory.
    Returns soft probabilities and masks, never inferred positions or truth.
    """
    q, r = np.array(alive, dtype=float, copy=True), np.array(home, dtype=float, copy=True)
    n = len(q)
    arrays = [r, frame, period, visible, home_distance, away_distance, ball_speed, player_count]
    if any(np.asarray(x).shape != (n,) for x in arrays):
        raise ValueError('unaligned anchor inputs')
    if not np.isfinite(fps) or fps <= 0 or not np.isfinite(max_gap_seconds) or max_gap_seconds <= 0:
        raise ValueError('invalid anchor time scale')
    if not np.isfinite(q).all() or not np.isfinite(r).all() or np.any((q<0)|(q>1)|(r<0)|(r>1)):
        raise ValueError('invalid anchor probabilities')
    visible = np.asarray(visible, dtype=bool)
    hd, ad, speed = map(np.asarray, (home_distance, away_distance, ball_speed))
    reliable = visible & (q >= .9) & np.isfinite(speed) & (speed <= 12) & (speed >= 0) & (np.asarray(player_count) >= 16)
    anchor = np.full(n, -1, dtype=np.int8)
    anchor[reliable & (hd < 2) & (ad-hd > 2) & (r >= .8)] = 1
    anchor[reliable & (ad < 2) & (hd-ad > 2) & (r <= .2)] = 0
    used = np.zeros(n, dtype=bool)
    unsupported = (~visible) & (np.asarray(player_count) == 0) & (np.asarray(period)>0)
    width = max(1, round(.2*fps))
    for a, b in continuous_blocks(frame, period):
        edges = np.flatnonzero(np.diff(np.r_[False, ~visible[a:b], False]))
        for lo, hi in edges.reshape(-1, 2):
            lo, hi = a+lo, a+hi
            if (hi-lo)/fps > max_gap_seconds or lo-width < a or hi+width > b:
                continue
            team = anchor[lo-1]
            if team < 0 or not np.all(anchor[lo-width:lo] == team) or not np.all(anchor[hi:hi+width] == team):
                continue
            if np.any(q[lo:hi] <= .1):
                continue
            distance = np.minimum(np.arange(1, hi-lo+1), np.arange(hi-lo, 0, -1))/fps
            weight = .5 * 2**(-distance/1.)
            q[lo:hi] = (1-weight)*q[lo:hi] + weight*min(q[lo-1], q[hi])
            anchor_r = min(r[lo-1], r[hi]) if team else max(r[lo-1], r[hi])
            r[lo:hi] = (1-weight)*r[lo:hi] + weight*anchor_r
            used[lo:hi] = True
    return q, r, dict(anchor=anchor, used=used, unsupported=unsupported)


def anchored(alive, home, batch):
    names = batch['feature_names'].tolist()
    def col(name):
        return batch['X'][:, names.index(name)]
    return anchor_probabilities(alive, home, frame=batch['frame'], period=batch['period'], fps=float(batch['fps']),
        visible=col('ball_visible') == 1, home_distance=col('home_d1_m'), away_distance=col('away_d1_m'),
        ball_speed=col('ball_speed_mps'), player_count=col('n_home_positions')+col('n_away_positions'))
