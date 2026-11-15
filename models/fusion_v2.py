"""Tracking-only possession observations and offline temporal decoding. No GT reads."""
import hashlib
import json
import os
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'starter')]
import games
from baseline import team_of_each_track, team_names
from load_smart_tagging import load

VERSION = 3
BLOCKS = [(s, e + 1) for _, s, e in games.half_video_blocks(games.GAME)]
DEFAULT = dict(decoder='viterbi', window=12, gap=50, tag_weight=.3,
               idle=.45, dead_bias=-.8, missing_weight=.5, team_penalty=25.,
               dead_penalty=20., sustain=14, dead_sustain=25, distance=3.2)

def runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))

def rolling(x, radius):
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    sums = np.r_[0., np.cumsum(np.where(ok, x, 0.))]
    counts = np.r_[0, np.cumsum(ok)]
    i = np.arange(len(x)); lo = np.maximum(0, i-radius); hi = np.minimum(len(x), i+radius+1)
    return np.divide(sums[hi]-sums[lo], counts[hi]-counts[lo],
                     out=np.full(len(x), np.nan), where=counts[hi] > counts[lo])

def signature(paths):
    return dict(version=VERSION, sources={str(Path(p).resolve()):
                hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths})

def features(data_dir, cache=None):
    paths = games.require(['atd_raw', 'atd_meta', 'smart_tagging'], data_dir=data_dir)
    sig = signature(paths)
    if cache and Path(cache).exists():
        with np.load(cache) as z:
            if json.loads(str(z['metadata'])) == sig:
                return {k: z[k] for k in z.files if k != 'metadata'}
    raw, meta, _ = paths
    teams = team_of_each_track(meta)
    n = games.GAMES[games.GAME]['n_video_frames']
    players = np.full((n, len(teams), 3), np.nan, np.float32)
    ball = np.full((n, 3), np.nan, np.float32)
    seen = np.zeros(n, bool)
    with open(raw) as handle:
        for line in handle:
            fields = line.strip().split(':')
            if len(fields) != 3:
                raise ValueError('Malformed ATD row')
            f = int(fields[0])-1
            if not 0 <= f < n or seen[f]:
                raise ValueError('Invalid or duplicate ATD frame')
            groups = fields[1].split(';')
            if len(groups) != len(teams):
                raise ValueError('ATD slot count differs from metadata')
            players[f] = [[float(v) for v in g.split(',')] for g in groups]
            ball[f] = [float(v) for v in fields[2].split(',')]
            seen[f] = True
    if not seen.all():
        raise ValueError('ATD has missing rows')
    scale = games.GAMES[games.GAME]['pitch']
    players[:, :, :2] *= scale
    ball[:, :2] *= scale
    # FIFA metadata explicitly declares speed_kph (not m/s).
    speed = players[:, :, 2] / 3.6
    valid = np.isfinite(speed)
    mean = np.divide(np.nansum(speed, axis=1), valid.sum(axis=1),
                     out=np.full(n, np.nan), where=valid.sum(axis=1)>0)
    moving = np.divide(((speed > 2) & valid).sum(axis=1), valid.sum(axis=1),
                       out=np.full(n, np.nan), where=valid.sum(axis=1)>0)
    tags = np.zeros((n, 3), np.float32)
    names = [name.casefold() for name in team_names(meta)]
    attack = {'attacking transition', 'build up', 'progression', 'ball in final third',
              'ball in the box', 'players in the box', 'creating chances', 'long ball'}
    for ev in load(data_dir=data_dir):
        s, e = max(0, ev['start_frame']), min(n, ev['end_frame'])
        code = ev['code'].strip().casefold()
        team = ev['tags'].get('Team', '').strip().casefold()
        if code == 'set pieces':
            tags[s:e, 2] = 1
        elif team in names and (code in attack or code.startswith('defending in')):
            side = names.index(team)
            if code.startswith('defending in'): side = 1-side
            tags[s:e, side] = 1
    result = dict(players=players[:, :, :2], teams=teams, ball=ball[:, :2],
                  mean_speed=mean, moving=moving, tags=tags)
    if cache:
        np.savez_compressed(cache, **result, metadata=json.dumps(sig))
    return result

def observations(f, cfg):
    ball = f['ball'].copy()
    present = np.isfinite(ball).all(axis=1)
    repaired = np.zeros(len(ball), bool)
    for lo, hi in BLOCKS:
        for a, b in runs(~present[lo:hi]):
            s, e = a+lo, b+lo
            if e-s <= cfg['gap'] and s > lo and e < hi:
                ball[s:e] = np.linspace(ball[s-1], ball[e], e-s+2)[1:-1]
                repaired[s:e] = True
    dist = np.sqrt(np.sum((f['players']-ball[:, None, :])**2, axis=2))
    dist = np.where(np.isfinite(dist), dist, np.inf)
    dh = dist[:, f['teams']==0].min(axis=1); da = dist[:, f['teams']==1].min(axis=1)
    valid = np.isfinite(dh) & np.isfinite(da)
    advantage = np.zeros(len(ball))
    advantage[valid] = np.clip((da[valid]-dh[valid])/cfg['distance'], -1, 1)
    proximity = ((dist[:, f['teams']==0] < 3).sum(axis=1) -
                 (dist[:, f['teams']==1] < 3).sum(axis=1))
    advantage = advantage * np.where(repaired, .5, 1) + .1*np.clip(proximity, -2, 2)
    tags = f['tags']
    advantage += cfg['tag_weight']*(tags[:, 0]-tags[:, 1])
    out = (ball[:, 0]<-1) | (ball[:, 0]>106.3) | (ball[:, 1]<-1) | (ball[:, 1]>69)
    speed = np.r_[np.nan, np.sqrt(np.sum(np.diff(ball, axis=0)**2, axis=1))*25]
    result = np.zeros((len(ball), 3))
    for lo, hi in BLOCKS:
        smooth = lambda x: rolling(x[lo:hi], cfg['window'])
        team = np.nan_to_num(smooth(advantage))
        idle = np.nan_to_num(cfg['idle']-smooth(f['mean_speed']))
        dead = (cfg['dead_bias'] + 6.48*idle + 2*smooth(out.astype(float)) +
                cfg['missing_weight']*smooth((~present).astype(float)) +
                cfg['tag_weight']*tags[lo:hi, 2] +
                .2*np.nan_to_num(1-2*smooth(f['moving'])) -
                .25*np.clip(np.nan_to_num(smooth(speed))/5, 0, 1))
        result[lo:hi] = np.column_stack((team, -team, dead))
    return result

def decode(scores, cfg):
    n = len(scores)
    if n == 0: return np.empty(0, np.int8)
    if not np.isfinite(scores).all(): raise ValueError('Nonfinite observation')
    if cfg['decoder'] == 'state':
        state = int(scores[0].argmax()); target = state; count = 0
        result = np.empty(n, np.int8)
        for i, row in enumerate(scores):
            best = int(row.argmax())
            count = count+1 if best == target else 1
            target = best
            threshold = cfg['dead_sustain'] if 2 in (state, best) else cfg['sustain']
            if best != state and count >= threshold:
                state = best
                result[max(0, i-count+1):i] = state
            result[i] = state
        return result
    penalty = np.array([[0, cfg['team_penalty'], cfg['dead_penalty']],
                        [cfg['team_penalty'], 0, cfg['dead_penalty']],
                        [cfg['dead_penalty'], cfg['dead_penalty'], 0]])
    back = np.zeros((n, 3), np.int8)
    value = scores[0].copy()
    for i in range(1, n):
        candidates = value[:, None]-penalty
        back[i] = candidates.argmax(axis=0)
        value = candidates[back[i], np.arange(3)]+scores[i]
        value -= value.max()
    result = np.empty(n, np.int8); result[-1] = value.argmax()
    for i in range(n-1, 0, -1): result[i-1] = back[i, result[i]]
    return result

def predict(f, cfg, blocks=None):
    unknown = set(cfg)-set(DEFAULT)
    if unknown: raise ValueError('Unknown configuration keys: '+str(sorted(unknown)))
    if cfg['decoder'] not in ('state', 'viterbi'):
        raise ValueError('decoder must be state or viterbi')
    for key in ('window', 'gap', 'sustain', 'dead_sustain'):
        if not isinstance(cfg[key], int) or cfg[key] < 0:
            raise ValueError(key+' must be a nonnegative integer')
    if cfg['distance'] <= 0 or min(cfg['team_penalty'], cfg['dead_penalty']) < 0:
        raise ValueError('distance must be positive and penalties nonnegative')
    obs = observations(f, cfg)
    labels = np.full(len(obs), 'X')
    for lo, hi in (BLOCKS if blocks is None else blocks):
        # Published kickoff anchor, independent of public labels.
        start = max(lo, 113742) if lo >= BLOCKS[1][0] else lo
        labels[lo:min(start, hi)] = 'D'
        if start < hi: labels[start:hi] = np.array(['H', 'A', 'D'])[decode(obs[start:hi], cfg)]
    return labels
