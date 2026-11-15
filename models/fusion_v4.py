"""Conservative V4: retain V2 team observations, add visible stoppage evidence."""
import json
import numpy as np
from fusion_v2 import ROOT, BLOCKS, observations as v2_observations, decode, rolling
from fusion_v3 import features, prepare as motion_features

BASE = dict(decoder='viterbi', window=12, gap=0, tag_weight=.2, idle=.5,
    dead_bias=-.8, missing_weight=1., team_penalty=60., dead_penalty=30.,
    sustain=20, dead_sustain=30, distance=2.)
DEFAULT = dict(boost=0., team_penalty=60., dead_penalty=30., support='movement')


def prepare(f):
    motion=motion_features(f,0)
    base=v2_observations(f,BASE)
    evidence={key:np.zeros(len(base)) for key in ['movement','stationary']}
    for lo,hi in BLOCKS:
        smooth=lambda x: rolling(np.asarray(x)[lo:hi],BASE['window'])
        visible=smooth(motion['present'].astype(float))
        slow=np.clip((smooth(f['slow'])-.4)/.4,0,1)
        idle=np.clip((.8-smooth(f['mean_speed']))/.4,0,1)
        speed=smooth(motion['speed'])
        motion_gate=np.clip(1-speed/5,0,1)
        station=smooth(motion['station'])
        evidence['movement'][lo:hi]=np.nan_to_num(visible*slow*idle*motion_gate)
        evidence['stationary'][lo:hi]=np.nan_to_num(visible*slow*idle*station)
    return dict(base=base,**evidence)


def predict(f,c,prepared=None):
    if set(c)!=set(DEFAULT): raise ValueError('V4 configuration schema mismatch')
    if c['support'] not in ('movement','stationary'): raise ValueError('Invalid support')
    for k in ['boost','team_penalty','dead_penalty']:
        if not np.isfinite(c[k]) or c[k]<0: raise ValueError('Invalid '+k)
    p=prepared if prepared is not None else prepare(f)
    obs=p['base'].copy()
    obs[:,2]+=c['boost']*p[c['support']]
    cfg=dict(BASE,team_penalty=c['team_penalty'],dead_penalty=c['dead_penalty'])
    labels=np.full(len(obs),'X')
    for lo,hi in BLOCKS:
        start=113742 if lo==110808 else lo
        labels[lo:min(start,hi)]='D'
        labels[start:hi]=np.array(['H','A','D'])[decode(obs[start:hi],cfg)]
    return labels
