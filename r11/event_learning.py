"""Observation-only inputs for conditional external event supervision.

Event metadata defines training targets only. The model cannot discover event
existence from positives alone, and these outputs are not complete possession.
"""
import numpy as np
from solution.hierarchical_decoder import continuous_blocks

OBSERVATION_FIELDS={'X','feature_names','frame','period','fps'}

def temporal_inputs(batch,offset_seconds):
    if set(batch)!=OBSERVATION_FIELDS:raise ValueError('only observation fields allowed')
    x=np.asarray(batch['X'],float);names=np.asarray(batch['feature_names']);frame=np.asarray(batch['frame']);period=np.asarray(batch['period']);fps=float(batch['fps'])
    if x.ndim!=2 or x.shape[0]!=len(frame) or x.shape[1]!=len(names) or not np.isfinite(fps) or fps<=0:raise ValueError('shape/fps')
    offsets=[round(float(t)*fps) for t in offset_seconds]
    out=np.full((len(x),len(offsets)*x.shape[1]),np.nan)
    for a,b in continuous_blocks(frame,period):
        for j,k in enumerate(offsets):
            lo=max(a,a-k);hi=min(b,b-k)
            if lo<hi:out[lo:hi,j*x.shape[1]:(j+1)*x.shape[1]]=x[lo+k:hi+k]
    out[~np.isfinite(out)]=np.nan
    return out


def supervised_rows(labels,batch):
    """Only explicit event positives/negatives with observed teams, never gap=0."""
    names=batch['feature_names'].tolist();x=batch['X']
    observed=(x[:,names.index('n_home_positions')]>0)&(x[:,names.index('n_away_positions')]>0)
    return (np.asarray(labels)>=0)&observed&(batch['period']>0)
