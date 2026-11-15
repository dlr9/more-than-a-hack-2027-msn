"""Label-independent observation interventions on public raw ball coordinates.

These simulate stale measurements and smooth measurement noise, not changes to
physical play. They are stress diagnostics, never independent unbiased accuracy.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from .r7_features import blind_intervals
from .quality import period_blocks

def intervene(ball_xy,period,fps,*,mode,seed=7204):
    if mode not in {'freeze','noise','combined'}:raise ValueError('unknown intervention')
    ball=np.array(ball_xy,dtype=float,copy=True);visible=np.isfinite(ball).all(axis=1)
    freeze,windows=blind_intervals(period,fps,scheme='long',seed=seed)
    if mode in {'noise','combined'}:
        for i,(a,b)in enumerate(period_blocks(period)):
            rng=np.random.default_rng(np.random.SeedSequence([seed,100,i]));n=b-a+1
            noise=gaussian_filter1d(rng.normal(size=(n,2)),sigma=max(1,.25*fps),axis=0,mode='reflect')
            # Fixed expected pointwise scale of 0.20m; no use of labels or ball motion.
            noise*=.2*np.sqrt(2*np.sqrt(np.pi)*max(1,.25*fps))
            ball[a:b+1]+=noise
    if mode in {'freeze','combined'}:
        for a,b in period_blocks(period):
            changes=np.diff(np.r_[False,freeze[a:b+1],False].astype(int))
            for lo,hi in zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1)):
                start=a+lo;stop=a+hi
                if visible[start]:ball[start:stop][visible[start:stop]]=ball[start]
    missing=np.zeros(len(ball),bool)
    if mode=='combined':
        missing,_=blind_intervals(period,fps,scheme='long',seed=seed+1);ball[missing]=np.nan
    ball[~visible]=np.nan
    return ball,dict(freeze_mask=freeze,missing_mask=missing,seed=seed,mode=mode)
