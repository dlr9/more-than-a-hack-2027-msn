"""Finite-horizon prior propagation for wholly unobserved intervals, not evidence.

Observed confident D clears team memory; periods/frame gaps reset everything.
Never accesses a label. This experimental fallback can miss hidden dead events.
"""
import numpy as np

def carry_priors(base,p_alive,p_home,*,team_memory=False):
    names=base['feature_names'].tolist();x=base['X'];fps=float(base['fps'])
    visible=x[:,names.index('ball_visible')]==1
    players=x[:,names.index('n_home_positions')]+x[:,names.index('n_away_positions')]
    speed_good=x[:,names.index('ball_speed_missing')]==0
    home_d=x[:,names.index('home_d1_m')];away_d=x[:,names.index('away_d1_m')]
    no_observation=(~visible)&(players==0)&(base['period']>0)
    a=np.array(p_alive,copy=True);h=np.array(p_home,copy=True);used_a=np.zeros(len(a),bool);used_h=used_a.copy()
    last_a=last_h=None
    for i in range(len(a)):
        if i==0 or base['period'][i]!=base['period'][i-1] or base['frame'][i]!=base['frame'][i-1]+1 or base['period'][i]<=0:
            last_a=last_h=None
        if base['period'][i]<=0:continue
        if visible[i] and speed_good[i]:
            if p_alive[i]<.1:
                last_a=(i,p_alive[i]);last_h=None
            elif p_alive[i]>.9:
                last_a=(i,p_alive[i])
                home_control=p_home[i]>.8 and home_d[i]<2 and away_d[i]-home_d[i]>2
                away_control=p_home[i]<.2 and away_d[i]<2 and home_d[i]-away_d[i]>2
                if home_control or away_control:last_h=(i,p_home[i])
        if not no_observation[i]:continue
        if last_a is not None:
            age=(i-last_a[0])/fps
            if age<=30:
                a[i]=.5+(last_a[1]-.5)*2**(-age/30);used_a[i]=True
        if team_memory and last_h is not None and a[i]>.5:
            age=(i-last_h[0])/fps
            if age<=30:
                h[i]=.5+(last_h[1]-.5)*2**(-age/10);used_h[i]=True
    return a,h,dict(alive_prior_used=used_a,home_prior_used=used_h,no_observation=no_observation)

def blackout_mask(period,fps,seed=7210):
    """Timeline-only outage schedule; never reads positions or labels."""
    from .quality import period_blocks
    mask=np.zeros(len(period),bool)
    for k,(a,b)in enumerate(period_blocks(period)):
        rng=np.random.default_rng(np.random.SeedSequence([seed,k]));i=a
        while i<=b:
            i+=round(rng.uniform(40,80)*fps)
            duration=round(float(rng.choice([1.,5.,10.,30.]))*fps)
            mask[i:min(b+1,i+duration)]=True;i+=duration
    return mask
