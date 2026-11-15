"""Training-label-only equal-budget sampling; never an inference feature."""
import numpy as np
from .hierarchical_decoder import continuous_blocks


def select(frame,period,truth,eligible,*,fps,seed):
    frame,period,truth,eligible=map(np.asarray,(frame,period,truth,eligible))
    if any(x.shape!=frame.shape for x in (period,truth,eligible))or frame.ndim!=1 or eligible.dtype!=bool:
        raise ValueError('unaligned training inputs')
    if fps<=0 or not np.isfinite(fps):raise ValueError('invalid fps')
    bins=np.floor(frame*2/fps).astype(int)
    uniform=eligible&(period>0)&np.r_[True,np.diff(bins)>0]
    budget=int(uniform.sum());rng=np.random.default_rng(seed)
    selected=np.zeros(len(frame),bool)
    if not budget:return selected,uniform
    selected[rng.choice(np.flatnonzero(uniform),size=budget//2,replace=False)]=True
    near=np.zeros(len(frame),bool);radius=round(.48*fps)
    for a,b in continuous_blocks(frame,period):
        for t in a+1+np.flatnonzero(truth[a+1:b]!=truth[a:b-1]):
            near[max(a,t-radius):min(b,t+radius+1)]=True
    pool=np.flatnonzero(near&eligible&(period>0)&~selected)
    need=budget-int(selected.sum());take=min(need,len(pool))
    selected[rng.choice(pool,size=take,replace=False)]=True
    need=budget-int(selected.sum())
    if need:
        pool=np.flatnonzero(eligible&(period>0)&~selected)
        selected[rng.choice(pool,size=need,replace=False)]=True
    assert int(selected.sum())==budget
    return selected,uniform
