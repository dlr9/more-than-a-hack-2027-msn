"""Label-free, team-pooled posterior acquisition probes. Not a decoder.

Future observations confirm an earlier change; no future labels are accepted.
Each channel uses identical proposal ranking and temporal windows, so a gain
cannot be attributed just to adding a new smoother to the baseline.
"""
import numpy as np
from solution.hierarchical_decoder import continuous_blocks
from solution.provider_neutral import RAW_FIELDS


def geometry_evidence(raw):
    if set(raw)-RAW_FIELDS:
        raise ValueError('geometry-only fields required')
    n=len(raw['frame']);ball=np.asarray(raw['ball_xy'],float)
    players=np.asarray(raw['players_xy'],float)
    if ball.shape!=(n,2) or players.ndim!=3 or players.shape[0]!=n or players.shape[2]!=2:
        raise ValueError('shape')
    if np.isinf(ball).any() or np.isinf(players).any():raise ValueError('infinite coordinate')
    team=np.broadcast_to(raw['player_team'],players.shape[:2])
    if not np.isin(team,[-1,0,1]).all():raise ValueError('0=H, 1=A, -1=unknown')
    present=np.isfinite(players).all(2)&np.isin(team,[0,1])
    distance=np.linalg.norm(players-ball[:,None,:],axis=2)
    distance[~present|~np.isfinite(distance)]=np.inf
    nearest=[];counts=[]
    for t in (0,1):
        nearest.append(np.min(np.where(team==t,distance,np.inf),axis=1) if players.shape[1] else np.full(n,np.inf))
        counts.append(np.sum(present&(team==t),axis=1))
    valid=np.isfinite(ball).all(1)&(counts[0]>=8)&(counts[1]>=8)
    # Both relative team advantage and absolute receiver proximity matter.
    # Fixed metre scales, no DFL label or target-derived parameter fitting.
    evidence=np.zeros((n,2))
    for t in (0,1):
        advantage=np.zeros(n)
        advantage[valid]=1/(1+np.exp(np.clip((nearest[t][valid]-nearest[1-t][valid])/2,-40,40)))
        evidence[:,t]=advantage*np.exp(-nearest[t]/5)
    return evidence,valid


def interval_mean(x,a,b):
    """Mean in [i+a,i+b), invalid edge windows are zero, never wrap."""
    n=len(x);out=np.zeros_like(x,dtype=float)
    lo=max(0,-a);hi=min(n,n-b+1)
    if hi<=lo:return out
    cs=np.vstack([np.zeros((1,x.shape[1])),np.cumsum(x,axis=0)])
    ix=np.arange(lo,hi);out[ix]=(cs[ix+b]-cs[ix+a])/(b-a)
    return out


def acquisition_strength(evidence,frame,period,fps):
    """Before opponent, subsequent team control, then independent late confirmation.

    Windows [-1,0), [0,.5), [.5,1.5), [1.5,3) seconds. The first future
    window localizes the acquisition; the two later windows check persistence.
    Missing observations carry zero support, never create a D state.
    """
    e=np.asarray(evidence,float)
    if e.shape!=(len(frame),2) or not np.isfinite(e).all() or np.any((e<0)|(e>1)):
        raise ValueError('bounded aligned evidence')
    if not np.isfinite(fps) or fps<2:raise ValueError('fps')
    out=np.zeros_like(e)
    for lo,hi in continuous_blocks(frame,period):
        x=e[lo:hi];past=interval_mean(x,-round(fps),0);before=past[:,::-1]
        onset=interval_mean(x,0,max(1,round(.5*fps)))
        future=interval_mean(x,round(.5*fps),round(1.5*fps))
        late=interval_mean(x,round(1.5*fps),round(3*fps))
        change=(past[:,::-1]>past)&(onset>onset[:,::-1])&(future>future[:,::-1])&(late>late[:,::-1])
        out[lo:hi]=before*onset*np.sqrt(future*late)*change
    return out


def rank_proposals(strength,frame,period,fps,limit=120):
    """Fixed per-continuous-block budget, 0.5s spacing; no GT-dependent selection."""
    proposals=[];radius=max(1,round(.5*fps))
    for block,(lo,hi) in enumerate(continuous_blocks(frame,period)):
        scores=np.max(strength[lo:hi],axis=1);team=np.argmax(strength[lo:hi],axis=1)
        # Exact ties between teams carry no directional information.
        scores=np.where(strength[lo:hi,0]!=strength[lo:hi,1],scores,0)
        used=np.zeros(hi-lo,bool)
        for i in np.argsort(-scores,kind='stable'):
            if scores[i]<=0:break
            if used[i]:continue
            proposals.append(dict(index=int(lo+i),team=int(team[i]),score=float(scores[i]),block=block))
            used[max(0,i-radius):min(hi-lo,i+radius+1)]=True
            if sum(p['block']==block for p in proposals)>=limit:break
    return proposals


def equal_budget(channels,frame,period):
    """Same available count in each block, never refill from labels."""
    out={k:[] for k in channels};budgets=[]
    for block,_ in enumerate(continuous_blocks(frame,period)):
        count=min(sum(p['block']==block for p in v) for v in channels.values())
        budgets.append(count)
        for k,vals in channels.items():out[k].extend([p.copy() for p in vals if p['block']==block][:count])
    return out,budgets


def negative_controls(proposals,frame,period,fps,seed=101603):
    rng=np.random.default_rng(seed);shuffled=[];shifted=[]
    for block,(lo,hi) in enumerate(continuous_blocks(frame,period)):
        vals=[p for p in proposals if p['block']==block]
        teams=rng.permutation([p['team'] for p in vals])
        for p,t in zip(vals,teams):
            shuffled.append({**p,'team':int(t)})
            shifted.append({**p,'index':int(lo+(p['index']-lo+round(30*fps))%(hi-lo))})
    return shuffled,shifted
