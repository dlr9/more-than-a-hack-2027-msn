"""Symmetric observation features, full-source targets and frozen-D decoder."""
import numpy as np
from scipy.ndimage import maximum_filter1d
from solution.hierarchical_decoder import continuous_blocks

FIELDS={'X','feature_names','frame','period','fps'}
PAIRS=('d1_m','d2_m','d3_mean_m','within_2p0m','within_5p0m','within_10p0m','local_speed_mean_mps')

def features(b,context=True):
    if set(b)!=FIELDS:raise ValueError('observation-only contract')
    names=b['feature_names'].tolist();x=b['X']
    h=np.column_stack([x[:,names.index('home_'+k)] for k in PAIRS]+[x[:,names.index('n_home_positions')]])
    a=np.column_stack([x[:,names.index('away_'+k)] for k in PAIRS]+[x[:,names.index('n_away_positions')]])
    # Missing paired values have no direction; symmetric missingness retained.
    valid=np.isfinite(h)&np.isfinite(a)
    s=np.where(valid,h-a,0.)
    n=np.column_stack([np.where(valid,h+a,0.),~valid,x[:,names.index('ball_visible')]])
    offsets=[-.8,0.,.8] if context else [0.]
    ss=[];nn=[]
    for t in offsets:
        k=round(t*float(b['fps']));v=np.zeros_like(s);w=np.zeros_like(n)
        for lo,hi in continuous_blocks(b['frame'],b['period']):
            p=max(lo,lo-k);q=min(hi,hi-k)
            if p<q:v[p:q]=s[p+k:q+k];w[p:q]=n[p+k:q+k]
        ss.append(v);nn.append(w)
    signed=np.column_stack(ss)
    hazard=np.column_stack(nn+[abs(v) for v in ss]+([ss[0]*ss[-1],abs(ss[-1]-ss[0])] if context else []))
    return np.nan_to_num(hazard),np.nan_to_num(signed)

def targets(y,frame,period,fps):
    """Training only: positive ±.2s; negatives alive >1s from any transition."""
    y=np.asarray(y);haz=np.full(len(y),-1,dtype=np.int8);direction=np.where(y=='A',1,np.where(y=='H',0,-1)).astype(np.int8)
    counts={'direct':0,'enter_D':0,'exit_D':0,'same_team_restart':0}
    for lo,hi in continuous_blocks(frame,period):
        yy=y[lo:hi];edge=np.r_[False,yy[1:]!=yy[:-1]]
        near=maximum_filter1d(edge.astype(int),2*round(fps)+1,mode='constant')>0
        local=np.full(len(yy),-1,dtype=np.int8);local[np.isin(yy,['H','A'])&~near]=0
        direct=edge&np.isin(yy,['H','A'])&np.r_[False,np.isin(yy[:-1],['H','A'])]
        positive=maximum_filter1d(direct.astype(int),2*round(.2*fps)+1,mode='constant')>0
        local[positive&np.isin(yy,['H','A'])]=1;haz[lo:hi]=local
        counts['direct']+=int(direct.sum());counts['enter_D']+=int((edge&(yy=='D')).sum());counts['exit_D']+=int((edge&np.r_[False,yy[:-1]=='D']).sum())
        bounds=np.r_[0,np.flatnonzero(edge),len(yy)]
        for a,b in zip(bounds[:-1],bounds[1:]):
            if yy[a]=='D' and a>0 and b<len(yy) and yy[a-1]==yy[b] and yy[b] in 'HA':counts['same_team_restart']+=1
    return haz,direction,counts

def rank(score,frame,period,fps,limit=120):
    """No labels/anchors: every frame eligible; fixed .5s suppression."""
    out=[]
    for bi,(a,b) in enumerate(continuous_blocks(frame,period)):
        chosen=[]
        for i in a+np.argsort(-score[a:b],kind='stable'):
            if all(abs(int(i)-j)>=round(.5*fps) for j in chosen):chosen.append(int(i))
            if len(chosen)>=limit:break
        out.extend((i,bi) for i in chosen)
    return sorted(out)

def decode(base,hazard,away,frame,period,fps):
    """Add confirmed switches within old spells; D and old boundaries fixed.

    .8 hazard; pre -.8s and post +.8s direction each >=.8 confident,
    opposite teams, consistent with old current team. Never edit before event,
    across old boundary, or a baseline possession shorter than one second.
    """
    base=np.asarray(base);out=base.copy();accepted=[];k=round(.8*fps)
    proposals=rank(hazard,frame,period,fps)
    for bi,(lo,hi) in enumerate(continuous_blocks(frame,period)):
        bounds=np.r_[lo,lo+1+np.flatnonzero(base[lo+1:hi]!=base[lo:hi-1]),hi]
        for i,block in proposals:
            if block!=bi or hazard[i]<.8 or i-k<lo or i+k>=hi:continue
            pos=np.searchsorted(bounds,i,side='right')-1;a,b=bounds[pos:pos+2]
            if base[i] not in 'HA' or b-a<fps:continue
            before,after=away[i-k],away[i+k]
            team='A' if after>=.8 else 'H' if after<=.2 else None
            prior='H' if before<=.2 else 'A' if before>=.8 else None
            if team is None or prior is None or team==prior or base[i]!=prior:continue
            if i==a:continue
            out[i:b]=team;accepted.append(i)
    assert np.array_equal(out=='D',base=='D')
    return out,accepted
