"""Label-free, conservative <=1 second reconstruction from raw metric positions.

Research view only: interpolated locations are proposals, never observations or
ground truth. A separate mask records every proposed coordinate.
"""
import numpy as np
from .provider_neutral import RAW_FIELDS
from .hierarchical_decoder import continuous_blocks


def reconstruct(raw, *, fps, max_seconds=1., max_speed=20., max_acceleration=60.):
    if set(raw)-RAW_FIELDS:
        raise ValueError('only raw positions and track identities accepted')
    if not np.isfinite([fps,max_seconds,max_speed,max_acceleration]).all() or min(fps,max_seconds,max_speed,max_acceleration)<=0:
        raise ValueError('invalid physical limits')
    ball=np.array(raw['ball_xy'],dtype=float,copy=True)
    players=np.asarray(raw['players_xy']);n=len(ball)
    if ball.shape!=(n,2)or players.ndim!=3 or players.shape[0]!=n or players.shape[2]!=2:
        raise ValueError('invalid coordinates')
    teams=np.broadcast_to(raw['player_team'],players.shape[:2])
    ids=np.broadcast_to(raw.get('player_track_id',np.arange(players.shape[1])),players.shape[:2])
    valid=np.isfinite(players).all(2)&np.isin(teams,[0,1])
    visible=np.isfinite(ball).all(1);used=np.zeros(n,bool);anchor_team=np.full(n,-1,dtype=np.int8)
    counts={k:0 for k in ['gaps','accepted','long_or_boundary','physical','player_support','control','identity','relative_velocity']}
    for a,b in continuous_blocks(raw['frame'],raw['period']):
        edges=np.flatnonzero(np.diff(np.r_[False,~visible[a:b],False]))
        for lo,hi in edges.reshape(-1,2):
            lo,hi=lo+a,hi+a;counts['gaps']+=1
            if (hi-lo)/fps>max_seconds or lo-2<a or hi+1>=b or not visible[lo-2:lo].all()or not visible[hi:hi+2].all():
                counts['long_or_boundary']+=1;continue
            step=(ball[hi]-ball[lo-1])/(hi-lo+1)
            velocity=step*fps;before=(ball[lo-1]-ball[lo-2])*fps;after=(ball[hi+1]-ball[hi])*fps
            speed=max(np.linalg.norm(x)for x in [velocity,before,after])
            accel=max(np.linalg.norm(velocity-before),np.linalg.norm(after-velocity))*fps
            if speed>max_speed or accel>max_acceleration:
                counts['physical']+=1;continue
            indices=np.arange(lo-1,hi+1)
            if not all(np.all(np.sum(valid[indices]&(teams[indices]==team),axis=1)>=8)for team in [0,1]):
                counts['player_support']+=1;continue
            proposal=ball[lo-1]+np.arange(hi-lo+2)[:,None]*step
            dist=np.linalg.norm(players[indices]-proposal[:,None,:],axis=2)
            dist[~valid[indices]]=np.inf
            nearest=dist.argmin(1);rows=np.arange(len(indices))
            control=teams[indices,nearest];own=dist[rows,nearest]
            opponent=np.min(np.where(teams[indices]!=control[:,None],dist,np.inf),axis=1)
            if not np.all(control==control[0])or np.any(own>2)or np.any(opponent-own<1):
                counts['control']+=1;continue
            # Require one continuous controlling track, not an inferred handover.
            track=ids[indices,nearest]
            if not np.all(track==track[0])or str(track[0])in ('-1','None',''):
                counts['identity']+=1;continue
            positions=players[indices,nearest]
            relative=np.linalg.norm(np.diff(positions,axis=0)*fps-velocity,axis=1)
            if np.any(relative>8):
                counts['relative_velocity']+=1;continue
            ball[lo:hi]=proposal[1:-1];used[lo:hi]=True;anchor_team[lo:hi]=control[0];counts['accepted']+=1
    return ball,dict(imputed=used,anchor_team=anchor_team,counts=counts)
