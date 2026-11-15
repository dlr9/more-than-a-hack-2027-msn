"""Observable control episodes and bounded H/A boundary proposals (research only).

Consumes metric geometry only. No status, event labels, or possession input.
Relative player/ball motion supplies evidence absent from the frozen 44 columns.
Scores are physical heuristics, not calibrated event probabilities.
"""
import numpy as np
from .hierarchical_decoder import continuous_blocks
from .provider_neutral import RAW_FIELDS


def control_episodes(raw, fps):
    if set(raw)-RAW_FIELDS: raise ValueError('geometry-only input required')
    if not np.isfinite(fps) or fps<=0: raise ValueError('fps')
    frame=np.asarray(raw['frame']);period=np.asarray(raw['period']);n=len(frame)
    ball=np.asarray(raw['ball_xy'],float);players=np.asarray(raw['players_xy'],float)
    if ball.shape!=(n,2) or players.ndim!=3 or players.shape[0]!=n or players.shape[2]!=2:raise ValueError('geometry alignment')
    if np.isinf(ball).any() or np.isinf(players).any():raise ValueError('infinite coordinate')
    if players.shape[1]==0:return [],np.full(n,-1,dtype=np.int8)
    team=np.broadcast_to(raw['player_team'],players.shape[:2])
    ids=np.broadcast_to(raw.get('player_track_id',np.arange(players.shape[1])),team.shape)
    present=np.isfinite(players).all(2)&np.isin(team,[0,1]);visible=np.isfinite(ball).all(1)
    distance=np.linalg.norm(players-ball[:,None,:],axis=2);distance[~present|~np.isfinite(distance)]=np.inf
    nearest=distance.argmin(1);rows=np.arange(n);own=distance[rows,nearest];side=team[rows,nearest]
    opposing=np.min(np.where(team!=side[:,None],distance,np.inf),axis=1)
    margin=np.full(n,-np.inf);finite=np.isfinite(own)&np.isfinite(opposing)
    margin[finite]=opposing[finite]-own[finite]
    chosen=players[rows,nearest];track=ids[rows,nearest]
    signal=np.full(n,-1,dtype=np.int8);events=[]
    for lo,hi in continuous_blocks(frame,period):
        if hi-lo<2:continue
        bv=np.diff(ball[lo:hi],axis=0)*fps;pv=np.diff(chosen[lo:hi],axis=0)*fps
        speed=np.linalg.norm(bv,axis=1);pspeed=np.linalg.norm(pv,axis=1);relative=np.linalg.norm(bv-pv,axis=1)
        valid=(visible[lo:hi-1]&visible[lo+1:hi]&(nearest[lo:hi-1]==nearest[lo+1:hi])&(track[lo:hi-1]==track[lo+1:hi])
            &(side[lo:hi-1]==side[lo+1:hi])&(own[lo+1:hi]<=2)&(margin[lo+1:hi]>=1)
            &(speed>=.5)&(speed<=12)&(pspeed<=12)&(relative<=3))
        for t in (0,1):valid&=np.sum(present[lo+1:hi]&(team[lo+1:hi]==t),axis=1)>=8
        values=np.where(valid,side[lo+1:hi],-1)
        # Consecutive observations of one controller establish control, not a touch alone.
        cuts=np.r_[0,np.flatnonzero((values[1:]!=values[:-1])|(track[lo+2:hi]!=track[lo+1:hi-1]))+1,len(values)]
        for a,b in zip(cuts[:-1],cuts[1:]):
            if values[a]<0 or b-a<max(2,int(np.ceil(.12*fps))):continue
            start,end=lo+1+a,lo+1+b
            signal[start:end]=values[a]
            events.append(dict(start=int(start),end=int(end),team='H' if values[a]==0 else 'A',
                support_seconds=(end-start)/fps,source='observed_relative_motion_v1',confidence='uncalibrated_physical_filter'))
    return events,signal


def relocate_boundaries(pred,home,frame,period,fps,events):
    """Finite local segment comparison; never create/remove D or a short spell."""
    pred=np.asarray(pred);home=np.asarray(home,float)
    if pred.shape!=home.shape or pred.shape!=np.asarray(frame).shape:raise ValueError('alignment')
    if not np.isin(pred,list('HAD')).all() or not np.isfinite(home).all() or np.any((home<0)|(home>1)):raise ValueError('labels/probabilities')
    out=pred.copy();trace=dict(eligible_boundaries=0,moved_boundaries=0,changed_frames=0,proposals=len(events))
    logs=np.log(np.clip(np.column_stack([1-home,home]),1e-12,1))/fps
    for lo,hi in continuous_blocks(frame,period):
        edges=np.r_[lo,np.flatnonzero(pred[lo+1:hi]!=pred[lo:hi-1])+lo+1,hi]
        proposed=edges.copy()
        for j in range(1,len(edges)-1):
            t=int(edges[j]);left,right=pred[t-1],pred[t]
            if left=='D' or right=='D':continue
            candidates=sorted(set(e['start'] for e in events if e['team']==right and abs(e['start']-t)<=round(fps) and edges[j-1]<e['start']<edges[j+1]))
            if not candidates:continue
            trace['eligible_boundaries']+=1
            # Fixed 0.2 second evidence bonus, not fitted probability; retain ties.
            a,b=int(edges[j-1]),int(edges[j+1]);lc=int(left=='H');rc=int(right=='H')
            def score(x):return logs[a:x,lc].sum()+logs[x:b,rc].sum()+(.2 if x in candidates else 0.)
            best=t;best_score=score(t)
            for x in candidates:
                s=score(x)
                if s>best_score+1e-12:best,best_score=x,s
            proposed[j]=best
        # Reject simultaneous collisions instead of deleting short possessions.
        bad=[j for j in range(1,len(edges)-1) if proposed[j]<=proposed[j-1] or proposed[j]>=proposed[j+1]]
        for j in bad:proposed[j]=edges[j]
        if np.any(np.diff(proposed)<=0):raise AssertionError('crossing boundaries')
        for a,b,s in zip(proposed[:-1],proposed[1:],pred[edges[:-1]]):out[a:b]=s
        trace['moved_boundaries']+=int(np.sum(proposed!=edges))
    np.testing.assert_array_equal(out=='D',pred=='D')
    trace['changed_frames']=int(np.sum(out!=pred))
    return out,trace
