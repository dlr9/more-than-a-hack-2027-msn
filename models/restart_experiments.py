"""Opt-in V5 restart latches. Thresholds fixed before diagnostic scoring."""
import numpy as np


def hull_area(points):
    p=sorted(set(map(tuple, points)))
    if len(p)<3:return 0.
    def cross(o,a,b):return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    def chain(seq):
        out=[]
        for x in seq:
            while len(out)>=2 and cross(out[-2],out[-1],x)<=0:out.pop()
            out.append(x)
        return out
    h=np.array(chain(p)[:-1]+chain(p[::-1])[:-1])
    return float(abs(np.dot(h[:,0],np.roll(h[:,1],1))-np.dot(h[:,1],np.roll(h[:,0],1)))/2)


def formation(players,teams):
    valid=np.isfinite(players).all(axis=1)
    ids=np.flatnonzero(valid);p=players[valid];t=teams[valid]
    # Do not interpret missing/duplicate tracks as all 22 players satisfying a rule.
    if len(p)!=22 or any((t==k).sum()!=11 for k in (0,1)):return None
    for gx in (0.,105.3):
        box=(abs(p[:,0]-gx)<16.5)&(abs(p[:,1]-34)<20.16)
        keeper=(abs(p[:,0]-gx)<2)&(abs(p[:,1]-34)<4.5)
        spot=np.array([11. if gx==0 else 94.3,34.])
        takers=np.flatnonzero(np.linalg.norm(p-spot,axis=1)<5)
        for k in np.flatnonzero(keeper):
            for taker in takers:
                if t[k]==t[taker]:continue
                others=np.ones(22,bool);others[[k,taker]]=False
                if not box[others].any():
                    return dict(type='penalty',slot=int(ids[taker]),team=int(t[taker]),
                        box_count=int(box.sum()),hull_area=hull_area(p))
        for gy in (0.,68.):
            near=np.flatnonzero(np.linalg.norm(p-[gx,gy],axis=1)<3)
            if len(near)==1 and box.sum()>14:
                j=near[0];dist=np.linalg.norm(p-p[j],axis=1);dist[j]=np.inf
                if dist.min()>5:
                    return dict(type='corner',slot=int(ids[j]),team=int(t[j]),
                        box_count=int(box.sum()),hull_area=hull_area(p))
    return None


def motion(f,blocks):
    p=f['players'];b=f['ball'];speed=np.full(p.shape[:2],np.nan);bs=np.full(len(b),np.nan)
    for lo,hi in blocks:
        # Five consecutive finite intervals: no bridging missing coordinates.
        for i in range(lo+5,hi):
            ok=np.isfinite(p[i-5:i+1]).all(axis=(0,2))
            speed[i,ok]=np.linalg.norm(p[i,ok]-p[i-5,ok],axis=1)*5
            if np.isfinite(b[i-5:i+1]).all():bs[i]=np.linalg.norm(b[i]-b[i-5])*5
    speed[speed>12]=np.nan;bs[bs>40]=np.nan
    return speed,bs


def masks(f,blocks,events,mode):
    speed,bs=motion(f,blocks);n=len(bs);mask=np.zeros(n,bool);audit=[]
    if mode=='player_only':
        for lo,hi in blocks:
            active=None;last_formation=lo;cooldown=lo
            for i in range(lo,hi,5):
                g=formation(f['players'][i],f['teams'])
                if active is None and i>=cooldown and g:
                    active=dict(g,start=i,slow=False);last_formation=i
                if active is None:continue
                slot=active['slot'];v=speed[i,slot]
                if np.isfinite(v) and v<.5:active['slow']=True
                if g and g['slot']==slot:last_formation=i
                reason=None
                if active['slow'] and np.isfinite(v) and v>3.5:reason='taker_sprint_proxy'
                elif i-last_formation>=25:reason='formation_lost_1s'
                elif i-active['start']>=750:reason='safety_timeout_30s'
                if reason:
                    audit.append(dict(active,end=i,release=reason));active=None;cooldown=i+50
                else:mask[i:min(i+5,hi)]=True
            if active:audit.append(dict(active,end=hi,release='half_end'))
    elif mode=='tag_head':
        for e in events:
            if e['code'].casefold()!='set pieces':continue
            for lo,hi in blocks:
                start=max(lo,e['start_frame']);end=min(hi,e['end_frame']);mid=(e['start_frame']+e['end_frame'])//2
                active=None;slow=np.zeros(speed.shape[1],bool)
                for i in range(start,end):
                    visible=np.isfinite(f['players'][i]).all(axis=1).sum()
                    if active is None:
                        if i>=mid:break
                        if visible<18 or not np.isfinite(f['mean_speed'][i]) or f['mean_speed'][i]>=.8:continue
                        # Visible ball identifies prospective taker; otherwise no arbitrary runner.
                        d=np.linalg.norm(f['players'][i]-f['ball'][i],axis=1)
                        near=np.flatnonzero(np.isfinite(d)&(d<3))
                        active=dict(start=i,type=e['tags'].get('Type','unknown'),slots=near.tolist())
                    slots=active['slots'];v=speed[i]
                    slow|=np.isfinite(v)&(v<.5)
                    sprint=any(slow[j] and np.isfinite(v[j]) and v[j]>3.5 for j in slots)
                    jump=np.isfinite(bs[i]) and bs[i]>8 and (i==start or not np.isfinite(bs[i-1]) or bs[i-1]<=8)
                    if jump or sprint:
                        audit.append(dict(active,end=i,release='ball_speed_jump' if jump else 'taker_sprint_proxy'));active=None;break
                    mask[i]=True
                if active:audit.append(dict(active,end=end,release='tag_end'))
    else:raise ValueError(mode)
    return mask,audit
