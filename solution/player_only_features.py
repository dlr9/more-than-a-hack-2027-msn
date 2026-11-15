"""Independent player-trajectory features; no ball, event, status or label access.

Forty-six aggregate columns. Offline centered windows reset at periods, missing
frames and wholly unobservable activity. IDs guard differences, never enter X.
Unknown/referee teams (-1) are excluded. Speeds use meters and seconds.
"""
import warnings
import numpy as np

FIELDS=frozenset({'frame','period','players_xy','player_team','player_track_id'})

def player_inputs(raw):
    """Explicit whitelist boundary; unrelated mapping values are never read."""
    return {k:raw[k] for k in FIELDS if k in raw}

def _mean(x,axis):
    count=np.isfinite(x).sum(axis=axis)
    return np.divide(np.nansum(x,axis=axis),count,out=np.full(count.shape,np.nan),where=count>0)

def _quantile(x,q,axis):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        return np.nanquantile(x,q,axis=axis)

def activity_windows(activity,frame,period,fps,seconds):
    """Centered mean/std, never borrowing across unavailable activity or gaps."""
    good=np.isfinite(activity)&(period>0)
    cut=np.r_[True,(period[1:]!=period[:-1])|(np.diff(frame)!=1)|~good[1:]|~good[:-1]]
    starts=np.flatnonzero(cut);ends=np.r_[starts[1:],len(frame)]
    mean=np.full(len(frame),np.nan);std=mean.copy();radius=max(1,round(seconds*fps/2))
    for a,b in zip(starts,ends):
        if not good[a]:continue
        origin=activity[a];x=activity[a:b]-origin;idx=np.arange(len(x));lo=np.maximum(0,idx-radius);hi=np.minimum(len(x),idx+radius+1)
        c=np.r_[0,np.cumsum(x)];c2=np.r_[0,np.cumsum(x*x)];n=hi-lo
        m=(c[hi]-c[lo])/n;mean[a:b]=m+origin;std[a:b]=np.sqrt(np.maximum(0,(c2[hi]-c2[lo])/n-m*m))
    return mean,std

def build_player_features(raw,*,fps,pitch=(105.,68.),units='centered_m'):
    if set(raw)-FIELDS:raise ValueError('player-only input forbids ball, status, labels and provider outputs')
    if not FIELDS.issubset(raw):raise ValueError('explicit tracklet IDs and player timeline required')
    if not np.isfinite(fps) or fps<=0:raise ValueError('invalid fps')
    frame,period=np.asarray(raw['frame']),np.asarray(raw['period'])
    xy=np.array(raw['players_xy'],dtype=float,copy=True)
    if frame.ndim!=1 or period.shape!=frame.shape or xy.ndim!=3 or xy.shape[0]!=len(frame) or xy.shape[2]!=2 or not len(frame):raise ValueError('unaligned player timeline')
    if not np.isfinite(frame).all() or not np.isfinite(period).all() or np.any(frame!=np.floor(frame)) or np.any(period!=np.floor(period)):raise ValueError('integer timeline required')
    if np.any((np.diff(frame)<=0)&(period[1:]==period[:-1])):raise ValueError('unordered frames')
    if np.isinf(xy).any():raise ValueError('infinite coordinate')
    if units=='normalized':xy=(xy-.5)*np.asarray(pitch)
    elif units!='centered_m':raise ValueError('unsupported units')
    teams=np.asarray(raw['player_team']);ids=np.asarray(raw['player_track_id'])
    if teams.shape not in {xy.shape[:2],(xy.shape[1],)} or ids.shape not in {xy.shape[:2],(xy.shape[1],)}:raise ValueError('invalid team/tracklet shape')
    teams=np.broadcast_to(teams,xy.shape[:2]);ids=np.broadcast_to(ids,xy.shape[:2])
    if not np.isin(teams,[-1,0,1]).all():raise ValueError('map referee/unknown to -1 before feature construction')
    visible=np.isfinite(xy).all(axis=2)&np.isin(teams,[0,1])&(period[:,None]>0)
    xy[~visible]=np.nan
    continuity=np.r_[False,(period[1:]==period[:-1])&(np.diff(frame)==1)&(period[1:]>0)]
    valid=np.zeros_like(visible)
    valid[1:]=visible[1:]&visible[:-1]&(teams[1:]==teams[:-1])&(ids[1:]==ids[:-1])&continuity[1:,None]
    velocity=np.full_like(xy,np.nan);velocity[1:]=np.diff(xy,axis=0)*fps;velocity[~valid]=np.nan
    speed=np.linalg.norm(velocity,axis=2)
    acceleration=np.full_like(speed,np.nan)
    acceleration[1:]=np.abs(np.diff(speed,axis=0))*fps
    acceleration[1:][~(valid[1:]&valid[:-1])]=np.nan
    cols={};centroids=[];radii=[]
    for team,prefix in [(0,'home'),(1,'away')]:
        mask=visible&(teams==team);s=np.where(mask,speed,np.nan);count=mask.sum(axis=1)
        pos=np.where(mask[:,:,None],xy,np.nan);center=_mean(pos,1);centroids.append(center)
        radius=_mean(np.linalg.norm(pos-center[:,None,:],axis=2),1);radii.append(radius)
        cols[prefix+'_count']=count.astype(float);cols[prefix+'_visible_fraction']=count/11.
        for q in [.1,.25,.5,.75,.9]:cols[f'{prefix}_speed_p{int(q*100)}_mps']=_quantile(s,q,1)
        ns=np.isfinite(s).sum(axis=1)
        for name,criterion in [('slow',s<1.),('fast',s>5.)]:
            cols[prefix+'_'+name+'_fraction']=np.divide(criterion.sum(axis=1),ns,out=np.full(len(frame),np.nan),where=ns>0)
        cols[prefix+'_accel_p50_mps2']=_quantile(np.where(mask,acceleration,np.nan),.5,1)
        cols[prefix+'_centroid_motion_mps']=np.linalg.norm(_mean(np.where(mask[:,:,None],velocity,np.nan),1),axis=1)
        cols[prefix+'_radius_m']=radius
        for axis,name in [(0,'length'),(1,'width')]:
            cols[prefix+'_'+name+'_m']=_quantile(pos[:,:,axis],1.,1)-_quantile(pos[:,:,axis],0.,1)
            cols[prefix+('_centroid_x_m' if axis==0 else '_centroid_y_m')]=center[:,axis]
        same=np.r_[False,np.all(mask[1:]==mask[:-1],axis=1)&np.all(~mask[1:]|(ids[1:]==ids[:-1]),axis=1)&continuity[1:]]
        dr=np.r_[np.nan,np.diff(radius)*fps];dr[~same]=np.nan;cols[prefix+'_expansion_mps']=dr
    # Bounded chunking avoids an N x P x P allocation for an entire match.
    pairmean=np.full(len(frame),np.nan);pairmin=pairmean.copy();rel50=pairmean.copy();rel90=pairmean.copy()
    for a in range(0,len(frame),1024):
        b=min(a+1024,len(frame));p=xy[a:b];v=velocity[a:b];t=teams[a:b]
        eligible=(t[:,:,None]==0)&(t[:,None,:]==1)
        d=np.linalg.norm(p[:,:,None,:]-p[:,None,:,:],axis=3);d[~eligible]=np.nan
        r=np.linalg.norm(v[:,:,None,:]-v[:,None,:,:],axis=3);r[~eligible]=np.nan
        pairmean[a:b]=_mean(d.reshape(b-a,-1),1);pairmin[a:b]=_quantile(d.reshape(b-a,-1),0,1)
        rel50[a:b]=_quantile(r.reshape(b-a,-1),.5,1);rel90[a:b]=_quantile(r.reshape(b-a,-1),.9,1)
    cols.update(opponent_distance_mean_m=pairmean,opponent_distance_min_m=pairmin,
                relative_speed_p50_mps=rel50,relative_speed_p90_mps=rel90,
                activity_mean_mps=_mean(speed,1),speed_observed_fraction=np.isfinite(speed).sum(axis=1)/22.)
    for seconds in [.5,2.,5.]:
        mean,std=activity_windows(cols['activity_mean_mps'],frame,period,fps,seconds)
        cols[f'activity_{seconds:g}s_mean_mps']=mean;cols[f'activity_{seconds:g}s_std_mps']=std
    names=sorted(cols)
    return dict(X=np.column_stack([cols[k]for k in names]),feature_names=np.asarray(names),frame=frame.copy(),period=period.copy(),
                fps=np.asarray(fps),schema_version=np.asarray('player_only_v1'),units=np.asarray('meters_seconds'),pitch=np.asarray(pitch))
