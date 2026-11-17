"""Optimistic ground-ball reachability. Never emits current ownership or edits.

Bounds must be declared from legal external motion data. Constant velocity ball
forecast and radial acceleration ignore turning/height and remain uncalibrated.
"""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class MotionBounds:
    max_speed_mps: float
    acceleration_mps2: float
    reaction_s: float
    provenance: str
    def __post_init__(self):
        if not self.provenance or not np.isfinite([self.max_speed_mps,self.acceleration_mps2,self.reaction_s]).all() or self.max_speed_mps<=0 or self.acceleration_mps2<=0 or self.reaction_s<0:raise ValueError('external motion bounds required')

def interception(players_m,velocity_mps,teams,ball_m,ball_velocity_mps,bounds,*,grounded=False,horizon_s=2.,step_s=.04):
    p=np.asarray(players_m,float);v=np.asarray(velocity_mps,float);teams=np.asarray(teams);ball=np.asarray(ball_m,float);bv=np.asarray(ball_velocity_mps,float)
    if p.ndim!=2 or p.shape[1]!=2 or v.shape!=p.shape or teams.shape!=(len(p),) or ball.shape!=(2,) or bv.shape!=(2,):raise ValueError('input shape')
    if horizon_s<=0 or step_s<=0:raise ValueError('forecast grid')
    arrival=np.full(len(p),np.nan);valid=np.isfinite(p).all(1)&np.isfinite(v).all(1)&np.isin(teams,[0,1])
    out=dict(arrival_s=arrival,team_arrival_s=np.full(2,np.nan),arrival_margin_s=np.nan,lane_clearance_m=np.nan,status='UNKNOWN',acquisition_probability=None)
    if not grounded or not np.isfinite(ball).all() or not np.isfinite(bv).all() or not valid.any():return out
    speed=np.linalg.norm(v,axis=1);valid&=speed<=bounds.max_speed_mps
    arrival[valid]=np.inf
    for t in np.arange(0,horizon_s+step_s/2,step_s):
        target=ball+bv*t;delta=target-p;distance=np.linalg.norm(delta,axis=1)
        direction=np.divide(delta,distance[:,None],out=np.zeros_like(delta),where=distance[:,None]>0)
        radial=np.maximum(0,np.sum(v*direction,axis=1));active=max(0,t-bounds.reaction_s)
        accel_time=np.minimum(active,(bounds.max_speed_mps-radial)/bounds.acceleration_mps2)
        reach=radial*np.minimum(t,bounds.reaction_s)+radial*accel_time+.5*bounds.acceleration_mps2*accel_time**2+bounds.max_speed_mps*np.maximum(0,active-accel_time)
        hit=valid&np.isinf(arrival)&(distance<=reach);arrival[hit]=t
    for k in (0,1):
        x=arrival[(teams==k)&valid];out['team_arrival_s'][k]=np.min(x) if len(x) else np.nan
    if np.isfinite(out['team_arrival_s']).all():out['arrival_margin_s']=float(out['team_arrival_s'][1]-out['team_arrival_s'][0])
    end=ball+bv*horizon_s;lane=end-ball;den=float(lane@lane)
    if den>0:
        u=np.clip((p-ball)@lane/den,0,1);d=np.linalg.norm(p-(ball+u[:,None]*lane),axis=1);out['lane_clearance_m']=float(np.min(d[valid])) if valid.any() else np.nan
    out['status']='FUTURE_REACHABILITY_NOT_CONTROL';return out
