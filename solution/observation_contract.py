"""Explicit physical observations; no labels, unit guessing or identity imputation."""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class ObservationContract:
    source: str
    fps: float
    frame_base: int
    pitch_m: tuple
    coord_units: str
    speed_units_raw: str
    speed_units_decoded: str
    homography_direction: str
    field_orientation: str
    team_map: tuple
    identity_type: str
    source_license: str
    quality_flags: tuple=()
    def __post_init__(self):
        if not self.source or not self.source_license:raise ValueError('source/license required')
        if not np.isfinite(self.fps) or self.fps<=0 or self.frame_base not in (0,1):raise ValueError('fps/frame base')
        if len(self.pitch_m)!=2 or not np.isfinite(self.pitch_m).all() or min(self.pitch_m)<=0:raise ValueError('pitch')
        if self.coord_units not in {'normalized','centered_m'}:raise ValueError('coordinates')
        if self.speed_units_raw not in {'m/s','km/h','unknown'} or self.speed_units_decoded not in {'m/s','unknown'}:raise ValueError('speed units')
        if self.homography_direction not in {'pitch_normalized_to_pixel','unavailable'}:raise ValueError('homography direction')
        if self.field_orientation!='fixed_pitch_axes' or self.identity_type not in {'tracklet','persistent','dynamic_slot_unknown'}:raise ValueError('identity/orientation')
        if tuple(self.team_map)!=('H','A'):raise ValueError('explicit canonical H/A mapping required')

def to_metres(xy,contract):
    x=np.asarray(xy,float).copy()
    if x.shape[-1]!=2 or np.isinf(x).any():raise ValueError('finite pairs or NaN required')
    x[~np.isfinite(x).all(-1)]=np.nan
    return (x-.5)*contract.pitch_m if contract.coord_units=='normalized' else x

def validate_timeline(frame,period,fps,timestamp=None):
    f=np.asarray(frame);p=np.asarray(period)
    if f.ndim!=1 or not len(f) or p.shape!=f.shape or not np.isfinite(f).all() or not np.isfinite(p).all() or np.any(f<0) or np.any(p<0) or np.any(f!=np.floor(f)) or np.any(p!=np.floor(p)):raise ValueError('timeline')
    same=p[1:]==p[:-1]
    if np.any(same&(np.diff(f)<=0)):raise ValueError('duplicate/reversed frames')
    contiguous=same&(np.diff(f)==1)&(p[1:]>0)
    if timestamp is not None:
        t=np.asarray(timestamp,float)
        if t.shape!=f.shape or not np.isfinite(t).all() or np.any(same&(np.diff(t)<=0)):raise ValueError('timestamp')
        if np.any(np.abs(np.diff(t)[contiguous]-1/fps)>1e-5):raise ValueError('frame/time disagreement')
    return contiguous

def track_velocity(xy,frame,period,ids,contract):
    x=to_metres(xy,contract);ids=np.asarray(ids)
    if x.ndim!=3 or ids.shape!=x.shape[:2]:raise ValueError('NxPx2 / NxP required')
    contiguous=validate_timeline(frame,period,contract.fps)
    if len(frame)!=len(x):raise ValueError('unaligned observations')
    valid=np.isfinite(x).all(-1);v=np.full_like(x,np.nan)
    pairs=contiguous[:,None]&valid[1:]&valid[:-1]&(ids[1:]==ids[:-1])&(ids[1:]>=0)
    if contract.identity_type=='dynamic_slot_unknown':pairs[:]=False
    v[1:]=np.where(pairs[...,None],np.diff(x,axis=0)*contract.fps,np.nan)
    age=np.zeros(valid.shape,int)
    for t in range(len(x)):
        age[t]=np.where(valid[t]&(ids[t]>=0),1,0)
        if t:age[t]=np.where(pairs[t-1],age[t-1]+1,age[t])
    return dict(velocity_mps=v,valid_observation=valid,identity_stability=np.isfinite(v).all(-1),tracklet_age=age)

def audit_speed(raw_speed,decoded_speed,position_speed,contract,metadata_evidence):
    if contract.speed_units_raw=='unknown' or contract.speed_units_decoded!='m/s' or not metadata_evidence:
        return dict(status='UNVERIFIED_UNITS',usable=False)
    raw,decoded,diff=map(lambda x:np.asarray(x,float),(raw_speed,decoded_speed,position_speed))
    if raw.shape!=decoded.shape or raw.shape!=diff.shape:raise ValueError('speed alignment')
    expected=raw/(3.6 if contract.speed_units_raw=='km/h' else 1.)
    valid=np.isfinite(expected)&np.isfinite(decoded)&np.isfinite(diff)
    if not valid.any():return dict(status='NO_COMPARABLE_PAIRS',usable=False)
    error=np.abs(decoded[valid]-diff[valid]);converted=np.allclose(expected[valid],decoded[valid],atol=1e-5,rtol=1e-4)
    return dict(status='CONVERSION_CHECKED' if converted else 'CONVERSION_MISMATCH',usable=bool(converted),n=int(valid.sum()),median_abs_position_difference_mps=float(np.median(error)),p95_abs_position_difference_mps=float(np.quantile(error,.95)),conversion_max_error_mps=float(np.max(np.abs(expected[valid]-decoded[valid]))),metadata_evidence=metadata_evidence,warning='Position differences and provider smoothing differ; conversion check is not proof of correct metadata or identity.')
