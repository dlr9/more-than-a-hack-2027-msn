"""Observation augmentation only. Missing is not Dead; conflicts do not overwrite."""
import numpy as np

def fuse(reference,other,alignment,*,enabled=False,validated=False,min_confidence=.9,max_disagreement_m=2.):
    raw=reference.ball
    # Exact no-op, even dtype/NaN bit patterns and object identity are preserved.
    if not enabled or not validated:return dict(ball=raw,added=np.zeros(len(raw),bool),conflict=np.zeros(len(raw),bool),coverage_gain=0.,status='OFF_OR_UNVALIDATED')
    idx=np.asarray(alignment['index']);conf=np.asarray(alignment['confidence'])
    if idx.shape!=(len(raw),) or conf.shape!=idx.shape or not np.isfinite(conf).all() or np.any((idx<-1)|(idx>=len(other.ball))):raise ValueError('alignment contract')
    if reference.pitch_m!=other.pitch_m:raise ValueError('pitch mismatch')
    out=raw.copy();added=np.zeros(len(raw),bool);conflict=added.copy();errors=[]
    for i in np.flatnonzero((idx>=0)&(conf>=min_confidence)):
        j=idx[i]
        if not other.ball_observed[j] or other.replay[j] or not other.homography_valid[j] or reference.period[i]!=other.period[j]:continue
        if reference.ball_observed[i]:
            e=float(np.linalg.norm((raw[i]-other.ball[j])*reference.pitch_m));errors.append(e);conflict[i]=e>max_disagreement_m
        elif other.ball_grounded is not None and other.ball_grounded[j] and reference.homography_valid[i]:
            out[i]=other.ball[j];added[i]=True
    return dict(ball=out,added=added,conflict=conflict,coverage_gain=float(added.mean()),intersection_ball_error_m=errors,status='OBSERVATIONS_ONLY_NO_CONTROL_BOUNDARY')
