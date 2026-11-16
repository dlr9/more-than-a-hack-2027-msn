"""Independent raw ball momentum-change channel, not a possession label.

Probe a receiver contact at the instant of a kick/deceleration/deflection;
unlike posterior proximity, no long future ownership persistence is required.
A deflection is deliberately a counterexample: contact need not change owner.
"""
import numpy as np
from r10.acquisition import geometry_evidence
from solution.hierarchical_decoder import continuous_blocks


def impulse_strength(raw,fps):
    if not np.isfinite(fps) or fps<2:raise ValueError('fps')
    proximity,valid=geometry_evidence(raw);ball=np.asarray(raw['ball_xy'],float)
    out=np.zeros_like(proximity);impulse=np.zeros(len(ball));k=max(1,round(.12*fps));dt=k/fps
    for lo,hi in continuous_blocks(raw['frame'],raw['period']):
        if hi-lo<=2*k:continue
        ix=np.arange(lo+k,hi-k)
        before=(ball[ix]-ball[ix-k])/dt;after=(ball[ix+k]-ball[ix])/dt
        incoming=np.linalg.norm(before,axis=1);outgoing=np.linalg.norm(after,axis=1)
        change=np.linalg.norm(after-before,axis=1)
        # Suppress stationary quantization noise and impossible tracking jumps;
        # constants are physical research priors, no target/config tuning.
        ok=valid[ix]&np.isfinite(change)&(change>=2)&(incoming<=45)&(outgoing<=45)
        v=np.zeros(len(ix));v[ok]=np.minimum(change[ok]/(incoming[ok]+outgoing[ok]+1),1)
        impulse[ix]=v
        out[ix]=proximity[ix]*v[:,None]
    return out,impulse
