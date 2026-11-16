"""R13 structural intervention: remember last owner across Dead; no cost tuning."""
import numpy as np
from solution.control_events import control_episodes
from solution.hierarchical_decoder import continuous_blocks

STATES=('H_control','H_flight_contested','A_control','A_flight_contested','Dead_after_H','Dead_after_A')
MAP=np.array(list('HHAADD'))

def anchors(raw,fps):
    events,signal=control_episodes(raw,fps)
    return [dict(e,visibility='observed_ball_and_player',independent_of_HGB=False,
        reliability='UNCALIBRATED_GEOMETRIC_SUPPORT',unknown=False,
        evidence=['persistent_same_identity','relative_comotion','opponent_separation','both_team_coverage'],
        limitation='Not independent sensor evidence; source complete HAD must audit precision') for e in events],signal

def decode(alive,home,signal,frame,period,fps,reference=None):
    """Sparse global bidirectional optimization; short spells permitted at every edge.

    The latent flight state retains team identity; cross-team acquisition costs
    less only at observed opponent control support. Dead uses separate frozen
    Alive probability, not set-piece intervals or missingness-as-Dead.
    All costs fixed before scoring, no claim of learned/calibrated parameters.
    """
    a=np.asarray(alive,float);h=np.asarray(home,float);s=np.asarray(signal)
    if a.shape!=h.shape or s.shape!=a.shape or not np.isin(s,[-1,0,1]).all():raise ValueError('alignment/signal')
    if not np.isfinite(a).all() or not np.isfinite(h).all() or np.any((a<0)|(a>1)|(h<0)|(h>1)):raise ValueError('probability')
    qh=np.where(s==0,.95,.1);qa=np.where(s==1,.95,.1)
    e=np.column_stack([a*h*qh,a*h*(1-qh),a*(1-h)*qa,a*(1-h)*(1-qa),1-a,1-a])
    log=np.log(np.clip(e,1e-9,1))/fps;prefix=np.vstack([np.zeros(6),np.cumsum(log,axis=0)])
    latent=np.full(len(a),4,np.int8);cols=np.arange(6);cells=0
    for lo,hi in continuous_blocks(frame,period):
        boundaries={int(lo),int(hi)};boundaries.update(range(int(lo),int(hi),max(1,round(.2*fps))))
        boundaries.update((lo+1+np.flatnonzero(s[lo+1:hi]!=s[lo:hi-1])).tolist())
        if reference is not None:boundaries.update((lo+1+np.flatnonzero(reference[lo+1:hi]!=reference[lo:hi-1])).tolist())
        edge=sorted(boundaries);back=np.zeros((len(edge)-1,6),np.int8);value=prefix[edge[1]]-prefix[edge[0]]
        for j,(start,end) in enumerate(zip(edge[1:-1],edge[2:]),1):
            cost=transition_cost(s[start])
            choices=value[:,None]-cost;back[j]=choices.argmax(0)
            value=choices[back[j],cols]+prefix[end]-prefix[start];value-=value.max()
        state=int(value.argmax())
        for j in range(len(edge)-2,-1,-1):
            latent[edge[j]:edge[j+1]]=state
            if j:state=int(back[j,state])
        cells+=len(edge)-1
    return MAP[latent],latent,dict(cells=cells,states=STATES,tag_input=False,visual_input=False)


def transition_cost(signal):
    """Preserve R12 costs; an owner change cannot escape through Dead.

    Dead memory does not prescribe the restart owner. A different restart owner
    remains possible but pays the same acquisition cost as a direct change.
    """
    cost=np.full((6,6),np.inf);np.fill_diagonal(cost,0)
    for old in range(4):
        for new in range(4):
            if old==new:continue
            cost[old,new]=.3 if old//2==new//2 else acquisition_cost(signal,new)
        cost[old,4+old//2]=.3
    for memory in range(2):
        for new in range(4):
            cost[4+memory,new]=.1+(0 if memory==new//2 else acquisition_cost(signal,new))
    return cost


def acquisition_cost(signal,new):
    return .3 if signal==new//2 and new%2==0 else 3.
