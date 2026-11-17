"""Privileged external-label interventions; NOT deployable predictors or bounds.

All segments are half-open. B supplies both times and count of true direct H/A
changes but retains model direction; C changes labels on fixed proposed segments.
A uses frozen conditional Home probabilities to fill baseline false-Dead frames.
D overwrites true short H/A intervals and therefore leaks both timing AND state.
These interventions can lose matches. Their effects are not additive.
"""
from dataclasses import dataclass
import numpy as np
from evaluation import possession_metrics as pm

@dataclass(frozen=True)
class ExternalTruth:
    labels: np.ndarray
    source: str
    def __post_init__(self):
        if self.source not in {'IDSSE_DFL_source_complete','synthetic'}:
            raise ValueError('Only registered external or synthetic truth')
        if np.asarray(self.labels).ndim!=1 or not np.isin(self.labels,list('HAD')).all():
            raise ValueError('Complete external H/A/D required')

def interventions(base,home,truth,blocks,fps):
    g=np.asarray(truth.labels);base=np.asarray(base);home=np.asarray(home)
    if base.shape!=g.shape or home.shape!=g.shape or not np.isfinite(home).all() or np.any((home<0)|(home>1)):
        raise ValueError('Aligned conditional probabilities required')
    if not np.isin(base,list('HAD')).all() or fps<=0:raise ValueError('invalid baseline/fps')
    output={}
    for flags in range(8):
        p=base.copy();names=[];proposed=[]
        if flags&1:
            names.append('A_dead');false_dead=(p=='D')&(g!='D')
            p[false_dead]=np.where(home[false_dead]>=.5,'H','A');p[g=='D']='D'
        if flags&2:
            names.append('B_times')
            for lo,hi in blocks:
                alive=p[lo:hi]!='D';cuts=np.flatnonzero(np.diff(np.r_[False,alive,False].astype(int)))+lo
                for start,end in zip(cuts[::2],cuts[1::2]):
                    boundaries=np.flatnonzero((g[start+1:end]!=g[start:end-1])&(g[start+1:end]!='D')&(g[start:end-1]!='D'))+start+1
                    edges=np.r_[start,boundaries,end]
                    for a,b in zip(edges[:-1],edges[1:]):
                        p[a:b]='H' if home[a:b].mean()>=.5 else 'A'
                        proposed.append((a,b))
        if flags&4:
            names.append('C_owner')
            segments=proposed if flags&2 else [(a,b) for a,b,k in pm.possession_spells(pm.label_codes(p),blocks)]
            for start,end in segments:
                h=np.sum(g[start:end]=='H');a=np.sum(g[start:end]=='A')
                if h!=a:p[start:end]='H' if h>a else 'A'
        output['+'.join(names) or 'baseline']=p
    for seconds in (1,5):
        for correct_dead in (False,True):
            p=output['A_dead' if correct_dead else 'baseline'].copy()
            for a,b,k in pm.possession_spells(pm.label_codes(g),blocks):
                if b-a<seconds*fps:p[a:b]=g[a:b]
            output[('A_dead+' if correct_dead else '')+f'D_short_lt{seconds}']=p
    return output
