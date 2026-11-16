"""Training-prior correction and neutral-preserving evidence pooling.

Prior correction assumes balanced-training posteriors approximate likelihoods.
It is not a claim of empirical calibration under provider/covariate shift.
"""
import numpy as np

def eligible_prior(counts,n_frames):
 positive=sum(counts['counts'][:2]);none=n_frames-counts['excluded']-positive
 if none<=0 or positive<=0:raise ValueError('empty class')
 p=np.array([positive/2,positive/2,none],float);return p/p.sum()

def correct(p,prior):
 p=np.asarray(p,float);prior=np.asarray(prior,float)
 if p.ndim!=2 or p.shape[1]!=3 or prior.shape!=(3,) or not np.isfinite(p).all() or np.any(p<0) or np.any(prior<=0):raise ValueError('invalid posterior/prior')
 q=p*prior;return q/q.sum(1,keepdims=True)

def pool(home,event):
 """Fixed unit log-odds opinion pool, not independent-sensor Bayes evidence."""
 event=np.asarray(event);h=np.clip(home,1e-9,1-1e-9)
 ratio=np.clip(event[:,0]+.5*event[:,2],1e-9,1)/np.clip(event[:,1]+.5*event[:,2],1e-9,1)
 odds=h/(1-h)*ratio
 return odds/(1+odds)
