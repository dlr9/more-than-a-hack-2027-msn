"""Permutation/team-swap invariant shape; no ball, labels or slot identity."""
import numpy as np

def fingerprint(players,teams):
    out=np.full((len(players),19),np.nan)
    for i,(p,t) in enumerate(zip(players,teams)):
        descriptors=[]
        for team in (0,1):
            x=p[(t==team)&np.isfinite(p).all(1)]
            if len(x)<3:break
            c=x.mean(0);cov=(x-c).T@(x-c)/len(x)
            descriptors.append(np.r_[c,np.sqrt(np.maximum(np.diag(cov),0)),cov[0,1],np.quantile(x[:,0],.9)-np.quantile(x[:,0],.1),np.quantile(x[:,1],.9)-np.quantile(x[:,1],.1)])
        if len(descriptors)!=2:continue
        descriptors.sort(key=lambda x:tuple(x))
        x=p[np.isfinite(p).all(1)&np.isin(t,[0,1])];d=np.linalg.norm(x[:,None]-x[None,:],axis=-1);np.fill_diagonal(d,np.inf)
        out[i]=np.r_[descriptors[0],descriptors[1],x.mean(0),np.quantile(d.min(1),[.1,.5,.9])]
    return out
