"""Ball-free permutation invariant physical team dynamics, not possession labels."""
import numpy as np
NAMES=('centroid_vx','centroid_vy','width_m','depth_m','shape_variance_m2','velocity_coherence','expansion_m2ps','speed_mean_mps')

def group_reactions(players_m,velocity_mps,teams):
    p=np.asarray(players_m,float);v=np.asarray(velocity_mps,float);teams=np.asarray(teams)
    if p.ndim!=3 or p.shape[-1]!=2 or v.shape!=p.shape or teams.shape!=p.shape[:2]:raise ValueError('aligned NxPx2 arrays')
    out=np.full((len(p),2,len(NAMES)),np.nan);counts=np.zeros((len(p),2),int)
    for k in (0,1):
        for i in range(len(p)):
            keep=(teams[i]==k)&np.isfinite(p[i]).all(1)&np.isfinite(v[i]).all(1);x=p[i,keep];u=v[i,keep];counts[i,k]=len(x)
            if len(x)<3:continue
            c=x.mean(0);mean=u.mean(0);s=np.linalg.norm(u,axis=1);coherence=np.linalg.norm(mean)/max(float(s.mean()),1e-12)
            out[i,k]=[*mean,np.ptp(x[:,1]),np.ptp(x[:,0]),np.mean(np.sum((x-c)**2,axis=1)),coherence,np.mean(np.sum((x-c)*(u-mean),axis=1)),s.mean()]
    return dict(features=out,valid=np.isfinite(out).all(2),player_count=counts,names=NAMES,independent_of_old_positions=False)
