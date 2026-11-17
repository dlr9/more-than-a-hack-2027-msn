"""Dense human-control supervision, observation-only inference, no event anchors.

Features describe identity-coherent player/ball relative velocity before/after a
frame. They remain geometric observations, not an independent visual sensor.
"""
import numpy as np
from solution.hierarchical_decoder import continuous_blocks,binary_decode

def features(raw,fps=25):
 ball=np.asarray(raw['ball_xy'],float);players=np.asarray(raw['players_xy'],float);n,k,_=players.shape
 team=np.broadcast_to(raw['player_team'],(n,k));ids=np.broadcast_to(raw['player_track_id'],(n,k))
 blocks=continuous_blocks(raw['frame'],raw['period']);pv=np.full_like(players,np.nan);bv=np.full_like(ball,np.nan)
 for a,b in blocks:
  if b-a<=10:continue
  v=(players[a+10:b]-players[a:b-10])/(10/fps)
  ok=(ids[a+10:b]==ids[a:b-10])&(ids[a+5:b-5]==ids[a:b-10]);v[~ok]=np.nan;pv[a+5:b-5]=v
  bv[a+5:b-5]=(ball[a+10:b]-ball[a:b-10])/(10/fps)
 dist=np.linalg.norm(players-ball[:,None],axis=2);dist[~np.isfinite(dist)]=np.inf
 views=[];rows=np.arange(n)
 for t in (0,1):
  td=np.where(team==t,dist,np.inf);nearest=td.argmin(1);d=td[rows,nearest];r=players[rows,nearest]-ball
  rv=pv[rows,nearest]-bv;norm=np.linalg.norm(rv,axis=1);radial=np.sum(r*rv,axis=1)/np.maximum(d,.1)
  chosen_speed=np.linalg.norm(pv[rows,nearest],axis=1)
  for v in (norm,radial,chosen_speed):v[~np.isfinite(d)]=np.nan
  speeds=np.linalg.norm(pv,axis=2);ok=(team==t)&np.isfinite(speeds)
  mean=np.sum(np.where(ok,speeds,0),axis=1)/np.maximum(ok.sum(1),1)
  count=np.sum((team==t)&np.isfinite(players).all(2),axis=1)
  x=np.column_stack([np.minimum(d,30),norm,radial,chosen_speed,mean,count,np.isfinite(d),np.isfinite(norm)])
  x=np.nan_to_num(x,nan=0,posinf=0,neginf=0);x=np.clip(x,-100,100);contexts=[]
  for offset in (-10,0,10):
   c=np.zeros_like(x)
   for a,b in blocks:
    ix=np.arange(a,b)+offset;valid=(ix>=a)&(ix<b);c[np.arange(a,b)[valid]]=x[ix[valid]]
   contexts.append(c)
  views.append(np.concatenate(contexts,axis=1))
 return np.stack(views,axis=1).astype(np.float32)

def flatten(x,swap=False):return x[:,::-1 if swap else 1,:].reshape(len(x),-1)

def probabilities(model,x):
 a=model.predict_proba(flatten(x));b=model.predict_proba(flatten(x,True))
 assert np.array_equal(model.classes_,[0,1,2])
 return .5*(a+b[:,[1,0,2]])

def fixed_dead_decode(home,base,frame,period,fps=25):
 """Same preexisting .3s binary decoder for all experimental arms, D frozen."""
 out=np.array(base,copy=True)
 for a,b in continuous_blocks(frame,period):
  live=base[a:b]!='D';edges=np.flatnonzero(np.diff(np.r_[False,live,False])).reshape(-1,2)
  for lo,hi in edges:
   h=binary_decode(home[a+lo:a+hi],fps,.3);out[a+lo:a+hi]=np.where(h,'H','A')
 np.testing.assert_array_equal(out=='D',base=='D')
 return out

def training_targets(g,frame,period,events,source_frame,kind):
 """Training only. Unknown transition neighborhoods excluded, not negatives."""
 n=len(g);target=np.full(n,2,np.int8);exclude=np.zeros(n,bool);positives=[]
 index={(int(p),int(f)):i for i,(p,f) in enumerate(zip(period,source_frame))}
 for a,b in continuous_blocks(frame,period):
  for i in a+1+np.flatnonzero(g[a+1:b]!=g[a:b-1]):exclude[max(a,i-25):min(b,i+26)]=True
  if kind=='source':
   for i in a+1+np.flatnonzero(g[a+1:b]!=g[a:b-1]):
    if g[i-1] in 'HA' and g[i] in 'HA':positives.append((i,int(g[i]=='A')))
 if kind!='source':
  for period_id,player,typ,f,receiver,receive in events:
   candidates=[]
   if receiver and receive is not None and np.isfinite(receive):candidates.append((receive,receiver))
   if typ in ['ball_recovery','keeper_pick_up','keeper_claim']:candidates.append((f,player))
   for sf,who in candidates:
    i=index.get((int(period_id),int(sf)))
    if i is None:continue
    exclude[max(0,i-25):min(n,i+26)]=True
    side='H' if who.startswith('home_') else 'A'
    # Human receipt is only a positive when training-source control persists.
    if i+10<=n and np.all(period[i:i+10]==period_id) and np.all(g[i:i+10]==side):positives.append((i,int(side=='A')))
 target[exclude]=-1;conflict=np.zeros(n,bool)
 for i,t in positives:
  lo=max(0,i-5);hi=min(n,i+6);ix=np.arange(lo,hi);ix=ix[period[ix]==period[i]]
  conflict[ix]|=(target[ix]>=0)&(target[ix]<2)&(target[ix]!=t);target[ix]=t
 target[conflict]=-1
 # Non-events sampled at fixed 5Hz, positives kept densely. No test selection.
 keep=(target>=0)&((target<2)|(np.arange(n)%5==0))
 return target,keep,dict(accepted_event_anchors=len(positives),counts=[int(np.sum(keep&(target==t))) for t in range(3)],excluded=int((target<0).sum()))
