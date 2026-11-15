"""V3: confidence-weighted control, visible-ball stoppages and dynamic decoding."""
import json
import numpy as np
from fusion_v2 import ROOT, BLOCKS, features as base_features, rolling, runs
import games

DEFAULT = dict(window=12, distance=2., confidence=1., attack_weight=.2,
    setpiece_weight=.2, missing_weight=1., idle=.5, dead_bias=-.8,
    visible_weight=.8, team_penalty=60., enter_penalty=30., exit_penalty=30.,
    dynamic=.5, gap=0)

def features(data_dir):
    f = base_features(data_dir, ROOT/'features_v2.npz')
    from fusion_v2 import signature
    raw, meta = games.require(['atd_raw','atd_meta'],data_dir=data_dir)
    sig = dict(source=signature([raw,meta]), version=1)
    path=ROOT/'features_v3_speed.npz'
    if path.exists():
        with np.load(path) as z:
            if json.loads(str(z['metadata']))==sig:
                f.update({k:z[k] for k in ['slow','speed_q25','speed_q75']})
                return f
    values=np.full((len(f['ball']),3),np.nan)
    with open(raw) as handle:
        for line in handle:
            frame,players,_=line.strip().split(':')
            speed=np.array([float(p.split(',')[2])/3.6 for p in players.split(';')])
            speed=speed[np.isfinite(speed)&(speed>=0)&(speed<15)]
            if len(speed): values[int(frame)-1]=[np.mean(speed<.5),*np.quantile(speed,[.25,.75])]
    extra=dict(slow=values[:,0],speed_q25=values[:,1],speed_q75=values[:,2])
    np.savez_compressed(path,**extra,metadata=json.dumps(sig)); f.update(extra)
    return f

def prepare(f,gap=0):
    b=f['ball'].copy(); original=np.isfinite(b).all(axis=1)
    repaired=np.zeros(len(b),bool)
    for lo,hi in BLOCKS:
        for a,c in runs(~original[lo:hi]):
            s,e=lo+a,lo+c
            if 0<e-s<=gap and s>lo and e<hi:
                # Only bridge short, physically plausible endpoint displacement.
                if np.linalg.norm(b[e]-b[s-1])*25/(e-s+1)<30:
                    b[s:e]=np.linspace(b[s-1],b[e],e-s+2)[1:-1]; repaired[s:e]=True
    dist=np.linalg.norm(f['players']-b[:,None,:],axis=2)
    dist=np.where(np.isfinite(dist),dist,np.inf)
    d=[dist[:,f['teams']==i].min(axis=1) for i in [0,1]]
    count=[np.isfinite(f['players'][:,f['teams']==i]).all(axis=2).sum(axis=1) for i in [0,1]]
    valid=np.isfinite(d[0])&np.isfinite(d[1])
    diff=np.zeros(len(b)); diff[valid]=d[1][valid]-d[0][valid]
    near=np.minimum(*d)
    confidence=np.exp(-np.maximum(near-2,0)/4)*np.minimum(np.minimum(*count)/8,1)
    confidence[~valid]=0
    proximity=np.clip((dist[:,f['teams']==0]<3).sum(axis=1)-(dist[:,f['teams']==1]<3).sum(axis=1),-2,2)
    speed=np.full(len(b),np.nan); station=np.full((len(b),2),np.nan)
    for lo,hi in BLOCKS:
        v=np.linalg.norm(np.diff(b[lo:hi],axis=0),axis=1)*25
        v[v>40]=np.nan; speed[lo+1:hi]=v
        for j,width in enumerate([25,50]):
            for i in range(lo,hi):
                chunk=b[max(lo,i-width//2):min(hi,i+width//2+1)]
                ok=np.isfinite(chunk).all(axis=1)
                if ok.mean()>=.8:
                    ranges=np.ptp(chunk[ok],axis=0)
                    station[i,j]=np.clip(1-np.linalg.norm(ranges)/3,0,1)
    return dict(diff=diff,proximity=proximity,confidence=confidence,repaired=repaired,
        present=original,speed=speed,station=np.nanmean(np.where(np.isfinite(station),station,0),axis=1),
        out=(b[:,0]<-1)|(b[:,0]>106.3)|(b[:,1]<-1)|(b[:,1]>69))

def observations(f,p,c):
    obs=np.zeros((len(f['ball']),3)); stopped=np.zeros(len(obs)); restart=stopped.copy()
    for lo,hi in BLOCKS:
        smooth=lambda a: np.nan_to_num(rolling(np.asarray(a)[lo:hi],c['window']))
        strength=(1-c['confidence'])+c['confidence']*p['confidence']
        team=smooth((np.clip(p['diff']/c['distance'],-1,1)+.1*p['proximity'])*strength*np.where(p['repaired'],.5,1))
        team+=c['attack_weight']*(f['tags'][lo:hi,0]-f['tags'][lo:hi,1])
        # Missing motion summaries contribute zero evidence, never fake stillness.
        idle=smooth(np.where(np.isfinite(f['mean_speed']),c['idle']-f['mean_speed'],np.nan))
        slow=smooth(np.where(np.isfinite(f['slow']),f['slow']-.5,np.nan))
        stationary=smooth(p['station']); visible=smooth(p['present'].astype(float))
        motion=smooth(p['speed'])
        visible_dead=visible*(1.5*stationary+2*slow+np.clip(idle*3,-1,1))
        dead=(c['dead_bias']+6.48*idle+2*smooth(p['out'].astype(float))+
            c['missing_weight']*smooth((~p['present']).astype(float))+
            c['setpiece_weight']*f['tags'][lo:hi,2]+c['visible_weight']*visible_dead+
            .2*smooth(1-2*f['moving'])-.25*np.clip(motion/5,0,1))
        obs[lo:hi]=np.column_stack([team,-team,dead])
        stopped[lo:hi]=np.clip(np.maximum(visible_dead,0),0,1)
        restart[lo:hi]=np.clip(motion/5,0,1)*smooth(p['confidence'])
    return obs,stopped,restart

def decode(obs,stop,restart,c):
    if not len(obs): return np.empty(0,np.int8)
    if not np.isfinite(obs).all(): raise ValueError('Nonfinite observation')
    back=np.zeros((len(obs),3),np.int8); value=obs[0].copy(); cols=np.arange(3)
    for i in range(1,len(obs)):
        enter=c['enter_penalty']*(1-c['dynamic']*stop[i])
        leave=c['exit_penalty']*(1-c['dynamic']*restart[i])
        costs=np.array([[0,c['team_penalty'],enter],[c['team_penalty'],0,enter],[leave,leave,0]])
        a=value[:,None]-costs; back[i]=a.argmax(axis=0)
        value=a[back[i],cols]+obs[i]; value-=value.max()
    result=np.empty(len(obs),np.int8); result[-1]=value.argmax()
    for i in range(len(obs)-1,0,-1): result[i-1]=back[i,result[i]]
    return result

def predict(f,c,prepared=None):
    if set(c)!=set(DEFAULT): raise ValueError('Configuration keys do not match V3 schema')
    if not all(np.isfinite(v) for v in c.values()): raise ValueError('Nonfinite config')
    if c['distance']<=0 or min(c[k] for k in ['team_penalty','enter_penalty','exit_penalty'])<0: raise ValueError('Invalid penalties or distance')
    if not 0<=c['dynamic']<=1 or not 0<=c['confidence']<=1: raise ValueError('Invalid weight')
    if any(not isinstance(c[k],int) or c[k]<0 for k in ['window','gap']): raise ValueError('Invalid window/gap')
    p=prepared if prepared is not None else prepare(f,c['gap'])
    obs,stop,restart=observations(f,p,c)
    labels=np.full(len(obs),'X')
    for lo,hi in BLOCKS:
        start=113742 if lo==110808 else lo
        labels[lo:min(start,hi)]='D'
        labels[start:hi]=np.array(['H','A','D'])[decode(obs[start:hi],stop[start:hi],restart[start:hi],c)]
    return labels
