"""One immutable all-timeline external experiment; no provider event anchors."""
import argparse,json,time,subprocess,warnings
from pathlib import Path
import numpy as np,joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score,f1_score
from r11p5.timeline import FIELDS,features,targets,rank,decode
from scripts.research_r7 import IDS,load,checked_path,write_new
from scripts.probe_r10_acquisition import hc_path,evaluate_proposals
from scripts.evaluate_r8_decoder import prediction_path,save_prediction
from scripts.evaluate_r11_roi import diagnostics
from solution.idsse import digest
from solution.evidence import labels_from_batch
from solution.hierarchical_decoder import continuous_blocks
from evaluation import possession_metrics as pm
ROOT=Path('local_artifacts/r11p5/e11a');PROTOCOL=Path('docs/R11P5_PROTOCOL.json')
DOMAINS=('natural','long');CODE=['r11p5/timeline.py','scripts/research_r11p5.py']

def freeze():
 inputs={str(p):digest(p) for m in IDS for d in DOMAINS for p in [checked_path(m,d),hc_path(m,d),prediction_path(m,'M3',d),Path(f'local_artifacts/r10-v3/c2/{m}-{d}.npz')]}
 write_new(PROTOCOL,dict(id='E11.5-A',hypothesis='Full-source direct handover positives and clean interior negatives identify ordered team-control changes autonomously; symmetric temporal contrast is incremental to instantaneous geometry.',folds={m:[x for x in IDS if x!=m] for m in IDS},exposure='All3 intensely consumed development matches,shared home team. WMX/WN1 previously exposed, no sealed data opened.',source='IDSSE DFL CC BY4.0 BallStatus/Possession, independently crosschecked; not official target GT',inputs=inputs,code={p:digest(p) for p in CODE},features='Only team-relative distances/counts/speed, visibility, coverage; no absolute coordinates,ball motion,status,phase,event metadata. Team-swap invariant hazard and explicitly antisymmetric direction. context[-.8,0,+.8] vs instant.',labels='directHA first frame +/- .2s; negative alive >1s from ANY transition; D and transition-near others excluded. Direction all alive. All-timeline inference, no anchor required.',sampling='Training only: all positives, negatives on fixed5Hz grid; direction5Hz. Both natural and pre-existing label-independent long occlusion training domains. No validation sampling.',model='StandardScaler+LogisticRegression C1 balanced lbfgs max_iter500 seed115; hazard plus direction per view per fold=12fits. Direction train mirrored and predict antisymmetrized.',decoder='Frozen HC_M3 D AND baseline boundaries. Add only hazard>=.8 and -.8/+ .8s opposite direction each confident>=.8,matching prior baseline. Within baseline spell >=1s until next old boundary; no threshold tuning.',controls='Instant feature ablation; same120/half .5s spacing proposal budget HC/M3 probability difference,R10C2,learned,30s shift,shuffled direction. Full candidate plus shift/shuffle/instant decoded.',gate='>=2/3 natural positive fullspell AND mean>=.02,noDmaskchange,no >.02 short-recall loss; even if pass TRANSFER_NOT_VALIDATED. StageA matched missed-boundary information exceeds bestold in>=2matches.',uncertainty='1000 match bootstrap seed115;3 reused matches not statistical confirmation',stop='Fail => stop family; no tuning. Keep all receipts.',scorer_sha=digest('evaluation/possession_metrics.py')))

def obs(b):return {k:b[k] for k in FIELDS}
def fit(x,y,mirror=False):
 if mirror:x=np.vstack([x,-x]);y=np.r_[y,1-y]
 model=make_pipeline(StandardScaler(),LogisticRegression(C=1.,class_weight='balanced',solver='lbfgs',max_iter=500,random_state=115))
 with warnings.catch_warnings():
  warnings.simplefilter('error',ConvergenceWarning);model.fit(x,y)
 return model

def prob(model,x,direction=False):
 p=model.predict_proba(x)[:,1]
 return .5*(p+1-model.predict_proba(-x)[:,1]) if direction else p

def full(b,g,p):
 blocks=list(continuous_blocks(b['frame'],b['period']));gc=pm.label_codes(g);pc=pm.label_codes(p)
 r=dict(spell=pm.spell_scores(gc,pc,blocks),macro_f1=float(f1_score(g,p,labels=list('HAD'),average='macro')),diagnostics=diagnostics(g,p,blocks,float(b['fps'])))
 transitions={}
 for name,select in [('direct',lambda t:t[1] in (0,1) and t[2] in (0,1)),('enter_D',lambda t:t[2]==pm.D),('exit_D',lambda t:t[1]==pm.D)]:
  count=np.zeros(3,int);lat=[]
  for block in blocks:
   gt=[t for t in pm.transitions(gc,[block]) if select(t)];pr=[t for t in pm.transitions(pc,[block]) if select(t)]
   pairs,mi,sp=pm.match_transitions(gt,pr,25,typed='pair');count+=len(pairs),len(sp),len(mi);lat.extend(abs(a[0]-z[0])/25 for a,z in pairs)
  transitions[name]=dict(tp=int(count[0]),fp=int(count[1]),fn=int(count[2]),mean_abs_s=float(np.mean(lat)) if lat else None)
 r['boundaries']=transitions
 return r

def proposal(score,away,b):return [dict(index=i,block=k,team=int(away[i]>=.5)) for i,k in rank(score,b['frame'],b['period'],float(b['fps']))]
def oldscore(p,b):
 out=np.zeros(len(p));k=20
 for a,z in continuous_blocks(b['frame'],b['period']):
  out[a+k:z-k]=abs(p[a+2*k:z]-p[a:z-2*k])
 return out

def run():
 p=json.loads(PROTOCOL.read_text());ROOT.mkdir(parents=True,exist_ok=True)
 for path,h in {**p['inputs'],**p['code']}.items():
  if digest(path)!=h:raise ValueError('drift '+path)
 rows=[];models=[]
 for held in IDS:
  done=ROOT/f'{held}-results.json'
  if done.exists():rows.extend(json.loads(done.read_text())['rows']);models.extend(json.loads(done.read_text())['models']);continue
  began=time.monotonic();trained={};foldmodels=[];counts={}
  for context,name in [(False,'instant'),(True,'context')]:
   hx=[];hy=[];dx=[];dy=[]
   for m in p['folds'][held]:
    for domain in DOMAINS:
     b=load(m,domain);g=labels_from_batch(b);h,t,c=targets(g,b['frame'],b['period'],float(b['fps']));a,s=features(obs(b),context);grid=np.arange(len(g))%5==0
     keep=(h==1)|((h==0)&grid);direction=(t>=0)&grid
     hx.append(a[keep]);hy.append(h[keep]);dx.append(s[direction]);dy.append(t[direction]);counts[m+'-'+domain]=dict(events=c,hazard_n=int(keep.sum()),positive=int((h==1).sum()),direction_n=int(direction.sum()),excluded=int((h<0).sum()))
   trained[name]=[]
   for target,x,y in [('hazard',np.vstack(hx),np.concatenate(hy)),('direction',np.vstack(dx),np.concatenate(dy))]:
    path=ROOT/f'{held}-{name}-{target}.joblib'
    if path.exists():raise FileExistsError('partialfit preserved; no silent repeat')
    model=fit(x,y,target=='direction');joblib.dump(model,path);trained[name].append(model);foldmodels.append(dict(path=str(path),sha256=digest(path),n=len(y),iterations=model[-1].n_iter_.tolist()))
    print('FIT',held,name,target,len(y),flush=True)
  foldrows=[]
  for domain in DOMAINS:
   b=load(held,domain);g=labels_from_batch(b);blocks=list(continuous_blocks(b['frame'],b['period']));hc=dict(np.load(hc_path(held,domain)));m3=dict(np.load(prediction_path(held,'M3',domain)));base=hc['decoded'];y,t,ec=targets(g,b['frame'],b['period'],25)
   probs={}
   for name in trained:
    a,s=features(obs(b),name=='context');probs[name]=(prob(trained[name][0],a),prob(trained[name][1],s,True))
   hazard,away=probs['context'];shift=[z.copy() for z in (hazard,away)];shuf=away.copy();rng=np.random.default_rng(115)
   for a,z in blocks:
    for j,v in enumerate((hazard,away)):shift[j][a:z]=np.roll(v[a:z],750)
    shuf[a:z]=rng.permutation(away[a:z])
   probs['shift']=tuple(shift);probs['shuffle_direction']=(hazard,shuf)
   comparisons={};arrays={};newproposals={}
   for name,(h,d) in probs.items():
    pred,accepted=decode(base,h,d,b['frame'],b['period'],25);comparisons[name]=dict(metrics=full(b,g,pred),accepted=len(accepted),changed_frames=int((pred!=base).sum()),hazard_AP_all_known=float(average_precision_score(y[y>=0],h[y>=0])))
    arrays.update({name+'_hazard':h.astype('float32'),name+'_away':d.astype('float32'),name+'_decoded':pred});newproposals[name]=proposal(h,d,b)
   # Equal budget for old cached physical proposals and dense scores.
   for name,z in [('M3',m3),('HC_M3',hc)]:newproposals[name]=proposal(oldscore(z['p_home'],b),1-z['p_home'],b)
   physical=np.load(f'local_artifacts/r10-v3/c2/{held}-{domain}.npz',allow_pickle=False)
   # Inspect schema once in freeze/setup; documented arrays are Nx4 index/team/score/block.
   physical_key='geometry'
   if physical_key not in physical.files:raise ValueError('physical cache schema '+str(physical.files))
   newproposals['R10C2']=[dict(index=int(v[0]),team=int(v[1]),block=int(v[3])) for v in physical[physical_key]]
   equal={k:[] for k in newproposals};budget=[]
   for bi,_ in enumerate(blocks):
    n=min(sum(q['block']==bi for q in v) for v in newproposals.values());budget.append(n)
    for name,v in newproposals.items():equal[name].extend([q for q in v if q['block']==bi][:n])
   visible=b['X'][:,b['feature_names'].tolist().index('ball_visible')]==1
   info={k:evaluate_proposals(v,pm.label_codes(g),pm.label_codes(base),blocks,25,visible,hc['p_home']) for k,v in equal.items()}
   path=ROOT/f'{held}-{domain}.npz';save_prediction(path,**arrays)
   row=dict(match=held,domain=domain,baseline=full(b,g,base),M3=full(b,g,m3['decoded']),candidates=comparisons,information=info,equal_budget=budget,source_events=ec,n_frames=len(g),natural_missing_frames=int((~visible).sum()),missing_direct_gt=sum(not visible[q[0]] for block in blocks for q in pm.transitions(pm.label_codes(g),[block]) if q[1] in (0,1) and q[2] in (0,1)),prediction_sha256=digest(path),code_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),protocol_sha=digest(PROTOCOL))
   foldrows.append(row);print('SCORE',held,domain,'base',row['baseline']['spell']['mean_f1'],'candidate',comparisons['context']['metrics']['spell']['mean_f1'],flush=True)
  record=dict(rows=foldrows,models=foldmodels,training_counts=counts,wall_s=time.monotonic()-began);write_new(done,record);rows.extend(foldrows);models.extend(foldmodels)
 natural=[r for r in rows if r['domain']=='natural'];d=np.array([r['candidates']['context']['metrics']['spell']['mean_f1']-r['baseline']['spell']['mean_f1'] for r in natural]);boot=np.random.default_rng(115).choice(d,(1000,3)).mean(axis=1)
 write_new('docs/R11P5_RESULTS.json',dict(rows=rows,models=models,delta_mean=float(d.mean()),match_bootstrap95=np.quantile(boot,[.025,.975]).tolist(),numeric_gate=bool((d>0).sum()>=2 and d.mean()>=.02),decision='RESEARCH_ONLY_PENDING_GATES' if (d>0).sum()>=2 and d.mean()>=.02 else 'REVERT_NO_FULL_SPELL_GAIN',transfer='TRANSFER_NOT_VALIDATED',new_fits=12))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['freeze','run']);v=a.parse_args();freeze() if v.action=='freeze' else run()
