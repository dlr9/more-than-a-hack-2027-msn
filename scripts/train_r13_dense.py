"""Human event timing versus same-feature source-boundary supervision.

Train only public J03WPY. Full WOH/WQQ timelines have no test event inputs.
"""
import argparse,json,subprocess,warnings
from pathlib import Path
import numpy as np,joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from r12.dense_control import features,flatten,probabilities,training_targets,fixed_dead_decode
from scripts.research_r7 import load,write_new
from scripts.evaluate_r8_interpolation import raw_input
from scripts.evaluate_r8_decoder import save_prediction
from scripts.research_r12 import matched_changes
from scripts.research_r11p5 import full
from scripts.verify_r12 import extra
from solution.evidence import labels_from_batch
from solution.hierarchical_decoder import continuous_blocks
from solution.idsse import digest
ROOT=Path('local_artifacts/r13/dense-v1');P=Path('docs/R13_DENSE_PROTOCOL.json')

def freeze():
 paths=['r12/dense_control.py','scripts/train_r13_dense.py','evaluation/possession_metrics.py','local_artifacts/r12/source-audit/J03WPY.parquet']
 for m in ['J03WPY','J03WOH','J03WQQ']:
  paths.append(f'local_artifacts/r6/prepared/{m}/raw.npz')
  for d in ['natural','long']:
   paths += [f'local_artifacts/r8/e08c/{m}-{d}.npz',f'local_artifacts/r12/global-v1/{m}-{d}.npz']
 write_new(P,dict(id='R13_DENSE_CONTROL_V1',hypothesis='Human-confirmed receipt timing plus identity-coherent relative velocity learns control evidence absent from source-boundary supervision; dense non-event negatives prevent conditional-event inference.',train=['J03WPY'],evaluation=['J03WOH','J03WQQ'],exposure='Previously consumed development matches, not blind confirmation. WPY training not scored for selection.',features='48 scalar features: team nearest-player distance, identity-coherent relative velocity norm/radial motion/player speed, collective player speed/count/availability; at -.4,0,+.4s. No events/status/GT input at inference; no absolute direction dependence. Existing geometric observations, NOT new sensor data.',supervision='Human receive times and recovery/keeper controls, corroborated by same team source HAD for next.4s; +/- .2s positives; exclude +/-1s event/transition ambiguity; remaining all-timeline5Hz none negatives. Same-feature source HA transition and shuffled positive-team supervision controls.',training='3 StandardScaler+LogisticRegression C1 balanced max_iter500 lbfgs, team-swap augmented. Natural and existing label-independent long-occlusion training views. Unknown class is not Dead.',inference='Predict3classes on EVERY frame; average mirrored outputs. Human potential = pH+.5*pUnknown. Equal .5 mixture with frozen C1 Home. Same fixed.3s binary decoder in frozen HC Dead segments for all arms. No new D changes. No geometry proposals required.',controls=['same_decoder_no_event','source_boundary_supervision','shuffled_positive_team_supervision','30s_shift_human_perhalf'],gate='Both evaluation natural matches improve full HC_M3 spellF1,mean>=.02,net true matches positive atIoU.5,short recall no decrease>.02,real beats same-decoder/source/shuffle/time controls. Otherwise NO_GO; any success still needs untouched cross-provider full labels.',stop='One configuration,3fits,no test-driven tuning; preserve failures and production HC/M3',hashes={p:digest(p) for p in paths}))

def observation(m,d):
 raw=raw_input(m);b=load(m,d);visible=b['X'][:,b['feature_names'].tolist().index('ball_visible')]==1
 raw['ball_xy']=raw['ball_xy'].copy();raw['ball_xy'][~visible]=np.nan
 return raw,b,visible

def run():
 protocol=json.loads(P.read_text());ROOT.mkdir(parents=True,exist_ok=True)
 for p,h in protocol['hashes'].items():assert digest(p)==h,p
 import duckdb
 events=duckdb.connect().execute('SELECT period_id,player_id,spadl_type,frame_id,receiver_id,receive_frame_id FROM read_parquet(?)',['local_artifacts/r12/source-audit/J03WPY.parquet']).fetchall()
 b=load('J03WPY','natural');g=labels_from_batch(b)
 with np.load('local_artifacts/r6/prepared/J03WPY/raw.npz',allow_pickle=False) as z:source=z['source_frame']
 labels={};counts={}
 for kind in ['human','source']:
  labels[kind]=training_targets(g,b['frame'],b['period'],events,source,kind);counts[kind]=labels[kind][2]
 rng=np.random.default_rng(1313);y,keep,_=labels['human'];shuffled=y.copy();positive=y<2;positive&=y>=0;shuffled[positive]=rng.permutation(y[positive]);labels['shuffle']=(shuffled,keep,{})
 data={k:[] for k in labels};ys={k:[] for k in labels}
 for d in ['natural','long']:
  raw,_,_=observation('J03WPY',d);x=features(raw)
  for kind,(y,keep,_) in labels.items():
   v=x[keep];target=y[keep];data[kind] += [flatten(v),flatten(v,True)];ys[kind] += [target,np.where(target<2,1-target,2)]
  del raw,x
 models={};receipts={}
 for kind in labels:
  path=ROOT/f'{kind}.joblib'
  if path.exists():raise FileExistsError('Preserve existing model; explicit recovery required: '+str(path))
  model=make_pipeline(StandardScaler(),LogisticRegression(C=1,class_weight='balanced',max_iter=500,solver='lbfgs',random_state=1313))
  with warnings.catch_warnings():
   warnings.simplefilter('error',ConvergenceWarning);model.fit(np.vstack(data[kind]),np.concatenate(ys[kind]))
  joblib.dump(model,path);models[kind]=model;receipts[kind]=dict(sha256=digest(path),n=sum(map(len,ys[kind])),iterations=model[-1].n_iter_.tolist());print('TRAINED',kind,receipts[kind],flush=True)
 del data,ys
 write_new(ROOT/'models.json',dict(models=receipts,training_counts=counts,protocol_sha=digest(P)))
 rows=[]
 for m in ['J03WOH','J03WQQ']:
  for d in ['natural','long']:
   raw,b,visible=observation(m,d);x=features(raw);g=labels_from_batch(b);blocks=list(continuous_blocks(b['frame'],b['period']));old=np.load(f'local_artifacts/r12/global-v1/{m}-{d}.npz')['HC_M3'];home=np.load(f'local_artifacts/r8/e08c/{m}-{d}.npz')['p_home']
   probs={k:probabilities(model,x) for k,model in models.items()};heads=dict(no_event=home)
   for k,q in probs.items():heads[k]=.5*home+.5*(q[:,0]+.5*q[:,2])
   shift=probs['human'].copy()
   for a,z in blocks:shift[a:z]=np.roll(shift[a:z],750,axis=0)
   heads['time_shift']=.5*home+.5*(shift[:,0]+.5*shift[:,2]);saved={k+'_prob':v for k,v in probs.items()};results={}
   for k,h in heads.items():
    pred=fixed_dead_decode(h,old,b['frame'],b['period']);saved[k]=pred
    results[k]=dict(metrics=full(b,g,pred),matched_changes_vs_HC=matched_changes(g,old,pred,blocks),additional=extra(g,pred,blocks,visible),dead_mask_exact=True)
   save_prediction(ROOT/f'{m}-{d}.npz',**saved);row=dict(match=m,domain=d,results=results,prediction_sha=digest(ROOT/f'{m}-{d}.npz'),protocol_sha=digest(P),git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
   write_new(ROOT/f'{m}-{d}.json',row);rows.append(row);print('SCORE',m,d,{k:round(v['metrics']['spell']['mean_f1'],6) for k,v in results.items()},flush=True)
 write_new('docs/R13_DENSE_RESULTS.json',dict(models=receipts,training_counts=counts,rows=rows,new_fits=3,transfer='NOT_VALIDATED'))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run']);a=p.parse_args();freeze() if a.action=='freeze' else run()
