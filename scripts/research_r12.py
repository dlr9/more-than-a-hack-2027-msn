"""Frozen full public-match raw-to-score reconstruction, no target inputs."""
import argparse,json,time,subprocess
from pathlib import Path
import joblib,numpy as np
from r12.control_global import anchors,decode
from scripts.research_r7 import IDS,load,model_path,checked_path,write_new
from scripts.evaluate_r8_interpolation import raw_input
from scripts.evaluate_r8_decoder import prediction_path,original,save_prediction
from scripts.probe_r10_acquisition import hc_path
from scripts.research_r11p5 import full
from scripts.train_idsse_r6 import legacy_v3_input
from solution.provider_neutral import provider_neutral_features_v3,FeatureContract
from solution.candidate import Candidate
from solution.evidence import labels_from_batch,period_reference
from solution.idsse import digest
from solution.hierarchical_decoder import continuous_blocks
from evaluation import possession_metrics as pm
ROOT=Path('local_artifacts/r12/global-v1');P=Path('docs/R12_PROTOCOL.json');DOMAINS=('natural','long')

def freeze():
 inputs={}
 for m in IDS:
  paths=[Path(f'local_artifacts/r6/prepared/{m}/raw.npz'),model_path(m,'M2'),Path(f'local_artifacts/r8/e08c/{m}.joblib')]
  for d in DOMAINS:paths += [checked_path(m,d),prediction_path(m,'M3',d),hc_path(m,d)]
  inputs.update({str(p):digest(p) for p in paths})
 v3='local_artifacts/v3/1874553/baseline.joblib';inputs[v3]=digest(v3)
 write_new(P,dict(id='R12_GLOBAL_V1',hypothesis='Global latent control/flight states can contain erroneous local anchors and retain ownership through flight better than hard suffix writes. This is structural information, NOT new independent sensor evidence.',source='Public IDSSE CC BY4.0 three consumed development matches; no blind or cross-provider claim',inputs=inputs,code={p:digest(p) for p in ['r12/control_global.py','scripts/research_r12.py','solution/control_events.py','evaluation/possession_metrics.py']},constants='0.2s grid plus every anchor and baseline edge; log emissions integrated per second; own-control .95 at anchor else .1; opposite acquisition transition .3 with support vs3 otherwise; control-flight .3; enterD .3 exitD .1. Single configuration, no search or tuning.',anchor='Existing persistent identity/comotion .12s physical evidence, UNCALIBRATED and NOT independent of HGB geometry; no single touch/nearest-only acquisition.',baselines=['reference_smooth50','frozen_PR1_V3_first_fold','M3','HC_M3','untuned_V2_idea_distance_plus_frozen_Alive'],controls=['no_anchors','shift30s_perhalf','shuffle_anchor_team_seed1201'],unavailable_ablations={'remove_visual':'NOT_APPLICABLE no visual input','remove_SmartTagging':'exact same no-tag execution, not independent ablation'},raw_pipeline='Recompute44 from raw allowed observation fields; compare cache feature equality; frozenM2 Alive+C1 Home. labels evaluator-only. OldV3 baseline alone uses source reported player speed for schema2 compatibility.',gate='>=2/3 natural improves HC,mean delta>=.02,positive mean5IoU TP and aggregateIoU.5 TP,no Dmiss rate increase>.02,no short recall loss>.02; all controls. Otherwise NO_GO. Any win research-only until independent complete provider validates.',new_fits=0,rollback='Untouched HC_M3/M3',stop='No parameter changes after scores'))

def matched_changes(g,base,pred,blocks):
 gs=pm.possession_spells(pm.label_codes(g),blocks);out={}
 flags=[]
 for p in (base,pred):
  ps=pm.possession_spells(pm.label_codes(p),blocks);ov=pm.spell_overlaps(gs,ps);flags.append({t:pm.matched_gt_flags(ov,len(gs),t) for t in pm.SPELL_IOUS})
 for t in pm.SPELL_IOUS:
  a,b=flags[0][t],flags[1][t];out[str(t)]=dict(gained=int((~a&b).sum()),lost=int((a&~b).sum()),net=int(b.sum()-a.sum()))
 return out

def run():
 protocol=json.loads(P.read_text());ROOT.mkdir(parents=True,exist_ok=True)
 for p,h in {**protocol['inputs'],**protocol['code']}.items():
  if digest(p)!=h:raise ValueError('drift '+p)
 rows=[];v3=joblib.load('local_artifacts/v3/1874553/baseline.joblib')
 for m in IDS:
  raw=raw_input(m);m2=joblib.load(model_path(m,'M2'));c1=joblib.load(f'local_artifacts/r8/e08c/{m}.joblib')
  with np.load(f'local_artifacts/r6/prepared/{m}/raw.npz',allow_pickle=False) as z:speed=z['reported_speed_mps']
  for domain in DOMAINS:
   dest=ROOT/f'{m}-{domain}.json'
   if dest.exists():rows.append(json.loads(dest.read_text()));continue
   start=time.monotonic();b=load(m,domain);visible=b['X'][:,b['feature_names'].tolist().index('ball_visible')]==1
   rebuilt=provider_neutral_features_v3(raw,FeatureContract(**json.loads(str(b['feature_contract_json']))),ball_dropout_mask=~visible if domain!='natural' else None)
   np.testing.assert_allclose(rebuilt['X'],b['X'],equal_nan=True,rtol=1e-10,atol=1e-10)
   masked={**raw,'ball_xy':raw['ball_xy'].copy()};masked['ball_xy'][~visible]=np.nan
   ev,signal=anchors(masked,25);a,_=Candidate.probabilities(m2,rebuilt);_,h=Candidate.probabilities(c1,rebuilt)
   base=np.load(hc_path(m,domain))['decoded'];m3=np.load(prediction_path(m,'M3',domain))['decoded'];g=labels_from_batch(b);blocks=list(continuous_blocks(b['frame'],b['period']))
   actual_base=original(a,h,b,'M3');np.testing.assert_array_equal(actual_base,base)
   signals={'global':signal,'no_anchors':np.full(len(signal),-1,np.int8),'shift':signal.copy(),'shuffle_team':signal.copy()};rng=np.random.default_rng(1201)
   for lo,hi in blocks:signals['shift'][lo:hi]=np.roll(signal[lo:hi],750)
   for e in ev:signals['shuffle_team'][e['start']:e['end']]=rng.integers(2)
   pred={};traces={};latent={}
   for name,s in signals.items():pred[name],latent[name],traces[name]=decode(a,h,s,b['frame'],b['period'],25,base)
   legacy=legacy_v3_input(rebuilt,{**masked,'reported_speed_mps':speed});pred['PR1_V3']=v3.predict(legacy)
   pred['reference_smooth50']=period_reference(b);pred['M3']=m3;pred['HC_M3']=base
   names=b['feature_names'].tolist();margin=b['X'][:,names.index('away_d1_m')]-b['X'][:,names.index('home_d1_m')]
   geometry=1/(1+np.exp(-np.clip(np.nan_to_num(margin)/2,-20,20)));pred['V2_idea_untuned']=original(a,geometry,b,'M3')
   scores={k:dict(metrics=full(b,g,p),match_changes_vs_HC=matched_changes(g,base,p,blocks)) for k,p in pred.items()}
   mask=signal>=0;truth=np.where(g=='H',0,np.where(g=='A',1,-1));audit=dict(n_anchors=len(ev),covered_frames=int(mask.sum()),coverage=float(mask.mean()),correct=int(((truth==signal)&mask).sum()),wrong_team=int((mask&(truth>=0)&(truth!=signal)).sum()),on_dead=int((mask&(truth<0)).sum()),precision=float(np.mean(truth[mask]==signal[mask])) if mask.any() else None,fully_correct_intervals=sum(bool(np.all(truth[e['start']:e['end']]==int(e['team']=='A'))) for e in ev),missing_ball_frames=int((~visible).sum()),independent=False)
   save_prediction(ROOT/f'{m}-{domain}.npz',**pred,**{k+'_latent':v for k,v in latent.items()})
   write_new(ROOT/f'{m}-{domain}-anchors.json',ev)
   row=dict(match=m,domain=domain,n_frames=len(g),scores=scores,anchor_audit=audit,traces=traces,features_recomputed_equal=True,HC_replay_exact=True,prediction_sha=digest(ROOT/f'{m}-{domain}.npz'),anchor_sha=digest(ROOT/f'{m}-{domain}-anchors.json'),wall_s=time.monotonic()-start,git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),protocol_sha=digest(P))
   write_new(dest,row);rows.append(row);print('SCORE',m,domain,{k:round(v['metrics']['spell']['mean_f1'],6) for k,v in scores.items()},'ANCHOR',audit,flush=True)
 write_new('docs/R12_RESULTS.json',dict(rows=rows,new_fits=0,transfer='NOT_VALIDATED',independent_observation_added=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run']);args=p.parse_args();freeze() if args.action=='freeze' else run()
