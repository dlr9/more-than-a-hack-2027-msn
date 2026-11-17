"""Position-only dynamic timing probe and honest input audit; no new classifier."""
import argparse,json,time,subprocess
from dataclasses import asdict
from pathlib import Path
import numpy as np
from solution.observation_contract import ObservationContract,track_velocity
from solution.group_reactions import group_reactions
from solution.multimodal_evidence import apply_edits
from scripts.research_r7 import IDS,load,write_new
from scripts.evaluate_r8_interpolation import raw_input
from r11p5.timeline import rank
from scripts.research_r11p5 import full
from solution.evidence import labels_from_batch
from solution.hierarchical_decoder import continuous_blocks
from solution.idsse import digest
from evaluation import possession_metrics as pm
P=Path('docs/R14_DYNAMIC_PROTOCOL_V2.json');ROOT=Path('local_artifacts/r14/dynamic-v1')

def freeze():
 files=['solution/observation_contract.py','solution/group_reactions.py','solution/multimodal_evidence.py','scripts/research_r14_dynamic.py','evaluation/possession_metrics.py']
 for m in IDS:
  files += [f'local_artifacts/r6/prepared/{m}/raw.npz',f'local_artifacts/r8/e08c/{m}-natural.npz']
  for d in ['natural','long']:files += [f'local_artifacts/r12/global-v1/{m}-{d}.npz']
 write_new(P,dict(id='R14_DYNAMIC_V2', implementation_repair='V1 halted on near-zero roundoff under permutation: maximum violating difference 4.04e-17; only assertion atol changed to 1e-12, no score or parameter changes; completed WPY cache retained with V1 provenance',source='3 consumed public DFL development matches; no target, no independent sensor',hypothesis='Ball-free collective velocity changes localize direct control changes missed by frozen probability-change timing.',signals='Sum over teams of centroid-velocity change norm at t-.4s vs t+.4s. All positions in metres, fps25, identity/gap/half reset. Teams exchange invariant. No labels/events in inputs.',baseline='Frozen C1 Home probability absolute change over same .8s window; HC_M3 hard-boundary misses evaluator only',controls=['shift30s_perhalf','reverse_time_perhalf','team_swap','slot_permutation'],budget='120 proposals per continuous block, .5s NMS, same for all methods; none filtered by GT',gate='>=2/3 matches missed-boundary-recall increase>=.05 vs probability AND shift AND reverse controls; otherwise STOP, no fits or tuning. Passing cannot promote without full independent spell gain.',dead_and_missing='All-timeline proposals; D false proposals reported. Natural ball missing zero; ball-free features identical under synthetic mask, no natural-missing evidence.',speed='Derive from explicit metres and25FPS source contract; reported S/3.6 only audited, not used. Raw-speed unit metadata and kloppy unavailable => UNVERIFIED, no claim teammate bug.',default_off='All three switches exact baseline bytes, 6 complete domain/match official scores; no edits because no independently calibrated edit gate.',hashes={p:digest(p) for p in files}))

def timing(proposals,g,base,blocks):
 gc=pm.label_codes(g);bc=pm.label_codes(base);out=dict(proposals=len(proposals),gt=0,matched=0,old_missed=0,recovered_old_missed=0,on_D=0);lat=[]
 for bi,block in enumerate(blocks):
  true=[x for x in pm.transitions(gc,[block]) if x[1]in(0,1) and x[2]in(0,1)];old=[x for x in pm.transitions(bc,[block]) if x[1]in(0,1) and x[2]in(0,1)]
  pred=[(i,0,1) for i,k in proposals if k==bi];pairs,_,_=pm.match_transitions(true,pred,25,typed='none');_,missed,_=pm.match_transitions(true,old,25,typed='none')
  out['gt']+=len(true);out['matched']+=len(pairs);out['old_missed']+=len(missed);out['recovered_old_missed']+=sum(t in missed for t,p in pairs);out['on_D']+=sum(g[i]=='D' for i,_,_ in pred);lat.extend(abs(t[0]-p[0])/25 for t,p in pairs)
 out.update(precision=out['matched']/max(1,len(proposals)),missed_recall=out['recovered_old_missed']/max(1,out['old_missed']),FP=len(proposals)-out['matched'],FN=out['gt']-out['matched'],mean_abs_s=float(np.mean(lat)) if lat else None);return out

def run():
 protocol=json.loads(P.read_text());ROOT.mkdir(parents=True,exist_ok=True)
 for p,h in protocol['hashes'].items():assert digest(p)==h,p
 rows=[];off=[]
 for m in IDS:
  done=ROOT/f'{m}.json'
  if done.exists():
   record=json.loads(done.read_text());rows.append(record['row']);off.extend(record['off']);continue
  start=time.monotonic();raw=raw_input(m);n,slots=raw['players_xy'].shape[:2];teams=np.broadcast_to(raw['player_team'],(n,slots));ids=np.broadcast_to(raw['player_track_id'],(n,slots))
  c=ObservationContract('IDSSE_DFL',25.,0,(105.,68.),'centered_m','unknown','m/s','unavailable','fixed_pitch_axes',('H','A'),'persistent','CC-BY-4.0',('no_video','raw_speed_metadata_unverified'))
  v=track_velocity(raw['players_xy'],raw['frame'],raw['period'],ids,c);reaction=group_reactions(raw['players_xy'],v['velocity_mps'],teams);a=reaction['features'];blocks=list(continuous_blocks(raw['frame'],raw['period']))
  signal=np.full(n,-np.inf);old=np.full(n,-np.inf);home=np.load(f'local_artifacts/r8/e08c/{m}-natural.npz')['p_home'];k=10
  for lo,hi in blocks:
   delta=a[lo+2*k:hi,:,:2]-a[lo:hi-2*k,:,:2];z=np.sum(np.linalg.norm(delta,axis=2),axis=1);signal[lo+k:hi-k]=np.where(np.isfinite(z),z,-np.inf);old[lo+k:hi-k]=np.abs(home[lo+2*k:hi]-home[lo:hi-2*k])
  shifted=signal.copy();reverse=signal.copy()
  for lo,hi in blocks:shifted[lo:hi]=np.roll(signal[lo:hi],750);reverse[lo:hi]=signal[lo:hi][::-1]
  # Exact invariance on an explicit temporal sample, without new fits.
  selection=np.arange(0,n,997);perm=np.arange(slots)[::-1];sw=group_reactions(raw['players_xy'][selection][:,perm],v['velocity_mps'][selection][:,perm],1-teams[selection][:,perm])['features'];np.testing.assert_allclose(sw,a[selection,::-1],equal_nan=True,atol=1e-12)
  channels={name:rank(z,raw['frame'],raw['period'],25) for name,z in [('dynamic',signal),('probability',old),('shift',shifted),('reverse',reverse)]}
  # Labels enter only after feature and proposal creation.
  batch=load(m,'natural');g=labels_from_batch(batch);base=np.load(f'local_artifacts/r12/global-v1/{m}-natural.npz')['HC_M3']
  scores={name:timing(p,g,base,blocks) for name,p in channels.items()}
  with np.load(f'local_artifacts/r6/prepared/{m}/raw.npz',allow_pickle=False) as z:reported=z['reported_speed_mps']
  physical=np.linalg.norm(v['velocity_mps'],axis=2);valid=np.isfinite(physical)&np.isfinite(reported);errors=np.abs(physical[valid]-reported[valid]);speed_audit=dict(status='RAW_METADATA_AND_KLOPPY_UNVERIFIED',n_pairs=int(valid.sum()),median_abs_decoded_vs_position_mps=float(np.median(errors)),p95_abs_decoded_vs_position_mps=float(np.quantile(errors,.95)),reported_speed_used_by_new_features=False,kloppy_installed=False)
  ball=raw['ball_xy'];ball_valid=np.isfinite(ball).all(1);step=np.linalg.norm(np.diff(ball,axis=0),axis=1)*25;cont=(np.diff(raw['frame'])==1)&(raw['period'][1:]==raw['period'][:-1]);stationary=cont&(step==0)
  coverage={x:dict(n=int(np.sum(g[1:]==x)),stationary_ball_fraction=float(np.mean(stationary[g[1:]==x]))) for x in ['H','A','D']}
  row=dict(match=m,results=scores,contract=asdict(c),speed_audit=speed_audit,ball_missing=int((~ball_valid).sum()),player_coverage_mean=float(v['valid_observation'].mean()),dynamic_valid_fraction=float(np.isfinite(signal).mean()),ball_stationary_by_source_state=coverage,slot_and_team_swap_equal_atol_1e12=True,features_sha256=None,wall_s=time.monotonic()-start)
  path=ROOT/f'{m}-timing.npz'
  with path.open('xb') as f:np.savez_compressed(f,**{name:np.array(p,int) for name,p in channels.items()})
  row['proposal_sha256']=digest(path);row['git_sha']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();row['protocol_sha256']=digest(P)
  local_off=[]
  for domain in ['natural','long']:
   b=load(m,domain);truth=labels_from_batch(b);baseline=np.load(f'local_artifacts/r12/global-v1/{m}-{domain}.npz')['HC_M3']
   for kw in [dict(),dict(new_feature=True,trust=0,model_off=False),dict(new_feature=True,trust=1,model_off=True)]:
    out=apply_edits(baseline,b['period'],[],**kw);assert out.tobytes()==baseline.tobytes()
   local_off.append(dict(match=m,domain=domain,three_switches_exact_bytes=True,metrics=full(b,truth,out),gained_lost_all_IoU=0))
  write_new(done,dict(row=row,off=local_off));rows.append(row);off.extend(local_off);print('DYNAMIC',m,scores,flush=True)
 gate=sum(all(r['results']['dynamic']['missed_recall']-r['results'][control]['missed_recall']>=.05 for control in ['probability','shift','reverse']) for r in rows)>=2
 write_new('docs/R14_FEATURE_INCREMENTAL_VALUE.json',dict(rows=rows,disabled_full_evaluations=off,new_fits=0,timing_gate=gate,decision='TIMING_PROBE_ONLY_NOT_PROMOTION' if gate else 'STOP_NO_TIMING_INCREMENT',scorer_sha256=digest('evaluation/possession_metrics.py')))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run']);a=p.parse_args();freeze() if a.action=='freeze' else run()
