"""One pre-frozen causal topology intervention; no parameter selection."""
import argparse,json,subprocess
from pathlib import Path
import numpy as np
from r12.control_memory import decode
from scripts.research_r7 import IDS,load,write_new
from scripts.evaluate_r8_decoder import prediction_path,save_prediction
from scripts.research_r12 import matched_changes
from scripts.research_r11p5 import full
from scripts.verify_r12 import extra
from solution.evidence import labels_from_batch
from solution.hierarchical_decoder import continuous_blocks
from solution.idsse import digest
ROOT=Path('local_artifacts/r13/memory-v1');P=Path('docs/R13_PROTOCOL.json')

def freeze():
 paths=['r12/control_memory.py','scripts/research_r13.py','evaluation/possession_metrics.py','docs/R12_RESULTS.json']
 for m in IDS:
  for d in ['natural','long']:
   paths += [str(prediction_path(m,'M3',d)),f'local_artifacts/r8/e08c/{m}-{d}.npz',f'local_artifacts/r12/global-v1/{m}-{d}.npz',f'local_artifacts/r12/global-v1/{m}-{d}-anchors.json']
 write_new(P,dict(id='R13_MEMORY_V1',hypothesis='Remembering last owner across Dead removes the proven cheap D bridge without changing any R12 numerical cost. This is a causal structural repair, NOT independent evidence.',hashes={p:digest(p) for p in paths},new_fits=0,intervention='Replace Dead by Dead_after_H/Dead_after_A; enter only matching memory, no switching inside D, exit different owner pays original acquisition cost plus original restart cost. All original costs, emissions, grid, anchors unchanged.',controls=['no_anchors','shift30s','shuffle_team_seed1201'],exposure='All3 development matches consumed; no blind-score claim or sealed data',gate='For mechanism: fewer false-D owner bridges than R12. Promotion additionally >=2/3 natural improves HC,mean >=.02,positive true-match net atIoU.5,no short recall loss>.02,no Dmiss rate increase>.02 and real controls superior; independent transfer still required.',stop='One intervention only; no retuning if fails. Keep original frozen R12 and HC/M3.'))

def run():
 protocol=json.loads(P.read_text());ROOT.mkdir(parents=True,exist_ok=True)
 for p,h in protocol['hashes'].items():assert digest(p)==h,p
 rows=[]
 for m in IDS:
  for d in ['natural','long']:
   dest=ROOT/f'{m}-{d}.json'
   if dest.exists():rows.append(json.loads(dest.read_text()));continue
   b=load(m,d);g=labels_from_batch(b);blocks=list(continuous_blocks(b['frame'],b['period']));n=len(g)
   old=dict(np.load(f'local_artifacts/r12/global-v1/{m}-{d}.npz'));a=np.load(prediction_path(m,'M3',d))['p_alive'];h=np.load(f'local_artifacts/r8/e08c/{m}-{d}.npz')['p_home']
   events=json.loads(Path(f'local_artifacts/r12/global-v1/{m}-{d}-anchors.json').read_text());signal=np.full(n,-1,np.int8)
   for e in events:signal[e['start']:e['end']]=int(e['team']=='A')
   signals=dict(memory=signal,no_anchors=np.full(n,-1,np.int8),shift=signal.copy(),shuffle_team=signal.copy());rng=np.random.default_rng(1201)
   for lo,hi in blocks:signals['shift'][lo:hi]=np.roll(signal[lo:hi],750)
   for e in events:signals['shuffle_team'][e['start']:e['end']]=rng.integers(2)
   result={};saved={};visible=b['X'][:,b['feature_names'].tolist().index('ball_visible')]==1
   for name,s in signals.items():
    pred,latent,trace=decode(a,h,s,b['frame'],b['period'],25,old['HC_M3']);saved[name]=pred;saved[name+'_latent']=latent
    result[name]=dict(metrics=full(b,g,pred),additional=extra(g,pred,blocks,visible),matched_changes_vs_HC=matched_changes(g,old['HC_M3'],pred,blocks),changed_from_R12=int((pred!=old['global']).sum()),trace=trace)
   save_prediction(ROOT/f'{m}-{d}.npz',**saved)
   row=dict(match=m,domain=d,results=result,prediction_sha=digest(ROOT/f'{m}-{d}.npz'),protocol_sha=digest(P),git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
   write_new(dest,row);rows.append(row);print(m,d,{k:round(v['metrics']['spell']['mean_f1'],6) for k,v in result.items()},flush=True)
 write_new('docs/R13_RESULTS.json',dict(rows=rows,new_fits=0,independent_information=False,transfer='NOT_VALIDATED'))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run']);a=p.parse_args();freeze() if a.action=='freeze' else run()
