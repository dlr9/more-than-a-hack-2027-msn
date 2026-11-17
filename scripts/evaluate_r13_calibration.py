"""Frozen semantic correction factorial, no additional training or search."""
import argparse,json,subprocess
from pathlib import Path
import numpy as np
from r12.control_calibration import eligible_prior,correct,pool
from r12.dense_control import fixed_dead_decode
from scripts.research_r7 import load,write_new
from scripts.research_r11p5 import full
from scripts.research_r12 import matched_changes
from scripts.evaluate_r8_decoder import save_prediction
from solution.hierarchical_decoder import continuous_blocks
from solution.evidence import labels_from_batch
from solution.idsse import digest
P=Path('docs/R13_CALIBRATION_PROTOCOL.json');ROOT=Path('local_artifacts/r13/calibration-v1')

def freeze():
 c=json.loads(Path('local_artifacts/r13/dense-v1/models.json').read_text())['training_counts'];n=len(load('J03WPY','natural')['frame'])
 priors={k:eligible_prior(v,n).tolist() for k,v in c.items()};priors['shuffle']=priors['human']
 paths=['r12/control_calibration.py','scripts/evaluate_r13_calibration.py','r12/dense_control.py','local_artifacts/r13/dense-v1/models.json']
 for m in ['J03WOH','J03WQQ']:
  for d in ['natural','long']:paths += [f'local_artifacts/r13/dense-v1/{m}-{d}.npz',f'local_artifacts/r8/e08c/{m}-{d}.npz',f'local_artifacts/r12/global-v1/{m}-{d}.npz']
 write_new(P,dict(id='R13_CALIBRATION_V1',reason='R13 dense training class_weight=balanced does not output population event priors. Arithmetic mixture changes baseline even for neutral event evidence.',primary='corrected: training eligible-class prior correction + neutral-preserving fixed unit log-odds pool. No validation-fitted coefficient.',factorial=['original uncorrected arithmetic in R13_DENSE_RESULTS','prior_only','neutral_only','corrected'],controls=['source_corrected','shuffle_corrected','time_shift_corrected','neutral_control'],priors=priors,limitations='Priors describe labeled eligible training population, not ambiguous intervals or a new provider. Shared geometric input means opinion pool is NOT independent-sensor Bayes evidence.',gate='Same R13 dense promotion gate. Primary fixed before scoring; do not select best factorial arm.',hashes={p:digest(p) for p in paths},new_fits=0,stop='No further probability weight/threshold tuning on these matches'))

def run():
 protocol=json.loads(P.read_text());ROOT.mkdir(parents=True,exist_ok=True)
 for p,h in protocol['hashes'].items():assert digest(p)==h,p
 rows=[]
 for m in ['J03WOH','J03WQQ']:
  for d in ['natural','long']:
   b=load(m,d);g=labels_from_batch(b);blocks=list(continuous_blocks(b['frame'],b['period']));z=dict(np.load(f'local_artifacts/r13/dense-v1/{m}-{d}.npz'));h=np.load(f'local_artifacts/r8/e08c/{m}-{d}.npz')['p_home'];base=np.load(f'local_artifacts/r12/global-v1/{m}-{d}.npz')['HC_M3']
   q={k:correct(z[k+'_prob'],protocol['priors'][k]) for k in ['human','source','shuffle']};v=q['human'];shift=v.copy()
   for a,c in blocks:shift[a:c]=np.roll(v[a:c],750,axis=0)
   heads=dict(corrected=pool(h,v),prior_only=.5*h+.5*(v[:,0]+.5*v[:,2]),neutral_only=pool(h,z['human_prob']),source_corrected=pool(h,q['source']),shuffle_corrected=pool(h,q['shuffle']),time_shift_corrected=pool(h,shift),neutral_control=pool(h,np.full((len(h),3),1/3)))
   results={};saved={}
   for k,home in heads.items():
    pred=fixed_dead_decode(home,base,b['frame'],b['period']);saved[k]=pred
    if k=='neutral_control':np.testing.assert_array_equal(pred,base)
    results[k]=dict(metrics=full(b,g,pred),matched_changes_vs_HC=matched_changes(g,base,pred,blocks),dead_mask_exact=True)
   save_prediction(ROOT/f'{m}-{d}.npz',**saved);row=dict(match=m,domain=d,results=results,prediction_sha=digest(ROOT/f'{m}-{d}.npz'),protocol_sha=digest(P),git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
   write_new(ROOT/f'{m}-{d}.json',row);rows.append(row);print(m,d,{k:round(v['metrics']['spell']['mean_f1'],6) for k,v in results.items()},flush=True)
 write_new('docs/R13_CALIBRATION_RESULTS.json',dict(rows=rows,primary='corrected',new_fits=0,transfer='NOT_VALIDATED'))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run']);a=p.parse_args();freeze() if a.action=='freeze' else run()
