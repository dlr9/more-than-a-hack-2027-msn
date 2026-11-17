"""Exact replay of all three frozen R13 experiments and official spell metrics."""
import json
from pathlib import Path
import numpy as np,joblib
from scripts.research_r7 import load,write_new
from scripts.evaluate_r8_decoder import prediction_path
from scripts.train_r13_dense import observation
from r12.dense_control import features,probabilities,fixed_dead_decode
from r12.control_memory import decode
from r12.control_calibration import correct,pool
from solution.hierarchical_decoder import continuous_blocks
from solution.evidence import labels_from_batch
from solution.idsse import digest
from r11.spell_roi import score
from scripts.evaluate_r11_roi import check
from evaluation import possession_metrics as pm

def run(output='docs/R13_VERIFICATION.json'):
 count=0;probs=0;scorings=0;bridges=[]
 models=json.loads(Path('docs/R13_DENSE_RESULTS.json').read_text())['models']
 for k,v in models.items():assert digest(f'local_artifacts/r13/dense-v1/{k}.joblib')==v['sha256']
 for family,result_file in [('memory','R13_RESULTS'),('dense','R13_DENSE_RESULTS'),('calibration','R13_CALIBRATION_RESULTS')]:
  for row in json.loads(Path('docs/'+result_file+'.json').read_text())['rows']:
   m,d=row['match'],row['domain'];b=load(m,d);g=labels_from_batch(b);blocks=list(continuous_blocks(b['frame'],b['period']));base=np.load(f'local_artifacts/r12/global-v1/{m}-{d}.npz')['HC_M3'];home=np.load(f'local_artifacts/r8/e08c/{m}-{d}.npz')['p_home']
   path=Path(f'local_artifacts/r13/{family}-v1/{m}-{d}.npz');assert digest(path)==row['prediction_sha'];z=dict(np.load(path))
   if family=='memory':
    a=np.load(prediction_path(m,'M3',d))['p_alive'];events=json.loads(Path(f'local_artifacts/r12/global-v1/{m}-{d}-anchors.json').read_text());signal=np.full(len(g),-1,np.int8)
    for e in events:signal[e['start']:e['end']]=int(e['team']=='A')
    signals=dict(memory=signal,no_anchors=np.full(len(g),-1,np.int8),shift=signal.copy(),shuffle_team=signal.copy());rng=np.random.default_rng(1201)
    for lo,hi in blocks:signals['shift'][lo:hi]=np.roll(signal[lo:hi],750)
    for e in events:signals['shuffle_team'][e['start']:e['end']]=rng.integers(2)
    for name,s in signals.items():
     pred,latent,_=decode(a,home,s,b['frame'],b['period'],25,base);np.testing.assert_array_equal(pred,z[name]);np.testing.assert_array_equal(latent,z[name+'_latent']);count+=1
    vals={}
    old=np.load(f'local_artifacts/r12/global-v1/{m}-{d}.npz')['global']
    for name,p in [('R12',old),('memory',z['memory'])]:
     n=0
     for lo,hi in blocks:
      for start,end,k in pm.segments(pm.label_codes(p),[(lo,hi)]):
       if k==pm.D and start>lo and end<hi and np.all(g[start:end]!='D') and p[start-1]!=p[end]:n+=1
     vals[name]=n
    bridges.append(dict(match=m,domain=d,false_D_opposite_owner_bridges=vals))
   else:
    cached=dict(np.load(f'local_artifacts/r13/dense-v1/{m}-{d}.npz'))
    if family=='dense':
     raw,_,_=observation(m,d);x=features(raw)
     for k in models:
      actual=probabilities(joblib.load(f'local_artifacts/r13/dense-v1/{k}.joblib'),x);np.testing.assert_array_equal(actual,cached[k+'_prob']);probs+=1
     heads={'no_event':home}
     for k in models:
      q=cached[k+'_prob'];heads[k]=.5*home+.5*(q[:,0]+.5*q[:,2])
     shift=cached['human_prob'].copy()
     for lo,hi in blocks:shift[lo:hi]=np.roll(shift[lo:hi],750,axis=0)
     heads['time_shift']=.5*home+.5*(shift[:,0]+.5*shift[:,2])
    else:
     priors=json.loads(Path('docs/R13_CALIBRATION_PROTOCOL.json').read_text())['priors'];q={k:correct(cached[k+'_prob'],priors[k]) for k in models};v=q['human'];shift=v.copy()
     for lo,hi in blocks:shift[lo:hi]=np.roll(v[lo:hi],750,axis=0)
     heads=dict(corrected=pool(home,v),prior_only=.5*home+.5*(v[:,0]+.5*v[:,2]),neutral_only=pool(home,cached['human_prob']),source_corrected=pool(home,q['source']),shuffle_corrected=pool(home,q['shuffle']),time_shift_corrected=pool(home,shift),neutral_control=pool(home,np.full((len(home),3),1/3)))
    for k,h in heads.items():
     pred=fixed_dead_decode(h,base,b['frame'],b['period']);np.testing.assert_array_equal(pred,z[k]);count+=1
   for k,v in row['results'].items():
    official=pm.spell_scores(pm.label_codes(g),pm.label_codes(z[k]),blocks);check(score(g.tolist(),z[k].tolist(),blocks=blocks),official)
    assert abs(official['mean_f1']-v['metrics']['spell']['mean_f1'])<1e-14;scorings+=1
 write_new(output,dict(models_verified=len(models),exact_probability_replays=probs,exact_full_prediction_replays=count,official_ROI_score_equivalences=scorings,bridges=bridges))
 print('VERIFIED',len(models),'models',probs,'probability arrays',count,'full predictions',scorings,'scores')
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--out',default='docs/R13_VERIFICATION.json');run(p.parse_args().out)
