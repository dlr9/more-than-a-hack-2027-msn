"""Post-hoc ownership agreement; no fitting, thresholds or candidate selection."""
import argparse,json
from pathlib import Path
import numpy as np,joblib
from scripts.train_r13_dense import observation
from scripts.research_r7 import load,write_new
from r12.dense_control import features,probabilities
from r12.control_calibration import correct
from solution.evidence import labels_from_batch
from solution.idsse import digest

def run(out):
 prior=json.loads(Path('docs/R13_CALIBRATION_PROTOCOL.json').read_text())['priors']['human'];rows=[]
 model='local_artifacts/r13/dense-v1/human.joblib'
 for match in ['J03WPY','J03WOH','J03WQQ']:
  batch=load(match,'natural');gt=labels_from_batch(batch)
  base=np.load(f'local_artifacts/r12/global-v1/{match}-natural.npz')['HC_M3']
  if match=='J03WPY':
   raw,_,_=observation(match,'natural');p=probabilities(joblib.load(model),features(raw))
  else:p=np.load(f'local_artifacts/r13/dense-v1/{match}-natural.npz')['human_prob']
  q=correct(p,prior);expert=np.where(q[:,0]>=q[:,1],'H','A');alive=gt!='D'
  selected=alive&(np.max(q[:,:2],axis=1)>q[:,2]);disagree=alive&(expert!=base)
  rows.append(dict(match=match,training_match=match=='J03WPY',selected_frames=int(selected.sum()),expert_correct_selected=int(np.sum(selected&(expert==gt))),baseline_correct_selected=int(np.sum(selected&(base==gt))),disagreement_frames=int(disagree.sum()),expert_correct_disagreement=int(np.sum(disagree&(expert==gt))),baseline_correct_disagreement=int(np.sum(disagree&(base==gt)))))
 write_new(out,dict(posthoc=True,selection='GT H/A and corrected event posterior max(H,A)>NONE; diagnostic only, not deployable gate',limitations='Event team versus persistent ownership are different tasks. Results do not prove that capacity or other sensors cannot help. No candidate changed.',model_sha256=digest(model),rows=rows))
 print(json.dumps(rows,indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',default='docs/R13_EVENT_DIAGNOSTIC.json');run(p.parse_args().out)
