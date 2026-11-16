"""Post-hoc false Dead bridge accounting. Does not modify predictions."""
import json,numpy as np
from scripts.research_r7 import load,write_new
from solution.evidence import labels_from_batch
from solution.hierarchical_decoder import continuous_blocks
from evaluation import possession_metrics as pm

def run():
 out=[]
 for r in json.load(open('docs/R12_RESULTS.json'))['rows']:
  b=load(r['match'],r['domain']);g=labels_from_batch(b);z=np.load('local_artifacts/r12/global-v1/'+r['match']+'-'+r['domain']+'.npz');v={}
  for name in ['global','HC_M3']:
   p=z[name];runs=[];bridge=0;short=0
   for lo,hi in continuous_blocks(b['frame'],b['period']):
    for a,c,k in pm.segments(pm.label_codes(p),[(lo,hi)]):
     if k==pm.D and a>lo and c<hi and np.all(g[a:c]!='D'):
      runs.append(c-a);short+=int(c-a<25);bridge+=int(p[a-1]!=p[c])
   v[name]=dict(wholly_false_D_runs=len(runs),short_false_D_runs=short,opposite_owner_false_D_bridges=bridge,maximum_false_D_seconds=max(runs,default=0)/25)
  out.append(dict(match=r['match'],domain=r['domain'],variants=v))
 write_new('docs/R12_DEAD_BRIDGE_DIAGNOSTIC.json',dict(posthoc_not_selection=True,rows=out))
if __name__=='__main__':run()
