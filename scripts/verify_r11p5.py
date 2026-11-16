"""Read-only prediction replay and full-timeline error attribution, no refit."""
import json
from pathlib import Path
import numpy as np,joblib
from sklearn.metrics import average_precision_score
from scripts.research_r11p5 import ROOT,DOMAINS,obs,prob,full
from scripts.research_r7 import IDS,load,write_new
from scripts.probe_r10_acquisition import hc_path
from r11p5.timeline import features,targets,decode
from solution.evidence import labels_from_batch
from solution.hierarchical_decoder import continuous_blocks
from solution.idsse import digest
from evaluation import possession_metrics as pm

def main():
 result=json.loads(Path('docs/R11P5_RESULTS.json').read_text());rows=[]
 for model in result['models']:assert digest(model['path'])==model['sha256']
 for m in IDS:
  for d in DOMAINS:
   b=load(m,d);g=labels_from_batch(b);gc=pm.label_codes(g);blocks=list(continuous_blocks(b['frame'],b['period']));cache=np.load(ROOT/f'{m}-{d}.npz');base=np.load(hc_path(m,d))['decoded'];pred=cache['context_decoded'];pc=pm.label_codes(pred);bc=pm.label_codes(base)
   for view in ('instant','context'):
    a,s=features(obs(b),view=='context')
    for target,x in [('hazard',a),('direction',s)]:
     model=joblib.load(ROOT/f'{m}-{view}-{target}.joblib');p=prob(model,x,target=='direction').astype('float32');np.testing.assert_array_equal(p,cache[view+('_away' if target=='direction' else '_hazard')])
   truth,_,_=targets(g,b['frame'],b['period'],25);all_target=(truth==1).astype(int)
   gs=pm.possession_spells(gc,blocks);bs=pm.possession_spells(bc,blocks);ps=pm.possession_spells(pc,blocks);bo=pm.spell_overlaps(gs,bs);po=pm.spell_overlaps(gs,ps);changes={}
   for threshold in pm.SPELL_IOUS:
    before=pm.matched_gt_flags(bo,len(gs),threshold);after=pm.matched_gt_flags(po,len(gs),threshold);changes[str(threshold)]=dict(newly_matched=int((after&~before).sum()),lost_matches=int((before&~after).sum()),net=int(after.sum()-before.sum()))
   visible=b['X'][:,b['feature_names'].tolist().index('ball_visible')]==1
   strata={}
   for name,mask in [('visible',visible),('missing',~visible)]:
    strata[name]=dict(frames=int(mask.sum()),gt_direct_positive_frames=int(all_target[mask].sum()),wrong_HA_before=int(((base!=g)&np.isin(g,['H','A'])&mask).sum()),wrong_HA_after=int(((pred!=g)&np.isin(g,['H','A'])&mask).sum()))
   cuts={};restarts={}
   for name,codes in [('base',bc),('candidate',pc)]:
    cuts[name]=sum(sum(k==pm.D for _,_,k in pm.segments(codes,[(a,z)])) for a,z,k in gs)
    good=total=0
    for block in blocks:
     seg=pm.segments(gc,[block])
     for j,(a,z,k) in enumerate(seg):
      if k==pm.D and j>0 and j+1<len(seg) and seg[j-1][2]==seg[j+1][2] and seg[j-1][2] in (pm.H,pm.A):
       total+=1;good+=int(np.any(codes[a:z]==pm.D) and codes[max(block[0],a-1)]==seg[j-1][2] and codes[min(block[1]-1,z)]==seg[j+1][2])
    restarts[name]=dict(source_same_team_restarts=total,exact_adjacent_owner_and_detected_D=good)
   rows.append(dict(match=m,domain=d,full_timeline_hazard_AP=float(average_precision_score(all_target,cache['context_hazard'])),positive_tolerance_s=.2,full_timeline_frames=len(g),spell_match_changes=changes,strata=strata,false_D_cuts=cuts,restarts=restarts,D_mask_exact=bool(np.array_equal(base=='D',pred=='D'))))
 write_new('docs/R11P5_VERIFICATION.json',dict(model_hashes=12,exact_probability_arrays_replayed=24,rows=rows,new_fits=0,warning='All-timeline evaluation includes D and all near-boundary negatives; training-only exclusion AP remains separately labelled in original results. No decision/threshold changed.'))
 print('Verified12 models,24 all-timeline arrays,6 complete candidate error budgets')
if __name__=='__main__':main()
