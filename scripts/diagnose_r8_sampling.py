"""Measure only each fold's training-label sample coverage; no test predictions."""
import json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.research_r7 import IDS,load,write_new,checked_path
from solution.evidence import labels_from_batch
from solution.quality import period_blocks
from evaluation import possession_metrics as pm
from solution.idsse import digest

def coverage(b,chosen):
    truth=pm.label_codes(labels_from_batch(b));fps=float(b['fps'])
    blocks=[(a,z+1)for a,z in period_blocks(b['period'])]
    spells=pm.possession_spells(truth,blocks)
    short=[(a,z)for a,z,_ in spells if (z-a)/fps<1]
    boundaries=pm.transitions(truth,blocks)
    counts={}
    for name,lo,hi in [('lt0p5',0,.5),('0p5to1',.5,1),('1to3',1,3),('ge3',3,np.inf)]:
        spans=[(a,z)for a,z,_ in spells if lo<=(z-a)/fps<hi]
        counts[name]=dict(n=len(spans),without_sample=sum(not chosen[a:z].any()for a,z in spans))
    selected=np.flatnonzero(chosen)
    near=[]
    for a,z in blocks:
        sel=selected[(selected>=a)&(selected<z)]
        for t in pm.transitions(truth,[(a,z)]):
            near.append(float(np.min(np.abs(sel-t[0]))/fps)if len(sel)else float('inf'))
    return dict(n_frames=len(truth),n_selected=int(chosen.sum()),ha_spell_sample_coverage=counts,
        short_spells=len(short),short_without_sample=sum(not chosen[a:z].any()for a,z in short),
        boundaries=len(boundaries),boundary_without_sample_within_0p12s=sum(x>.12 for x in near),
        max_boundary_sample_distance_seconds=max(near)if near else None)

def main():
    folds=[]
    for test in IDS:
        training=[]
        for m in IDS:
            if m==test:continue
            b=load(m,'natural');bins=np.floor(b['frame']*2/float(b['fps'])).astype(int)
            keep=np.r_[True,np.diff(bins)>0]&(b['period']>0)&(b['y_alive']>=0)
            training.append(dict(match=m,input_sha256=digest(checked_path(m,'natural')),coverage=coverage(b,keep)))
        # Fixed diagnosis criterion: at least10% of <1s training HA spells have no 2Hz sample.
        n=sum(t['coverage']['short_spells']for t in training);miss=sum(t['coverage']['short_without_sample']for t in training)
        folds.append(dict(heldout_not_inspected=test,training=training,short_spell_miss_rate=miss/n if n else None,trigger=bool(n and miss/n>=.1)))
    result=dict(folds=folds,trigger_at_least_two_folds=sum(f['trigger']for f in folds)>=2,
        caveat='Uniform sampling can preserve frame proportions while omitting brief spells; this diagnostic does not establish a general bias or model improvement.')
    write_new('local_artifacts/r8/sampling-diagnostic.json',result);write_new('docs/R8_SAMPLING_DIAGNOSTIC.json',result)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
