"""Evaluate finite state memory under public label-independent whole-scene outage."""
import json,sys,time
from pathlib import Path
import numpy as np,joblib,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from solution.observation_memory import carry_priors,blackout_mask
from solution.provider_neutral import RAW_FIELDS,FeatureContract,provider_neutral_features_v3
from solution.idsse import digest
from solution.r7_metrics import extended
from solution.evidence import visibility_strata,labels_from_batch
from solution.decoder import emission_probabilities
from scripts.research_r7 import IDS,load,model_path,write_new
from scripts.research_r7b import ROOT,identity,decode,baseline

OUT=ROOT/'memory-v1'
def main():
    OUT.mkdir(parents=True,exist_ok=True);rows=[];checks=[]
    for m in IDS:
        for domain in ['natural','dropout']:
            b=load(m,domain);old=baseline(m,domain);a,h,info=carry_priors(b,old['p_alive'],old['p_home'],team_memory=True)
            assert np.array_equal(a,old['p_alive'])and np.array_equal(h,old['p_home'])
            checks.append(dict(match=m,domain=domain,probabilities_unchanged=True))
        path=OUT/f'{m}-prepared.npz';base=load(m,'natural')
        if path.exists():
            meta=json.loads(path.with_suffix('.json').read_text());assert digest(path)==meta['sha256'];b=dict(np.load(path))
        else:
            source=Path('local_artifacts/r6/prepared')/m/'raw.npz'
            with np.load(source,allow_pickle=False)as z:raw={k:z[k]for k in RAW_FIELDS}
            mask=blackout_mask(raw['period'],25.);raw['ball_xy']=raw['ball_xy'].copy();raw['players_xy']=raw['players_xy'].copy()
            raw['ball_xy'][mask]=np.nan;raw['players_xy'][mask]=np.nan
            b=provider_neutral_features_v3(raw,FeatureContract(**json.loads(str(base['feature_contract_json']))))
            for k in ['y_alive','y_home','match_id','label_semantics']:b[k]=base[k]
            np.savez_compressed(path,**b);write_new(path.with_suffix('.json'),dict(sha256=digest(path),source_sha256=digest(source),masked_frames=int(mask.sum()),**identity()))
        frozen=joblib.load(model_path(m,'M2'));x=pd.DataFrame(b['X'],columns=b['feature_names'].tolist());ba=frozen.alive.predict_proba(x)[:,1];bh=frozen.home.predict_proba(x)[:,1]
        for variant,memory in [('M2',None),('T1',False),('T2',True)]:
            resultpath=OUT/f'{m}-{variant}.json'
            if resultpath.exists():rows.append(json.loads(resultpath.read_text()));continue
            if memory is None:a,h,info=ba,bh,{}
            else:a,h,info=carry_priors(b,ba,bh,team_memory=memory)
            pred=decode(a,h,b);arg=np.array(list('HAD'))[emission_probabilities(a,h).argmax(axis=1)]
            row=dict(match=m,variant=variant,domain='scene_blackout',metrics=dict(decoded=extended(b,a,h,pred),argmax=extended(b,a,h,arg)),usage={k:int(v.sum())for k,v in info.items()})
            np.savez_compressed(resultpath.with_suffix('.npz'),p_alive=a,p_home=h,decoded=pred,argmax=arg);write_new(resultpath,row);rows.append(row)
            print('MEMORY',m,variant,row['metrics']['decoded']['full_spell_f1'],row['metrics']['decoded']['dead_errors'],flush=True)
    b=dict(np.load('local_artifacts/r7/transfer-v2/prepared.npz'));old=dict(np.load('local_artifacts/r7/transfer-v2/M2.npz'));partial=[]
    for variant,memory in [('T1',False),('T2',True)]:
        a,h,info=carry_priors(b,old['p_alive'],old['p_home'],team_memory=memory);pred=decode(a,h,b);strata={}
        for name,mask in visibility_strata(b).items():
            k=mask&(b['y_alive']==1)&(b['y_home']>=0);n=int(k.sum())
            strata[name]=dict(n=n,false_dead=int(np.sum(pred[k]=='D')),ha_accuracy=float(np.mean(pred[k]==labels_from_batch(b)[k]))if n else None,
                alive_prior_used=int(info['alive_prior_used'][k].sum()),home_prior_used=int(info['home_prior_used'][k].sum()))
        cache=OUT/f'SC-{variant}.npz'
        if cache.exists():
            with np.load(cache)as saved:
                if not all(np.array_equal(saved[k],v)for k,v in [('p_alive',a),('p_home',h),('decoded',pred)]):raise ValueError('preserved SC memory replay mismatch')
        else:np.savez_compressed(cache,p_alive=a,p_home=h,decoded=pred)
        partial.append(dict(variant=variant,strata=strata,full_spell_f1=None));print('SC MEMORY',variant,strata['missing_long'],flush=True)
    path=Path('docs/R7B_MEMORY_RESULTS.json')
    if not path.exists():write_new(path,dict(rows=rows,partial=partial,unchanged_domains=checks,**identity(),extra_hashes={p:digest(p)for p in ['scripts/memory_r7b.py','solution/observation_memory.py','docs/R7B_MEMORY_PROTOCOL.json']}))
if __name__=='__main__':main()
