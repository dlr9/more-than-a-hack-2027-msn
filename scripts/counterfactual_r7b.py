"""Real HGB training with label-independent ball observation degradation."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np,joblib,pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from solution.ball_counterfactual import intervene
from solution.provider_neutral import RAW_FIELDS,FeatureContract,provider_neutral_features_v3
from solution.idsse import digest
from solution.r7_metrics import extended
from solution.decoder import emission_probabilities
from scripts.research_r7 import IDS,PARAMS,load,model_path,write_new
from scripts.research_r7b import ROOT,identity,record,baseline,decode
from scripts.calibrate_r7b import sample

OUT=ROOT/'counterfactual-v1'
def ident():return {**identity(),'counterfactual_hashes':{p:digest(p)for p in ['solution/ball_counterfactual.py','scripts/counterfactual_r7b.py','docs/R7B_COUNTERFACTUAL_PROTOCOL.json']}}
def prepared(m,d):
    p=OUT/'prepared'/f'{m}-{d}.npz';meta=json.loads(p.with_suffix('.json').read_text())
    if digest(p)!=meta['sha256']:raise ValueError('CF cache drift')
    return dict(np.load(p,allow_pickle=False))
def prepare():
    for m in IDS:
        base=load(m,'natural');source=Path('local_artifacts/r6/prepared')/m/'raw.npz'
        with np.load(source,allow_pickle=False)as z:raw={k:z[k]for k in RAW_FIELDS}
        contract=FeatureContract(**json.loads(str(base['feature_contract_json'])))
        for mode in ['combined','freeze','noise']:
            path=OUT/'prepared'/f'{m}-{mode}.npz';path.parent.mkdir(parents=True,exist_ok=True)
            if path.with_suffix('.json').exists():prepared(m,mode);continue
            if path.exists():raise FileExistsError('incomplete CF cache')
            started=time.monotonic();ball,info=intervene(raw['ball_xy'],raw['period'],25.,mode=mode)
            b=provider_neutral_features_v3({**raw,'ball_xy':ball},contract)
            for k in ['y_alive','y_home','match_id','label_semantics']:b[k]=base[k]
            np.savez_compressed(path,**b)
            meta=dict(match=m,mode=mode,sha256=digest(path),raw_sha256=digest(source),wall_s=time.monotonic()-started,
                freeze_frames=int(info['freeze_mask'].sum()),masked_frames=int(info['missing_mask'].sum()),**ident())
            write_new(path.with_suffix('.json'),meta);print('CF PREPARED',m,mode,meta['wall_s'],flush=True)
def train():
    for m in IDS:
        path=OUT/f'{m}.joblib';path.parent.mkdir(parents=True,exist_ok=True)
        if path.with_suffix('.json').exists():assert digest(path)==json.loads(path.with_suffix('.json').read_text())['model_sha256'];continue
        if path.exists():raise FileExistsError('incomplete CF model receipt')
        started=time.monotonic();xs=[];ys=[];training=[x for x in IDS if x!=m]
        for t in training:
            for b in [load(t,'natural'),prepared(t,'combined')]:
                s=sample(b);xs.append(b['X'][s]);ys.append(b['y_alive'][s])
        estimator=HistGradientBoostingClassifier(**PARAMS).fit(np.vstack(xs),np.concatenate(ys));joblib.dump(estimator,path)
        meta=dict(stage='CF1_fit',heldout=m,train=training,n=len(np.concatenate(ys)),model_sha256=digest(path),wall_s=time.monotonic()-started,**ident())
        write_new(path.with_suffix('.json'),meta);record(meta);print('CF TRAINED',m,meta['wall_s'],flush=True)
def evaluate():
    rows=[]
    for m in IDS:
        cf=joblib.load(OUT/f'{m}.joblib');old=joblib.load(model_path(m,'M2'))
        for domain in ['natural','dropout','long','freeze','noise']:
            b=prepared(m,domain)if domain in {'freeze','noise'}else load(m,domain)
            if domain in {'freeze','noise'}:
                x=pd.DataFrame(b['X'],columns=b['feature_names'].tolist());ba=old.alive.predict_proba(x)[:,1];h=old.home.predict_proba(x)[:,1]
            else:
                prev=baseline(m,domain);ba=prev['p_alive'];h=prev['p_home']
            for variant,a in [('M2',ba),('CF1',cf.predict_proba(b['X'])[:,1])]:
                if variant=='M2'and domain not in {'freeze','noise'}:continue # Frozen reference metrics already published.
                path=OUT/f'{m}-{variant}-{domain}.json'
                if path.exists():rows.append(json.loads(path.read_text()));continue
                pred=decode(a,h,b);arg=np.array(list('HAD'))[emission_probabilities(a,h).argmax(axis=1)]
                row=dict(match=m,variant=variant,domain=domain,metrics=dict(decoded=extended(b,a,h,pred),argmax=extended(b,a,h,arg)))
                np.savez_compressed(path.with_suffix('.npz'),p_alive=a,p_home=h,decoded=pred,argmax=arg);write_new(path,row);rows.append(row)
                print('CF SCORE',m,variant,domain,row['metrics']['decoded']['full_spell_f1'],flush=True)
    path=Path('docs/R7B_COUNTERFACTUAL_RESULTS.json')
    if not path.exists():write_new(path,dict(rows=rows,**ident()));record(dict(stage='CF1_evaluation',n_rows=len(rows),**ident()))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','train','evaluate']);a=p.parse_args();globals()[a.stage]()
