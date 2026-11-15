"""Resumable public-only independent player HGB fits and fixed-quality fusion."""
import argparse,json,subprocess,sys,time
from pathlib import Path
import joblib,numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from solution.player_only_features import build_player_features,player_inputs
from solution.quality_experts import fuse
from solution.r7_skillcorner import read_positions
from solution.idsse import digest
from solution.r7_metrics import extended
from solution.evidence import score_sequence,labels_from_batch,visibility_strata
from solution.calibration import probability_metrics
from solution.decoder import emission_probabilities,viterbi_decode
from solution.quality import period_blocks
from scripts.research_r7 import IDS,PARAMS,load,write_new,js

ROOT=Path('local_artifacts/r7b')
def identity():
    paths=['scripts/research_r7b.py','solution/player_only_features.py','solution/quality_experts.py','solution/r7_metrics.py','solution/decoder.py','evaluation/possession_metrics.py','docs/R7B_PROTOCOL.json']
    return dict(git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),hashes={p:digest(p)for p in paths})
def record(row):
    p=Path('docs/R7B_EXPERIMENTS.json');rows=json.loads(p.read_text())if p.exists()else[]
    rows.append(row);p.write_text(js(rows)+'\n')
def prepared(m):
    p=ROOT/'prepared'/f'{m}.npz';meta=json.loads(p.with_suffix('.json').read_text())
    if digest(p)!=meta['sha256']:raise ValueError('player feature cache drift')
    if digest('solution/player_only_features.py')!=meta['code_sha256']:raise ValueError('player feature definition changed')
    return dict(np.load(p,allow_pickle=False))
def prepare():
    for m in [*IDS,'1874553']:
        out=ROOT/'prepared'/f'{m}.npz';out.parent.mkdir(parents=True,exist_ok=True)
        if out.with_suffix('.json').exists():prepared(m);print('verified prepared',m,flush=True);continue
        if out.exists():raise FileExistsError('incomplete cache: preserve and inspect')
        started=time.monotonic()
        if m in IDS:
            source=Path('local_artifacts/r6/prepared')/m/'raw.npz'
            with np.load(source,allow_pickle=False)as z:raw=player_inputs(z)
            base=load(m,'natural');pitch=base['pitch'];fps=25.
            sources={str(source):digest(source)}
        else:
            raw,pitch=read_positions('local_artifacts/external/skillcorner/1874553');raw=player_inputs(raw);fps=10.
            sources=json.loads(Path('docs/R7_TRANSFER_PROTOCOL.json').read_text())['inputs']
        batch=build_player_features(raw,fps=fps,pitch=pitch)
        if len(batch['feature_names'])!=46:raise ValueError('feature count drift')
        np.savez_compressed(out,**batch)
        info=dict(match=m,sha256=digest(out),code_sha256=digest('solution/player_only_features.py'),sources=sources,
                  wall_s=time.monotonic()-started,n=len(batch['X']),n_features=46,**identity())
        write_new(out.with_suffix('.json'),info);record(dict(stage='prepare',**info));print('prepared',m,info['wall_s'],flush=True)
def model(m):
    p=ROOT/'p1-v1'/f'{m}.joblib';meta=json.loads(p.with_suffix('.json').read_text())
    if digest(p)!=meta['model_sha256']:raise ValueError('model drift')
    return joblib.load(p)
def train():
    for heldout in IDS:
        out=ROOT/'p1-v1'/f'{heldout}.joblib';out.parent.mkdir(parents=True,exist_ok=True)
        if out.with_suffix('.json').exists():model(heldout);print('verified trained',heldout,flush=True);continue
        if out.exists():raise FileExistsError('incomplete model receipt; preserve and inspect')
        started=time.monotonic();training=[m for m in IDS if m!=heldout];xs=[];ys=[];inputs={}
        for m in training:
            b=prepared(m);base=load(m,'natural');assert np.array_equal(b['frame'],base['frame'])and np.array_equal(b['period'],base['period'])
            bins=np.floor(b['frame']*2/float(b['fps'])).astype(int);keep=(base['y_alive']>=0)&(b['period']>0)&np.r_[True,np.diff(bins)>0]
            xs.append(b['X'][keep]);ys.append(base['y_alive'][keep]);inputs[m]=digest(ROOT/'prepared'/f'{m}.npz')
        x,y=np.vstack(xs),np.concatenate(ys);estimator=HistGradientBoostingClassifier(**PARAMS).fit(x,y)
        bundle=dict(alive=estimator,names=b['feature_names'].tolist(),schema='player_only_v1',train=training,heldout=heldout,
                    provider='IDSSE_DFL',observation_view='available_unknown',fps=25.,promoted=False,**identity())
        joblib.dump(bundle,out)
        meta=dict(stage='P1_fit',heldout=heldout,train=training,n=len(y),class_counts=np.bincount(y).tolist(),
                  feature_inputs=inputs,model_path=str(out),model_sha256=digest(out),wall_s=time.monotonic()-started,**identity())
        write_new(out.with_suffix('.json'),meta);record(meta);print('TRAINED P1',heldout,meta['n'],meta['wall_s'],flush=True)
def baseline(m,domain):
    root=Path('local_artifacts/r7')/('core-v1'if domain in {'natural','dropout'}else'stress-v1')
    return dict(np.load(root/f'{m}-M2-{domain}.npz',allow_pickle=False))
def decode(a,h,b):return viterbi_decode(emission_probabilities(a,h),period_blocks(b['period']),3*float(b['fps'])/10,float(b['fps'])/10)
def evaluate():
    out=ROOT/'p1-eval-v1';out.mkdir(parents=True,exist_ok=True);rows=[]
    for m in IDS:
        players=prepared(m);bundle=model(m);pa=bundle['alive'].predict_proba(players['X'])[:,1]
        for domain in ['natural','dropout','short','long']:
            b=load(m,domain);old=baseline(m,domain);mixed,use=fuse(b,players,old['p_alive'],pa)
            for v,a in [('P1',pa),('G1',mixed)]:
                path=out/f'{m}-{v}-{domain}.json'
                if path.exists():rows.append(json.loads(path.read_text()));continue
                started=time.monotonic();h=old['p_home'];pred=decode(a,h,b);arg=np.array(list('HAD'))[emission_probabilities(a,h).argmax(axis=1)]
                row=dict(match=m,variant=v,domain=domain,metrics=dict(decoded=extended(b,a,h,pred),argmax=extended(b,a,h,arg)),
                    gate_frames=int(use.sum()),wall_s=time.monotonic()-started,model_sha256=digest(ROOT/'p1-v1'/f'{m}.joblib'))
                np.savez_compressed(path.with_suffix('.npz'),p_alive=a,p_home=h,decoded=pred,argmax=arg)
                row['prediction_sha256']=digest(path.with_suffix('.npz'));write_new(path,row);rows.append(row)
                print('SCORED',m,v,domain,row['metrics']['decoded']['full_spell_f1'],row['metrics']['decoded']['dead_errors'],flush=True)
    result=dict(stage='P1_G1_evaluation',rows=rows,**identity(),decision='Pending fixed guardrail; no target promotion')
    path=Path('docs/R7B_P1_RESULTS.json')
    if not path.exists():write_new(path,result);record({k:v for k,v in result.items()if k!='rows'})
def transfer():
    out=ROOT/'transfer-v1';out.mkdir(parents=True,exist_ok=True)
    b=dict(np.load('local_artifacts/r7/transfer-v2/prepared.npz'));players=prepared('1874553');bundle=model('J03WPY')
    assert np.array_equal(b['frame'],players['frame'])and np.array_equal(b['period'],players['period'])
    old=dict(np.load('local_artifacts/r7/transfer-v2/M2.npz'));pa=bundle['alive'].predict_proba(players['X'])[:,1]
    mixed,use=fuse(b,players,old['p_alive'],pa);rows=[]
    for v,a in [('P1',pa),('G1',mixed)]:
        path=out/f'{v}.json'
        if path.exists():rows.append(json.loads(path.read_text()));continue
        h=old['p_home'];pred=decode(a,h,b);arg=np.array(list('HAD'))[emission_probabilities(a,h).argmax(axis=1)];strata={}
        for name,mask in visibility_strata(b).items():
            keep=mask&(b['y_alive']==1)&(b['y_home']>=0);n=int(keep.sum())
            strata[name]=dict(n=n,false_dead=int(np.sum(pred[keep]=='D')),argmax_dead=int(np.sum(arg[keep]=='D')),
                ha_accuracy=float(np.mean(pred[keep]==labels_from_batch(b)[keep]))if n else None,
                alive=probability_metrics(b['y_alive'][keep],a[keep]),home=probability_metrics(b['y_home'][keep],h[keep]))
        row=dict(variant=v,strata=strata,gate_frames=int(use.sum()),full_spell_f1=None,
                 waiver='Explicit research-only provider/view/fps transfer; production compatibility gates unchanged',model_sha256=digest(ROOT/'p1-v1/J03WPY.joblib'))
        np.savez_compressed(path.with_suffix('.npz'),p_alive=a,p_home=h,decoded=pred,argmax=arg);row['prediction_sha256']=digest(path.with_suffix('.npz'))
        write_new(path,row);rows.append(row);print('TRANSFER',v,strata['missing_long'],flush=True)
    path=Path('docs/R7B_TRANSFER_RESULTS.json')
    if not path.exists():write_new(path,dict(rows=rows,**identity()));record(dict(stage='partial_transfer',rows=rows,**identity()))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','train','evaluate','transfer']);a=p.parse_args();globals()[a.stage]()
