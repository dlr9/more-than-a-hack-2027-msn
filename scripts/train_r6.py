"""R6 preregistered, real CPU training on existing public external matches."""
import copy,datetime,hashlib,json,subprocess,sys,time
from pathlib import Path
import joblib,numpy as np,pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from solution.modeling import load_prepared
from scripts.research_v3 import diagnostics
from solution.candidate import Candidate

IDS=['1874553','1886347','1899585','1927964']
OUT=Path('local_artifacts/r6/fallback-v1')

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fit(batches,target,period=None):
    xs,ys=[],[];names=batches[0]['feature_names'].astype(str).tolist()
    for b in batches:
        if b['feature_names'].astype(str).tolist()!=names:raise ValueError('feature mismatch')
        if int(b['schema_version'])!=2:raise ValueError('this control uses frozen legacy schema2 only')
        y=b[target];mask=(y>=0)&(b['period']>0)
        if target=='y_home':mask&=b['y_alive']==1
        if period is not None:mask&=b['period']==period
        bins=np.floor(b['frame']*2/float(b['fps'])).astype(int)
        mask&=np.r_[True,np.diff(bins)>0]
        xs.append(b['X'][mask]);ys.append(y[mask])
    x,y=np.vstack(xs),np.concatenate(ys)
    if set(np.unique(y))!={0,1}:raise ValueError('both known classes required')
    model=HistGradientBoostingClassifier(max_iter=60,max_leaf_nodes=15,learning_rate=.1,l2_regularization=1.,early_stopping=False,random_state=2027)
    model.fit(pd.DataFrame(x,columns=names),y)
    return model,dict(n=len(y),class_counts=np.bincount(y,minlength=2).tolist(),sample_weight='natural_unweighted')

def period_subset(b,period):
    n=len(b['frame']);mask=b['period']==period
    return {k:(v[mask] if v.ndim>0 and len(v)==n and k not in {'feature_names'}else v)for k,v in b.items()}

def main():
    start=time.monotonic()
    if OUT.exists():raise FileExistsError('preserve prior run; choose a new preregistered run ID')
    protocol=json.loads(Path('docs/R6_EXPERIMENT_PROTOCOL.json').read_text())
    assert protocol['status']=='frozen_before_training'
    hashes=json.loads(Path('docs/R6_INPUT_MANIFEST.json').read_text())
    for p,h in hashes.items():
        if digest(p)!=h:raise ValueError('input drift: '+p)
    OUT.mkdir(parents=True)
    cache={(m,v):load_prepared(f'local_artifacts/v2/{m}-{v}.npz')for m in IDS for v in ['observed','dropout']}
    metrica=load_prepared('local_artifacts/v2/metrica.npz')
    weak_alive,alive_training=fit([metrica],'y_alive',period=1)
    joblib.dump(weak_alive,OUT/'weak_alive_metrica_period1.joblib')
    print('F2 weak Alive trained on Metrica period1',alive_training,flush=True)
    test_metrica=period_subset(metrica,2)
    rows=[];models=[]
    for i,m in enumerate(IDS):
        train=[j for j in IDS if j not in [m,IDS[(i+1)%4]]]
        home,home_training=fit([cache[j,'observed']for j in train],'y_home')
        baseline=joblib.load(f'local_artifacts/v3/{m}/baseline.joblib')
        f1=copy.copy(baseline);f1.home=home;f1.metadata={**baseline.metadata,'model_version':'R6-F1','home_weighting':'natural','promoted':False}
        f2=copy.copy(baseline);f2.alive=weak_alive;f2.metadata={**baseline.metadata,'model_version':'R6-F2','alive_training':'MetricaGame3 period1 weak event interiors','observation_view':'explicit_cross_provider_heads','promoted':False}
        for variant,model in [('F1',f1),('F2',f2)]:joblib.dump(model,OUT/f'{m}-{variant}.joblib')
        models.append(dict(heldout=m,train=train,unused_calibration=IDS[(i+1)%4],F1_training=home_training,F1_sha256=digest(OUT/f'{m}-F1.joblib'),F2_sha256=digest(OUT/f'{m}-F2.joblib')))
        for domain,b in [('natural',cache[m,'observed']),('dropout_stress',cache[m,'dropout']),('metrica_period2',test_metrica)]:
            for variant,model in [('baseline',baseline),('F1',f1),('F2',f2)]:
                a,h=model.probabilities(b);pred=model.predict(b)
                d=diagnostics(b,a,h,pred)
                rows.append(dict(heldout=m,domain=domain,variant=variant,diagnostics=d))
                np.savez_compressed(OUT/f'{m}-{domain}-{variant}.npz',p_alive=a,p_home=h,labels=pred)
        print('completed heldout',m,flush=True)
    def ds(m,domain,v):return next(r['diagnostics']for r in rows if r['heldout']==m and r['domain']==domain and r['variant']==v)
    brier_delta=[ds(m,'natural','F1')['home']['brier']-ds(m,'natural','baseline')['home']['brier']for m in IDS]
    ha_delta=[ds(m,'natural','F1')['ha_macro_f1']-ds(m,'natural','baseline')['ha_macro_f1']for m in IDS]
    macro_delta=[ds(m,'natural','F1')['primary']['partial_macro_f1']-ds(m,'natural','baseline')['primary']['partial_macro_f1']for m in IDS]
    keep1=bool(sum(d<0 for d in brier_delta)>=3 and np.mean(ha_delta)>=-.005 and min(macro_delta)>=-.01)
    bd=[];dd=[];fd=[]
    for m in IDS:
        b,f=ds(m,'metrica_period2','baseline'),ds(m,'metrica_period2','F2')
        bd.append(f['alive_proxy']['brier']-b['alive_proxy']['brier']);dd.append(f['primary']['per_class']['D']['f1']-b['primary']['per_class']['D']['f1'])
        b,f=ds(m,'natural','baseline'),ds(m,'natural','F2')
        fd.append((f['primary']['live_to_dead_proxy_errors']-b['primary']['live_to_dead_proxy_errors'])/b['primary']['n_known'])
    keep2=bool(np.mean(bd)<0 and np.mean(dd)>0 and max(fd)<=.01)
    result=dict(run_id='R6-fallback-v1',protocol_sha256=digest('docs/R6_EXPERIMENT_PROTOCOL.json'),git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),code_sha256=digest(__file__),input_hashes=hashes,alive_training=alive_training,models=models,rows=rows,decisions={'F1':dict(decision='EXPERIMENTAL'if keep1 else'REVERT',brier_delta=brier_delta,ha_macro_delta=ha_delta,partial_macro_delta=macro_delta),'F2':dict(decision='EXPERIMENTAL'if keep2 else'REVERT',metrica_alive_brier_delta=bd,metrica_D_f1_delta=dd,skillcorner_false_dead_rate_delta=fd)},full_spell_f1=None,promoted=False,elapsed_s=time.monotonic()-start)
    (OUT/'results.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
    Path('docs/R6_FALLBACK_RESULTS.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
    entry={k:result[k]for k in ['run_id','git_sha','code_sha256','input_hashes','decisions','elapsed_s','full_spell_f1']}
    entry.update(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),command='OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 LOKY_MAX_CPU_COUNT=2 .venv/bin/python scripts/train_r6.py',failure_reason='full independent HAD labels unavailable in these sources; Metrica validation within match only',next_todo='IDSSE verified full period parsing and independent-match training; neutral schema raw rebuild')
    with Path('docs/R6_EXPERIMENT_LEDGER.jsonl').open('a')as f:f.write(json.dumps(entry)+'\n')
    print(json.dumps(result['decisions']),flush=True)
if __name__=='__main__':main()
