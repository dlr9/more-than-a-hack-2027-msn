"""Frozen three-match IDSSE leave-one-match-out HGB and raw-dropout ablation."""
import copy,datetime,hashlib,json,subprocess,sys,time
from pathlib import Path
import joblib,numpy as np,pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from solution.modeling import load_prepared
from solution.provider_neutral import FeatureContract,assert_compatible
from solution.idsse_model import IDSSECandidate
from solution.evidence import period_reference
from scripts.evaluate_r6 import evaluate

IDS=['J03WPY','J03WOH','J03WQQ'];ROOT=Path('local_artifacts/r6/prepared');OUT=Path('local_artifacts/r6/idsse-m1-v1')
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def fit(batches,target):
    xs,ys=[],[];names=batches[0]['feature_names'].tolist()
    contracts=[FeatureContract(**json.loads(str(b['feature_contract_json'])))for b in batches]
    waivers=[assert_compatible(contracts[:1],c,purpose='train',allow_cross_pitch=True)for c in contracts]
    for b in batches:
        if b['feature_names'].tolist()!=names or int(b['schema_version'])!=3:raise ValueError('feature contract mismatch')
        y=b[target];bins=np.floor(b['frame']*2/float(b['fps'])).astype(int)
        mask=(y>=0)&(b['period']>0)&np.r_[True,np.diff(bins)>0]
        if target=='y_home':mask&=b['y_alive']==1
        xs.append(b['X'][mask]);ys.append(y[mask])
    x,y=np.vstack(xs),np.concatenate(ys)
    if set(np.unique(y))!={0,1}:raise ValueError('both known classes required')
    model=HistGradientBoostingClassifier(max_iter=60,max_leaf_nodes=15,learning_rate=.1,l2_regularization=1.,early_stopping=False,random_state=2027)
    model.fit(pd.DataFrame(x,columns=names),y)
    return model,dict(n=len(y),counts=np.bincount(y,minlength=2).tolist(),weighting='natural',waivers=waivers)

def legacy_v3_input(batch,raw):
    # Explicit provider adapter for frozen schema2 semantics: reported player S km/h->m/s.
    # Reconstruct its two local-speed columns; all other base geometry/ball differences match.
    b={**batch,'X':batch['X'].copy(),'schema_version':np.asarray(2)}
    distance=np.linalg.norm(raw['players_xy']-raw['ball_xy'][:,None,:],axis=2)
    for team,prefix in [(0,'home'),(1,'away')]:
        valid=(distance<=5)&(raw['player_team'][None,:]==team)&np.isfinite(raw['reported_speed_mps'])
        speeds=raw['reported_speed_mps'];n=valid.sum(axis=1)
        avg=np.divide(np.where(valid,speeds,0).sum(axis=1),n,out=np.full(len(n),np.nan),where=n>0)
        b['X'][:,b['feature_names'].tolist().index(prefix+'_local_speed_mean_mps')]=avg
    return b

def main():
    started=time.monotonic()
    if OUT.exists():raise FileExistsError('retain all prior results')
    protocol=json.loads(Path('docs/R6_IDSSE_PROTOCOL.json').read_text())
    manifest=json.loads(Path('docs/R6_IDSSE_INPUTS.json').read_text())
    for path,sha in manifest.items():
        if digest(path)!=sha:raise ValueError('input drift '+path)
    OUT.mkdir(parents=True)
    cache={(m,v):load_prepared(ROOT/m/(v+'.npz'))for m in IDS for v in ['natural','dropout']}
    rows=[];models=[]
    for heldout in IDS:
        train=[m for m in IDS if m!=heldout]
        natural=cache[heldout,'natural']
        truth_complete=bool((natural['y_alive']>=0).all()and((natural['y_alive']==0)|(natural['y_home']>=0)).all())
        if not truth_complete:raise ValueError('frozen full-period experiment requires complete labels')
        raw=dict(np.load(ROOT/heldout/'raw.npz',allow_pickle=False))
        old=joblib.load('local_artifacts/v3/1874553/baseline.joblib') # Preselected first fold, no target ranking.
        legacy=legacy_v3_input(natural,raw);a,h=old.probabilities(legacy);p=old.predict(legacy)
        rows.append(dict(heldout=heldout,train=old.metadata['train'],variant='V3_frozen_legacy_adapter',domain='natural',diagnostics=evaluate(natural,a,h,p,crosschecked=True)))
        reference=period_reference(natural)
        # Reference has no probabilistic heads; score_sequence below avoids fabricated confidence scores.
        from solution.evidence import labels_from_batch,score_sequence
        r=score_sequence(labels_from_batch(natural),reference,natural['period'],25.,semantics='independent_complete')
        rows.append(dict(heldout=heldout,train=[],variant='reference_smooth50',domain='natural',diagnostics=r))
        for variant in ['M1','M2_dropout']:
            training=[cache[m,'natural']for m in train]
            if variant=='M2_dropout':training += [cache[m,'dropout']for m in train]
            alive,astats=fit(training,'y_alive');home,hstats=fit(training,'y_home')
            model=IDSSECandidate(alive,home,natural['feature_names'].tolist(),dict(schema_version=3,model_version='R6-'+variant,train=train,heldout=heldout,promoted=False,decoder_time_normalized=True,training_contracts=[json.loads(str(cache[m,'natural']['feature_contract_json']))for m in train]))
            file=OUT/f'{heldout}-{variant}.joblib';joblib.dump(model,file)
            models.append(dict(heldout=heldout,variant=variant,sha256=digest(file),alive=astats,home=hstats))
            for domain in ['natural','dropout']:
                b=cache[heldout,domain];a,h=model.probabilities(b);decoded=model.predict(b)
                argmax=np.array(list('HAD'))[np.argmax(np.column_stack([a*h,a*(1-h),1-a]),axis=1)]
                rows.append(dict(heldout=heldout,train=train,variant=variant,domain=domain,diagnostics=evaluate(b,a,h,decoded,crosschecked=True),argmax=evaluate(b,a,h,argmax,crosschecked=True)))
                np.savez_compressed(OUT/f'{heldout}-{variant}-{domain}.npz',p_alive=a,p_home=h,decoded=decoded,argmax=argmax)
            print('trained/evaluated',heldout,variant,flush=True)
        del raw
    def get(m,v,d):return next(r['diagnostics']for r in rows if r['heldout']==m and r['variant']==v and r['domain']==d)
    dn=[get(m,'M2_dropout','natural')['full_spell_f1']-get(m,'M1','natural')['full_spell_f1']for m in IDS]
    ds=[get(m,'M2_dropout','dropout')['full_spell_f1']-get(m,'M1','dropout')['full_spell_f1']for m in IDS]
    keep=bool(sum(x>0 for x in ds)>=2 and np.mean(ds)>=.01 and min(dn)>=-.01)
    result=dict(run_id='R6-IDSSE-M1-v1',git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),protocol_sha256=digest('docs/R6_IDSSE_PROTOCOL.json'),code_sha256=digest(__file__),models=models,rows=rows,decision=dict(M1='KEEP_RESEARCH_BASELINE',M2='EXPERIMENTAL'if keep else'REVERT',natural_spell_delta=dn,dropout_spell_delta=ds,competition_promoted=False),elapsed_s=time.monotonic()-started)
    for path in [OUT/'results.json',Path('docs/R6_IDSSE_RESULTS.json')]:path.write_text(json.dumps(result,separators=(',',':'))+'\n')
    with Path('docs/R6_EXPERIMENT_LEDGER.jsonl').open('a')as f:f.write(json.dumps({k:v for k,v in result.items()if k not in {'rows','models'}}|dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),input_manifest='docs/R6_IDSSE_INPUTS.json',command='OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 LOKY_MAX_CPU_COUNT=2 .venv/bin/python scripts/train_idsse_r6.py',next_todo='review independent-match full spell errors before another mechanism'))+'\n')
    print(json.dumps(result['decision']),flush=True)
if __name__=='__main__':main()
