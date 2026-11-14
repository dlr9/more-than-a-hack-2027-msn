"""Frozen public-only R7 experiments. Run core, prepare-stress, then stress."""
import argparse, datetime, hashlib, json, resource, subprocess, sys, time
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from solution.idsse import digest
from solution.modeling import load_prepared
from solution.provider_neutral import RAW_FIELDS, FeatureContract, provider_neutral_features_v3
from solution.r7_features import project, AblatedCandidate, blind_intervals
from solution.r7_metrics import extended
from solution.evidence import period_reference
from scripts.train_idsse_r6 import legacy_v3_input

IDS=('J03WPY','J03WOH','J03WQQ');ROOT=Path('local_artifacts/r7');OLD=Path('local_artifacts/r6')
PARAMS=dict(max_iter=60,max_leaf_nodes=15,learning_rate=.1,l2_regularization=1.,early_stopping=False,random_state=2027)
VARIANTS={'M4':('no_ball_motion',False),'M5':('no_ball_motion',True),'M6':('geometry_context',True)}

def js(value):return json.dumps(value,indent=2,default=lambda x:x.item() if isinstance(x,np.generic)else x)
def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x')as f:f.write(js(value)+'\n')
def record(row):
    p=Path('docs/R7_EXPERIMENTS.json');records=json.loads(p.read_text());records.append(row);p.write_text(js(records)+'\n')
def source_identity():
    files=['scripts/research_r7.py','solution/r7_features.py','solution/r7_metrics.py','solution/provider_neutral.py','solution/features.py','solution/decoder.py','evaluation/possession_metrics.py']
    return dict(git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),code_sha256={p:digest(p)for p in files},protocol_sha256=digest('docs/R7_PROTOCOL.json'),input_manifest_sha256=digest('docs/R7_INPUTS.json'))
def checked_path(match, domain):
    if match not in IDS or domain not in {'natural','dropout','short','long'}:raise ValueError('unregistered public research input')
    p=(OLD/'prepared'/match/(domain+'.npz'))if domain in {'natural','dropout'}else ROOT/'prepared'/match/(domain+'.npz')
    if p.is_symlink()or not p.resolve().is_relative_to(Path('local_artifacts').resolve()):raise ValueError('input escape')
    return p

def verify_inputs():
    for p,sha in json.loads(Path('docs/R7_INPUTS.json').read_text()).items():
        if digest(p)!=sha:raise ValueError('frozen input drift '+p)
    for m in IDS:
        report=json.loads((OLD/f'floodlight_crosscheck_{m}.json').read_text())
        if report['match']!=m or len(report['halves'])!=2 or not all(h['labels_equal']and h['ball_xy_equal']and h['transition_positions_equal']for h in report['halves']):raise ValueError('independent source check required')
        for p,sha in report['files'].items():
            if digest(p)!=sha:raise ValueError('source check drift')

def load(match,domain):
    p=checked_path(match,domain)
    if domain in {'short','long'}:
        manifest=json.loads((ROOT/'prepared'/match/'manifest.json').read_text())
        if digest(p)!=manifest['domains'][domain]['prepared_sha256']:raise ValueError('corruption cache drift')
    b=load_prepared(p)
    if str(b['match_id'])!=match or str(b['provider'])!='IDSSE_DFL' or str(b['label_semantics'])!='DFL_source_complete':raise ValueError('unapproved source content')
    return b

def fit(batches,target,view):
    xs=[];ys=[];names=None
    for base in batches:
        b=project(base,view);names=b['feature_names'].tolist();y=b[target]
        bins=np.floor(b['frame']*2/float(b['fps'])).astype(int)
        keep=(y>=0)&(b['period']>0)&np.r_[True,np.diff(bins)>0]
        if target=='y_home':keep&=b['y_alive']==1
        xs.append(b['X'][keep]);ys.append(y[keep])
    x,y=np.vstack(xs),np.concatenate(ys)
    if set(np.unique(y))!={0,1}:raise ValueError('both classes required')
    model=HistGradientBoostingClassifier(**PARAMS).fit(pd.DataFrame(x,columns=names),y)
    return model,dict(n=len(y),class_counts=np.bincount(y,minlength=2).tolist())

def eval_model(model,base,variant):
    b=project(base,VARIANTS[variant][0])if variant in VARIANTS else base
    a,h=model.probabilities(b);pred=model.predict(b)
    raw=np.array(list('HAD'))[np.argmax(np.column_stack([a*h,a*(1-h),1-a]),axis=1)]
    # Evaluation uses full base geometry for fixed strata even when a model removes columns.
    return dict(decoded=extended(base,a,h,pred),argmax=extended(base,a,h,raw)),dict(p_alive=a,p_home=h,decoded=pred)

def model_path(m,v):
    if v in VARIANTS:return ROOT/'core-v1'/f'{m}-{v}.joblib'
    if v=='M3':return OLD/'missing-stability-v1'/f'{m}.joblib'
    name='M2_dropout'if v=='M2'else'M1'
    return OLD/'idsse-m1-v1'/f'{m}-{name}.joblib'

def old_prediction(m,v,domain):
    if v=='M3':return OLD/'missing-stability-v1'/f'{m}-{domain}.npz'
    name='M2_dropout'if v=='M2'else'M1'
    return OLD/'idsse-m1-v1'/f'{m}-{name}-{domain}.npz'

def core():
    started=time.monotonic();cpu=time.process_time();out=ROOT/'core-v1'
    if out.exists():raise FileExistsError('preserve core run')
    verify_inputs();out.mkdir(parents=True);identity=source_identity();rows=[];models=[]
    cache={(m,d):load(m,d)for m in IDS for d in ['natural','dropout']}
    for m in IDS:
        train=[x for x in IDS if x!=m]
        for v in ['M1','M2','M3','M4','M5','M6']:
            if v in VARIANTS:
                view,aug=VARIANTS[v];training=[cache[x,d]for x in train for d in (['natural','dropout']if aug else['natural'])]
                a,astats=fit(training,'y_alive',view);h,hstats=fit(training,'y_home',view)
                names=project(cache[m,'natural'],view)['feature_names'].tolist()
                model=AblatedCandidate(a,h,names,dict(schema_version=4,feature_view=view,train=train,heldout=m,promoted=False,decoder_time_normalized=True,model_version='R7-'+v,training_contracts=[json.loads(str(cache[x,'natural']['feature_contract_json']))for x in train]))
                path=out/f'{m}-{v}.joblib';joblib.dump(model,path)
                models.append(dict(heldout=m,variant=v,train=train,path=str(path),sha256=digest(path),bytes=path.stat().st_size,alive=astats,home=hstats))
            else:
                path=model_path(m,v)
                model=joblib.load(path)
            for domain in ['natural','dropout']:
                score,arrays=eval_model(model,cache[m,domain],v)
                if v not in VARIANTS:
                    path=old_prediction(m,v,domain)
                    with np.load(path)as saved:
                        for name in saved.files:
                            if name in arrays and not np.array_equal(saved[name],arrays[name]):raise ValueError('R6 replay mismatch')
                np.savez_compressed(out/f'{m}-{v}-{domain}.npz',**arrays)
                rows.append(dict(match=m,variant=v,domain=domain,metrics=score))
            print('core',m,v,rows[-2]['metrics']['decoded']['full_spell_f1'],rows[-1]['metrics']['decoded']['full_spell_f1'],flush=True)
        # External rule reference only; no fabricated head calibration.
        from solution.evidence import score_sequence, labels_from_batch
        b=cache[m,'natural'];ref=period_reference(b)
        rows.append(dict(match=m,variant='smooth50',domain='natural',metrics=dict(decoded=score_sequence(labels_from_batch(b),ref,b['period'],25.,semantics='independent_complete'))))
        raw=dict(np.load(OLD/'prepared'/m/'raw.npz',allow_pickle=False));v3=joblib.load('local_artifacts/v3/1874553/baseline.joblib')
        legacy=legacy_v3_input(b,raw);a,h=v3.probabilities(legacy)
        rows.append(dict(match=m,variant='V3',domain='natural',metrics=dict(decoded=extended(b,a,h,v3.predict(legacy)))))
        del raw
    result=dict(experiment_id='E01_E02_E04_core',**identity,models=models,rows=rows,cost=dict(wall_s=time.monotonic()-started,cpu_s=time.process_time()-cpu,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),fits=18,competition_promoted=False)
    write_new(out/'results.json',result);write_new('docs/R7_CORE_RESULTS.json',result)
    record({k:v for k,v in result.items()if k!='rows'}|dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),decision='diagnostic; stress comparison pending',next='E03 independent label-blind masks and E05 complete attribution'))

def prepare_stress():
    verify_inputs();started=time.monotonic()
    for m in IDS:
        out=ROOT/'prepared'/m
        if out.exists():raise FileExistsError('preserve stress inputs')
        out.mkdir(parents=True);b=load(m,'natural')
        with np.load(OLD/'prepared'/m/'raw.npz',allow_pickle=False)as z:raw={k:z[k]for k in RAW_FIELDS if k in z.files}
        contract=FeatureContract(**json.loads(str(b['feature_contract_json'])))
        manifest=dict(match=m,source_raw_sha256=digest(OLD/'prepared'/m/'raw.npz'),code=source_identity(),domains={})
        for scheme,seed in [('short',7101),('long',7102)]:
            mask,windows=blind_intervals(raw['period'],25.,scheme=scheme,seed=seed)
            batch=provider_neutral_features_v3(raw,contract,ball_dropout_mask=mask)
            for k in ['y_alive','y_home','match_id','label_semantics']:batch[k]=b[k]
            path=out/f'{scheme}.npz';np.savez_compressed(path,**batch)
            np.savez_compressed(out/f'{scheme}-mask.npz',mask=mask,windows=np.asarray(windows))
            manifest['domains'][scheme]=dict(seed=seed,windows=len(windows),masked_frames=int(mask.sum()),fraction=float(mask.mean()),mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),prepared_sha256=digest(path))
            print('raw stress rebuilt',m,scheme,flush=True)
        write_new(out/'manifest.json',manifest)
    record(dict(experiment_id='E03_prepare',**source_identity(),wall_s=time.monotonic()-started,decision='KEEP label-blind fixed processes',next='score all frozen models without further fitting'))

def stress():
    verify_inputs();started=time.monotonic();out=ROOT/'stress-v1'
    if out.exists():raise FileExistsError('preserve stress run')
    out.mkdir(parents=True);rows=[]
    for m in IDS:
        cache={d:load(m,d)for d in ['short','long']}
        for v in ['M1','M2','M3','M4','M5','M6']:
            path=model_path(m,v)
            model=joblib.load(path)
            for domain,b in cache.items():
                score,arrays=eval_model(model,b,v);np.savez_compressed(out/f'{m}-{v}-{domain}.npz',**arrays)
                rows.append(dict(match=m,variant=v,domain=domain,metrics=score))
            print('stress scored',m,v,flush=True)
    result=dict(experiment_id='E03_E05_stress',**source_identity(),rows=rows,input_manifests={m:json.loads((ROOT/'prepared'/m/'manifest.json').read_text())for m in IDS},wall_s=time.monotonic()-started,competition_promoted=False)
    write_new(out/'results.json',result);write_new('docs/R7_STRESS_RESULTS.json',result)
    record({k:v for k,v in result.items()if k not in {'rows','input_manifests'}}|dict(decision='evaluate frozen retention gate next'))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['core','prepare-stress','stress']);a=p.parse_args()
    {'core':core,'prepare-stress':prepare_stress,'stress':stress}[a.stage]()
