"""One frozen new-task Logistic experiment on public weak event anchors.

No old HGB refits. Held-out match is previously team-consumed, not fresh blind.
Full spell is undefined here; no absence of event is treated as a negative/D.
"""
import argparse,csv,datetime,json,resource,subprocess,sys,time,warnings
from pathlib import Path
import joblib,numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,roc_auc_score,confusion_matrix,balanced_accuracy_score
from sklearn.exceptions import ConvergenceWarning
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from r11.acquisition_events import adapt_skillcorner_events,anchor_labels
from r11.event_learning import temporal_inputs,supervised_rows,OBSERVATION_FIELDS
from solution.idsse import digest
from scripts.research_r7 import write_new
from scripts.evaluate_r8_decoder import save_prediction
MATCHES=('1874553','1886347','1899585','1927964');HELD='1874553';REV='4340d274572876239c154c90bc507a9b3250a656'
ROOT=Path('local_artifacts/r11/event-logistic-v1');PROTOCOL=Path('docs/R11_EVENT_PROTOCOL.json')
CODE=['r11/acquisition_events.py','r11/event_learning.py','scripts/train_r11_events.py']

def source_paths(m):
    root=Path('local_artifacts/external/skillcorner')/m
    return Path(f'local_artifacts/v2/{m}-observed.npz'),root/f'{m}_dynamic_events.csv',root/f'{m}_match.json'

def load(m):
    bp,ep,mp=source_paths(m)
    with np.load(bp,allow_pickle=False) as z:
        assert str(z['observation_view'])=='observed';b={k:z[k] for k in OBSERVATION_FIELDS}
    assert float(b['fps'])==10
    with ep.open() as f:events,audit=adapt_skillcorner_events(csv.DictReader(f),json.loads(mp.read_text()),dict(source_kind='public_other_match',revision=REV,sha256=digest(ep),match_id=m))
    labels=anchor_labels(events,b['frame'],b['period'])
    return b,labels,audit

def freeze():
    inputs={str(p):digest(p) for m in MATCHES for p in source_paths(m)}
    for p in ['local_artifacts/r9-v2/e09a2/SC1874553-M3-M2.npz','local_artifacts/r9-v2/e09a2/SC1874553-M3-C1.npz']:
        inputs[p]=digest(p)
    write_new(PROTOCOL,dict(id='R11-H2-EVENT-LOGISTIC-V1',frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
      hypothesis='New external weak player_possession event supervision can distinguish acquisition from explicit same-team reception and add team evidence at old model errors.',
      ground_truth_provenance='SkillCorner dynamic events combines tracking with Wyscout event alignment; weak provider anchor labels, not independent complete official HAD',
      train=[m for m in MATCHES if m!=HELD],heldout=HELD,exposure='All four previously consumed by team V3; new target-task out-of-match fit only, not fresh blind or independent transfer',
      features='Only observed X44 and timeline; context [-1,0,+1]sec vs matched single-frame ablation. No actor/team/start_type/first/end_type/outcome/pressure/EPV/phase/y_* metadata in X.',
      supervision='acquisition positive: matched nonfullyextrapolated interception/recovery+first+previousopponent; negative: explicit matched successful same-team targeted reception chain; other unknown. Team labels at eligible anchors only.',
      stage1='NOT_IDENTIFIABLE_NO_CONTACT_NEGATIVES; conditional on supplied provider event times; not an autonomous acquisition detector',
      training=dict(estimator='SimpleImputer(median,add_indicator=True,keep_empty_features=True)+StandardScaler+LogisticRegression',C=1.,class_weight='balanced',max_iter=1000,solver='lbfgs',seed=111601,targets=['acquisition','team_away'],views={'single':[0.],'context':[-1.,0.,1.]},fits=4),
      selection='Fixed threshold.5 and fixed top20% per-half anchor budget (minimum1). No hyperparameter search. Train imputation/scaling only on three other matches.',
      controls='single-frame same labels/model; frozen M3/HC at exact same provider anchors; shuffled predicted acquisition probabilities and team probabilities per half; +30s roll within each half before anchor scoring. One seed only, not significance.',
      gate='Research only if context AP and balanced accuracy exceed single frame; directional correction must exceed deterioration at old errors. Regardless of result, no complete HAD or natural-missing gate => SCIENTIFICALLY_INCONCLUSIVE, no promotion.',
      risk='Selected provider anchors mostly ball-visible; event quality is not detection quality; exclude frames with no observed players for either team. Do not imply accuracy on SC195zero-observation frames.',
      full_spell_f1=None,new_weights_expected=4,inputs=inputs,code_hashes={p:digest(p) for p in CODE},rollback='Untouched frozen M3/HC_M3; learned head optional research only'))

def binary_metrics(y,p):
    pred=p>=.5
    return dict(n=len(y),positive=int(y.sum()),average_precision=float(average_precision_score(y,p)),roc_auc=float(roc_auc_score(y,p)),balanced_accuracy=float(balanced_accuracy_score(y,pred)),confusion=confusion_matrix(y,pred,labels=[0,1]).tolist())

def evaluate(b,labels,pa,pt,base):
    am=supervised_rows(labels['acquisition'],b);tm=supervised_rows(labels['team'],b);a=binary_metrics(labels['acquisition'][am],pa[am]);t=binary_metrics(labels['team'][tm],pt[tm]);chosen=[]
    for half in (1,2):
        ix=np.flatnonzero(am&(b['period']==half));k=max(1,int(np.ceil(.2*len(ix))));chosen.extend(ix[np.argsort(-pa[ix],kind='stable')[:k]])
    chosen=np.asarray(chosen,int);a['fixed_budget']=dict(n=len(chosen),true_acquisitions=int(labels['acquisition'][chosen].sum()),correct_team_acquisitions=int(np.sum((labels['acquisition'][chosen]==1)&((pt[chosen]>=.5)==labels['team'][chosen]))))
    truth=np.where(labels['team'][tm]==0,'H','A');pred=np.where(pt[tm]>=.5,'A','H');old={}
    for name,p in base.items():
        ref=p[tm];old[name]=dict(n=len(truth),correct=int((ref==truth).sum()),false_D=int((ref=='D').sum()),old_errors=int((ref!=truth).sum()),corrected=int(((pred==truth)&(ref!=truth)).sum()),deteriorated=int(((pred!=truth)&(ref==truth)).sum()),candidate_correct=int((pred==truth).sum()),scope='conditional H/A evidence only; not written over frozen D labels')
    return dict(acquisition=a,team=t,against_frozen=old,full_spell_f1=None,labels='partial_weak_provider_event_anchors')

def run():
    protocol=json.loads(PROTOCOL.read_text());ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'results.json').exists():raise FileExistsError('preserve completed fit; use receipts/replay, no repeat')
    for p,h in {**protocol['inputs'],**protocol['code_hashes']}.items():
        if digest(p)!=h:raise ValueError('drift '+p)
    data={m:load(m) for m in MATCHES};b,l,au=data[HELD]
    base={v:np.load(f'local_artifacts/r9-v2/e09a2/SC1874553-M3-{h}.npz')['decoded'] for v,h in [('M3','M2'),('HC_M3','C1')]}
    assert all(len(z)==len(b['frame']) for z in base.values());rows=[];models=[];counts={};start=time.monotonic();cpu=time.process_time()
    for view,offset in protocol['training']['views'].items():
        xs={m:temporal_inputs(z[0],offset) for m,z in data.items()};prob={}
        for target,label in [('acquisition','acquisition'),('team_away','team')]:
            path=ROOT/f'{view}-{target}.joblib'
            if path.exists():raise FileExistsError('partial model preserved; explicit recovery needed')
            trainx=[];trainy=[];nper={}
            for m in protocol['train']:
                bb,ll,_=data[m];keep=supervised_rows(ll[label],bb);trainx.append(xs[m][keep]);trainy.append(ll[label][keep]);nper[m]=dict(n=int(keep.sum()),positive=int(ll[label][keep].sum()))
            x=np.vstack(trainx);y=np.concatenate(trainy)
            model=make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),StandardScaler(),LogisticRegression(C=1.,class_weight='balanced',max_iter=1000,solver='lbfgs',random_state=111601))
            with warnings.catch_warnings():
                warnings.simplefilter('error',ConvergenceWarning);model.fit(x,y)
            joblib.dump(model,path);prob[target]=model.predict_proba(xs[HELD])[:,1]
            models.append(dict(path=str(path),sha256=digest(path),train_counts=nper,view=view,target=target,iterations=model[-1].n_iter_.tolist()))
            print('FIT',view,target,'samples',len(y),'features',x.shape[1],flush=True)
        save_prediction(ROOT/f'{view}-probabilities.npz',p_acquisition=prob['acquisition'],p_away=prob['team_away'])
        rows.append(dict(view=view,metrics=evaluate(b,l,prob['acquisition'],prob['team_away'],base),prediction_sha=digest(ROOT/f'{view}-probabilities.npz')))
        if view=='context':
            rng=np.random.default_rng(111601);sh=[z.copy() for z in prob.values()];shift=[z.copy() for z in prob.values()]
            for half in (1,2):
                ix=np.flatnonzero(b['period']==half);perm=rng.permutation(len(ix))
                for j,z in enumerate(prob.values()):sh[j][ix]=z[ix[perm]];shift[j][ix]=np.roll(z[ix],300)
            for name,z in [('shuffled_time',sh),('shifted_time',shift),('shuffled_team',[prob['acquisition'],sh[1]])]:
                save_prediction(ROOT/f'{name}-probabilities.npz',p_acquisition=z[0],p_away=z[1]);rows.append(dict(view=name,metrics=evaluate(b,l,z[0],z[1],base),prediction_sha=digest(ROOT/f'{name}-probabilities.npz')))
    for m,(bb,ll,audit) in data.items():
        am=supervised_rows(ll['acquisition'],bb);tm=supervised_rows(ll['team'],bb);visible=bb['X'][:,bb['feature_names'].tolist().index('ball_visible')]==1
        counts[m]=dict(source_audit=audit,acquisition_anchors=int(am.sum()),team_anchors=int(tm.sum()),team_anchor_missing_ball=int((tm&~visible).sum()))
    result=dict(id=protocol['id'],models=models,rows=rows,counts=counts,code_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),protocol_sha=digest(PROTOCOL),code_hashes=protocol['code_hashes'],fits=4,full_spell_f1=None,decision='SCIENTIFICALLY_INCONCLUSIVE_PARTIAL_EVENT_SUPERVISION_ONLY',competition_promoted=False,
      wall_s=time.monotonic()-start,cpu_s=time.process_time()-cpu,max_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write_new(ROOT/'results.json',result);write_new('docs/R11_EVENT_RESULTS.json',result)
    for r in rows:print(r['view'],json.dumps(r['metrics']),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run']);a=p.parse_args();freeze() if a.action=='freeze' else run()
