"""One preregistered equal-budget boundary-sampled HGB pair per DFL LOMO fold."""
import argparse,json,hashlib,resource,subprocess,sys,time
from pathlib import Path
import joblib,numpy as np,pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.research_r7 import IDS,PARAMS,load,model_path,write_new
from scripts.evaluate_r8_decoder import original,metrics,save_prediction,append_ledger
from scripts.diagnose_r8_sampling import coverage
from solution.boundary_sampling import select
from solution.idsse_model import IDSSECandidate
from solution.idsse import digest
from solution.evidence import labels_from_batch,score_sequence,visibility_strata
from solution.provider_neutral import FeatureContract,assert_compatible
from solution.candidate import Candidate

ROOT=Path('local_artifacts/r8/e08c');PROTOCOL=Path('docs/R8_E08C_PROTOCOL.json')
def save_or_verify(path,**arrays):
    """Recover an interrupted receipt write only after exact array agreement."""
    if path.exists():
        with np.load(path,allow_pickle=False)as saved:
            if set(saved.files)!=set(arrays):raise ValueError('unreceipted output fields differ')
            for key,value in arrays.items():np.testing.assert_array_equal(saved[key],value)
        print('RECOVERED exact existing prediction; file retained',path,flush=True)
    else:save_prediction(path,**arrays)
def identity():
    return dict(code_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),protocol_sha=digest(PROTOCOL),
        code_hashes={p:digest(p)for p in ['solution/boundary_sampling.py','scripts/train_r8_boundaries.py','solution/decoder.py']})
def verify():
    protocol=json.loads(PROTOCOL.read_text())
    for path,sha in protocol['inputs'].items():
        if digest(path)!=sha:raise ValueError('input drift '+path)
    ROOT.mkdir(parents=True,exist_ok=True);return protocol
def train():
    verify()
    for heldout in IDS:
        out=ROOT/f'{heldout}.joblib';receipt=out.with_suffix('.json')
        if receipt.exists():
            assert digest(out)==json.loads(receipt.read_text())['model_sha256'];continue
        if out.exists():raise FileExistsError('unreceipted model; preserve and inspect')
        start=time.monotonic();cpu=time.process_time();models=[];selection={};training=[m for m in IDS if m!=heldout]
        for target in ['y_alive','y_home']:
            xs=[];ys=[];stats=[]
            for m in training:
                for d in ['natural','dropout']:
                    b=load(m,d);eligible=(b[target]>=0)&(b['period']>0)
                    if target=='y_home':eligible &= b['y_alive']==1
                    # Seed fixed by source and domain, not fold scores or heldout labels.
                    seed=8203+IDS.index(m)*10+int(d=='dropout')
                    chosen,uniform=select(b['frame'],b['period'],labels_from_batch(b),eligible,fps=float(b['fps']),seed=seed)
                    xs.append(b['X'][chosen]);ys.append(b[target][chosen]);key=f'{target}-{m}-{d}'
                    selection[key]=np.flatnonzero(chosen)
                    stats.append(dict(match=m,domain=d,n=int(chosen.sum()),baseline_n=int(uniform.sum()),selection_sha256=hashlib.sha256(selection[key].tobytes()).hexdigest(),coverage=coverage(b,chosen)))
            x,y=np.vstack(xs),np.concatenate(ys);names=b['feature_names'].tolist()
            estimator=HistGradientBoostingClassifier(**PARAMS).fit(pd.DataFrame(x,columns=names),y)
            models.append((estimator,dict(target=target,n=len(y),class_counts=np.bincount(y).tolist(),selection=stats)))
        frozen=joblib.load(model_path(heldout,'M2'))
        model=IDSSECandidate(models[0][0],models[1][0],names,{**frozen.metadata,'model_version':'R8-C1-boundary-budget','train':training,'heldout':heldout,'competition_promoted':False})
        with out.open('xb')as f:joblib.dump(model,f)
        save_prediction(ROOT/f'{heldout}-selection.npz',**selection)
        row=dict(experiment_id='E08C_fit',heldout=heldout,train_matches=training,eval_matches=[],fits=2,params=PARAMS,heads=[x[1]for x in models],
            model_sha256=digest(out),path=str(out),bytes=out.stat().st_size,selection_sha256=digest(ROOT/f'{heldout}-selection.npz'),
            wall_seconds=time.monotonic()-start,cpu_memory=dict(cpu_seconds=time.process_time()-cpu,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),**identity())
        write_new(receipt,row)
        append_ledger(dict(experiment_id=f'E08C_fit/{heldout}',hypothesis='Equal-budget boundary sampling may recover omitted short possession transitions',source_match='public DFL training folds',
            train_matches=training,eval_matches=[],model_sha256=row['model_sha256'],prediction_sha256=None,code_sha=row['code_sha'],protocol_sha=row['protocol_sha'],wall_seconds=row['wall_seconds'],cpu_memory=row['cpu_memory'],
            metrics_by_match=row['heads'],decision='FITTED_NOT_EVALUATED',revert_path=str(model_path(heldout,'M2')),next='frozen full25Hz paired evaluation'))
        print('TRAINED C1',heldout,'two HGB fits',row['wall_seconds'],'s',row['model_sha256'],flush=True)
def evaluate():
    verify();rows=[]
    for m in IDS:
        model=joblib.load(ROOT/f'{m}.joblib')
        assert digest(ROOT/f'{m}.joblib')==json.loads((ROOT/f'{m}.json').read_text())['model_sha256']
        for d in ['natural','dropout','short','long']:
            out=ROOT/f'{m}-{d}.json'
            if out.exists():rows.append(json.loads(out.read_text()));continue
            start=time.monotonic();b=load(m,d);a,h=model.probabilities(b);pred=original(a,h,b,'M2')
            score=metrics(b,a,h,pred);raw=np.array(list('HAD'))[np.column_stack([a*h,a*(1-h),1-a]).argmax(1)]
            save_or_verify(out.with_suffix('.npz'),p_alive=a,p_home=h,decoded=pred)
            row=dict(experiment_id='E08C',match=m,domain=d,variant='C1',metrics=score,argmax_metrics=metrics(b,a,h,raw),prediction_sha256=digest(out.with_suffix('.npz')),
                model_sha256=digest(ROOT/f'{m}.joblib'),wall_seconds=time.monotonic()-start,**identity())
            write_new(out,row);rows.append(row);print('C1',m,d,score['full_spell_f1'],score['dead_errors'],flush=True)
    controls=json.loads(Path('local_artifacts/r8/e08a/results.json').read_text())['rows'];checks=[]
    for d in ['natural','dropout','short','long']:
        for control in ['M2','M3','G1']:
            per=[]
            for r in rows:
                if r['domain']!=d:continue
                ref=next(x['metrics']['decoded']for x in controls if x['match']==r['match']and x['domain']==d and x['variant']==control)
                sc=r['metrics'];rd=ref['dead_errors'];sd=sc['dead_errors']
                per.append(dict(match=r['match'],spell_delta=sc['full_spell_f1']-ref['full_spell_f1'],dead_miss_rate_delta=(sd['missed_dead_frames']-rd['missed_dead_frames'])/rd['gt_dead_frames'],missed_segments_delta=sd['missed_segments']-rd['missed_segments']))
            safe=all(p['dead_miss_rate_delta']<=.02 and p['missed_segments_delta']<=0 for p in per)
            benefit=all(p['spell_delta']>=-.01 for p in per)if d=='natural'else sum(p['spell_delta']>0 for p in per)>=2 and np.mean([p['spell_delta']for p in per])>=.02
            checks.append(dict(domain=d,control=control,per_match=per,pass_gate=bool(safe and benefit)))
    transferred=[];model=joblib.load(ROOT/'J03WPY.joblib')
    for source,path in [('SC1874553','local_artifacts/r7/transfer-v2/prepared.npz'),('MetricaGame3','local_artifacts/r7b/metrica-v1/base.npz')]:
        b=dict(np.load(path));sources=[FeatureContract(**c)for c in model.metadata['training_contracts']]
        waiver=assert_compatible(sources,FeatureContract(**json.loads(str(b['feature_contract_json']))),purpose='evaluate',allow_cross_provider=True,allow_cross_view=True,allow_cross_fps=True,allow_cross_pitch=True)
        a,h=Candidate.probabilities(model,b);pred=original(a,h,b,'M2');truth=labels_from_batch(b)
        save_or_verify(ROOT/f'{source}-C1.npz',p_alive=a,p_home=h,decoded=pred)
        strata={}
        for name,mask in visibility_strata(b).items():
            keep=mask&np.isin(truth,['H','A']);n=int(keep.sum());strata[name]=dict(known_live=n,false_dead=int(np.sum(pred[keep]=='D')),ha_accuracy=float(np.mean(pred[keep]==truth[keep]))if n else None)
        transferred.append(dict(source=source,metrics=score_sequence(truth,pred,b['period'],float(b['fps']),semantics='partial_source_proxy'),strata=strata,compatibility_waiver=waiver,
            full_spell_f1=None,false_dead=int(np.sum((pred=='D')&np.isin(truth,['H','A']))),missed_weak_dead=int(np.sum((truth=='D')&(pred!='D'))),prediction_sha256=digest(ROOT/f'{source}-C1.npz')))
    result=dict(rows=rows,transfer=transferred,checks=checks,decision='RESEARCH_GATE_R_ONLY'if all(c['pass_gate']for c in checks)else'REVERT_AS_REPLACEMENT',competition_promoted=False,**identity())
    write_new(ROOT/'results.json',result);write_new('docs/R8_E08C_RESULTS.json',result)
    append_ledger(dict(experiment_id='E08C_decision',hypothesis='Equal-budget training-only boundary sampling',source_match='public DFL plus partial cross-provider',train_matches=list(IDS),eval_matches=list(IDS),
        model_sha256={m:digest(ROOT/f'{m}.joblib')for m in IDS},prediction_sha256={f"{r['match']}/{r['domain']}":r['prediction_sha256']for r in rows},code_sha=result['code_sha'],protocol_sha=result['protocol_sha'],
        wall_seconds=sum(r['wall_seconds']for r in rows),cpu_memory=dict(peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),metrics_by_match=checks,
        decision=result['decision'],revert_path='frozen M2/M3; no production default changes',next='freeze evidence, publish Draft, prioritize human offline P0; no duration model without further evidence'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['train','evaluate']);a=p.parse_args();globals()[a.stage]()
