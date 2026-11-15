"""E08B: label-blind raw reconstruction first, then frozen-head full evaluation."""
import argparse,json,resource,subprocess,sys,time
from dataclasses import replace
from pathlib import Path
import joblib,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from solution.idsse import digest
from solution.short_gap_ball import reconstruct
from solution.provider_neutral import RAW_FIELDS,FeatureContract,provider_neutral_features_v3,assert_compatible
from solution.candidate import Candidate
from scripts.research_r7 import IDS,load,model_path,write_new
from scripts.evaluate_r8_decoder import prediction_path,original,metrics,save_prediction,append_ledger

ROOT=Path('local_artifacts/r8/e08b');PROTOCOL=Path('docs/R8_E08B_PROTOCOL.json')

def identity():
    return dict(code_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),protocol_sha=digest(PROTOCOL),
        code_hashes={p:digest(p)for p in ['solution/short_gap_ball.py','scripts/evaluate_r8_interpolation.py','solution/provider_neutral.py','solution/features.py','solution/decoder.py']})

def raw_input(m):
    with np.load(f'local_artifacts/r6/prepared/{m}/raw.npz',allow_pickle=False)as z:
        return {k:z[k]for k in RAW_FIELDS if k in z.files}

def verify():
    p=json.loads(PROTOCOL.read_text())
    for path,sha in p['inputs'].items():
        if digest(path)!=sha:raise ValueError('input drift '+path)
    ROOT.mkdir(parents=True,exist_ok=True)
    return p

def diagnose():
    verify()
    for m in IDS:
        raw=raw_input(m)
        for d in ['short','dropout','long']:
            out=ROOT/f'{m}-{d}-reconstruction.json'
            if out.exists():
                receipt=json.loads(out.read_text())
                if digest(out.with_suffix('.npz'))!=receipt['sha256']:raise ValueError('reconstruction drift')
                continue
            started=time.monotonic();b=load(m,d)
            missing=b['X'][:,b['feature_names'].tolist().index('ball_visible')]!=1
            masked={**raw,'ball_xy':raw['ball_xy'].copy()};masked['ball_xy'][missing]=np.nan
            ball,masks=reconstruct(masked,fps=float(b['fps']))
            used=masks['imputed'];error=np.linalg.norm(ball[used]-raw['ball_xy'][used],axis=1)
            # Diagnostic geometric control disagreement uses hidden coordinates only AFTER proposals.
            dist=np.linalg.norm(raw['players_xy'][used]-raw['ball_xy'][used,None,:],axis=2)
            teams=np.broadcast_to(raw['player_team'],raw['players_xy'].shape[:2])[used]
            dist[~np.isfinite(dist)|~np.isin(teams,[0,1])]=np.inf
            valid=np.isfinite(dist).any(1);near=dist.argmin(1) if len(dist)else np.zeros(0,dtype=int)
            actual=teams[np.arange(len(dist)),near]
            wrong=int(np.sum(actual[valid]!=masks['anchor_team'][used][valid]))
            save_prediction(out.with_suffix('.npz'),ball_xy=ball,imputed=used,anchor_team=masks['anchor_team'])
            row=dict(match=m,domain=d,counts=masks['counts'],missing_frames=int(missing.sum()),imputed_frames=int(used.sum()),
                error_m=dict(mean=float(error.mean())if len(error)else None,median=float(np.median(error))if len(error)else None,
                    p95=float(np.quantile(error,.95))if len(error)else None,max=float(error.max())if len(error)else None),
                wrong_geometric_control=wrong,control_diagnostic_denominator=int(valid.sum()),
                caveat='Geometric nearest-team disagreement, not independently labeled effective-control error',
                sha256=digest(out.with_suffix('.npz')),wall_seconds=time.monotonic()-started,**identity())
            write_new(out,row);print('RECONSTRUCT',m,d,row['counts'],'frames',row['imputed_frames'],'error',row['error_m'],flush=True)

def evaluate():
    protocol=verify();rows=[]
    for m in IDS:
        raw=raw_input(m);model=joblib.load(model_path(m,'M2'))
        for d in ['natural','dropout','short','long']:
            out=ROOT/f'{m}-{d}-score.json'
            if out.exists():
                row=json.loads(out.read_text());assert digest(out.with_suffix('.npz'))==row['prediction_sha256'];rows.append(row);continue
            start=time.monotonic();cpu=time.process_time();b=load(m,d);old=dict(np.load(prediction_path(m,'M2',d)))
            reconstruction=None;waiver=None
            if d!='natural':
                rp=ROOT/f'{m}-{d}-reconstruction.json';reconstruction=json.loads(rp.read_text())
                if digest(rp.with_suffix('.npz'))!=reconstruction['sha256']:raise ValueError('reconstruction drift')
                z=dict(np.load(rp.with_suffix('.npz')))
            if d=='natural' or not reconstruction['imputed_frames']:
                a,h,pred=old['p_alive'],old['p_home'],old['decoded'];feature_hash=None
            else:
                contract=replace(FeatureContract(**json.loads(str(b['feature_contract_json']))),observation_view='r8_short_gap_proposals')
                rebuilt=provider_neutral_features_v3({**raw,'ball_xy':z['ball_xy']},contract)
                for k in ['y_alive','y_home','match_id','label_semantics']:rebuilt[k]=b[k]
                sources=[FeatureContract(**c)for c in model.metadata['training_contracts']]
                waiver=assert_compatible(sources,contract,purpose='evaluate',allow_cross_view=True,allow_cross_pitch=True)
                # Explicit research-only view waiver above; production model API still rejects this view.
                a,h=Candidate.probabilities(model,rebuilt);pred=original(a,h,rebuilt,'M2')
                import hashlib
                feature_hash=hashlib.sha256(rebuilt['X'].tobytes()).hexdigest()
            save_prediction(out.with_suffix('.npz'),p_alive=a,p_home=h,decoded=pred)
            score=metrics(b,a,h,pred) # Strata use original observations, never imputed visibility.
            row=dict(experiment_id='E08B',match=m,domain=d,variant='B1',metrics=score,raw_metrics=metrics(b,a,h,np.array(list('HAD'))[np.column_stack([a*h,a*(1-h),1-a]).argmax(1)]),
                reconstruction=reconstruction,compatibility_waiver=waiver,recomputed_44_sha256=feature_hash,
                model_sha256=digest(model_path(m,'M2')),prediction_sha256=digest(out.with_suffix('.npz')),
                wall_seconds=time.monotonic()-start,cpu_memory=dict(cpu_seconds=time.process_time()-cpu,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),**identity())
            write_new(out,row);rows.append(row)
            append_ledger(dict(experiment_id=f'E08B/{m}/{d}',hypothesis=protocol['hypothesis'],source_match=m,train_matches=[x for x in IDS if x!=m],eval_matches=[m],
                model_sha256=row['model_sha256'],prediction_sha256=row['prediction_sha256'],code_sha=row['code_sha'],protocol_sha=row['protocol_sha'],
                wall_seconds=row['wall_seconds'],cpu_memory=row['cpu_memory'],metrics_by_match={m:score},decision='PENDING_PAIRED_GATE',revert_path='unchanged M2 raw observed view',next='complete B1 paired evaluation'))
            print('SCORED B1',m,d,score['full_spell_f1'],score['dead_errors'],flush=True)
    controls=json.loads(Path('local_artifacts/r8/e08a/results.json').read_text())['rows'];lookup={(r['match'],r['domain'],r['variant']):r['metrics']['decoded']for r in controls}
    checks=[]
    for d in ['natural','dropout','short','long']:
        for control in ['M2','M3','G1']:
            per=[]
            for r in rows:
                if r['domain']!=d:continue
                ref=lookup[r['match'],d,control];score=r['metrics'];dead=score['dead_errors'];rd=ref['dead_errors']
                per.append(dict(match=r['match'],spell_delta=score['full_spell_f1']-ref['full_spell_f1'],
                    dead_miss_rate_delta=(dead['missed_dead_frames']-rd['missed_dead_frames'])/rd['gt_dead_frames'],missed_segments_delta=dead['missed_segments']-rd['missed_segments']))
            safe=all(p['dead_miss_rate_delta']<=.02 and p['missed_segments_delta']<=0 for p in per)
            benefit=all(p['spell_delta']>=-.01 for p in per)if d=='natural'else sum(p['spell_delta']>0 for p in per)>=2 and np.mean([p['spell_delta']for p in per])>=.02
            checks.append(dict(domain=d,control=control,per_match=per,pass_gate=bool(safe and benefit)))
    # Long gaps are intentionally unsupported; required benefit is short and original dropout.
    passed=all(c['pass_gate']for c in checks if c['domain']in('natural','short','dropout'))
    result=dict(rows=rows,checks=checks,decision='RESEARCH_GATE_R_ONLY'if passed else'REVERT_AS_REPLACEMENT',competition_promoted=False,**identity())
    write_new(ROOT/'results.json',result);write_new('docs/R8_E08B_RESULTS.json',result)
    append_ledger(dict(experiment_id='E08B_decision',hypothesis=protocol['hypothesis'],source_match='public DFL',train_matches=list(IDS),eval_matches=list(IDS),
        model_sha256=protocol['models'],prediction_sha256={f"{r['match']}/{r['domain']}":r['prediction_sha256']for r in rows},
        code_sha=result['code_sha'],protocol_sha=result['protocol_sha'],wall_seconds=sum(r['wall_seconds']for r in rows),cpu_memory='per-row receipts',metrics_by_match=checks,
        decision=result['decision'],revert_path='frozen M2 without interpolation',next='measure training-only boundary sample coverage before deciding E08C fits'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['diagnose','evaluate']);a=p.parse_args();globals()[a.stage]()
