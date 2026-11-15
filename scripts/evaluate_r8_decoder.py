"""Exclusive, resumable E08A evaluation of frozen public-other-match caches."""
import json, resource, subprocess, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from solution.hierarchical_decoder import decode, anchored
from solution.decoder import emission_probabilities, viterbi_decode
from solution.idsse import digest
from solution.quality import period_blocks
from solution.r7_metrics import extended
from solution.evidence import labels_from_batch, score_sequence, visibility_strata
from evaluation import possession_metrics as pm
from scripts.research_r7 import IDS, load, write_new, js, model_path, checked_path

ROOT=Path('local_artifacts/r8/e08a')
PROTOCOL=Path('docs/R8_PROTOCOL.json')
DOMAINS=('natural','dropout','short','long')

def prediction_path(m,v,d):
    if v=='G1':return Path('local_artifacts/r7b/p1-eval-v1')/f'{m}-G1-{d}.npz'
    return Path('local_artifacts/r7')/('core-v1' if d in ('natural','dropout') else 'stress-v1')/f'{m}-{v}-{d}.npz'

def identity():
    return dict(code_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        protocol_sha=digest(PROTOCOL), code_hashes={p:digest(p) for p in ['solution/hierarchical_decoder.py','scripts/evaluate_r8_decoder.py','solution/r7_metrics.py','evaluation/possession_metrics.py']})

def checked(path,protocol):
    if digest(path)!=protocol['inputs'][str(path)]:raise ValueError('frozen input changed: '+str(path))
    return dict(np.load(path,allow_pickle=False))

def original(a,h,b,variant):
    visible=b['X'][:,b['feature_names'].tolist().index('ball_visible')]==1
    scale=np.where(visible & np.r_[True,visible[:-1]],1.,4.) if variant=='M3' else None
    return viterbi_decode(emission_probabilities(a,h),period_blocks(b['period']),3*float(b['fps'])/10,float(b['fps'])/10,dead_scale=scale)

def metrics(b,a,h,p):
    result=extended(b,a,h,p)
    truth=pm.label_codes(labels_from_batch(b));pred=pm.label_codes(p);fps=float(b['fps'])
    blocks=[(lo,hi+1)for lo,hi in period_blocks(b['period'])]
    result['boundaries_pair_0p48s']=pm.match_transitions(
        pm.transitions(truth,blocks),pm.transitions(pred,blocks),round(.48*fps),typed='pair')
    # Retain aggregate counts only, not per-frame transitions in the public report.
    pairs,missed,spurious=result['boundaries_pair_0p48s']
    result['boundaries_pair_0p48s']=dict(n_matched=len(pairs),n_missed=len(missed),n_spurious=len(spurious))
    return result

def save_prediction(path,**arrays):
    with path.open('xb')as f:np.savez_compressed(f,**arrays)

def append_ledger(row):
    path=Path('docs/R8_EXPERIMENT_LEDGER.jsonl')
    with path.open('a')as f:f.write(json.dumps(row,default=lambda x:x.item() if isinstance(x,np.generic)else x)+'\n')

def run_row(m,d,v,b,cache,protocol):
    path=ROOT/f'{m}-{v}-{d}.json'
    ident=identity()
    if path.exists():
        row=json.loads(path.read_text())
        if row['protocol_sha']!=ident['protocol_sha'] or row['code_hashes']!=ident['code_hashes']:
            raise ValueError('saved run identity drift; retain output and use an explicit new run')
        if digest(path.with_suffix('.npz'))!=row['prediction_sha256']:raise ValueError('saved prediction drift')
        return row
    start=time.monotonic();cpu=time.process_time()
    old=cache[v] if v in cache else cache['G1']
    a,h=old['p_alive'],old['p_home'];diagnostics={}
    if v in cache:
        pred=original(a,h,b,v)
        np.testing.assert_array_equal(pred,old['decoded'])
        diagnostics['exact_old_decode']=True
    else:
        if v=='H2':
            a,h,masks=anchored(a,h,b)
            diagnostics={k:int(np.sum(x>=0)) if k=='anchor' else int(x.sum()) for k,x in masks.items()}
            unsupported=masks['unsupported'];diagnostics['unsupported_spans']=int(np.sum(unsupported & ~np.r_[False,unsupported[:-1]]))
        pred=decode(a,h,b['frame'],b['period'],float(b['fps']))
    joint=np.array(list('HAD'))[emission_probabilities(a,h).argmax(1)]
    factor=np.where(a>=.5,np.where(h>=.5,'H','A'),'D')
    score=dict(decoded=metrics(b,a,h,pred),joint_argmax=metrics(b,a,h,joint),factorized_argmax=metrics(b,a,h,factor))
    source=prediction_path(m,v if v in cache else 'G1',d)
    model_hashes={str(model_path(m,'M2')):digest(model_path(m,'M2'))}
    if v in ('G1','H1','H2'):
        p=Path('local_artifacts/r7b/p1-v1')/f'{m}.joblib';model_hashes[str(p)]=digest(p)
    save_prediction(path.with_suffix('.npz'),p_alive=a,p_home=h,decoded=pred,joint_argmax=joint,factorized_argmax=factor)
    row=dict(experiment_id='E08A',match=m,variant=v,domain=d,source_match='public IDSSE DFL',train_matches=[x for x in IDS if x!=m],
        eval_matches=[m],model_sha256=model_hashes,source_prediction_sha256=digest(source),prediction_sha256=digest(path.with_suffix('.npz')),
        diagnostics=diagnostics,metrics=score,wall_seconds=time.monotonic()-start,
        cpu_memory=dict(cpu_seconds=time.process_time()-cpu,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),**ident)
    write_new(path,row)
    append_ledger(dict(experiment_id=f'E08A/{m}/{d}/{v}',hypothesis=protocol['candidates'].get(v,'exact H0 replay'),
        source_match='public IDSSE DFL',train_matches=row['train_matches'],eval_matches=[m],model_sha256=model_hashes,
        prediction_sha256=row['prediction_sha256'],code_sha=ident['code_sha'],protocol_sha=ident['protocol_sha'],
        wall_seconds=row['wall_seconds'],cpu_memory=row['cpu_memory'],metrics_by_match={m:score['decoded']},decision='PENDING_PAIRED_GATE',
        revert_path='frozen R6 M1/M2/M3 and R7B G1',next='finish all paired domains; no parameter retuning'))
    print(m,d,v,'spell',score['decoded']['full_spell_f1'],'D',score['decoded']['dead_errors'],flush=True)
    return row

def decisions(rows):
    lookup={(r['match'],r['domain'],r['variant']):r['metrics']['decoded'] for r in rows};out={}
    for v in ('H1','H2'):
        checks=[]
        for d in DOMAINS:
            for control in ('M2','M3','G1'):
                per=[]
                for m in IDS:
                    c,ref=lookup[m,d,v],lookup[m,d,control];cd,rd=c['dead_errors'],ref['dead_errors']
                    per.append(dict(match=m,spell_delta=c['full_spell_f1']-ref['full_spell_f1'],
                        missed_dead_rate_delta=(cd['missed_dead_frames']-rd['missed_dead_frames'])/rd['gt_dead_frames'],
                        missed_segments_delta=cd['missed_segments']-rd['missed_segments'],false_cuts_delta=cd['false_cuts']-rd['false_cuts']))
                passes=(sum(p['spell_delta']>0 for p in per)>=2 and np.mean([p['spell_delta']for p in per])>=.02
                    and all(p['missed_dead_rate_delta']<=.02 and p['missed_segments_delta']<=0 for p in per)
                    and (d!='natural' or all(p['spell_delta']>=-.01 for p in per)))
                checks.append(dict(domain=d,control=control,pass_gate=bool(passes),per_match=per,mean_spell_delta=float(np.mean([p['spell_delta']for p in per]))))
        out[v]=dict(checks=checks,gate_R=all(c['pass_gate']for c in checks if c['control']in('M3','G1')),
            decision='REVERT_AS_REPLACEMENT' if not all(c['pass_gate']for c in checks if c['control']in('M3','G1'))else'RESEARCH_GATE_R_ONLY',competition_promoted=False)
    return out

def transfer(protocol):
    rows=[]
    for source,basepath,predroot in [
        ('SC1874553','local_artifacts/r7/transfer-v2/prepared.npz','local_artifacts/r7b/transfer-v1'),
        ('MetricaGame3','local_artifacts/r7b/metrica-v1/base.npz','local_artifacts/r7b/metrica-v1')]:
        path=ROOT/f'{source}-transfer.json'
        if path.exists():rows.extend(json.loads(path.read_text())['rows']);continue
        b=checked(Path(basepath),protocol);old=checked(Path(predroot)/'G1.npz',protocol);these=[]
        for v in ('G1','H1','H2'):
            a,h=old['p_alive'],old['p_home'];diag={}
            if v=='H2':
                a,h,masks=anchored(a,h,b);diag={k:int((x>=0).sum()) if k=='anchor'else int(x.sum())for k,x in masks.items()}
            pred=original(a,h,b,'G1') if v=='G1'else decode(a,h,b['frame'],b['period'],float(b['fps']))
            truth=labels_from_batch(b);known=truth!='X'
            score=score_sequence(truth,pred,b['period'],float(b['fps']),semantics='partial_source_proxy')
            strata={}
            for name,mask in visibility_strata(b).items():
                keep=mask&np.isin(truth,['H','A']);n=int(keep.sum())
                strata[name]=dict(known_live=n,false_dead=int(np.sum(pred[keep]=='D')),ha_accuracy=float(np.mean(pred[keep]==truth[keep]))if n else None)
            npz=ROOT/f'{source}-{v}.npz';save_prediction(npz,p_alive=a,p_home=h,decoded=pred)
            these.append(dict(source=source,variant=v,full_spell_f1=None,metrics=score,strata=strata,diagnostics=diag,
                false_dead=int(np.sum((pred=='D')&np.isin(truth,['H','A']))),missed_weak_dead=int(np.sum((truth=='D')&(pred!='D'))),
                known_frames=int(known.sum()),prediction_sha256=digest(npz),waiver='Research-only fixed WPY-heldout fold; provider/view/fps differ; no production compatibility waiver',**identity()))
        write_new(path,dict(rows=these));rows.extend(these)
    return rows

def main():
    protocol=json.loads(PROTOCOL.read_text());ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'results.json').exists():raise FileExistsError('E08A is complete; preserve original results')
    # All inputs are allowlisted, frozen before scoring, and checked before use.
    for path,sha in protocol['inputs'].items():
        if digest(path)!=sha:raise ValueError('input drift: '+path)
    rows=[]
    for m in IDS:
        for d in DOMAINS:
            b=load(m,d);cache={v:checked(prediction_path(m,v,d),protocol) for v in ('M2','M3','G1')}
            for v in ('M2','M3','G1','H1','H2'):rows.append(run_row(m,d,v,b,cache,protocol))
    result=dict(rows=rows,decisions=decisions(rows),transfer=transfer(protocol),**identity())
    write_new(ROOT/'results.json',result);write_new('docs/R8_E08A_RESULTS.json',result)
    append_ledger(dict(experiment_id='E08A_decision',hypothesis='Factorization and bounded anchors improve full spells without missing true D',
        source_match='public IDSSE DFL / partial SC / weak Metrica',train_matches=list(IDS),eval_matches=list(IDS),
        model_sha256=protocol['models'],prediction_sha256={f"{r['match']}/{r['domain']}/{r['variant']}":r['prediction_sha256']for r in rows},
        code_sha=result['code_sha'],protocol_sha=result['protocol_sha'],wall_seconds=sum(r['wall_seconds']for r in rows),cpu_memory='per-row receipts',
        metrics_by_match='R8_E08A_RESULTS.json',decision=result['decisions'],revert_path='frozen R6 M1/M2/M3',next='E08B fixed short-gap physical interpolation if gate fails'))

if __name__=='__main__':main()
