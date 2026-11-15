"""Summarize frozen public experiment receipts without fitting or selecting anew."""
import json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from evaluation import possession_metrics as pm
from solution.evidence import labels_from_batch
from solution.quality import period_blocks
from solution.calibration import probability_metrics
from solution.idsse import digest
from scripts.research_r7 import IDS,load,write_new,js
from scripts.research_r7b import ROOT,prepared,identity

def boundaries(b,pred,tolerance):
    gt=pm.label_codes(labels_from_batch(b));pr=pm.label_codes(pred);fps=float(b['fps']);out={}
    for name,select in [('enter_D',lambda t:t[2]==pm.D),('exit_D',lambda t:t[1]==pm.D),('HA',lambda t:t[1]in[pm.H,pm.A]and t[2]in[pm.H,pm.A])]:
        ng=npred=0;offsets=[]
        for a,z in period_blocks(b['period']):
            g=[t for t in pm.transitions(gt,[(a,z+1)])if select(t)];p=[t for t in pm.transitions(pr,[(a,z+1)])if select(t)]
            pairs,_,_=pm.match_transitions(g,p,round(tolerance*fps),typed='pair');ng+=len(g);npred+=len(p);offsets.extend((y[0]-x[0])/fps for x,y in pairs)
        out[name]=dict(n_gt=ng,n_pred=npred,n_matched=len(offsets),precision=len(offsets)/npred if npred else None,recall=len(offsets)/ng if ng else None,
            mean_signed_s=float(np.mean(offsets))if offsets else None,mean_abs_s=float(np.mean(np.abs(offsets)))if offsets else None)
    return out

def main():
    reports={name:json.loads(Path(f'docs/{name}.json').read_text())for name in ['R7B_P1_RESULTS','R7B_CALIBRATION_RESULTS','R7B_COUNTERFACTUAL_RESULTS','R7B_MEMORY_RESULTS']}
    refs=[]
    for name in ['R7_CORE_RESULTS','R7_STRESS_RESULTS']:
        refs.extend(r for r in json.loads(Path(f'docs/{name}.json').read_text())['rows']if r['variant']=='M2')
    def metric(row):return row['metrics']['decoded']
    def find(rows,m,v,d):return next(r for r in rows if r['match']==m and r['variant']==v and r['domain']==d)
    def comparison(rows,v,reference,rv,domains):
        result=[]
        for d in domains:
            for m in IDS:
                a=metric(find(rows,m,v,d));b=metric(find(reference,m,rv,d));ae,be=a['dead_errors'],b['dead_errors']
                result.append(dict(match=m,domain=d,spell_delta=a['full_spell_f1']-b['full_spell_f1'],
                    dead_miss_frame_rate_delta=ae['missed_dead_frames']/ae['gt_dead_frames']-be['missed_dead_frames']/be['gt_dead_frames'],
                    dead_miss_segment_rate_delta=ae['missed_segment_rate']-be['missed_segment_rate'],false_cuts_delta=ae['false_cuts']-be['false_cuts']))
        return result
    p1=reports['R7B_P1_RESULTS']['rows'];cal=reports['R7B_CALIBRATION_RESULTS']['rows'];cf=reports['R7B_COUNTERFACTUAL_RESULTS']['rows'];mem=reports['R7B_MEMORY_RESULTS']['rows']
    decisions=dict(G1=dict(decision='EXPERIMENTAL_NOT_PROMOTED',reason='Partial SkillCorner zero-player long gaps remain 195/195 D; observation-specific DFL gains insufficient',deltas=comparison(p1,'G1',refs,'M2',['natural','dropout','short','long'])),
        P1=dict(decision='KEEP_DIAGNOSTIC_EXPERT_ONLY',reason='Independent player evidence exists, but pure replacement reduces complete spell; zero-player .982 Alive is unsupported missing-branch prior'),
        G2_cal=dict(decision='REVERT_AS_REPLACEMENT',reason='Disjoint-match sigmoid calibration does not improve consistently vs same-training uncalibrated control',deltas=comparison(cal,'G2_cal',cal,'G2_uncal',['natural','dropout','long'])),
        CF1=dict(decision='REVERT_AS_REPLACEMENT',reason='Noise robustness improves but natural and missing domains degrade',deltas=comparison(cf,'CF1',refs,'M2',['natural','dropout','long'])),
        T2=dict(decision='EXPERIMENTAL_NOT_PROMOTED',reason='195 knownHA falseD reduced to7, but complete blackout trueDmiss guardrails fail',deltas=comparison(mem,'T2',mem,'M2',['scene_blackout'])),
        blind_evaluation=dict(decision='NOT_OPENED',reason='No new candidate clears frozen guardrails; both R7 reserved matches already consumed; WOY/WR9 remain unscored and require team-overlap check before a new protocol'))
    registry=[]
    for folder,variant in [('p1-v1','P1'),('calibration-v1','G2'),('counterfactual-v1','CF1')]:
        for m in IDS:
            path=ROOT/folder/f'{m}.joblib';receipt=json.loads(path.with_suffix('.json').read_text());assert digest(path)==receipt['model_sha256']
            registry.append(dict(candidate=variant,heldout=m,path=str(path),bytes=path.stat().st_size,sha256=digest(path),receipt=receipt))
    write_new('docs/R7B_MODEL_REGISTRY.json',dict(models=registry,hgb_fits=12,sigmoid_calibrators=6,preserved_old_model_count=39,competition_promoted=False))
    write_new('docs/R7B_DECISIONS.json',dict(**identity(),decisions=decisions))
    # Missing first early fit receipt is imported without duplicating any prior ledger event.
    ledger_path=Path('docs/R7B_EXPERIMENTS.json');ledger=json.loads(ledger_path.read_text());existing={r.get('model_sha256')for r in ledger}
    for r in registry:
        if r['sha256']not in existing:ledger.append(r['receipt'])
    ledger.append(dict(stage='decisions_and_registry',**identity(),decision_file='docs/R7B_DECISIONS.json',hgb_fits=12,failed_replacements=['G2_cal','CF1'],remaining_experimental=['G1','T2']))
    ledger_path.write_text(js(ledger)+'\n')
    # Typed .48s and 1s boundaries plus independent binary Alive quality metrics.
    supplements=[]
    for m in IDS:
        players=prepared(m);pn=players['feature_names'].tolist();counts=players['X'][:,pn.index('home_count')]+players['X'][:,pn.index('away_count')]
        for domain in ['natural','dropout','short','long']:
            b=load(m,domain)
            for v in ['P1','G1']:
                path=ROOT/'p1-eval-v1'/f'{m}-{v}-{domain}.npz';p=dict(np.load(path));ya=b['y_alive'];a=p['p_alive'];quality={}
                for label,mask in [('all',np.ones(len(a),bool)),('zero_players',counts==0),('under16_players',counts<16),('at_least16_players',counts>=16)]:
                    live=mask&(ya==1);dead=mask&(ya==0);tp=int(np.sum(dead&(a<.5)));fp=int(np.sum(live&(a<.5)));fn=int(np.sum(dead&(a>=.5)))
                    quality[label]=dict(n=int(mask.sum()),probability=probability_metrics(ya[mask],a[mask]),D_f1=2*tp/(2*tp+fp+fn)if 2*tp+fp+fn else None,
                        D_recall=tp/int(dead.sum())if dead.any()else None,falseD_rate=fp/int(live.sum())if live.any()else None,missedD=fn)
                supplements.append(dict(match=m,variant=v,domain=domain,prediction_sha256=digest(path),binary_alive=quality,
                    boundaries={str(t):{mode:boundaries(b,p[mode],t)for mode in ['decoded','argmax']}for t in [.48,1.]}))
    # Episode support, not frame-IID confidence. Same frozen four SC episodes.
    b=dict(np.load('local_artifacts/r7/transfer-v2/prepared.npz'));diagnosis=json.loads(Path('docs/R7B_MISSING_DIAGNOSIS.json').read_text());episodes=[]
    for v,path in [('P1',ROOT/'transfer-v1/P1.npz'),('G1',ROOT/'transfer-v1/G1.npz'),('T1',ROOT/'memory-v1/SC-T1.npz'),('T2',ROOT/'memory-v1/SC-T2.npz')]:
        p=dict(np.load(path))
        for e in diagnosis['episodes']:
            k=(b['frame']>=e['start_frame'])&(b['frame']<e['stop_frame_exclusive'])&(b['y_alive']==1)&(b['y_home']>=0)
            episodes.append(dict(variant=v,start_frame=e['start_frame'],duration_s=e['duration_s'],n=int(k.sum()),
                p_alive_mean=float(p['p_alive'][k].mean()),p_home_mean=float(p['p_home'][k].mean()),false_dead=int(np.sum(p['decoded'][k]=='D')),
                correct_ha=int(np.sum(p['decoded'][k]==labels_from_batch(b)[k]))))
    write_new('docs/R7B_BOUNDARIES_AND_QUALITY.json',dict(rows=supplements,skillcorner_episodes=episodes,**identity(),script_sha256=digest(__file__)))
    print('Registered9bundles/12HGBfits/6calibrators; fixed gates evaluated; no candidate promoted')
if __name__=='__main__':main()
