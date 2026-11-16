"""E10-C1 one-shot full external probe; labels enter evaluator only."""
import argparse,datetime,json,resource,subprocess,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from r10.acquisition import geometry_evidence,acquisition_strength,rank_proposals,equal_budget,negative_controls
from solution.control_events import control_episodes
from solution.hierarchical_decoder import continuous_blocks
from solution.evidence import labels_from_batch
from solution.idsse import digest
from evaluation import possession_metrics as pm
from scripts.research_r7 import IDS,load,checked_path,write_new
from scripts.evaluate_r8_interpolation import raw_input
from scripts.evaluate_r8_decoder import prediction_path,metrics,save_prediction
from scripts.diagnose_r9_spell_loss import cache_path,DOMAINS
PROTOCOL=Path('docs/R10_V3_PROTOCOL_C1.json');ROOT=Path('local_artifacts/r10-v3/c1')
CODE=['r10/acquisition.py','scripts/probe_r10_acquisition.py','solution/control_events.py','solution/r7_metrics.py','evaluation/possession_metrics.py']


def hc_path(m,d):return Path(f'local_artifacts/r9-v2/e09a/{m}-{d}-M2A-C1H.npz')
def freeze():
    inputs={}
    for m in IDS:
        paths=[Path(f'local_artifacts/r6/prepared/{m}/raw.npz')]
        for d in DOMAINS:
            paths += [checked_path(m,d),hc_path(m,d),cache_path(m,'C1',d),cache_path(m,'H1',d)]
            paths += [prediction_path(m,v,d) for v in ['M2','M3']]
        inputs.update({str(p):digest(p) for p in paths})
    models=json.loads(Path('docs/R9_V2_MODEL_REGISTRY.json').read_text())['inherited_model_hashes']
    for p,h in models.items():
        if digest(p)!=h:raise ValueError('frozen model changed '+p)
    write_new(PROTOCOL,dict(id='E10-C1',frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),base='9f701a4f193d776515a6c2129d5809b6e736b8b2',
      hypothesis='Team-pooled future receiver proximity confirms acquisitions missed by HC_M3; test signal before fitting.',
      input_contract='Raw metric geometry only; 0=Home/1=Away; DFL complete truth evaluator-only. Future observations allowed; future truth forbidden.',
      evidence='Existing 44 columns contain instantaneous distance; new information is ordered opponent-to-team proximity and two subsequent confirmation windows, invariant to IDs/slots.',
      sources='public IDSSE DFL J03WPY/J03WOH/J03WQQ; already used development matches, shared home team; not blind or transfer',
      tag_path='NO_INDEPENDENT_COMPLETE_LABELS for same-source automatic tactical tags. Metrica event-derived weak labels are not independent.',
      methods=['M2 probability same-window control','C1 probability same-window strong control','R9 E1fix observed motion episodes','C1 posterior pooled geometry','per-half shuffled-team negative','per-half +30s time-shift negative'],
      parameters=dict(past_s=1,onset_s=.5,future_s=[.5,1.5,3],distance_margin_scale_m=2,distance_decay_m=5,min_players_each=8,proposal_limit_per_block=120,spacing_s=.5,seed=101603,match_tolerance_s=1),
      equal_budget='For each block use minimum available proposal count across M2prob/C1prob/E1fix/geometry, capped120; preserve exact counts for negatives. Also report unrestricted capped channel coverage; no label-based proposal selection.',
      gate=dict(advance_min_matches=2,required_missed_boundary_recall_gain_vs_best_old=.10,max_precision_loss_vs_best_old=.05,required_gain_vs_each_negative=.05,negative_requirement='all three natural matches',stress='report all three predefined missing-ball domains; no tuning'),
      stop='If gate fails, do not train hazard or complex decoder or tune windows. No candidate predictions changed by this probe.',
      fits=0,inputs=inputs,models=models,code_hashes={p:digest(p) for p in CODE},scorer_sha=digest('evaluation/possession_metrics.py'),
      evaluation='Complete baseline spell5IoU, TP/FP/FN, short spell, macro/D/boundary metrics per match/domain; information precision/recall and missed baseline boundary recovery; partial proxies cannot promote.',
      resource='2 CPU threads; no new raw downloads/weights; sparse proposal outputs only; preserve all history',rollback='unchanged HC_M3/M3; official isolated reference remains human-owned'))
    print('FROZEN',digest(PROTOCOL),'inputs',len(inputs),'models',len(models),flush=True)


def ha_transitions(codes,block):return [t for t in pm.transitions(codes,[block]) if t[1]!=pm.D and t[2]!=pm.D]

def evaluate_proposals(proposals,truth,reference,blocks,fps,visible,home):
    totals=dict(n_gt=0,n_proposals=len(proposals),n_matched=0,old_missed=0,recovered_old_missed=0,
      proposed_on_gt_D=0,proposed_ball_missing=0,matched_ball_missing=0,uncertain_proposals=0,uncertain_matched=0)
    for bi,block in enumerate(blocks):
        gt=ha_transitions(truth,block);old=ha_transitions(reference,block)
        _,missed,_=pm.match_transitions(gt,old,round(fps),typed='pair')
        ps=[(p['index'],1-p['team'],p['team']) for p in proposals if p['block']==bi]
        pairs,_,_=pm.match_transitions(gt,ps,round(fps),typed='pair')
        totals['n_gt']+=len(gt);totals['n_matched']+=len(pairs);totals['old_missed']+=len(missed)
        totals['recovered_old_missed']+=sum(g in missed for g,p in pairs)
        totals['proposed_on_gt_D']+=sum(truth[p[0]]==pm.D for p in ps)
        totals['proposed_ball_missing']+=sum(not visible[p[0]] for p in ps)
        totals['matched_ball_missing']+=sum(not visible[p[0]] for g,p in pairs)
        totals['uncertain_proposals']+=sum(.35<=home[p[0]]<=.65 for p in ps)
        totals['uncertain_matched']+=sum(.35<=home[p[0]]<=.65 for g,p in pairs)
    totals.update(precision=totals['n_matched']/max(1,totals['n_proposals']),recall=totals['n_matched']/max(1,totals['n_gt']),
      missed_recall=totals['recovered_old_missed']/max(1,totals['old_missed']),n_spurious=totals['n_proposals']-totals['n_matched'])
    return {k:(v.item() if isinstance(v,np.generic) else v) for k,v in totals.items()}


def gate(rows):
    natural=[r for r in rows if r['domain']=='natural'];checks=[]
    for row in natural:
        a=row['equal_budget']['geometry'];old=[row['equal_budget'][k] for k in ['M2prob','C1prob','E1fix']]
        best_recall=max(x['missed_recall'] for x in old);best_precision=max(x['precision'] for x in old)
        neg=[row['equal_budget'][k]['missed_recall'] for k in ['shuffled_team','shifted_time']]
        checks.append(dict(match=row['match'],gain_vs_best_old=a['missed_recall']-best_recall,precision_loss=best_precision-a['precision'],
          gain_vs_best_negative=a['missed_recall']-max(neg)))
    advance=sum(c['gain_vs_best_old']>=.10 and c['precision_loss']<=.05 for c in checks)>=2 and all(c['gain_vs_best_negative']>=.05 for c in checks)
    return dict(checks=checks,advance=advance,decision='ADVANCE_MINIMAL_HEAD' if advance else 'REVERT_PROPOSAL_ROUTE_NO_FIT',competition_promoted=False,transfer='TRANSFER_NOT_ESTABLISHED')


def run():
    protocol=json.loads(PROTOCOL.read_text());ROOT.mkdir(parents=True,exist_ok=True)
    for p,h in {**protocol['inputs'],**protocol['models'],**protocol['code_hashes']}.items():
        if digest(p)!=h:raise ValueError('drift '+p)
    rows=[]
    for m in IDS:
      raw=raw_input(m)
      for d in DOMAINS:
        path=ROOT/f'{m}-{d}.json'
        if path.exists():
            row=json.loads(path.read_text())
            if row['protocol_sha']!=digest(PROTOCOL) or digest(path.with_suffix('.npz'))!=row['proposal_sha']:raise ValueError('resume drift')
            rows.append(row);continue
        start=time.monotonic();cpu=time.process_time();b=load(m,d);fps=float(b['fps']);blocks=continuous_blocks(b['frame'],b['period'])
        assert np.array_equal(raw['frame'],b['frame']) and np.array_equal(raw['period'],b['period'])
        visible=b['X'][:,b['feature_names'].tolist().index('ball_visible')]==1
        masked={**raw,'ball_xy':raw['ball_xy'].copy()};masked['ball_xy'][~visible]=np.nan
        m2=dict(np.load(prediction_path(m,'M2',d)));m3=dict(np.load(prediction_path(m,'M3',d)));c1=dict(np.load(cache_path(m,'C1',d)))
        h1=dict(np.load(cache_path(m,'H1',d)));hc=dict(np.load(hc_path(m,d)))
        geo,valid=geometry_evidence(masked);old_events,old_signal=control_episodes(masked,fps)
        channels={}
        for name,ev in [('M2prob',np.column_stack([m2['p_home'],1-m2['p_home']])),('C1prob',np.column_stack([c1['p_home'],1-c1['p_home']])),('geometry',geo)]:
            channels[name]=rank_proposals(acquisition_strength(ev,b['frame'],b['period'],fps),b['frame'],b['period'],fps)
        # R9 E1 supports use the original physical episodes, ranked by observed duration.
        old_strength=np.zeros((len(visible),2))
        for e in old_events:old_strength[e['start'],int(e['team']=='A')]=e['support_seconds']
        channels['E1fix']=rank_proposals(old_strength,b['frame'],b['period'],fps)
        paired,budgets=equal_budget(channels,b['frame'],b['period'])
        sh,ti=negative_controls(paired['geometry'],b['frame'],b['period'],fps)
        paired.update(shuffled_team=sh,shifted_time=ti)
        # Store label-free proposals before loading evaluator truth; no new hard labels.
        arrays={k:np.array([[p['index'],p['team'],p['score'],p['block']] for p in v],dtype=float).reshape(-1,4) for k,v in paired.items()}
        save_prediction(path.with_suffix('.npz'),**arrays)
        truth=pm.label_codes(labels_from_batch(b));reference=pm.label_codes(hc['decoded'])
        eval_args=(truth,reference,blocks,fps,visible,c1['p_home'])
        scores={k:evaluate_proposals(v,*eval_args) for k,v in paired.items()}
        all_scores={k:evaluate_proposals(v,*eval_args) for k,v in channels.items()}
        baseline={}
        for name,a,h,p in [('M2',m2['p_alive'],m2['p_home'],m2['decoded']),('M3',m3['p_alive'],m3['p_home'],m3['decoded']),('HC_M3',m3['p_alive'],c1['p_home'],hc['decoded']),('H1',h1['p_alive'],h1['p_home'],h1['decoded'])]:
            baseline[name]=metrics(b,a,h,p)
        row=dict(id='E10-C1',match=m,domain=d,equal_budget=scores,unrestricted_capped=all_scores,budgets_per_block=budgets,baseline=baseline,
          availability=dict(frames=len(visible),geometry_valid=int(valid.sum()),missing_ball=int((~visible).sum())),
          protocol_sha=digest(PROTOCOL),code_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),code_hashes=protocol['code_hashes'],
          proposal_sha=digest(path.with_suffix('.npz')),models=protocol['models'],new_fits=0,candidate_predictions_changed=False,
          wall_cpu_memory=dict(wall_s=time.monotonic()-start,cpu_s=time.process_time()-cpu,max_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        write_new(path,row);rows.append(row)
        print(m,d,'budgets',budgets,{k:(v['recovered_old_missed'],round(v['precision'],3)) for k,v in scores.items()},flush=True)
    result=dict(rows=rows,decision=gate(rows),new_fits=0)
    write_new(ROOT/'results.json',result);write_new('docs/R10_V3_RESULTS_C1.json',result)
    print(json.dumps(result['decision']),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run']);a=p.parse_args();freeze() if a.action=='freeze' else run()
