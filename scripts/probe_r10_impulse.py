"""E10-C2: one different observable channel after C1 information-gate failure."""
import argparse,datetime,json,resource,subprocess,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from r10.impulse import impulse_strength
from r10.acquisition import acquisition_strength,rank_proposals,equal_budget,negative_controls
from scripts.probe_r10_acquisition import evaluate_proposals,gate,hc_path
from solution.control_events import control_episodes
from solution.hierarchical_decoder import continuous_blocks
from solution.evidence import labels_from_batch
from solution.idsse import digest
from evaluation import possession_metrics as pm
from scripts.research_r7 import IDS,load,write_new
from scripts.evaluate_r8_interpolation import raw_input
from scripts.evaluate_r8_decoder import prediction_path,save_prediction
from scripts.diagnose_r9_spell_loss import cache_path,DOMAINS
PROTOCOL=Path('docs/R10_V3_PROTOCOL_C2.json');ROOT=Path('local_artifacts/r10-v3/c2')
CODE=['r10/impulse.py','r10/acquisition.py','scripts/probe_r10_impulse.py','scripts/probe_r10_acquisition.py','solution/control_events.py','evaluation/possession_metrics.py']

def freeze():
    parent=json.loads(Path('docs/R10_V3_PROTOCOL_C1FIX.json').read_text());results=json.loads(Path('docs/R10_V3_RESULTS_C1FIX.json').read_text())
    if results['decision']['advance']:raise ValueError('C2 requires documented C1 failure')
    inputs=parent['inputs'].copy()
    for p in [Path('docs/R10_V3_RESULTS_C1FIX.json')]+[Path(f'local_artifacts/r10-v3/c1fix/{m}-{d}.json') for m in IDS for d in DOMAINS]:inputs[str(p)]=digest(p)
    write_new(PROTOCOL,dict(id='E10-C2',frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),parent_protocol=digest('docs/R10_V3_PROTOCOL_C1FIX.json'),
      reason='C1 temporal proximity failed: natural missed-recall gains 5.29/2.67/1.24pp vs old and WPY shuffled-team recovered more. Need an action channel, not longer smoothing.',
      hypothesis='Ball momentum-change at a nearby team detects reception/kick instants which sustained proximity and old probability miss; contact alone may be a deflection.',
      new_information='Vector incoming/outgoing ball velocity change at +-0.12s, paired with team receiver proximity. No long future persistence; no provider speed/status/possession fields.',
      parameters=dict(momentum_sample_s=.12,min_delta_v_mps=2,max_ball_speed_mps=45,seed=101603,limit_per_half=120,proposal_spacing_s=.5),
      gate=parent['gate'],equal_budget=parent['equal_budget'],sources=parent['sources'],input_contract=parent['input_contract'],
      controls='identical M2prob/C1prob/E1fix proposal code; new impulse timestamps with shuffled-team and +30s time-shift controls. geometry result key now means impulse channel; no candidate decoder changes.',
      stop='If gate fails, stop geometry-heuristic proposal training; independent action observations/labels required. No third threshold or smoothing trial.',
      inputs=inputs,models=parent['models'],code_hashes={p:digest(p) for p in CODE},fits=0,
      baseline='Reuse exact C1FIX full M2/M3/HC_M3/H1 metric receipts after hash verification; do not count copied scores as new independent evaluations.',rollback='unchanged HC_M3/M3'))
    print('FROZEN',digest(PROTOCOL),flush=True)

def run():
    protocol=json.loads(PROTOCOL.read_text());ROOT.mkdir(parents=True,exist_ok=True);rows=[]
    for p,h in {**protocol['inputs'],**protocol['models'],**protocol['code_hashes']}.items():
        if digest(p)!=h:raise ValueError('drift '+p)
    for m in IDS:
      raw=raw_input(m)
      for d in DOMAINS:
        path=ROOT/f'{m}-{d}.json'
        if path.exists():
            row=json.loads(path.read_text())
            if row['protocol_sha']!=digest(PROTOCOL) or digest(path.with_suffix('.npz'))!=row['proposal_sha']:raise ValueError('resume drift')
            rows.append(row);continue
        start=time.monotonic();cpu=time.process_time();b=load(m,d);fps=float(b['fps']);blocks=continuous_blocks(b['frame'],b['period'])
        visible=b['X'][:,b['feature_names'].tolist().index('ball_visible')]==1;masked={**raw,'ball_xy':raw['ball_xy'].copy()};masked['ball_xy'][~visible]=np.nan
        assert np.array_equal(raw['frame'],b['frame']) and np.array_equal(raw['period'],b['period'])
        m2=dict(np.load(prediction_path(m,'M2',d)));c1=dict(np.load(cache_path(m,'C1',d)));hc=dict(np.load(hc_path(m,d)))
        strength,impulse=impulse_strength(masked,fps);channels={'geometry':rank_proposals(strength,b['frame'],b['period'],fps)}
        for name,z in [('M2prob',m2),('C1prob',c1)]:
            ev=np.column_stack([z['p_home'],1-z['p_home']]);channels[name]=rank_proposals(acquisition_strength(ev,b['frame'],b['period'],fps),b['frame'],b['period'],fps)
        events,_=control_episodes(masked,fps);e1=np.zeros((len(visible),2))
        for e in events:e1[e['start'],int(e['team']=='A')]=e['support_seconds']
        channels['E1fix']=rank_proposals(e1,b['frame'],b['period'],fps)
        paired,budgets=equal_budget(channels,b['frame'],b['period']);sh,ti=negative_controls(paired['geometry'],b['frame'],b['period'],fps);paired.update(shuffled_team=sh,shifted_time=ti)
        save_prediction(path.with_suffix('.npz'),**{k:np.array([[p['index'],p['team'],p['score'],p['block']] for p in v],dtype=float).reshape(-1,4) for k,v in paired.items()})
        truth=pm.label_codes(labels_from_batch(b));reference=pm.label_codes(hc['decoded']);args=(truth,reference,blocks,fps,visible,c1['p_home'])
        old=Path(f'local_artifacts/r10-v3/c1fix/{m}-{d}.json');prior=json.loads(old.read_text())
        row=dict(id='E10-C2',match=m,domain=d,equal_budget={k:evaluate_proposals(v,*args) for k,v in paired.items()},unrestricted_capped={k:evaluate_proposals(v,*args) for k,v in channels.items()},
          budgets_per_block=budgets,baseline=prior['baseline'],baseline_receipt_sha=digest(old),baseline_metrics_reused=True,
          channel='raw ball momentum impulse, not posterior proximity',impulse_frames=int((impulse>0).sum()),
          protocol_sha=digest(PROTOCOL),proposal_sha=digest(path.with_suffix('.npz')),code_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),code_hashes=protocol['code_hashes'],new_fits=0,candidate_predictions_changed=False,
          wall_cpu_memory=dict(wall_s=time.monotonic()-start,cpu_s=time.process_time()-cpu,max_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        write_new(path,row);rows.append(row)
        print(m,d,{k:(v['recovered_old_missed'],round(v['precision'],3)) for k,v in row['equal_budget'].items()},flush=True)
    result=dict(rows=rows,decision=gate(rows),new_fits=0)
    write_new(ROOT/'results.json',result);write_new('docs/R10_V3_RESULTS_C2.json',result);print(json.dumps(result['decision']),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run']);a=p.parse_args();freeze() if a.action=='freeze' else run()
