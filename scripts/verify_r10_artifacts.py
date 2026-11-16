"""Read-only exact replay of frozen R10 probes; never refit or write predictions."""
import json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from r10.acquisition import geometry_evidence,acquisition_strength,rank_proposals,equal_budget,negative_controls
from r10.impulse import impulse_strength
from solution.control_events import control_episodes
from solution.hierarchical_decoder import continuous_blocks
from solution.evidence import labels_from_batch
from solution.idsse import digest
from evaluation import possession_metrics as pm
from scripts.research_r7 import IDS,load
from scripts.evaluate_r8_interpolation import raw_input
from scripts.evaluate_r8_decoder import prediction_path
from scripts.diagnose_r9_spell_loss import cache_path,DOMAINS
from scripts.probe_r10_acquisition import hc_path,evaluate_proposals


def main():
    seen={};narrays=nfiles=nscores=0
    for exp in ['C1','C1FIX','C2']:
        p=json.loads(Path(f'docs/R10_V3_PROTOCOL_{exp}.json').read_text())
        for key,value in {**p['inputs'],**p['models']}.items():
            if key in seen and seen[key]!=value:raise ValueError('hash disagreement')
            seen[key]=value
        # Historical C1 code differs intentionally; do not reinterpret its result.
        if exp!='C1':
            for key,value in p['code_hashes'].items():
                if digest(key)!=value:raise ValueError('inference source changed '+key)
    for p,h in seen.items():
        if digest(p)!=h:raise ValueError('input/model drift '+p)
    for m in IDS:
      raw=raw_input(m)
      for d in DOMAINS:
        b=load(m,d);fps=float(b['fps']);blocks=continuous_blocks(b['frame'],b['period']);visible=b['X'][:,b['feature_names'].tolist().index('ball_visible')]==1
        masked={**raw,'ball_xy':raw['ball_xy'].copy()};masked['ball_xy'][~visible]=np.nan
        m2=dict(np.load(prediction_path(m,'M2',d)));m3=dict(np.load(prediction_path(m,'M3',d)));c1=dict(np.load(cache_path(m,'C1',d)));hc=dict(np.load(hc_path(m,d)));h1=dict(np.load(cache_path(m,'H1',d)))
        truth=pm.label_codes(labels_from_batch(b));ref=pm.label_codes(hc['decoded']);controls={}
        for name,z in [('M2prob',m2),('C1prob',c1)]:
            e=np.column_stack([z['p_home'],1-z['p_home']]);controls[name]=rank_proposals(acquisition_strength(e,b['frame'],b['period'],fps),b['frame'],b['period'],fps)
        es=np.zeros((len(visible),2));events,_=control_episodes(masked,fps)
        for e in events:es[e['start'],int(e['team']=='A')]=e['support_seconds']
        controls['E1fix']=rank_proposals(es,b['frame'],b['period'],fps)
        for exp in ['c1fix','c2']:
            if exp=='c1fix':
                ev,_=geometry_evidence(masked);s=acquisition_strength(ev,b['frame'],b['period'],fps)
            else:s,_=impulse_strength(masked,fps)
            paired,_=equal_budget({**controls,'geometry':rank_proposals(s,b['frame'],b['period'],fps)},b['frame'],b['period'])
            sh,ti=negative_controls(paired['geometry'],b['frame'],b['period'],fps);paired.update(shuffled_team=sh,shifted_time=ti)
            path=Path(f'local_artifacts/r10-v3/{exp}/{m}-{d}.json');r=json.loads(path.read_text());z=np.load(path.with_suffix('.npz'))
            if digest(path.with_suffix('.npz'))!=r['proposal_sha']:raise ValueError('proposal drift')
            for k,v in paired.items():
                arr=np.array([[p['index'],p['team'],p['score'],p['block']] for p in v],float).reshape(-1,4);np.testing.assert_array_equal(z[k],arr);narrays+=1
                if evaluate_proposals(v,truth,ref,blocks,fps,visible,c1['p_home'])!=r['equal_budget'][k]:raise ValueError('metric drift')
            nfiles+=1
            if exp=='c1fix':
                for name,c in [('M2',m2),('M3',m3),('HC_M3',hc),('H1',h1)]:
                    got=pm.spell_scores(truth,pm.label_codes(c['decoded']),blocks)
                    if abs(got['mean_f1']-r['baseline'][name]['full_spell_f1'])>1e-14:raise ValueError('baseline score drift')
                    nscores+=1
        print('REPLAY',m,d,nfiles,narrays,flush=True)
    invalid=json.loads(Path('local_artifacts/r10-v3/c1/results.json').read_text())['rows']
    for r in invalid:
        p=f"local_artifacts/r10-v3/c1/{r['match']}-{r['domain']}.npz"
        if digest(p)!=r['proposal_sha']:raise ValueError('invalid historical output changed')
    print('PASS',nfiles,'valid proposal files',narrays,'exact arrays',nscores,'official full baseline spell replays; invalid12 preserved; frozen15 model hashes; zero fits',flush=True)
if __name__=='__main__':main()
