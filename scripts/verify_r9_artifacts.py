"""Read-only exact replay of R9 predictions; never fit or touch target assets."""
import json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from solution.idsse import digest
from solution.control_events import control_episodes,relocate_boundaries,decode_control_windows
from solution.hierarchical_decoder import decode,continuous_blocks
from solution.evidence import labels_from_batch
from evaluation import possession_metrics as pm
from scripts.research_r7 import IDS,load
from scripts.evaluate_r8_interpolation import raw_input
from scripts.evaluate_r8_decoder import original
from scripts.diagnose_r9_spell_loss import cache_path,DOMAINS
ROOT=Path('local_artifacts/r9-v2')

def verify_saved(path,pred,b,expected_spell=None):
    r=json.loads(path.read_text())
    if digest(path.with_suffix('.npz'))!=r['prediction_sha']:raise ValueError('prediction hash drift '+str(path))
    with np.load(path.with_suffix('.npz')) as z:np.testing.assert_array_equal(pred,z['decoded'])
    if expected_spell is not None:
        score=pm.spell_scores(pm.label_codes(labels_from_batch(b)),pm.label_codes(pred),continuous_blocks(b['frame'],b['period']))['mean_f1']
        if abs(score-expected_spell)>1e-14:raise ValueError('score replay drift')

def main():
    seen={};models={};count=0
    for p in sorted(Path('docs').glob('R9_V2_*_PROTOCOL.json')):
        protocol=json.loads(p.read_text())
        if protocol['scorer_sha']!=digest('evaluation/possession_metrics.py'):raise ValueError('scorer changed')
        for k,v in protocol['inputs'].items():
            if k in seen and seen[k]!=v:raise ValueError('input protocol disagreement')
            seen[k]=v
        models.update(protocol['models'])
    for p,h in {**seen,**models}.items():
        if digest(p)!=h:raise ValueError('frozen input/model drift '+p)
    print('Verified',len(seen),'inputs and',len(models),'model hashes; no refits',flush=True)
    for m in IDS:
      raw=raw_input(m)
      for d in DOMAINS:
        b=load(m,d);masked={**raw,'ball_xy':raw['ball_xy'].copy()};names=b['feature_names'].tolist();masked['ball_xy'][b['X'][:,names.index('ball_visible')]!=1]=np.nan
        events,signal=control_episodes(masked,float(b['fps']))
        base=dict(np.load(cache_path(m,'M3',d)));c1=dict(np.load(cache_path(m,'C1',d)))
        for parent in ['M3','H1']:
          c=dict(np.load(cache_path(m,parent,d)))
          for exp,fn in [('e09e1fix',relocate_boundaries),('e09e2',decode_control_windows)]:
            args=[c['decoded'],c['p_home'],b['frame'],b['period'],float(b['fps']),events]
            if exp=='e09e2':args.append(signal)
            pred,_=fn(*args);path=ROOT/exp/f'{m}-{d}-{parent}.json';r=json.loads(path.read_text());verify_saved(path,pred,b,r['metrics']['full_spell_f1']);count+=1
        for aa,z in [('M2',base),('C1',c1)]:
          for hh,zz in [('M2',base),('C1',c1)]:
            pred=original(z['p_alive'],zz['p_home'],b,'M3');path=ROOT/'e09a'/f'{m}-{d}-{aa}A-{hh}H.json';r=json.loads(path.read_text());verify_saved(path,pred,b,r['metrics']['full_spell_f1']);count+=1
        for hh,z in [('M2',base),('C1',c1)]:
            pred=decode(base['p_alive'],z['p_home'],b['frame'],b['period'],float(b['fps']));path=ROOT/'e09a2'/f'{m}-{d}-{hh}.json';r=json.loads(path.read_text());verify_saved(path,pred,b,r['metrics']['full_spell_f1']);count+=1
        print('Exact replay',m,d,'count',count,flush=True)
    protocol=json.loads(Path('docs/R9_V2_E09A2_PROTOCOL.json').read_text())
    for source,spec in protocol['transfer'].items():
        b=dict(np.load(spec['batch']));base=dict(np.load(spec['M2']));c1=dict(np.load(spec['C1']))
        for decoder in ['M3','factorized_H1']:
          for hh,z in [('M2',base),('C1',c1)]:
            pred=original(base['p_alive'],z['p_home'],b,'M3') if decoder=='M3' else decode(base['p_alive'],z['p_home'],b['frame'],b['period'],float(b['fps']))
            verify_saved(ROOT/'e09a2'/f'{source}-{decoder}-{hh}.json',pred,b);count+=1
    invalid=json.loads((ROOT/'e09e1/results.json').read_text())['rows']
    for r in invalid:
        path=ROOT/'e09e1'/f"{r['match']}-{r['domain']}-{r['parent']}.npz"
        if digest(path)!=r['prediction_sha']:raise ValueError('invalid historical run not preserved')
    print('PASS',count,'new predictions exact; official full spell scores replayed; 24 invalid-contract outputs preserved, not reinterpreted; zero fits',flush=True)
if __name__=='__main__':main()
