"""Frozen full external diagnostic, 3 consumed matches x 2 domains x 12 arms."""
import argparse,json,subprocess,time
from pathlib import Path
import numpy as np
from research_only.oracle_bottleneck import ExternalTruth,interventions
from scripts.research_r7 import IDS,load,checked_path,write_new
from scripts.research_r11p5 import full
from scripts.research_r12 import matched_changes
from scripts.verify_r12 import extra
from solution.evidence import labels_from_batch
from solution.hierarchical_decoder import continuous_blocks
from solution.idsse import digest
from evaluation import possession_metrics as pm
from r11.spell_roi import score
from scripts.evaluate_r11_roi import check
P=Path('docs/R14_ORACLE_PROTOCOL_V2.json');ROOT=Path('local_artifacts/r14/oracle-v1')

def freeze():
 paths=['research_only/oracle_bottleneck.py','scripts/research_r14_oracle.py','evaluation/possession_metrics.py']
 for m in IDS:
  for d in ['natural','long']:paths += [str(checked_path(m,d)),f'local_artifacts/r12/global-v1/{m}-{d}.npz',f'local_artifacts/r8/e08c/{m}-{d}.npz']
 write_new(P,dict(id='R14_ORACLE_V2',pre_score_fix='Synthetic test exposed loss of B candidate cuts when equal model labels merged; C now retains B latent proposed intervals. V1 never scored.',source='IDSSE_DFL_source_complete; legal external consumed development matches; not blind validation',baseline='Frozen HC_M3 exact R12 full predictions; no new model fit',A='Set true D mask; fill removed false D with frozen conditional Home argmax; other HA unchanged',B='Within current predicted Alive intervals, use ONLY true direct HA boundaries and mean frozen Home for each segment. Removes old HA cuts; gives both time AND event count; direction not oracle.',C='Reassign each current predicted HA segment by true majority ownership; ties retain label; Dead unchanged. Frame-majority is not spell-optimal.',D='Overwrite complete true HA spells shorter than1s or5s; reveals existence,timing and direction. Not a deployable proposal.',factorial='All8 A/B/C combinations in A→B→C order; D_lt1,D_lt5 with/withoutA, total12. Noncommutative interactions reported, not additive upper bounds.',controls='baseline identity; full A+B+C must equal external truth; official-vs-ROI equivalence every arm',gate='DIAGNOSTIC_ONLY; no candidate promotion possible from oracle scores. Rank engineering by mean gains and matched-loss interactions, not target GT.',resources='No downloads/training; public caches only; threads2; small compressed results; no sealed WOY/WR9',hashes={p:digest(p) for p in paths}))

def run():
 protocol=json.loads(P.read_text());ROOT.mkdir(parents=True,exist_ok=True)
 for p,h in protocol['hashes'].items():assert digest(p)==h,p
 rows=[];start=time.monotonic()
 for m in IDS:
  for domain in ['natural','long']:
   dest=ROOT/f'{m}-{domain}.json'
   if dest.exists():rows.append(json.loads(dest.read_text()));continue
   b=load(m,domain);g=labels_from_batch(b);blocks=list(continuous_blocks(b['frame'],b['period']));base=np.load(f'local_artifacts/r12/global-v1/{m}-{domain}.npz')['HC_M3'];home=np.load(f'local_artifacts/r8/e08c/{m}-{domain}.npz')['p_home'];visible=b['X'][:,b['feature_names'].tolist().index('ball_visible')]==1
   outputs=interventions(base,home,ExternalTruth(g,'IDSSE_DFL_source_complete'),blocks,float(b['fps']))
   np.testing.assert_array_equal(outputs['baseline'],base);np.testing.assert_array_equal(outputs['A_dead+B_times+C_owner'],g)
   results={}
   for k,p in outputs.items():
    metrics=full(b,g,p);check(score(g.tolist(),p.tolist(),blocks=blocks),metrics['spell'])
    results[k]=dict(metrics=metrics,gained_lost=matched_changes(g,base,p,blocks),additional=extra(g,p,blocks,visible),changed_frames=int(np.sum(p!=base)))
   path=ROOT/f'{m}-{domain}.npz'
   with path.open('xb') as f:np.savez_compressed(f,**outputs)
   row=dict(match=m,domain=domain,n_frames=len(g),results=results,prediction_sha256=digest(path),protocol_sha256=digest(P),git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
   write_new(dest,row);rows.append(row);print('SCORE',m,domain,{k:round(v['metrics']['spell']['mean_f1'],6) for k,v in results.items()},flush=True)
 write_new('docs/R14_ORACLE_BOTTLENECK.json',dict(rows=rows,wall_s=time.monotonic()-start,official_ROI_equivalences=72,decision='DIAGNOSTIC_ONLY_NOT_MODEL_PERFORMANCE'))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run']);a=p.parse_args();freeze() if a.action=='freeze' else run()
