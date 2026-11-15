"""Frozen R9 physical control experiment, public other matches only, no fits."""
import argparse,json,resource,subprocess,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from solution.control_events import control_episodes,relocate_boundaries,decode_control_windows
from solution.hierarchical_decoder import continuous_blocks
from solution.evidence import labels_from_batch
from solution.idsse import digest
from evaluation import possession_metrics as pm
from scripts.research_r7 import IDS,load,write_new
from scripts.evaluate_r8_interpolation import raw_input
from scripts.evaluate_r8_decoder import metrics,save_prediction
from scripts.diagnose_r9_spell_loss import cache_path,DOMAINS


def execute(experiment):
    protocol_path=Path(f'docs/R9_V2_{experiment.upper()}_PROTOCOL.json');protocol=json.loads(protocol_path.read_text())
    root=Path('local_artifacts/r9-v2')/experiment;root.mkdir(parents=True,exist_ok=True)
    for p,h in protocol['inputs'].items():
        if digest(p)!=h:raise ValueError('input drift '+p)
    code={p:digest(p) for p in ['solution/control_events.py','scripts/evaluate_r9_control.py']}
    rows=[]
    for m in IDS:
      raw=raw_input(m)
      for d in DOMAINS:
        b=load(m,d);names=b['feature_names'].tolist();fps=float(b['fps'])
        masked={**raw,'ball_xy':raw['ball_xy'].copy()};masked['ball_xy'][b['X'][:,names.index('ball_visible')]!=1]=np.nan
        events,signal=control_episodes(masked,fps)
        for v in ['M3','H1']:
          out=root/f'{m}-{d}-{v}.json'
          if out.exists():
            row=json.loads(out.read_text())
            if row['protocol_sha']!=digest(protocol_path) or row['code_hashes']!=code or digest(out.with_suffix('.npz'))!=row['prediction_sha']:raise ValueError('run drift')
            rows.append(row);continue
          start=time.monotonic();cpu=time.process_time();path=cache_path(m,v,d);c=dict(np.load(path,allow_pickle=False))
          pred,trace=(decode_control_windows(c['decoded'],c['p_home'],b['frame'],b['period'],fps,events,signal) if experiment=='e09e2' else relocate_boundaries(c['decoded'],c['p_home'],b['frame'],b['period'],fps,events))
          # No labels enter geometry or inference. Full truth used only below.
          g=pm.label_codes(labels_from_batch(b));pc=pm.label_codes(pred);old=pm.label_codes(c['decoded']);blocks=continuous_blocks(b['frame'],b['period'])
          score=metrics(b,c['p_alive'],c['p_home'],pred)
          reference=pm.spell_scores(g,old,blocks)
          perhalf={str(i+1):dict(candidate=pm.spell_scores(g,pc,[block]),reference=pm.spell_scores(g,old,[block])) for i,block in enumerate(blocks)}
          event_eval=dict(n=len(events),frames=int(np.sum(signal>=0)),GT_D_episodes=sum(np.mean(g[e['start']:e['end']]==pm.D)>.5 for e in events),
              majority_wrong_team=sum(np.mean(g[e['start']:e['end']]==(pm.H if e['team']=='H' else pm.A))<.5 for e in events),
              improved_frames=int(np.sum((pc==g)&(old!=g))),worsened_frames=int(np.sum((pc!=g)&(old==g))),
              delta_matched={str(t):score['spell_by_iou'][t]['n_matched']-reference['per_iou'][t]['n_matched'] for t in pm.SPELL_IOUS})
          save_prediction(out.with_suffix('.npz'),decoded=pred)
          row=dict(experiment_id=experiment,match=m,domain=d,parent=v,metrics=score,reference_spell=reference,per_half=perhalf,trace=trace,event_evaluation=event_eval,
              prediction_sha=digest(out.with_suffix('.npz')),source_prediction_sha=digest(path),protocol_sha=digest(protocol_path),code_hashes=code,
              code_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),fits=0,wall_cpu_memory=dict(wall_s=time.monotonic()-start,cpu_s=time.process_time()-cpu,max_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
          write_new(out,row);rows.append(row)
          print(m,d,v,'spell',round(score['full_spell_f1'],6),'delta',round(score['full_spell_f1']-reference['mean_f1'],6),trace,event_eval,flush=True)
    if not (root/'results.json').exists():write_new(root/'results.json',dict(rows=rows,competition_promoted=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('experiment',choices=['e09e1fix','e09e2']);a=p.parse_args();execute(a.experiment)
