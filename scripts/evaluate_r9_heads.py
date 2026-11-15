"""Single fixed 2x2 attribution, frozen M3 decoder, no new fits or grids."""
import json,resource,subprocess,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from solution.idsse import digest
from solution.evidence import labels_from_batch
from solution.hierarchical_decoder import continuous_blocks
from evaluation import possession_metrics as pm
from scripts.research_r7 import IDS,load,write_new
from scripts.evaluate_r8_decoder import original,metrics,save_prediction
from scripts.diagnose_r9_spell_loss import cache_path,DOMAINS
ROOT=Path('local_artifacts/r9-v2/e09a');PROTOCOL=Path('docs/R9_V2_E09A_PROTOCOL.json')

def mix_heads(m,c,alive_source,home_source):
    if alive_source not in ('M2','C1') or home_source not in ('M2','C1'):raise ValueError('unregistered cell')
    sources={'M2':m,'C1':c}
    a=sources[alive_source]['p_alive'];h=sources[home_source]['p_home']
    if a.shape!=h.shape:raise ValueError('alignment')
    return a,h

def main():
    p=json.loads(PROTOCOL.read_text());ROOT.mkdir(parents=True,exist_ok=True)
    for path,h in p['inputs'].items():
        if digest(path)!=h:raise ValueError('input drift '+path)
    rows=[]
    for m in IDS:
      for d in DOMAINS:
        b=load(m,d);m2=dict(np.load(cache_path(m,'M3',d)));c1=dict(np.load(cache_path(m,'C1',d)))
        # Same source emissions; verify both existing decoders before the four-cell score.
        np.testing.assert_array_equal(original(m2['p_alive'],m2['p_home'],b,'M3'),m2['decoded'])
        np.testing.assert_array_equal(original(c1['p_alive'],c1['p_home'],b,'M2'),c1['decoded'])
        for aa,hh in [('M2','M2'),('M2','C1'),('C1','M2'),('C1','C1')]:
          path=ROOT/f'{m}-{d}-{aa}A-{hh}H.json'
          if path.exists():
            r=json.loads(path.read_text())
            if r['protocol_sha']!=digest(PROTOCOL) or r['script_sha']!=digest(__file__) or digest(path.with_suffix('.npz'))!=r['prediction_sha']:raise ValueError('run identity drift')
            rows.append(r);continue
          start=time.monotonic();cpu=time.process_time();a,h=mix_heads(m2,c1,aa,hh);pred=original(a,h,b,'M3')
          score=metrics(b,a,h,pred);gt=pm.label_codes(labels_from_batch(b));pc=pm.label_codes(pred)
          halves={str(i+1):pm.spell_scores(gt,pc,[block]) for i,block in enumerate(continuous_blocks(b['frame'],b['period']))}
          save_prediction(path.with_suffix('.npz'),decoded=pred)
          r=dict(match=m,domain=d,alive_source=aa,home_source=hh,decoder='M3',metrics=score,per_half=halves,source_heads={k:digest(cache_path(m,k,d)) for k in ['M3','C1']},prediction_sha=digest(path.with_suffix('.npz')),protocol_sha=digest(PROTOCOL),script_sha=digest(__file__),code_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),fits=0,wall_cpu_memory=dict(wall_s=time.monotonic()-start,cpu_s=time.process_time()-cpu,max_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
          write_new(path,r);rows.append(r);print(m,d,aa,hh,round(score['full_spell_f1'],6),'Dmiss',score['dead_errors']['missed_dead_frames'],flush=True)
    if not (ROOT/'results.json').exists():write_new(ROOT/'results.json',dict(rows=rows,decoder_note='C1A-C1H uses M3, unlike original C1 M2 decoder; not relabeled original C1 score',competition_promoted=False))
if __name__=='__main__':main()
