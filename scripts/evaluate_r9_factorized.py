"""One fixed factorized decoder ablation and partial-source risk check."""
import json,sys,subprocess,time,resource
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from solution.hierarchical_decoder import decode,continuous_blocks
from solution.evidence import labels_from_batch,score_sequence,visibility_strata
from solution.idsse import digest
from evaluation import possession_metrics as pm
from scripts.research_r7 import IDS,load,write_new
from scripts.evaluate_r8_decoder import original,metrics,save_prediction
from scripts.diagnose_r9_spell_loss import cache_path,DOMAINS
ROOT=Path('local_artifacts/r9-v2/e09a2');PROTOCOL=Path('docs/R9_V2_E09A2_PROTOCOL.json')

def identity():return dict(protocol_sha=digest(PROTOCOL),script_sha=digest(__file__),code_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),fits=0)

def main():
    protocol=json.loads(PROTOCOL.read_text());ROOT.mkdir(parents=True,exist_ok=True)
    for p,h in protocol['inputs'].items():
        if digest(p)!=h:raise ValueError('input drift '+p)
    rows=[]
    for m in IDS:
      for d in DOMAINS:
        b=load(m,d);base=dict(np.load(cache_path(m,'M3',d)));c=dict(np.load(cache_path(m,'C1',d)))
        for hh,z in [('M2',base),('C1',c)]:
          out=ROOT/f'{m}-{d}-{hh}.json'
          if out.exists():
            r=json.loads(out.read_text())
            if r['protocol_sha']!=digest(PROTOCOL) or r['script_sha']!=digest(__file__) or digest(out.with_suffix('.npz'))!=r['prediction_sha']:raise ValueError('output drift')
            rows.append(r);continue
          start=time.monotonic();cpu=time.process_time();a,h=base['p_alive'],z['p_home'];pred=decode(a,h,b['frame'],b['period'],float(b['fps']))
          g=pm.label_codes(labels_from_batch(b));pc=pm.label_codes(pred);score=metrics(b,a,h,pred)
          half={str(i+1):pm.spell_scores(g,pc,[block]) for i,block in enumerate(continuous_blocks(b['frame'],b['period']))}
          save_prediction(out.with_suffix('.npz'),decoded=pred)
          r=dict(match=m,domain=d,alive_source='M2',home_source=hh,decoder='factorized_H1',metrics=score,per_half=half,prediction_sha=digest(out.with_suffix('.npz')),wall_cpu_memory=dict(wall_s=time.monotonic()-start,cpu_s=time.process_time()-cpu,max_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),**identity())
          write_new(out,r);rows.append(r);print(m,d,hh,round(score['full_spell_f1'],6),'Dmiss',score['dead_errors']['missed_dead_frames'],flush=True)
    transfer=[]
    for source in ['SC1874553','MetricaGame3']:
      spec=protocol['transfer'][source];b=dict(np.load(spec['batch']));base=dict(np.load(spec['M2']));c=dict(np.load(spec['C1']))
      for decoder in ['M3','factorized_H1']:
       for hh,z in [('M2',base),('C1',c)]:
        out=ROOT/f'{source}-{decoder}-{hh}.json'
        if out.exists():
            r=json.loads(out.read_text())
            if r['protocol_sha']!=digest(PROTOCOL) or r['script_sha']!=digest(__file__) or digest(out.with_suffix('.npz'))!=r['prediction_sha']:raise ValueError('output drift')
            transfer.append(r);continue
        a,h=base['p_alive'],z['p_home'];pred=original(a,h,b,'M3') if decoder=='M3' else decode(a,h,b['frame'],b['period'],float(b['fps']))
        truth=labels_from_batch(b);score=score_sequence(truth,pred,b['period'],float(b['fps']),semantics='partial_source_proxy')
        strata={}
        for name,mask in visibility_strata(b).items():
            k=mask&np.isin(truth,['H','A']);strata[name]=dict(known_alive=int(k.sum()),false_D=int(np.sum(pred[k]=='D')),wrong_HA=int(np.sum(np.isin(pred[k],['H','A'])&(pred[k]!=truth[k]))))
        save_prediction(out.with_suffix('.npz'),decoded=pred)
        r=dict(source=source,decoder=decoder,alive_source='M2',home_source=hh,full_spell_f1=None,metrics=score,strata=strata,
            false_D=int(np.sum((pred=='D')&np.isin(truth,['H','A']))),missed_weak_D=int(np.sum((truth=='D')&(pred!='D'))),known_live=int(np.isin(truth,['H','A']).sum()),known_weak_D=int(np.sum(truth=='D')),prediction_sha=digest(out.with_suffix('.npz')),waiver='Fixed WPY-heldout cache, research-only provider/view/fps transfer; labels partial/weak, no full spell or promotion',**identity())
        write_new(out,r);transfer.append(r);print(source,decoder,hh,'falseD',r['false_D'],'weakDmiss',r['missed_weak_D'],flush=True)
    if not (ROOT/'results.json').exists():write_new(ROOT/'results.json',dict(rows=rows,transfer=transfer,competition_promoted=False))
if __name__=='__main__':main()
