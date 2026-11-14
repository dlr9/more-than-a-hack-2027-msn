"""One preregistered missing-observation transition-cost ablation, no refitting."""
import datetime,hashlib,json,subprocess,sys
from pathlib import Path
import joblib,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from solution.modeling import load_prepared
from solution.decoder import emission_probabilities,viterbi_decode
from solution.quality import period_blocks
from solution.idsse_model import MissingStableCandidate
from scripts.evaluate_r6 import evaluate

ROOT=Path('local_artifacts/r6/idsse-m1-v1');OUT=Path('local_artifacts/r6/missing-stability-v1');IDS=['J03WPY','J03WOH','J03WQQ']
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    if OUT.exists():raise FileExistsError('preserve prior run')
    OUT.mkdir(parents=True);rows=[];hashes={}
    previous=json.loads((ROOT/'results.json').read_text())
    for m in IDS:
        old=joblib.load(ROOT/f'{m}-M2_dropout.joblib')
        new=MissingStableCandidate(old.alive,old.home,old.feature_names,{**old.metadata,'model_version':'R6-M3-missing-stability-experimental'})
        for domain in ['natural','dropout']:
            batch=load_prepared(f'local_artifacts/r6/prepared/{m}/{domain}.npz')
            path=ROOT/f'{m}-M2_dropout-{domain}.npz';hashes[str(path)]=digest(path)
            with np.load(path)as data:a,h,baseline=data['p_alive'],data['p_home'],data['decoded']
            visible=batch['X'][:,batch['feature_names'].tolist().index('ball_visible')]==1
            scale=np.where(visible&np.r_[True,visible[:-1]],1.,4.)
            decoded=viterbi_decode(emission_probabilities(a,h),period_blocks(batch['period']),7.5,2.5,dead_scale=scale)
            if domain=='natural':assert np.array_equal(decoded,baseline)
            oldscore=next(r['diagnostics']for r in previous['rows']if r['heldout']==m and r['variant']=='M2_dropout'and r['domain']==domain)
            score=evaluate(batch,a,h,decoded,crosschecked=True)
            rows.append(dict(match=m,domain=domain,baseline=oldscore,modified=score,labels_changed=int(np.sum(decoded!=baseline))))
            np.savez_compressed(OUT/f'{m}-{domain}.npz',decoded=decoded)
        joblib.dump(new,OUT/f'{m}.joblib')
        print('decoded',m,flush=True)
    stress=[r for r in rows if r['domain']=='dropout'];delta=[r['modified']['full_spell_f1']-r['baseline']['full_spell_f1']for r in stress]
    cuts=[r['modified']['source_proxy_fragmentation']['false_dead_cuts']-r['baseline']['source_proxy_fragmentation']['false_dead_cuts']for r in stress]
    keep=bool(sum(d>0 for d in delta)>=2 and np.mean(delta)>=.01 and max(cuts)<=0)
    result=dict(run_id='R6-M3-missing-stability-v1',git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),protocol_sha256=digest('docs/R6_DECODER_PROTOCOL.json'),input_hashes=hashes,rows=rows,decision='EXPERIMENTAL'if keep else'REVERT',spell_delta=delta,false_dead_cut_delta=cuts,competition_promoted=False)
    for p in [OUT/'results.json',Path('docs/R6_DECODER_RESULTS.json')]:p.write_text(json.dumps(result,separators=(',',':'))+'\n')
    with Path('docs/R6_EXPERIMENT_LEDGER.jsonl').open('a')as f:f.write(json.dumps({k:v for k,v in result.items()if k!='rows'}|dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),command='.venv/bin/python scripts/ablate_r6_decoder.py',next_todo='freeze research artifacts and external-only reproducibility package; no further cost search on these folds'))+'\n')
    print(result['decision'],delta,cuts)
if __name__=='__main__':main()
