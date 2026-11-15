"""One implementation correction, exclusive outputs; no threshold reselection."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import scripts.evaluate_r8_decoder as run
from scripts.research_r7 import IDS,load,write_new
from solution.idsse import digest

def main():
    run.ROOT=Path('local_artifacts/r8/e08a-anchor-fix')
    run.PROTOCOL=Path('docs/R8_E08A_ANCHOR_FIX_PROTOCOL.json')
    protocol=json.loads(run.PROTOCOL.read_text());run.ROOT.mkdir(parents=True,exist_ok=True)
    if (run.ROOT/'results.json').exists():raise FileExistsError('preserve corrected-anchor results')
    for path,sha in protocol['inputs'].items():
        if digest(path)!=sha:raise ValueError('input drift '+path)
    old=json.loads(Path('local_artifacts/r8/e08a/results.json').read_text())
    rows=[]
    for m in IDS:
        for d in run.DOMAINS:
            b=load(m,d);cache={v:run.checked(run.prediction_path(m,v,d),protocol)for v in ('M2','M3','G1')}
            rows.append(run.run_row(m,d,'H2',b,cache,protocol))
    paired=[r for r in old['rows']if r['variant']!='H2']+rows
    result=dict(experiment_id='E08A_anchor_implementation_fix',rows=rows,decisions=run.decisions(paired),transfer=run.transfer(protocol),
        original_results_sha256=digest('local_artifacts/r8/e08a/results.json'),wrapper_sha256=digest(__file__),**run.identity())
    write_new(run.ROOT/'results.json',result);write_new('docs/R8_E08A_ANCHOR_FIX_RESULTS.json',result)
    run.append_ledger(dict(experiment_id='E08A_anchor_fix_decision',hypothesis=protocol['amendment'],source_match='public other-match DFL/SC/Metrica',
        train_matches=list(IDS),eval_matches=list(IDS),model_sha256=protocol['models'],prediction_sha256={f"{r['match']}/{r['domain']}":r['prediction_sha256']for r in rows},
        code_sha=result['code_sha'],protocol_sha=result['protocol_sha'],wall_seconds=sum(r['wall_seconds']for r in rows),cpu_memory='per-row receipts',metrics_by_match='R8_E08A_ANCHOR_FIX_RESULTS.json',
        decision=result['decisions']['H2'],revert_path='original G1/M2/M3; preserve first H2 failure',next='E08C only if training coverage diagnostic supports it'))

if __name__=='__main__':main()
