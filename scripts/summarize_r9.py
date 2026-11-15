"""Publish bounded aggregate R9 results and append-only experiment receipts."""
import json,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from solution.idsse import digest
from scripts.research_r7 import write_new
ROOT=Path('local_artifacts/r9-v2')
EXPS=['e09d','e09e1','e09e1fix','e09e2','e09a','e09a2']
DECISIONS={
'e09d':'KEEP_DIAGNOSTIC: HA dominates; sparse correction can fragment spells; oracle not deployable',
'e09e1':'INVALID_INPUT_CONTRACT: reversed home/away encoding; preserve all 24 outputs, no method inference',
'e09e1fix':'REVERT_AS_REPLACEMENT: small mixed timing changes, WPY consistently worse',
'e09e2':'REVERT_AS_REPLACEMENT: new short segments did not improve natural; stop event weight search',
'e09a':'KEEP_RESEARCH_COMPONENT HC_M3 (M2 Alive + C1 Home + M3); C1 Alive REVERT; not universal replacement for H1',
'e09a2':'REVERT_FACTORISATION: Metrica missed weak D 25380/25619; HC_M3 retained as research only'}

def compact(m):
    keys=['full_spell_f1','partial_macro_f1','dead_errors','boundaries_pair_1s','spell_by_iou','spell_lengths','short_spell_recall','confusion']
    out={k:m[k] for k in keys}
    out['probabilities']={k:{j:v for j,v in x.items() if j!='reliability'} for k,x in m['probabilities'].items()}
    out['strata']={k:{j:v for j,v in x.items() if j in ['n','n_known','n_gt_dead','false_dead','missed_dead','home_head_wrong']} for k,x in m['strata'].items()}
    return out

def main():
    reports={};summary=[];ledger=Path('docs/R9_V2_EXPERIMENT_LEDGER.jsonl')
    existing={json.loads(s)['experiment_id'] for s in ledger.read_text().splitlines()} if ledger.exists() else set()
    for exp in EXPS:
        source=ROOT/exp/'results.json';data=json.loads(source.read_text());protocol=Path(f'docs/R9_V2_{exp.upper()}_PROTOCOL.json');p=json.loads(protocol.read_text())
        public=[]
        for r in data['rows']:
            if exp=='e09d':public.append(r);continue
            rr={k:v for k,v in r.items() if k!='metrics'};rr['metrics']=compact(r['metrics']);public.append(rr)
            summary.append(dict(experiment=exp,match=r['match'],domain=r['domain'],variant=r.get('parent',r.get('alive_source','')+'A-'+r.get('home_source','')+'H'),
                spell=r['metrics']['full_spell_f1'],macro=r['metrics']['partial_macro_f1'],false_cuts=r['metrics']['dead_errors']['false_cuts'],missed_D=r['metrics']['dead_errors']['missed_dead_frames'],missed_D_segments=r['metrics']['dead_errors']['missed_segments'],HA_spurious=r['metrics']['boundaries_pair_1s']['ha_switch']['n_spurious']))
        target=Path(f'docs/R9_V2_{exp.upper()}_RESULTS.json')
        if not target.exists():write_new(target,dict(rows=public,transfer=data.get('transfer',[]),decision=DECISIONS[exp],source_results_sha=digest(source),competition_promoted=False))
        reports[exp]=str(target)
        predictions={str(q):digest(q) for q in sorted((ROOT/exp).glob('*.npz'))}
        if exp not in existing:
            r0=data['rows'][0];cost=[r.get('wall_cpu_memory',{}) for r in data['rows']]
            row=dict(experiment_id=exp,parent_checkpoint=p.get('parent_checkpoint'),hypothesis=p['hypothesis'],protocol_sha=digest(protocol),code_sha=sorted({r['code_sha'] for r in data['rows']}),source_provenance='Public other matches only; pinned R7_DATA_PROVENANCE / R7B_MODEL_CARDS; no target/official GT; no new dataset downloaded',data_hashes=p['inputs'],label_semantics='DFL_source_complete; SC partial known HA / Metrica weak event proxy only in A2',exact_model_hashes=p['models'],saved_prediction_hashes=predictions,fits=0,sample_units='3 repeated shared-home matches x2 halves; 4 correlated observation domains; no frame IID; SC1match/4zero-observation episodes',
                wall_cpu_memory=dict(row_wall_s=sum(c.get('wall_s',0) for c in cost),row_cpu_s=sum(c.get('cpu_s',0) for c in cost),peak_row_rss_bytes_mac=max([c.get('max_rss',0) for c in cost]),scope='per-row decode/score only; excludes raw load/proposal extraction and reused rows; total process wall not measured'),per_match_metrics=str(target),counterexamples=DECISIONS[exp],decision=DECISIONS[exp],fallback='Untouched frozen M3/M2/R6-M1; official offline reference chosen by authorized human',next_no_retrain_command='OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python scripts/verify_r9_artifacts.py')
            with ledger.open('a') as f:f.write(json.dumps(row)+'\n')
    # Failures are explicit, not silently removed after repairs.
    failures=[('r9-freeze-helper-failure','Initial metadata freeze helper assumed registry new_models; actual key models. KeyError before scoring; corrected, model hash15 verified; no old artifacts modified.'),
      ('e09e2-synthetic-fixture-failure','Initial short-control fixture expected endpoints fixed while event +/-1s window covered whole clip. One assertion failed; shell sequence incorrectly continued to commit/evaluate. Preserved logs, enlarged fixture to put asserted endpoints outside window; added no algorithm changes. Full152 tests subsequently passed; never label initial preflight as passed.')]
    for eid,why in failures:
        if eid not in existing:
            with ledger.open('a') as f:f.write(json.dumps(dict(experiment_id=eid,parent_checkpoint='see frozen experiment commits',hypothesis='recovery integrity',protocol_sha=None,code_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_provenance='synthetic/code only',data_hashes={},label_semantics='none',exact_model_hashes={},saved_prediction_hashes={},fits=0,sample_units='none',wall_cpu_memory=None,per_match_metrics=None,counterexamples=why,decision='FIXED_WITH_HISTORY_PRESERVED',fallback='frozen historical checkpoint',next_no_retrain_command='.venv/bin/python -m unittest discover -s tests -v'))+'\n')
    if not Path('docs/R9_V2_SUMMARY.json').exists():write_new('docs/R9_V2_SUMMARY.json',dict(rows=summary,reports=reports))
    if not Path('docs/R9_V2_DECISIONS.json').exists():write_new('docs/R9_V2_DECISIONS.json',dict(experiments=DECISIONS,Gate_R='HC_M3 conditional evidence vs M3, but not a uniform improvement vs H1; keep Pareto research component',Gate_T='TRANSFER_NOT_ESTABLISHED',Gate_S='HUMAN_ONLY_UNKNOWN',competition_promoted=False,stop='One event mechanism revision and one head decoder followup completed; no further same-match tuning. Need independent complete crossvendor labels / external event boundary supervision.'))
    if not Path('docs/R9_V2_MODEL_REGISTRY.json').exists():write_new('docs/R9_V2_MODEL_REGISTRY.json',dict(new_fits=0,new_weight_files=0,inherited_model_hashes=p['models'],candidates={k:dict(decision=v,prediction_receipts=reports[k]) for k,v in DECISIONS.items()},scorer_sha=p['scorer_sha'],competition_promoted=False))
    lines=['# R9 V2 实际实验结果','', '三场公开其他比赛，共享主队、反复使用的开发比赛；不是新盲测。新增拟合0次。无目标数据或官方5分钟GT。','', '|实验|比赛|域|版本|spellF1|macroF1|假D切断|漏真D帧|完整漏D段|伪HA边界|','|---|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        if r['experiment']=='e09e1':continue
        lines.append(f"|{r['experiment']}|{r['match']}|{r['domain']}|{r['variant']}|{r['spell']:.6f}|{r['macro']:.6f}|{r['false_cuts']}|{r['missed_D']}|{r['missed_D_segments']}|{r['HA_spurious']}|")
    lines+=['','完整5IoU匹配、半场结果、短spell、概率指标及哈希见对应RESULTS。E1错误合同版本独立保留，不与有效候选排名。Oracle只保留汇总，未保存真值修正标签。']
    target=Path('docs/R9_V2_FINDINGS.md')
    if not target.exists():target.write_text('\n'.join(lines)+'\n')
    print('Summarized',len(summary),'candidate/reference rows; append ledger, no refits/no overwrites')
if __name__=='__main__':main()
