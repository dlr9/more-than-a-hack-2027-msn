"""Append R10 receipts and evaluator-only posthoc error categories, no selection."""
import json,subprocess,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from evaluation import possession_metrics as pm
from solution.idsse import digest
from solution.evidence import labels_from_batch
from solution.hierarchical_decoder import continuous_blocks
from scripts.research_r7 import load,write_new
from scripts.probe_r10_acquisition import hc_path,ha_transitions


def breakdown(row,root):
    b=load(row['match'],row['domain']);g=pm.label_codes(labels_from_batch(b));old=pm.label_codes(np.load(hc_path(row['match'],row['domain']))['decoded']);fps=float(b['fps'])
    z=np.load(root/f"{row['match']}-{row['domain']}.npz");out={}
    for name in z.files:
        values=z[name];stats={k:dict(old_missed=0,recovered=0,opposite_direction_only_nearby=0) for k in ['lt1','one_to3','ge3']}
        for bi,block in enumerate(continuous_blocks(b['frame'],b['period'])):
            truth=ha_transitions(g,block);_,miss,_=pm.match_transitions(truth,ha_transitions(old,block),round(fps),typed='pair')
            props=sorted((int(p[0]),1-int(p[1]),int(p[1])) for p in values if int(p[3])==bi)
            pairs,_,_=pm.match_transitions(truth,props,round(fps),typed='pair');matched={t for t,p in pairs}
            durations={a:(end-a)/fps for a,end,s in pm.segments(g,[block])}
            for t in miss:
                duration=durations[t[0]];key='lt1' if duration<1 else 'one_to3' if duration<3 else 'ge3';v=stats[key];v['old_missed']+=1;v['recovered']+=t in matched
                nearby=[p for p in props if abs(p[0]-t[0])<=round(fps)]
                v['opposite_direction_only_nearby']+=bool(nearby) and all(p[2]!=t[2] for p in nearby)
        out[name]=stats
    return dict(match=row['match'],domain=row['domain'],by_next_GT_spell_length=out)


def main():
    ledger=Path('docs/R10_V3_DECISIONS.jsonl');existing={json.loads(x)['id'] for x in ledger.read_text().splitlines()} if ledger.exists() else set();diagnostics=[]
    for exp in ['C1','C1FIX','C2']:
        root=Path('local_artifacts/r10-v3')/exp.lower();rp=root/'results.json';results=json.loads(rp.read_text());pp=Path(f'docs/R10_V3_PROTOCOL_{exp}.json');protocol=json.loads(pp.read_text())
        prediction_hashes={str(p):digest(p) for p in root.glob('*.npz')}
        decision='INVALID_EVALUATOR_ORDER_PRESERVED' if exp=='C1' else results['decision']['decision']
        item=dict(id='E10-'+exp,protocol_sha=digest(pp),code_sha=sorted({r['code_sha'] for r in results['rows']}),code_hashes=protocol['code_hashes'],
          hypothesis=protocol['hypothesis'],source='public IDSSE DFL; 3 already used matches x4 correlated domains; complete source HAD, not blind/transfer',
          input_hashes=protocol['inputs'],models=protocol['models'],proposal_hashes=prediction_hashes,results_sha=digest(rp),public_results=f'docs/R10_V3_RESULTS_{exp}.json',
          new_fits=0,new_weights=0,predicted_possessions_changed=False,decision=decision,rollback='unchanged HC_M3/M3/M2; separate human-owned official reference',
          actual_runs=12,baseline_metrics='48 freshly evaluated in C1FIX; copied with verified receipts in C2; invalid C1 boundary metrics not scientific evidence',
          cost=dict(wall_s=sum(r['wall_cpu_memory']['wall_s'] for r in results['rows']),cpu_s=sum(r['wall_cpu_memory']['cpu_s'] for r in results['rows']),peak_rss_bytes_mac=max(r['wall_cpu_memory']['max_rss'] for r in results['rows']),scope='row processing; excludes upfront hash checks and raw loading'),
          failure='Ranked proposals passed unsorted to chronological matcher' if exp=='C1' else 'Insufficient missed-boundary gain versus strong/no-information and negative controls; do not fit a hazard on failed proposals',
          next='independent action observations and compatible labels required; no more same-match threshold searches')
        if item['id'] not in existing:
            with ledger.open('a') as f:f.write(json.dumps(item)+'\n')
        if exp!='C1':
            diagnostics.append(dict(experiment=exp,rows=[breakdown(row,root) for row in results['rows']]))
    diag=dict(status='POSTHOC_EVALUATOR_ONLY_NOT_A_SELECTION_OR_TUNING_PROTOCOL',purpose='Explain failure by next true spell duration and wrong-direction nearby signals; never export GT arrays',experiments=diagnostics,source_sha=digest(__file__))
    write_new('docs/R10_V3_ERROR_CATEGORIES.json',diag)
    models=json.loads(Path('docs/R9_V2_MODEL_REGISTRY.json').read_text())['inherited_model_hashes']
    for p,h in models.items():
        if digest(p)!=h:raise ValueError('model drift')
    write_new('docs/R10_V3_MODEL_REGISTRY.json',dict(inherited_base='9f701a4f193d776515a6c2129d5809b6e736b8b2',inherited_model_hashes=models,new_fits=0,new_weights=0,new_deployable_models=0,
      candidates={'C1':'INVALID_EVALUATOR_ORDER','C1FIX':'REVERT_PROPOSAL_ROUTE','C2':'REVERT_PROPOSAL_ROUTE','T0':'KEEP_CODE_ONLY_INTERFACE'},competition_promoted=False,transfer='TRANSFER_NOT_ESTABLISHED'))
    lines=['# R10 V3 实际结果','', '合法其他比赛，三场均已反复开发；新训练0次。两条探针未达信息门槛，不生成新的possession预测，也不凭片段压缩制造提升。C1初版匹配顺序无效，只保留失败；以下仅C1FIX/C2。','',
      '|探针|比赛|域|原HC漏换手|旧概率最佳找回|新信号找回|打乱队伍找回|平移时间找回|新信号precision|','|---|---|---|---:|---:|---:|---:|---:|---:|']
    for exp in ['C1FIX','C2']:
        r=json.loads(Path(f'docs/R10_V3_RESULTS_{exp}.json').read_text())
        for row in r['rows']:
            e=row['equal_budget'];g=e['geometry'];best=max(e[k]['recovered_old_missed'] for k in ['M2prob','C1prob','E1fix'])
            lines.append(f"|{exp}|{row['match']}|{row['domain']}|{g['old_missed']}|{best}|{g['recovered_old_missed']}|{e['shuffled_team']['recovered_old_missed']}|{e['shifted_time']['recovered_old_missed']}|{g['precision']:.4f}|")
    lines+=['','每场每个信号240个提议（两半各120），相同预算/后处理。旧概率对照也用相同future算子；E1沿用旧物理事件。单次打乱/30s循环平移有自相关，不能解释为显著性检验。','',
      'C1FIX：未来接近关系比旧E1容易覆盖边界，但相对强旧概率的自然增益仅5.29/2.67/1.24百分点；WPY打乱球队29比真信息28还多。原始距离的时序重表达不是独立传感源。两个未来窗要求持久控制，对短控球有结构偏差；遮球时0证据惩罚整窗，不代表所有物理信息无用。','',
      'C2：改变为原始球速度向量突变的动作通道，没有要求后续长控球。自然找回21/15/14，最佳旧信号17/18/14；总提议精度仅15.8/14.6/13.3%，显著低于旧概率36.3/30.8/39.2%。触球/踢球/偏转并非可控换手。两个冻结门槛均失败，停止训练这些提议的hazard；没有再搜阈值。','',
      '短spell及方向反例见ERROR_CATEGORIES（评价器事后分层，不反向修改算法）；完整旧基线结果如下。候选没有改变D或H/A输出，故不得把提议找回当作实际spell提升。', '',
      '|比赛|域|基线|spellF1|macroF1|TP/FP/FN @.5|短spell TP/GT @.5|假D切断|漏D帧/整段|', '|---|---|---|---:|---:|---|---|---:|---|']
    r=json.loads(Path('docs/R10_V3_RESULTS_C1FIX.json').read_text())
    for row in r['rows']:
        for v,m in row['baseline'].items():
            s=m['spell_by_iou']['0.5'];short=m['short_spell_recall']['lt1@0.5'];dead=m['dead_errors']
            lines.append(f"|{row['match']}|{row['domain']}|{v}|{m['full_spell_f1']:.6f}|{m['partial_macro_f1']:.6f}|{s['n_matched']}/{s['n_spurious']}/{s['n_missed']}|{short['n_matched']}/{short['n_gt']}|{dead['false_cuts']}|{dead['missed_dead_frames']}/{dead['missed_segments']}|")
    lines+=['','完整5IoU TP/FP/FN、D、短spell、宏观与换手指标在RESULTS；C2复用相同哈希的基线成绩，并非额外独立验证。15个历史模型hash保持。没有目标资料、五分钟GT或封存新比赛参与。','',
      '后续独立信息路线已核查：B自动战术标签仅完成安全schema迁移，缺公开其他比赛同构自动tag和独立完整HAD，BLOCKED_DATA；Metrica事件不能同时当特征及自己的验证真值；ELASTIC纠正事件标注不是自动可部署tag，不下载以伪造增益。当前盘余量约200–300MB，不下载大型视频或重下20场SkillCorner。下一步由C确认已存合法外部事件/追踪缓存，可用后另冻协议；正式提交由A/B独立完成。']
    Path('docs/R10_V3_FINDINGS.md').write_text('\n'.join(lines)+'\n')
    print('Recorded 3 runs (one invalid), 24 valid probes, 48 full baseline scores, posthoc categories; zero fits')
if __name__=='__main__':main()
