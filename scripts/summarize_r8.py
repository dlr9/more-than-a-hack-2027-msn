"""Generate immutable R8 aggregate handoff from completed receipts, not estimates."""
import json,subprocess,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.research_r7 import IDS,write_new
from solution.idsse import digest

def text_new(path,text):
    with Path(path).open('x')as f:f.write(text)

def main():
    root=Path('local_artifacts/r8')
    a=json.loads((root/'e08a/results.json').read_text());fix=json.loads((root/'e08a-anchor-fix/results.json').read_text())
    b=json.loads((root/'e08b/results.json').read_text());c=json.loads((root/'e08c/results.json').read_text())
    records=[]
    for r in a['rows']:
        if r['variant']=='H2':continue
        records.append(dict(match=r['match'],domain=r['domain'],variant=r['variant'],metrics=r['metrics']['decoded']))
    for r in fix['rows']:records.append(dict(match=r['match'],domain=r['domain'],variant='H2_fixed',metrics=r['metrics']['decoded']))
    for result in (b,c):
        for r in result['rows']:records.append(dict(match=r['match'],domain=r['domain'],variant=r['variant'],metrics=r['metrics']))
    for file in ['docs/R7_CORE_RESULTS.json','docs/R7_STRESS_RESULTS.json']:
        for r in json.loads(Path(file).read_text())['rows']:
            if r['variant']=='M1':records.append(dict(match=r['match'],domain=r['domain'],variant='M1_frozen_reference',metrics=r['metrics']['decoded']))
    means={v:{d:float(np.mean([r['metrics']['full_spell_f1']for r in records if r['variant']==v and r['domain']==d]))for d in ['natural','dropout','short','long']}for v in sorted(set(r['variant']for r in records))}
    models=[json.loads((root/'e08c'/f'{m}.json').read_text())for m in IDS]
    for model in models:assert digest(model['path'])==model['model_sha256']
    registry=dict(new_fits=6,new_bundles=3,models=models,inherited_models=json.loads(Path('docs/R8_PROTOCOL.json').read_text())['models'],
        no_fit_candidates={'H1':'frozen G1 emissions + binary factors','H2_fixed':'same G1 + bounded observable priors','B1':'frozen M2, reconstructed raw short gaps'},
        competition_promoted=False,scorer_sha256=digest('evaluation/possession_metrics.py'))
    write_new('docs/R8_MODEL_REGISTRY.json',registry)
    write_new('docs/R8_SUMMARY.json',dict(means=means,rows=records,decisions={'H1':a['decisions']['H1'],'H2_fixed':fix['decisions']['H2'],'B1':b['decision'],'C1':c['decision']},
        result_sha256={stage:digest(root/stage/'results.json')for stage in ['e08a','e08a-anchor-fix','e08b','e08c']},competition_promoted=False))
    lines=['# R8 实测闭环与决策\n',
        '仅公开其他比赛。三场DFL为反复使用的开发场且共享主队；LOMO避免训练比赛泄漏，但不是新盲测。源完整H/A/D不等同赛事定义。所有候选 competition_promoted=false；未打开WOY/WR9，未重用WMX/WN1作新盲测。\n',
        '## 完整外部 spellF1 均值\n','|候选|自然|原遮球|短遮球|长遮球|','|---|---:|---:|---:|---:|']
    for v,ds in means.items():lines.append('|'+v+'|'+ '|'.join(f'{ds[d]:.6f}'for d in ['natural','dropout','short','long'])+'|')
    lines += ['\n## 每场完整指标\n','|比赛|域|候选|spellF1|macroF1|假D切断|漏真D帧|完全漏D段|伪H/A转换±1s|','|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in sorted(records,key=lambda x:(x['match'],x['domain'],x['variant'])):
        q=r['metrics'];d=q['dead_errors'];ha=q['boundaries_pair_1s']['ha_switch']
        lines.append(f"|{r['match']}|{r['domain']}|{r['variant']}|{q['full_spell_f1']:.6f}|{q['partial_macro_f1']:.6f}|{d['false_cuts']}|{d['missed_dead_frames']}|{d['missed_segments']}|{ha['n_spurious']}|")
    lines += ['\n完整5个IoU、每类F1、短控球召回、预测控球数量/长度、原始argmax与解码、头部概率与可观测分层均保存在各RESULTS JSON。±1秒是逐半场typed pair匹配；±.48秒总计另存。M1为冻结历史参照，其余H0三模型36组解码在本轮重放；没有把历史M1数值称为新训练。\n',
        '## E08A：分层与锚点\n',
        '假设：联合H/A概率分摊导致部分q>.5仍判D，分层可减少此结构偏置。协议8086ec6先于评分；60组完整25Hz评分，0次拟合。H1原遮球获益，但短遮球不稳定超过M3，自然增益不足；SC195个零观测已知H/A仍全D，Metrica弱D漏检3891→4967。REVERT替代资格，保留诊断代码。\n',
        '首版H2触发0帧，是恢复首帧后向速度NaN导致右端锚点不可达的工程缺陷，不能据此判断锚点方法无效。保留首版结果，单独冻结修复协议，新增回归测试，仅用相邻可见样本的前向速度修复，所有阈值不变。修复版12组完整重测共使用'+str(sum(r['diagnostics']['used']for r in fix['rows']))+'帧软锚点，收益仍不足。两个protocol SHA区分初版和修复版账本记录。\n',
        '## E08B：真实原始轨迹短缺球重建\n',
        '假设：<=1秒且两队球员可见、有连续控制几何的缺口可保守补充位置。先隐藏既有标签盲mask内的真实球坐标，只根据两端位置/速度/加速度/球员支持与同一轨迹提出候选，再评分重建误差；之后重算全部44维，固定M2两个头和旧decoder。独立imputed mask保留，生产兼容闸门不放宽；研发显式记录view waiver，评分分层仍按原始可见性。\n',
        '|比赛|域|插补帧/原缺球帧|平均误差m|P95误差m|错误最近队伍锚点/可测帧|','|---|---|---:|---:|---:|---:|']
    for m in IDS:
        for d in ['short','dropout','long']:
            r=json.loads((root/'e08b'/f'{m}-{d}-reconstruction.json').read_text());e=r['error_m']
            lines.append(f"|{m}|{d}|{r['imputed_frames']}/{r['missing_frames']}|{e['mean']}|{e['p95']}|{r['wrong_geometric_control']}/{r['control_diagnostic_denominator']}|")
    lines += ['\n最近队伍误差是几何代理，不是独立有效控制真值。B1短遮球改善但未超过强对照门槛，长缺口全部拒绝；REVERT替代资格，回退未经插补M2。\n',
        '## E08C：等预算真实监督重训\n',
        '训练标签诊断发现各训练折17.5%–21.3%的<1秒控球片段没有2Hz样本。新协议固定一半均匀样本、一半±.48秒边界附近样本，不放回且总量逐头/逐源与M2完全相等；仍用原44维、60轮15叶HGB和原解码，3个LOMO包共6次实际拟合。训练时没有读取heldout标签选样，没有按验证排名改seed/比例。\n',
        '|留出比赛|模型SHA256|两头拟合和加载秒数|','|---|---|---:|']
    for r in models:lines.append(f"|{r['heldout']}|{r['model_sha256']}|{r['wall_seconds']:.3f}|")
    lines += ['\nC1决定：'+c['decision']+'。自然域改善不能抵消遮球时漏真D增加；完整对照和门槛未修改。边界抽样改变了训练状态分布，缺球无运动证据时Alive更偏高；这只是本轮错误归因，不把它当独立校准成功。回退M2/M3原件。\n',
        '## 跨供应商风险（没有完整spell资格）\n','|来源|候选|已知H/A假D帧|漏弱D帧|','|---|---|---:|---:|']
    for r in fix['transfer']:lines.append(f"|{r['source']}|{r['variant']}|{r['false_dead']}|{r['missed_weak_dead']}|")
    for r in c['transfer']:lines.append(f"|{r['source']}|C1|{r['false_dead']}|{r['missed_weak_dead']}|")
    lines += ['\nSC只评价已知H/A，不能检验真实D；Metrica的D是事件弱代理，full spell保持null。没有模型通过完整R/T门槛，不打开封存数据。\n',
        '## 下一步与回退\n',
        '保留全部源码、权重、概率、失败、协议和哈希。当前没有足够证据证明加duration模型会解决漏D，E08D不启动；E08E是可选泛化接口，已有隔离离线入口满足本轮交付保障，不接触真实Smart Tagging。优先A/C完成正式资产与截止时区的人类离线确认，B冻结这一轮，不继续在同三场和195帧上搜索。下一研究需新的完整跨供应商观测/监督支持或独立标注的缺球边界证据。\n']
    text_new('docs/R8_FINDINGS_AND_DECISIONS.md','\n'.join(lines)+'\n')
    text_new('docs/R8_MODEL_CARDS.md',f'''# R8 模型卡

所有候选未通过正式比赛晋级，不自动接入离线runner。代码/协议/模型/预测哈希见R8_MODEL_REGISTRY、各RESULTS及追加账本。外部DFL/SC/Metrica来源许可及固定版本沿用R7_DATA_PROVENANCE与R7B_MODEL_CARDS；无目标素材或官方5分钟GT使用。

|版本|已证实优势|缺点/适用域|决定|回退|
|---|---|---|---|---|
|H1|原遮球mean spell {means['H1']['dropout']:.6f}，减少部分假D切断|自然基本不变；短遮球低于M3；SC零观测仍D，Metrica弱D漏检增加|REVERT替代；KEEP分层诊断代码|G1/M2/M3冻结概率与旧decoder|
|H2首版|发现速度可达性bug|0帧锚点，不能解释为算法因果反证|保留失败记录，修复后另测|原H1|
|H2_fixed|触发真实观测软锚点，严格拒绝长桥接|平均收益微小；同样不能识别长零观测内隐藏D|REVERT替代|原H1/G1|
|B1|短遮球mean spell {means['B1']['short']:.6f}，支持原始短轨迹重建|只覆盖少量有球员支持缺口；MSE不等于有效控制；未超强对照|REVERT替代；KEEP可审查物理拒绝模块|未插补M2|
|C1|真实6次HGB；自然mean spell {means['C1']['natural']:.6f}|重采样增加缺球时漏真D；不适合无运动证据域|{c['decision']}|冻结M2/M3，R6-M1研究回退|

原R6-M1、M2/M3、V3、R7B P1/G1与所有失败模型继续保留，不改变原卡片历史。三场开发共享主队、被重复评估；SC4个相关episode及Metrica弱标签都不足以证明泛化。新C1固定WPY留出折迁移，未按结果选折；显式研发兼容waiver不改变生产默认拒绝。

A：审阅Draft增量、远端SHA/CI，保留PR2/R7B双方历史，人工决定合并。B：不继续盲调M3、记忆期限或新采样比例；复核每场D守门与短控球；新数据支持后再立协议。C：仅由授权成员在非Codex、无云同步且断网环境检查Tactical ATD，运行官方参考与199319行格式验证，核实11/17邮件时区。正式提交/回执由人办理，不上传输入、目标预测或私人报告给Codex。
''')

if __name__=='__main__':main()
