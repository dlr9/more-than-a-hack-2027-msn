# R7 输入、特征与证据合同

`FEATURE_PROVENANCE.csv`逐列记录R6的44列来源、单位、冻结风险和消融去留。R6 schema3及其模型保持原样；R7 schema4保留原始供应商合同，并添加特征视图及列名哈希。旧模型拒绝schema4，新模型拒绝其他视图、列序、provider、观测口径和fps。跨供应商诊断使用显式记录的waiver，未放宽生产候选接口。

- M4/M5删除5列：ball_speed_mps、ball_speed_missing和0.2/0.5/1秒球位移。剩余39列仍可能包含冻结球坐标造成的空间代理，不能称为彻底无捷径。
- M6额外删除双方local_speed_mean及三个窗口的双方d1_mean，共31列。保留瞬时几何与可见性；这不是另一个供应商等价性证明。
- 特征基础输入仅为原始球/球员坐标、球队、帧/半场，以及只用于差分连续性检查的tracklet ID。没有BallStatus、BallPossession、事件、phase、球高Z或供应商S速度作为输入。
- DFL原始位置以球场中心米计，位置差分为m/s；球场归一化坐标用各自长宽，距离用米。SkillCorner只使用is_detected=True的球员重新计算位置速度，缺失/换半场处不计算跨界速度。ID不进入特征。
- 球遮挡从原始坐标发生，所有球距离、差分、轨迹位移和窗口重新计算。位移要求窗口内连续可见；历史距离均值仍保留窗口内此前有效观测，这一点不是“缺球处完全清零”，也不会跨半场/帧缺口。
- 新短/长遮挡由period、fps、固定seed生成，与H/A/D、坐标及预测无关；两域各自保留mask、窗口和哈希在本地。完整源标签不变，不将人工遮挡称为真实目标缺失分布。

## 源级隔离

R7开发入口只接受固定三场ID和固定域名，验证R6原始/准备数据的SHA以及每半场独立parser报告，再核对match_id、provider和完整标签语义。不能通过参数读任意路径或目标GT。封存入口另有仅两场ID的固定公开源和独立核验；跨供应商入口另有固定SkillCorner原始文件和标签缓存SHA。未知标注禁止进入完整spell；不使用任何目标视频/ATD/五分钟GT。

完整spell调用未改动的官方 `evaluation/possession_metrics.py`，全半场边界分开。R7边界诊断使用官方支持的typed='pair'；R6历史调用typed='from_to'实际按目标状态匹配，现保留为legacy字段，不篡改旧报告。本修正只改变新边界诊断，不影响历史spell或模型。

## 原始数据解析差异

封存比赛J03WMX包含明确TeamId=referee对象。原parser正确拒绝了未知团队；现仅排除明确裁判对象并记录数量，其他未知/冲突球员仍报错。该结构修复发生在任何封存评分之前，并有合成回归测试；不会把裁判当成两队球员或改变标签定义。独立floodlight核验必须通过后才能评分。

JSON沿用旧接口键`partial_macro_f1`；在DFL完整源标签行该值覆盖全部半场帧，在SkillCorner行只覆盖已知标签。应与coverage、n_known及full_spell_status一起读取，不能把后者部分标签指标当完整分数。
