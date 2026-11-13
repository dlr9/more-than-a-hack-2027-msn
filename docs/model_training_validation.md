# 双模型训练与验证报告

> 训练日期：2026-11-13  
> 数据版本标签：`external-training-samples-v1`  
> 训练代码 commit：`dc74509e35dc2bc5ebae11b5522f8af1035aecc7`  
> 随机种子：`2027`

## 1. 结论

`model_v1` 通过预先约定的全部外部晋级门槛，模型元数据为 `promoted=true`：

- SkillCorner validation 平均 spellF1 从 `0.2798` 提升到 `0.5059`，提升 `0.2261`。
- 3/3 validation 比赛均未下降，满足至少 70% 不下降的要求。
- Metrica Game 3 高置信 H/A macroF1 从 `0.6118` 提升到 `0.6456`，提升 `0.0338`。
- 晋级后首次且仅一次读取 SkillCorner test：平均 spellF1 从 `0.3001` 提升到 `0.4886`，3/3 比赛均提升。
- 目标比赛候选通过官方 validator，重复生成两次 SHA256 完全相同。

按照冻结的外部选择规则，`model_v1` 可以进入最终候选。公开五分钟未参与训练、调参或版本选择。

## 2. 训练数据

- SkillCorner train：14 场。
- SkillCorner validation：3 场。
- SkillCorner test：3 场。
- alive/dead 训练帧：train 中 alive `328,121`、dead `353,896`。
- H/A 训练帧：train 中 H `129,675`、A `128,553`。
- 共同特征：44 个。
- Metrica Game 3：只用于跨供应商晋级检查，不参与模型拟合。

严格训练标签为 `y_alive/y_home`；SkillCorner phase 构造的完整弱监督时间线 `eval_y_alive/eval_y_home` 只用于粗粒度 spell/Viterbi 验证。详细清洗规则见 `docs/external_data_audit.md`。

## 3. 概率模型选择

两个模型均为 `HistGradientBoostingClassifier`，使用逆类别频率 sample weight、early stopping 和固定种子。

### 3.1 Alive/dead 模型

- balanced accuracy：`0.941686`
- log loss：`0.186500`
- max_leaf_nodes：`15`
- learning_rate：`0.1`
- l2_regularization：`0.0`
- max_iter 上限：`300`

### 3.2 Home/away 模型

- balanced accuracy：`0.929729`
- log loss：`0.165535`
- max_leaf_nodes：`31`
- learning_rate：`0.1`
- l2_regularization：`0.0`
- max_iter 上限：`300`

## 4. Viterbi 选择

选中的参数：

```text
H↔A penalty   = 5.0
H/A↔D penalty = 3.0
```

在三场 validation 上：

- 平均 spellF1：`0.505867`
- 平均 macroF1：`0.861351`
- 平均预测 segments：`583.0`

需要注意，最优参数位于预设网格上界 `(5, 3)`，并且 spellF1 随惩罚总体上升，说明外部验证仍偏好更强的时序平滑。本轮没有扩大网格：原执行计划已经冻结该搜索范围，扩大范围会构成新的模型选择实验。

## 5. SkillCorner validation

| match | baseline spellF1 | model spellF1 | baseline macroF1 | model macroF1 |
|---:|---:|---:|---:|---:|
| 1925299 | 0.3201 | 0.5279 | 0.6264 | 0.8806 |
| 2013725 | 0.2803 | 0.4938 | 0.5193 | 0.8457 |
| 2017461 | 0.2389 | 0.4959 | 0.5868 | 0.8577 |
| **mean** | **0.2798** | **0.5059** | **0.5775** | **0.8614** |

晋级条件：

- 平均 spellF1 提升至少 0.02：**通过**，实际提升 `0.2261`。
- 至少 70% 比赛不下降：**通过**，实际为 `100%`。

## 6. Metrica 跨供应商检查

- 高置信 H/A 帧：`72,606`。
- nearest-player + smooth50 H/A macroF1：`0.611810`。
- 模型 H/A macroF1：`0.645642`。
- 差值：`+0.033833`。
- 允许最低差值：`-0.02`。
- 结果：**通过**。

这项结果说明模型没有只记住 SkillCorner 的坐标或标签特征；在另一供应商的高置信 PASS/CARRY 内部帧上仍有正向提升。

## 7. SkillCorner test（晋级后首次读取）

| match | baseline spellF1 | model spellF1 | baseline macroF1 | model macroF1 |
|---:|---:|---:|---:|---:|
| 1996435 | 0.3498 | 0.5193 | 0.6013 | 0.8895 |
| 2011166 | 0.2842 | 0.4808 | 0.6129 | 0.8683 |
| 2015213 | 0.2662 | 0.4657 | 0.5002 | 0.8444 |
| **mean** | **0.3001** | **0.4886** | **0.5715** | **0.8674** |

3/3 test 比赛均提升。test 结果只用于最终一次独立确认，不再用于新的参数选择。

## 8. 目标比赛候选

文件：`evaluation/results/model_v1.csv`

标签数量：

```text
H = 46,142
A = 30,389
D = 82,326
X = 40,462
```

- 总行数：`199,319`。
- frame：`0..199318` 连续。
- validator：`OK`。
- 报告 possession 数：`699`，平均 `4.4 s`。
- 连续运行两次 SHA256 相同。

候选 SHA256：

```text
2790393ff3260fbf6350b349a94ee8295e9038c5c26caad46dec13a0f0077905
```

## 9. 公开五分钟：一次性管线检查

该检查在模型、Viterbi 参数和晋级结论全部冻结后执行，只评分 `model_v1`，没有与其他候选一起进行公开集排序或选参。

结果：

- macroF1：`0.759681`
- spellF1_mean：`0.253333`
- H F1：`0.848712`
- A F1：`0.722743`
- D F1：`0.707590`
- alive/dead F1：`0.917301`
- 预测 home share 误差：`0.003651`
- GT possessions：`17`
- 预测 possessions：`43`
- fragmentation：`3.0`

解释：帧级分类、alive/dead 和控球比例均显示主客映射及时间对齐没有明显错误；但预测 possession 数偏多，导致公开 spellF1 低于 reference smooth50 的 `0.325581`。这 5 分钟只有 17 个 possession，官方文档明确不允许用它调参。因此该结果只登记为“目标域时序碎片化风险”，不反向修改模型或 Viterbi。

## 10. 产物与哈希

```text
alive.joblib
SHA256 73b7d4da710e6cd4ec23412fbe8a86c2627fd07db9ee0575693d9f943c23c267

home.joblib
SHA256 6a510ce0bdcb442e80672de6b086f55bea9e96cef61f57ad86cff87efc2f8fdf

training_report.json
SHA256 455b024e65337af4c45e231e99d8121fab2f772c8991b829574528382b986167

model_v1.csv
SHA256 2790393ff3260fbf6350b349a94ee8295e9038c5c26caad46dec13a0f0077905
```

模型与训练报告位于 `local_artifacts/models/`，被 Git 忽略。提交仓库只保存代码、配置和本报告。

## 11. 复现命令

```powershell
conda activate hack2027

python -m solution.train fit `
  --prepared local_artifacts\skillcorner_clean `
  --split local_artifacts\skillcorner_clean\split.json `
  --metrica local_artifacts\metrica_game3.npz `
  --out local_artifacts\models

python -m solution.run --mode final `
  --model-dir local_artifacts\models `
  --out evaluation\results\model_v1.csv

python evaluation\validate_submission.py evaluation\results\model_v1.csv
```

## 12. 当前决策

- `model_v1`：按外部晋级规则进入最终候选。
- `reference_smooth50`：继续作为确定性最低回退点。
- `heuristic_v1`：仍是未完成独立晋级验证的规则候选。
- 不根据公开五分钟修改当前版本。
- 如果设计 `model_v2`，必须先写出新的外部数据实验假设和选择规则，不能使用公开五分钟或已读取的 test 结果挑参数。
