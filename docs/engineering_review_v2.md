# PR #1 工程审核与 V2 研发记录（2026-11-14）

结论：沿用现有 `solution/`，完成公开代码环境、关键解析/评估修复及外部实验。本轮候选均不晋级，不合并主分支、不生成或提交比赛文件。完整 H/A/D 标签语义及跨供应商完整时序门槛仍未建立。

## 审核范围与证据边界

- 审核对象：[团队 PR #1](https://github.com/dlr9/more-than-a-hack-2027-msn/pull/1)，head `83b262a9420c67eb494b39ed43ece90bd4e66623`，team main `f8f2629ff1e0985f81d7e59d8413678514ee1530`。
- 完整阅读交接包中的 `CODEX_NEXT_ITERATION_MASTER_CN.md`。实施以用户明确要求的合规边界为准。
- 仅按白名单取得公开代码；没有完整拉取该 PR 的文件树内容。未读取 `docs/hack2027_solution_technical_report.html`、`docs/model_training_validation.md`、目标受限视频/ATD/Smart Tagging/Homography/衍生逐帧文件/私人实验报告，也未读取官方公开五分钟 GT。
- 独立安全快照保存了 #1 的代码；旧工作区未被覆盖。新修复分支以 team main 为基点带入 #1 的公开实现及修复，不修改原 PR 的报告或历史。原报告须由团队成员自行隔离/审查。
- 当前官方 [规则](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/RULES.md)、[FAQ](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/FAQ.md)、[评估说明](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/EVALUATION.md) 已复核。对应 blob：`d26a3835a93e800cc2039723943a250d90a44c20`、`c45ff9141f2f59e5ec75e7dace55b575b4519878`、`0c281c41787cdbb77b1f8b6aba9403f28cf9c4a2`。
- 其他比赛数据固定版本与下载 SHA256 见 [source manifest](external_sources_v2.json)。SkillCorner 使用原训练池的四场比赛做分组交叉验证，不伪称全新未见测试集。原测试集三场继续封存。

## R0：公开工程与数据协议修复

**研究假设：** 严格解析、观测来源区分、独立时序边界和关闭不足证据的晋级路径，能消除静默错误及评估偏差；合成测试只能验证行为，不能证明比赛精度。

**主要发现及修改：**

| 问题 | 修改文件 | 结果 |
|---|---|---|
| ATD 跳过坏行、重复帧覆盖、截断/多槽位不严格 | `solution/atd.py` | 拒绝坏行、非有限无穷值、缺帧/乱序/重复、未知球队、槽位错误；坐标格式显式区分 |
| Metrica 名单不等于当前轨迹槽位 | `solution/atd.py`, `solution/external.py` | 根据 XML channel→player→team 和 11 个有效区间解析换人；按元数据半场独立解码 |
| 缺失片段/半场间错误派生速度或位移 | `solution/features.py`, `solution/atd.py`, `solution/external.py` | 时间特征按半场重建；位移不跨越缺球；速度不跨半场 |
| 外推球员被当作可见球员 | `solution/external.py`, `solution/features.py` | 默认仅使用 `is_detected=true`；保留 all 消融视图；单独记录检测/外推/未知计数；44 维保留，计数列改为 positions，schema=2 |
| phase 补集直接成为 D | `solution/external.py`, `solution/quality.py` | 默认补集未知，保存来源/置信类别/掩码；legacy 补集仅显式诊断 |
| 标签覆盖岛决定 Viterbi 重置点 | `solution/modeling.py` | 解码边界仅来自半场；缺标区间仍参与推断，不从标签覆盖泄漏边界 |
| HGB 随机逐帧 early stopping；拆分重叠与缓存混用 | `solution/modeling.py`, `solution/train.py` | 禁用随机逐帧 early stopping；检查比赛分组/身份/schema/view；封存旧测试；prepare 要求明确比赛 ID |
| H/A 门槛冒充完整 H/A/D 门槛 | `solution/modeling.py`, `solution/run.py` | 记录部分 H/A/D，完整序列为 not_tested；禁止自动晋级；final 入口关闭 |
| 目标开球硬编码、宽泛事件标签直接写 D | `solution/config.json`, `solution/heuristic.py`, `solution/run.py` | 移除目标帧锚点；set-piece 标签本身不再断言死球；目标入口须团队离线明确确认 |
| 双头未校准、固定转移成本 | `solution/calibration.py`, `solution/decoder.py` | 独立比赛拟合 logistic 校准并记录可靠性；保留固定基线，增加实验性观测控制代理 |
| 输出标签被 U1 截断 | `solution/submission.py` | 在转换前验证一维完整标签，UTF-8/LF 输出 |
| Python 版本与依赖环境不一致 | `environment-solution.yml`, `.github/workflows/public-solution.yml` | 固定已实测 Python 3.12.5；公共代码稀疏检出、合成测试 CI；路径审查不打开疑似私有文件 |

**实际测试：** 原 PR 安全快照的 10 项测试通过。最终 `python -m unittest discover -s tests -v` 为 32 项通过；覆盖帧契约、换人槽位、观测/缺球、半场、标签独立解码、概率、校准隔离、输出和公共文件路径。官方平滑器对照仅用随机/边界合成序列，没有读取任何目标 GT 或逐帧结果。`pip check` 与 `git diff --check` 通过。远端 CI 状态另见 PR，不将本地通过等同远端通过。

**失败记录：** 首次旧环境缺 joblib，创建独立 Python 3.12 环境后恢复；初始修改出现局部变量赋值顺序错误，新增测试捕获后修正；严格 Metrica 解析依次暴露二元坐标及换人槽位问题，按公开 XML 修复后完整解析成功。

**晋级/回退：** 工程修复可进入团队代码评审；模型不晋级。保留官方参考及原 PR 公开代码安全快照作对照，不能把已发现不可靠的旧晋级判据恢复为正式门槛。

## R1：预先固定的四场分组消融

**研究假设：** 检测球员视图、连续缺球增强、分组校准和控制事件代理可能改善分布变化时的稳健性，但必须同时检查 alive 与 H/A，不能只看 H/A 分数。

**修改文件：** `solution/external.py`, `solution/quality.py`, `solution/calibration.py`, `solution/decoder.py`, `scripts/fetch_external_v2.py`, `scripts/experiment_v2.py`, `scripts/summarize_v2.py`。

**实际执行：** 四场 SkillCorner × 两场训练/一场校准/一场验证轮换；三种观测/增强变体；校准前后；固定及控制代理两种解码。另在同一公开 Metrica Game 3 上作跨供应商部分标签诊断。72 组域/变体/校准记录全部完成。固定参数与命令见 [protocol](experiment_v2_protocol.md)，完整结果见 [外部验证](external_validation_v2.md) / [JSON](external_validation_v2.json)。

**实际指标：**

- 自然 SkillCorner H/A macro F1：all 0.9229，observed 0.9183，双头 dropout 0.9176（未校准均值）。这不支持“过滤外推必然提升原供应商精度”。
- 人工丢球压力集 H/A：all 0.8786，observed 0.8898，dropout 0.8880。观测来源变化与丢球共同作用，不能把这个差异全部归因于丢球增强。
- 自然缺球 H/A Brier：observed 0.4786，dropout 0.3377；样本极少且存在标签选择偏差，只能作诊断。
- 同一 Metrica 比赛、四个训练模型的描述均值：observed 固定解码部分 HAD macro F1 0.8697，双头 dropout 0.5104；D 类 F1 分别约 0.8868 与 0.0139。参考平滑的部分 HAD 为 0.8721。不是完整比赛 spell 指标。
- 校准未稳定改善 SkillCorner H/A：observed 的 Brier 0.05546→0.05620，ECE 0.01752→0.02404。Metrica observed 部分 HAD 有轻微改善，但不能因此选胜者。控制代理收益接近零，尚无有效事件约束的验证证据。

**phase 语义审核：** [固定版本官方 README](https://github.com/SkillCorner/opendata/blob/4340d274572876239c154c90bc507a9b3250a656/README.md) 将 phase 与活球关联，但没有明确保证逐帧覆盖完整。官方详细 PDF 下载未成功，不能据此声称语义已证实。四场有 397 个 phase 空档；1886347 的空档内部（距边界超过 1 秒）仍有 47 帧与同供应商动态活动重合。活动本身也不是独立真值。保守策略没有可确认 D 负类，故其 alive 训练会明确失败，而非把未知补成 D。

**失败原因：** 双头增强使跨供应商 alive 判断严重偏向活球（平均活球概率约 0.66→0.84），D 识别崩溃。数据域变化、弱标签和增强分布共同作用；仅凭本轮不能确定单一因果。校准集只有一场/折，且 alive 标签同样是弱标签。控制代理仅是近球关系，不能视作已验证的抢断、拦截或有效控制事件。

**晋级：** 全部不晋级。**回退：** 取消双头增强候选，保留 all/observed 固定基线作研发对照；最终仍使用官方参考回退方案。**下一步：** R2 只增强 H/A 头，保持 alive 输入不变，检验伤害是否来自对 alive 头的增强；明确记录为事后诊断。

## R2：仅增强 H/A 头的事后诊断

**研究假设：** R1 的跨供应商 D 崩溃主要与 alive 头也接收了人工丢球样本有关；将增强限定于 H/A 头可能保留缺球概率改善，同时避免伤害 alive。

**修改文件：** `scripts/experiment_v2.py` 加入显式 `--followup` 路径，分别构造两个头的训练输入；不改首轮三种配置或覆盖首轮结果。

**实际运行：** `OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python scripts/experiment_v2.py --followup`，相同四折、24 组域/校准结果完成；这是使用已看过的研发比赛做诊断，不能算独立确认。数值见 [R2 JSON](external_followup_v2.json)。

**实际指标及失败原因：** 未校准 Metrica 部分 HAD macro F1 从双头增强的 0.5104 恢复到 0.8697，D 类 F1 从 0.0139 恢复到 0.8870；alive Brier 恢复为 0.04743，H/A macro F1 仍为 0.8868。校准后部分 HAD 为 0.8730，但同一比赛的微小差异不能用于晋级。H/A 在自然 SkillCorner / 人工遮挡下分别为 0.9176 / 0.8880；没有证据表明综合结果超过参考。活球头输入恢复后结果恢复，支持将两头增强分开，但弱标签与跨供应商覆盖问题未解决。

**晋级：** 不晋级。R2 仅支持后续把增强限定于 H/A 头进行验证，不授权模型替换或发布。**回退：** 一键不传 `--followup` 即保留首轮协议；目标 final 仍关闭。**下一步：** 先解决 D 真值、自然缺失覆盖及全时序门槛，再设计新的封存验证。

## 尚未闭合的验收项

- 独立确认 phase 覆盖与 D 标签，或取得其他比赛完整独立 H/A/D 标注；当前缺证据，不能训练/验证真正的保守双头候选。
- 自然缺球与实际检测视图的更广泛独立比赛覆盖；当前人工遮挡不能替代自然缺失监督。
- 校准需要更多按比赛隔离的样本。alive 校准受弱标签污染；没有自动启用校准。
- 控制事件代理没有证明收益；尚未实现经过独立验证的抢断/拦截控制状态机或持续时间模型，跨帧率转移成本也需要专门验证。
- 完整 H/A/D spell 门槛、全新封存测试与团队批准尚未满足；禁止根据官方公开目标 GT 排序。
- 本轮没有宣称训练出优于官方参考的可提交系统。所有外部实验结果和失败保留，后续不得当作未见测试重新使用。

## 交付状态

本地修复分支、纯代码交付包及补丁已准备。GitHub 插件在创建 tree 时返回 403 `Resource not accessible by integration`；本机没有可用 `gh`。因此尚未创建远端修复 PR、未运行远端 CI。现有 PR #1 已附到任务；未合并、未修改其报告/历史。需为当前连接授予团队仓库 Contents / Pull requests 写权限后，才能发布已准备的草稿；PR 正文保存在 `docs/pr_v2_draft.md`。
