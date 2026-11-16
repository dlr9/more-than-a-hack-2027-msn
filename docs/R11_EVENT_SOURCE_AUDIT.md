# R11 H2：真实事件监督源与准备回执

本轮已完整读取新 R11 master。复用四场已存在的公开 SkillCorner 缓存，未重新下载 20 场。实现 `r11/acquisition_events.py`，运行 12 个合成测试和四场真实 CSV→追踪起点对齐；本项准备任务没有拟合或评价模型。后续训练由主研发执行方单独冻结协议和记账。

## 真实来源与含义

来源固定为 [SkillCorner/opendata@4340d274572876239c154c90bc507a9b3250a656](https://github.com/SkillCorner/opendata/tree/4340d274572876239c154c90bc507a9b3250a656)，本地 `local_artifacts/external/manifest.json` 记录文件 URL、大小和 SHA256。本轮核对四场 metadata 与 events 文件的实际字节哈希；1874553 的 tracking 也重新核对哈希。仓库 [LICENSE](https://github.com/SkillCorner/opendata/blob/4340d274572876239c154c90bc507a9b3250a656/LICENSE) 是 MIT（Git blob `c1b414c6a4553572fec4a3e4c87ceddcdc73ffd9`）；保留 SkillCorner 署名，不把软件许可解释为可任意转发第三方私人原始事件。

[原厂 Dynamic Events CSV 规格](https://26560301.fs1.hubspotusercontent-eu1.net/hubfs/26560301/Guides/Dynamic%20Events/20250216%20-%20Dynamic%20Events%20CSV%20Specifications.pdf) 将 player possession 定义为能传球、射门或带球的控制，从首次触球到末次触球；未建立控制的封堵不算该事件。它结合 tracking 和 Wyscout 事件，起点另有供应商模型及距离匹配。因此提供新的监督语义，但绝非独立人工控制边界真值；追踪与事件仍可能共享误差。规格还说明事件坐标可能按进攻方向镜像，不能直接当原始追踪坐标。

[公开数据说明](https://skillcorner.github.io/opendata/) 指出 tracking 是 10 FPS，is_detected 区分检测与外推；pressure、difficulty、EPV 是 player_possession 上的派生属性。本适配器不将这些属性、event_type、start_type、phase、team/actor 身份、比赛状态或事件坐标提供给预测器。它们只生成弱监督/来源记录；推断特征必须另从观测轨迹构造。

## 实际文件与样本

每场固定路径：

- 事件：`local_artifacts/external/skillcorner/{match}/{match}_dynamic_events.csv`
- 元数据：同目录 `{match}_match.json`
- 原始追踪：同目录 `{match}_tracking_extrapolated.jsonl`，使用时必须单独执行 is_detected 过滤
- 已有 44 维观测缓存：`local_artifacts/v2/{match}-observed.npz`
- 新弱监督小缓存：`local_artifacts/r11/events/{match}-supervision-v1.json` 和 `-audit-v1.json`

| 比赛 | player_possession | 起点 matched 且非完全外推 | 弱 acquisition=1 | 显式同队传接 acquisition=0 | unknown=-1 |
|---|---:|---:|---:|---:|---:|
| 1874553 | 959 | 936 | 130 | 647 | 182 |
| 1886347 | 990 | 976 | 133 | 686 | 171 |
| 1899585 | 898 | 880 | 158 | 550 | 190 |
| 1927964 | 1140 | 1122 | 131 | 853 | 156 |

所有 unknown 均保留，阶段缺口从未填 D。表中正负标签只针对有明确事件链的起点，其余帧一律 unknown；没有事件不等于负样本，也不是 D。contact 标签只有正例/未知，无法据此识别阶段一 contact 检测器的阴性分布。完整 spellF1 为 `null`。

1874553 的 4,915 条动态事件包括 959 PP、2,543 passing_option、878 on_ball_engagement、535 off_ball_run。PP 中有 672 pass_reception、83 pass_interception、61 recovery、54 keep_possession；209 个为单帧控制。reception_difficulty 有值 849 条、overall_pressure 865、time_to_impact 865、pass_reception_epv_for 803。这些统计只证明字段存在，不是特征效用或模型成绩。

1874553 源事件 SHA256：`92507490abbbeec7697227581c864cfb0a96c29b214dd63dcaa48929bb6c6408`；metadata：`e6974e402779d8be44e75bf88ef2a3acfd4d025b87b3fc57f518fa826a645ad8`；tracking：`ea97f58f8eaad925feaeacc6395ec24860dd80027af8a89450276adebd29d265`。其余源哈希见既存固定 manifest 及本轮各场 audit；不复制原始数据到 Git。

## 合同与实现

`adapt_skillcorner_events(rows, metadata, provenance)` 是纯函数，文件由调用方读取；输出 `(events, audit)`。`anchor_labels(events, frame, period)` 用实际 global frame ID 对齐，而非依赖行号或用比赛钟乘 FPS。输出 team=0(Home)/1(Away)、acquisition=-1/0/1；没有任何完整序列填充。

1874553 元数据映射 Home=1802(Brisbane)、Away=866(Adelaide)，全部 959 个 PP actor/team 与元数据一致。其 CSV `frame_start` 与 R7 tracking frame 直接对齐；`time_start` 是比赛钟，例如第二半场从约45分钟开始，不能当全视频时间。`duration=(frame_end-frame_start)/10` 在全部 PP 成立；frame_end 等于 frame_start 的单帧事件必须保留，不以零 duration 丢弃。

acquisition=1 要求前一有效 PP 属于对手、同半场、时间不重叠、双方无显式 interruption，当前 start_type 为 pass_interception/recovery 且 first_player_possession_in_team_possession=True。acquisition=0 要求前一同队 PP 明确成功 pass、target player 与当前 actor 相同、当前为 matched pass_reception 且非新队首控。其他情况不猜测。在 source-defined 事件条件下标签有意义，但正负选择机制不同，训练结果必须标为条件任务，不能冒充全帧 acquisition 检测。

拒绝混合比赛、未知球队、演员球队冲突、重复/同帧歧义、跨半场、错误 10 FPS 时长及未知布尔语义。事件中所有衍生属性都未复制成特征。函数不接受目标输入授权，也不打开文件。

四场现有缓存 `X` 为 float64，形状分别 60301/59061/60531/59071 ×44；frame 为 int32 0..N-1，period 为 int8，fps=10。调用方仅可取 `X/feature_names/frame/period/fps` 及经审查的观测质量字段作为输入。缓存还含 `y_* / eval_y_* / raw_y_* / phase_y_home / label_* / home_stable / phase_team_conflict`：这些标签或标签代理不得入特征。

## 选择偏差与暴露历史

实际流式检查 1874553 的 936 个合格起点：**936/936 球可见**，924 个 actor 在该帧实际检测到，11 帧没有检测到任何球员。非完全外推是事件级标志，不能替代起点 actor 的检测质量。质量回执 `1874553-anchor-quality-v1.json` 耗时约1.606秒；只消费 frame/ball_data/player_data，未消费 tracking possession 或 phase。监督准备/对齐耗时约0.378秒，无拟合。

因此这些事件当前不能验证自然缺球取得，也不能解决 SC 195 个零球零人帧；matched 过滤引入明显可见性选择偏差。把 all-visible 事件的随机缺球增强称为真实 missing 验证是不成立的。

四场均已被 V3 旋转训练/校准/验证消耗。`docs/candidate_runbook_v3.md` 记录旧冻结交接折训练1899585/1927964、校准1886347、验证1874553；PR #1 还已处理20场。新 acquisition 模型可按比赛划分 fit/eval，但只能声称新任务的 out-of-match diagnostic，不能声称团队首次独立盲测。动态事件与 phase/追踪 possession 不是独立完整 H/A/D 标签，不可互相当真值宣称跨供应商晋级。

## SoccerTrack v2

[公开 dataset card](https://huggingface.co/datasets/atomscott/soccertrack-v2) 明示须登录并同意共享身份信息后访问文件。此处只读取公开卡片，状态 **ACCESS_BLOCKED**；未申请、同意、读取或绕过任何 gated 文件。卡片声明 CC BY4.0、10场934分钟、v1.2 共21,428个 BAS 事件、约145GB。

`bas.position` 是各半场首视频帧起毫秒，`bas.frame` 与视频/GSR 对齐。`ball/` 由已标注 BAS 锚点和事件间线性插值重建：禁止用于从 BAS 监督预测事件，否则发生标签泄漏。将来获授权也需先核验独立原始观测；tackle actor 不自动等于成功新控球队。

## 实际测试与下一步边界

已运行 `OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python -m unittest tests.test_r11_acquisition_events -v`：12/12 通过。覆盖队向、精确起点、不填未知、显式正负链、跨半场/中断拒绝、检测/外推、单帧控制、时间合同、重复/混合比赛、派生字段不进入输出特征和来源要求。

准备代码基于 Git `3fa13c4e66c8c04dffad5a4b0c56e9e4aeda3678`；适配器 SHA256 `114099b82a3e5e8547ad95cc7558ab6073b1d25366dc18250248b69765de2d33`。准备结果保留为 weak-supervision component，不晋级任何正式模型。下一步由主执行方冻结单次 out-of-match 条件 acquisition 实验，保留强基线、等数量无事件对照、同半场时间/球队打乱及 UNKNOWN 语义；没有独立完整标签时不算正式全场 spell 或真 D 成绩。
