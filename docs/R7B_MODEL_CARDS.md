# R7B 模型卡与保留决定

这是公开其他比赛研发成果，**不是经过目标比赛验证的正式模型**。所有模型 `competition_promoted=false`。没有用官方五分钟GT、赛事受限资产或已消费封存比赛选模型。三场开发共享主队；逐场结果是重复使用开发场上的跨比赛验证，不是新盲测。SC为1场4个已知H/A缺球episode，Metrica为弱事件窗口，均没有完整源H/A/D评分资格。

数据与许可：DFL IDSSE CC BY4.0、固定HF修订 `a715a38dfbaf5f58e431727c2b78d174101a703c`，原始来源及许可链接见 `R7_DATA_PROVENANCE.md`。SC固定公开仓库修订 `4340d274572876239c154c90bc507a9b3250a656`；Metrica Sample Game3固定修订 `e706dd506b360d69d9d123d5b8026e7294b13996`，来源许可记录沿用 `external_sources_v2.json` / `external_data_audit.md`。公开Git只提交源码、摘要和哈希；不提交原始数据、逐帧输出或joblib。数据许可不代表可上传赛事受限输入。

所有新版模型的 `git_sha`、训练名单、输入hash、模型SHA256、路径、字节数见 `R7B_MODEL_REGISTRY.json`。P1/CF1均按WPY/WOH/WQQ留一、其余两场拟合。G2固定一场fit、一场calibration、第三场evaluate，具体循环见registry。SC/Metrica均固定WPY-heldout折，不按分数选折。冻结模型哈希见R6/R7原注册表。版本入口：源码冻结 `1a823b6`，校准 `10ff15b`，反事实 `e7dfe4c`，时序记忆 `d69062a`。

|candidate_id / parent|feature_schema|优点与适用域|缺点、已知边界|决定 / fallback / owner|
|---|---|---|---|---|
|official smooth50 / official reference|规则|免训练、授权离线保底路径已有合成验证|死球弱；当前目标资产和本地baseline未检查、未执行|FROZEN reference / 官方运行器 / A+C|
|V3 / 历史SC模型|schema2|完整保留、可恢复|phase弱标签、跨源适配非等价、不能替代完整D监督|FROZEN / 原V3 bundle / B|
|R6-M1 / V3研究线|schema3/44|源完整监督；研究回退|DFL冻结依赖；SC缺球HA只有9.74%|KEEP research / 原M1 / B|
|R6-M2 / M1|schema3/44|原始遮球训练提高DFL压力域|SC零球员195帧全D；不具备真实跨源稳健性|EXPERIMENTAL / M1 / B|
|R6-M3 / M2固定概率|schema3/44|DFL遮球时减少部分D切断|×4只改转移成本，不修Alive；195帧全D|EXPERIMENTAL / M2或M1 / B|
|R7-M4 / M1|schema4/39|完成去五列因子诊断|删掉真实运动信息；不等于捷径因果贡献|DIAGNOSTIC / M1 / B|
|R7-M5,M6 / M4|schema4/39或31|真实消融及失败证据|自然/压力/迁移整体退化，不能作为player-only|REVERT replacement / M1 / B|
|P1-playeronly-alive / 新独立Alive+冻结M2 Home|player_only_v1/46|原始球员运动、队形、相对运动；无球或标签特征；缺球但球员可观测时有用|纯替代自然spell下降；SC零球员输出.982只是缺失分支，不是运动证据；Metrica假D增加|KEEP diagnostic expert / M2或M1 / B|
|G1 / M2+P1|44与46两个独立专家|观测质量门控；DFL自然不变，遮球/长缺球提升|SC零观测仍全D；Metrica弱D漏检从1911增到3891|EXPERIMENTAL / M2或M1 / B|
|G2_uncal,G2_cal / 重新拟合的独立专家|44+46+独立sigmoid|严格fit/cal/test按比赛分离，提供同训练预算对照|少量比赛、calibrated并未稳定提升；不能与G1归因比较|G2_cal REVERT replacement / G1 / B|
|CF1 / M2 Home+反事实增强Alive|schema3/44|噪声域spell .529高于M2 .388，揭示冻结敏感性|自然及原遮球下降；训练扰动是压力研究，非真实供应商分布|REVERT replacement / M2 / B|
|T1,T2 / 冻结M2+有限观测先验|44+30秒状态记忆|T2在SC195帧将D降为7，HA准确率96.41%；完整blackout各场spell改善|会错过真正死球；WPY/WOH漏D帧率增加2.12/4.78百分点；每场漏掉2段D；相同Alive下Home记忆才修正队伍错误|EXPERIMENTAL / G1或M1研究回退 / B|

## 已证实逐场成绩

比赛顺序固定为 J03WPY / J03WOH / J03WQQ。下表为完整外部源spellF1，非官方目标分数。

|模型|自然域（逐场）|原遮球（逐场）|长遮球（逐场）|
|---|---|---|---|
|M1|.550500 / .551090 / .581509|.395285 / .416568 / .409588|见冻结R7_STRESS_RESULTS|
|M2|.557107 / .551485 / .585319|.428283 / .435386 / .444492|见冻结R7_STRESS_RESULTS|
|M3|同M2|.458491 / .461647 / .463252|见冻结R7_STRESS_RESULTS|
|P1|.430268 / .455889 / .464804|.416432 / .412698 / .392017|.407109 / .421968 / .413072|
|G1|.557107 / .551485 / .585319|.488837 / .458427 / .437821|.512655 / .494787 / .483690|
|G2_cal|.556413 / .546798 / .580383|.473479 / .441196 / .437821|.507882 / .483372 / .479376|
|CF1|.518483 / .506220 / .555789|.406438 / .418696 / .426891|.430911 / .440275 / .448182|

全场景同时缺失：M2 .502930/.497378/.527845；T1 .511877/.509506/.541809；T2 .519004/.513111/.546766。T2漏D帧1491/3238/1090，对照120/117/107；真D帧分母64751/65269/57993。T2假D帧3495/756/1729，对照10050/6615/8616。减少假D与漏真D的代价同时展示，不采用仅live-only指标的晋级。

完整逐场macroF1、D-F1、五个IoU、raw vs Viterbi、假D切断/漏D、短片段、typed边界±.48/1秒与Brier/log-loss/ECE见 `R7B_EXPERIMENTS.md`、各RESULTS及 `R7B_BOUNDARIES_AND_QUALITY.json`。ECE仅描述验证集可靠性；未经独立校准的概率不称已校准。M4/M5/M6及历史基线逐场结果保留在 `R7_CORE_RESULTS.json`，不重复修改。

## 三人审核 TODO

- A：核验公开增量、远端SHA、CI、48个历史/新增bundle哈希；保持生产兼容闸门，P1/G1当前是研究API，不能静默替换官方离线入口。
- B：确认G1和T2不通过跨源/真D门槛；冻结这一轮参数，不用195帧继续网格调参。下个优先假设是零观测期间可识别性与更多完整供应商标签，不启动大型网络。
- C：在隔离本地确认官方资产包、ATD及截止时区，执行已审查的offline preflight/reference与官方格式验证。不要将任何目标资产或结果回传云端。新外部验证先检查球队重叠与标签语义，再预注册。
