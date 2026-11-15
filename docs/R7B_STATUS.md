# R7B 恢复状态 — 真实研发闭环完成，准备Draft发布

- 分支：`codex/r7b-player-only-missing-alive`，从R7 `e498e9390189bdf22ea3549ebf5f839b45915dbc`接续；未重写main，未修改PR2分支。
- last_verified_sha: `b4c3159`（完整研发结果checkpoint；最终发布SHA由publication receipt核验，避免文档自引用SHA）。
- 当前锁/PID：所有训练、特征准备、评分已完成，无需等待或重启模型任务。未发布源码/报告即本分支当前增量，发布前必须commit和安全审查。
- 已保存：9个新增bundle，包含12次HGB拟合与6个sigmoid校准器；原R6的30个注册模型及R7的9个bundle全部哈希一致。保留V3和所有旧概率/实验cache。
- 本轮测试：109项本地测试通过；109项干净无数据副本测试也通过，最终发布回执见下方追加状态。9个新bundle哈希及30组完整概率/解码精确重放通过；pip check通过；官方scorer未改动。

## 已验证结果与决定

1. 195帧来自SC1874553的4个episode，全无检测/外推球员。M2/M3 Alive均值.42721，argmax与Viterbi均全D；不存在发现的frame/period/标签错位。详见ROOT_CAUSE和MISSING_DIAGNOSIS。
2. 真Player-only 46维/3次留一Alive训练完成。纯替代自然spell均值.45032，低于M2；零观测高Alive是missing分支响应，不视为证据。
3. G1质量门控：DFL自然均值.564637不变，原遮球.461695（M2 .436054），长遮球.497044；但SC仍195/195 D，Metrica弱D漏检增多。**EXPERIMENTAL，不晋级**。
4. 独立比赛校准6次HGB+6校准器完成，无稳定增益；CF1反事实增强3次Alive拟合完成，噪声提高、自然及缺球下降。均REVERT替代资格，文件保留。
5. T2有限观测先验：SC误D降到7/195、HA准确率96.41%；完整DFL同时遮挡球和球员时，WPY/WOH漏真D率增加超过.02上限，各场完全漏2段D。**EXPERIMENTAL，不晋级**。不再调阈值搜195帧。
6. 两场R7封存WMX/WN1已消费，本轮未作新盲测；没有候选通过门槛，因此未消耗WOY/WR9。完整跨供应商D真值和目标效果仍未知。

## 恢复入口与输入哈希

工作目录：团队公开源码副本。原始公共数据/模型/cache不移动、不覆盖。

- P1模型：`local_artifacts/r7b/p1-v1/{J03WPY,J03WOH,J03WQQ}.joblib`
- 独立校准：`local_artifacts/r7b/calibration-v1/`
- 反事实：`local_artifacts/r7b/counterfactual-v1/`
- 场景消失记忆压力：`local_artifacts/r7b/memory-v1/`
- SC、Metrica诊断：`local_artifacts/r7b/transfer-v1/`、`metrica-v1/`
- 原始输入/feature/model/Git哈希及实际成本：`R7B_MODEL_REGISTRY.json`、`R7B_EXPERIMENTS.json`、各prepared receipt。首次P1-WQQ提前拟合的receipt已补入账本，未重拟合。

下一条精确恢复命令（不训练、不消耗盲测）：

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python scripts/verify_r7b_artifacts.py
```

首次完整复现顺序见 `R7B_REPRODUCE.md`；不要重新运行已有成功模型阶段。各完整评分与失败决定见 `R7B_EXPERIMENTS.md`、`R7B_DECISIONS.json`，模型适用域和回退见 `R7B_MODEL_CARDS.md`。

## 失败与回退

拟合前窗口方差浮点相消已修复并测试；没有按验证分数改特征。当前主要阻碍是零观测状态不可识别与跨供应商标签/观测机制差异，非网络或算力失败。G1、T2保留研究组件，G2cal、CF1不作为替代。保留R6-M1研究回退、M2/M3/V3原件。不存在已证实的目标比赛提升。

R7 Git HTTPS曾超时，原对象已逐SHA上传至e498e93；R7B采用安全Git对象API增量，创建新分支，不force、不merge。公开Draft发布/CI尚需以远端回执确认，不把“本地commit”称作已发布。

## A/B/C 协作TODO

- A：核验Draft增量、远端SHA与CI；保持所有旧产物；不用新研究模型静默替换离线入口。
- B：审查真D漏检与假D切断的权衡；下轮先找更多完整外部标签/零观测episode，不继续在195帧上调参、不启动大网络。
- C：由授权成员在工作区外断网环境检查正式资产、执行preflight和官方baseline/validate_submission，核实邮件截止时区。若仅两个MP4则仍缺Tactical ATD XML/TXT。Codex未打开任何目标资产、未声称已跑正式baseline；目标结果不得回传云端。

正式比赛模型选择、合并和提交均留给团队，当前仅授权Draft PR。

## 发布前检查

完整公开增量历史扫描通过：20 commits / 143 text blobs（该检查时head=b4c3159）。无原始数据/权重/逐帧输出进入Git。新对齐拒绝闸门额外验证后，G1的WPY自然/原遮球保存概率仍逐元素一致。远端PR2仍Draft/open且head=093361a3，未修改。新PR创建后追加URL与远端SHA。
