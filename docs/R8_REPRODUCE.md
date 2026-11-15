# R8 恢复与精确复现

本轮从已核验远端R7B `ccc4b2a1382c20112624e86a4dcdf50921673560`增量继续。原R7B结果与模型不修改；R8分支 `codex/r8-possession-centric-20261115`。仅合法公开其他比赛和合成输入，不需要、不得放入正式赛事素材、GT、衍生预测或私人报告。

Python3.12.5；numpy2.4.6、pandas3.1.1、scipy1.17.1、scikit-learn1.7.2、joblib1.6.0。安装锁定版本见requirements-solution.txt。所有训练/解码使用CPU，两线程：

```bash
cd /Users/dailuren/Documents/ChatGPT/barca/review-pr1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 LOKY_MAX_CPU_COUNT=2
.venv/bin/python scripts/verify_r7_artifacts.py
.venv/bin/python scripts/verify_r8_artifacts.py
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m pip check
.venv/bin/python scripts/check_public_tree.py
git diff --check
```

首选以上只读核验，不重跑已完成拟合。R8保存3个C1包/6个HGB，R6/R7/R7B旧模型继续在原目录。模型SHA、训练名单和样本量在R8_MODEL_REGISTRY；缓存/模型/source哈希在各PROTOCOL及receipt。核验器同时检查96组已保存全场解码、12组C1双头概率、36组历史对照spell数值与原模型等预算；官方评分器SHA固定为`341bad48904b489d6e875397b27687f5d6a4d22a397f3dc8af3a00d48f16874d`。

## 实际执行的阶段与恢复边界

- E08A初版协议/源码checkpoint `8086ec6`，结果在 `local_artifacts/r8/e08a/`。当前代码已修复H2首个重现球观测的速度可达性，因此拒绝把修复实现重新标为初版。初版精确算法在历史checkpoint保留；核验保存解码无需恢复旧工作树。
- 原版A结果、B协议/源码checkpoint `7b36628`。B先重建坐标，再重算44维，用冻结M2发射/解码；`local_artifacts/r8/e08b/`不含新拟合模型。不要再次运行diagnose覆盖记录。
- 锚点修复、训练采样诊断和C协议checkpoint `ffa9e3d`。修复版A保存于 `local_artifacts/r8/e08a-anchor-fix/`，不覆盖初版。所有阈值不变。独立运行入口 `scripts/replay_r8_anchor_fix.py`。
- C1三个模型与选择索引在 `local_artifacts/r8/e08c/`，fit receipt记录真实SHA。训练样本预算与R6-M2逐头一致。唯一一次评分中断留下WOH-dropout NPZ，没有traceback，根因未能确认；已保存原文件并精确重放一致后恢复receipt，未重训。恢复防护合成测试保证不一致文件不被覆盖。

如重新建立**全新隔离公开研发副本**，先按R6/R7/R7B runbook恢复合法公共缓存并核验哈希，不重复下载已有成功文件。在新副本保留原结果目录只读后按阶段重建；历史E08A用8086ec6的源码，之后用当前修复源码和独立目录继续。不要对当前工作树执行reset/clean/rebase或删除现有结果。

```bash
# 初版E08A仅在8086ec6的独立安全源码副本及空e08a目录首次运行：
.venv/bin/python scripts/evaluate_r8_decoder.py
# 当前修复实现，空e08a-anchor-fix目录；引用完成的初版对照：
.venv/bin/python scripts/replay_r8_anchor_fix.py
# 新空e08b目录；顺序不能交换：
.venv/bin/python scripts/evaluate_r8_interpolation.py diagnose
.venv/bin/python scripts/evaluate_r8_interpolation.py evaluate
# 诊断/协议已冻结后，新空e08c目录才需训练；已有成功receipt不再拟合：
.venv/bin/python scripts/train_r8_boundaries.py train
.venv/bin/python scripts/train_r8_boundaries.py evaluate
```

已有完整results的阶段应使用verify入口。若仅有部分receipt，先核对当前源码/协议SHA与原receipt；不能跨代码版本直接当作同一次运行。输出文件默认exclusive；C评分的无receipt预测仅在数组逐元素相同后复用。账本标识以experiment_id和protocol_sha共同定位，锚点修复具有独立协议。`summarize_r8.py`只用于首次从完整本地结果生成公开摘要，已生成后拒绝覆盖。

## 人工离线交付P0

已有 `official_assets_preflight.command`、`run_offline_baseline.command`、`run_offline_reference.command` 与R6_OFFLINE_RUNBOOK/R7_OFFLINE_TEAM保持不变。本轮仅合成回归测试，没有运行目标预检或Baseline。只有两个MP4仍不足，需Tactical ATD XML/TXT。由授权成员在非Codex、无云同步的隔离离线环境确认文件、官方199319行格式验证、报名邮件截止时区并人工提交；不回传目标输入、逐帧输出或私人报告。建议11/16北京时间18:00前冻结可运行路径，正式11/17截止以报名邮件为准。
