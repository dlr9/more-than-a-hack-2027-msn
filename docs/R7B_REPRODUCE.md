# R7B 恢复与复现

只在公开源码/合法公开其他比赛工作区执行。不要把正式比赛资产放入这里。Python3.12.5，`requirements-solution.txt`；核心环境沿用 `R7_ENVIRONMENT.json`。线程上限2。可恢复状态先看 `R7B_STATUS.md`，已有成功模型与cache不重拟合。

## 本机已有产物：只核验，不重训练

```bash
git status --short
git log -12 --oneline
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python scripts/verify_r7_artifacts.py
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python scripts/verify_r7b_artifacts.py
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python -m unittest discover -s tests
```

`verify_r7b_artifacts.py`检查9个新bundle哈希、fit/calibration/test分离，精确重放P1/G1/CF1/G2uncal/G2cal共30组自然/原遮球概率与解码。它不是重训练或新盲测。已有公开报告不覆盖。

## 从干净研发副本重建

先按照 `R6_REPRODUCE.md`及 `R7_HANDOFF.md` 获取/准备公开外部源与冻结R6/R7缓存。DFL固定HF版本、LFS/Git哈希与floodlight独立对账必须完成；不需要下载任何目标数据。SC/Metrica的公共源来源/哈希见 `local_artifacts/external/manifest.json` 和公开来源文档。不要运行整批下载脚本覆盖已有成功数据，缺失文件按固定来源单独恢复。

以下阶段在全新 `local_artifacts/r7b` 中重建；已有receipt的阶段会核验并跳过成功拟合。若存在模型/NPZ但缺少receipt，脚本停止，先核实保存过程，不删除旧产物。

```bash
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 LOKY_MAX_CPU_COUNT=2
.venv/bin/python scripts/diagnose_missing_alive.py --output local_artifacts/r7b/diagnosis-reproduced.json
.venv/bin/python scripts/research_r7b.py prepare
.venv/bin/python scripts/research_r7b.py train
.venv/bin/python scripts/research_r7b.py transfer
.venv/bin/python scripts/research_r7b.py evaluate
.venv/bin/python scripts/calibrate_r7b.py
.venv/bin/python scripts/counterfactual_r7b.py prepare
.venv/bin/python scripts/counterfactual_r7b.py train
.venv/bin/python scripts/counterfactual_r7b.py evaluate
.venv/bin/python scripts/memory_r7b.py
.venv/bin/python scripts/transfer_metrica_r7b.py
.venv/bin/python scripts/verify_r7b_artifacts.py
```

新首次运行按准备→训练→评分顺序执行。不要多进程写同一个实验目录或JSON账本。`summarize_r7b.py`是首次冻结摘要的工具，使用exclusive输出；当前摘要已经生成，无需再运行。重现实验时比较新local cache与公开哈希/报告，不能把不同平台joblib序列化哈希差异直接当作模型分数差异；本轮本机重放要求逐元素完全相等。

训练输出：`p1-v1`（3个Alive头）、`calibration-v1`（3对独立专家+6校准器）、`counterfactual-v1`（3个Alive头）。P1准备约15分钟以内、拟合秒级；44维反事实准备每视图约45秒。实际耗时/配置/输入hash见各local receipt与registry。新bundle不上传Git。无数据测试无需这些缓存。

## 正式比赛的人工离线入口

由获授权成员将审查后的代码带到Codex工作区以外、关闭云同步/断网的独立本地环境。先手动执行 `scripts/official_assets_preflight.command`，再按 `R6_OFFLINE_RUNBOOK.md` / `R7_OFFLINE_TEAM.md` 执行 `scripts/run_offline_baseline.command`。运行器拒绝继承Codex环境、强制macOS网络沙箱、拒绝覆盖结果。这里仅做过合成输入测试，**没有检查或运行目标资产**。

如果只有两个视频，仍缺Tactical ATD XML/TXT，不能宣称官方baseline已运行。只有授权成员在隔离本地完成官方 `validate_submission.py` 严格检查的199319行结果才可称可提交文件。R7B研究模型不自动接入该入口；最终模型/提交由三人团队决定。禁止回传受限输入、目标输出或私人实验报告。

## 发布恢复

当前分支从R7 `e498e93`接续。PR base应为PR2研发分支 `codex/pr1-v2-integration-20261114`，不合并main。公开历史审计：

```bash
.venv/bin/python scripts/audit_public_release.py --out local_artifacts/r7b/release-audit-new.json
```

Git HTTPS超时时，`scripts/release_git_objects.py --uploaded-base e498e9390189bdf22ea3549ebf5f839b45915dbc`可复用已逐SHA验证的远端对象；仅创建新codex分支，逐blob/tree/commit核验，不force、不merge。实际发布receipt和PR URL以STATUS为准；已有分支默认拒绝移动；仅对本轮自己的研发分支，可显式使用 `--update-existing`，它在本地验证祖先关系并让服务端以 `force=false` 执行快进，拒绝重写历史。
