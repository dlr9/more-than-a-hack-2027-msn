# V3 候选恢复与团队交接

状态：研发冻结，未晋级；不合并主分支，不生成或提交赛事结果。历史 model_v1 的源代码保留在 `974216d`；原 runner 未存模型二进制，不能声称恢复了旧权重。本轮重新拟合的基线已逐折复现旧 H/A 指标，所有新模型和概率保存于忽略目录。

## 已有本地环境和候选

工作目录 `review-pr1`，Python 3.12.5，依赖精确版本见 `requirements-solution.txt`，已通过 pip check。四个折各保留 baseline.joblib、missing_expert.joblib、alive_calibration.joblib，以及三种域的概率缓存。完整模型哈希和训练/校准/验证名单见 `docs/external_validation_v3.json` 的 manifest。

固定交接入口是第一个预声明折 `local_artifacts/v3/1874553/baseline.joblib`，不按验证结果挑折：训练 1899585、1927964，校准比赛 1886347，验证 1874553。基线不使用校准器；专家为单独实验文件。其余三折保留用于复核，不能当作相互独立的四个 Metrica 测试。正式候选应在新的完整外部标注协议下重新确认。

在仓库根运行（仅限可信的本地 joblib 文件）：

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python scripts/predict_external_candidate.py \
  --bundle local_artifacts/v3/1874553/baseline.joblib \
  --prepared local_artifacts/v2/1874553-observed.npz \
  --out local_artifacts/v3/1874553/my-external-check.npz
```

这是其他比赛研发预测，不是比赛提交文件。输出路径已存在会拒绝覆盖。专家对照只替换 bundle 为 missing_expert.joblib；回退替换回 baseline.joblib。bundle 检查 schema 和特征顺序。默认固定 Viterbi；时间归一化是 E3 实验开关，冻结基线未启用。不要据目标比赛表现修改参数或选择折。

## 从干净公开代码环境复现

远端主仓库含赛事公开 GT；云端研发不要普通完整检出。使用 `--no-checkout --filter=blob:none` 和与 `.github/workflows/public-solution.yml` 相同的非 cone 白名单稀疏检出。添加所需 `scripts/*.py` 和本轮安全 `docs/*v3*`、外部来源清单；不添加 `evaluation/gt`、`evaluation/results`、`data` 或目标报告。Git 传输不可用时，使用本地安全代码归档或白名单 GitHub API，不能为省事整包下载上游。

新环境命令（Python 3.12.5）：

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-solution.txt
.venv/bin/python -m pip check
.venv/bin/python scripts/check_public_tree.py
OMP_NUM_THREADS=2 LOKY_MAX_CPU_COUNT=2 .venv/bin/python -m unittest discover -s tests -v
```

重现实验必须在**新空运行目录/安全检出**中执行，保留原结果目录。下载器固定公开其他比赛版本，后续 manifest 需与本轮输入哈希核对，不接入目标数据：

```sh
.venv/bin/python scripts/fetch_external_v2.py
.venv/bin/python -c 'from scripts.experiment_v2 import prepare; prepare()'
.venv/bin/python scripts/audit_evaluation_v3.py
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 LOKY_MAX_CPU_COUNT=2 .venv/bin/python scripts/research_v3.py
.venv/bin/python scripts/summarize_v3.py
```

research_v3 拒绝覆盖既有 research_results.json。需要重跑时建立新的隔离检出并记录原因，不删除历史记录。无需重跑旧 72/24 组训练。新实验只拟合预声明的 8 个基线头和 4 个专家，另拟合轻量校准器。代码后续补充的训练入口防护不改变本次 runner 使用的 load_prepared/_weights 行为；实验记录包含实际执行时 runner/evidence/candidate 的 SHA256。

## 提交前的人类离线步骤

1. 团队审阅 PR #2 的代码和本轮证据；PR #1 及私人实验报告由团队自行隔离审查。任何合并必须另获团队批准。
2. 先用公开其他比赛获得连续、独立裁定的 H/A/D 与边界标注，冻结新协议和真正新测试。现有候选不满足正式晋级要求。
3. 由团队成员在完全离线且隔离的环境装入允许的目标输入。云端不读取输入、输出、逐帧衍生物或私人目标实验报告。现有 `solution.run --offline-team-ack` 只限团队离线入口；`final` 模式仍关闭。新 Candidate 是外部研发接口，未通过目标运行认证，不能直接当成赛事提交器。
4. 最终应按届时官方规则、框架和验证器检查 frame/label、帧数、半场外 X、格式、版本与哈希；这些可先用合成输入核验。官方五分钟 GT 不用于训练、调参或模型排序。没有合格候选时保留官方 reference 回退方案。
5. 团队批准提交前不上传比赛文件；不要把目标运行结果传给云端以选择模型。

## 发布与历史

本机 gh 已完成发布，原先 403/网络失败不再作为阻塞。原安全提交 974216d、98ac1ad 保留原 SHA，远端整合分支另有合并提交保留 team main 历史。V3 协议提交 8de0db7 在训练之前。远端 PR #2 是 Draft，PR #1 未关闭，main 未改动。Git HTTPS fetch 失败后使用 gh Git Data API 上传对象并逐一核对 SHA；没有 force push。最新远端 SHA 和 CI 状态在最终交接消息/PR 中核对，避免在提交本身写入循环自引用哈希。
