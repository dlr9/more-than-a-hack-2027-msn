# R6 公开数据训练复现

所有命令从安全源码目录运行，Python 3.12.5。不要完整检出官方 GT、baseline CSV、赛事 data 或私人报告。推荐交接的白名单公共源码 ZIP；若使用 Git，沿用 PR2 的非 cone 稀疏检出规则，并添加公开 docs 和 requirements。源码 ZIP 内的 `SOURCE_REVISION.txt` 记录实际交付提交。

## 保留已有产物

本机当前 `local_artifacts` 已有全部实际结果。不要在这里重复启动训练或删除输出目录。重跑使用另一个全新的安全源码副本；所有新结果都记录自己的 SHA。源码 ZIP 无 `.git`，需要先 `git init` 并提交白名单源码，供实验记录 Git SHA；这个重建提交不是原始训练提交，原 SHA 在随包文件和历史 ledger 中保留。

模型 ZIP 只包含本轮外部训练模型和原 V3 模型/校准器，不包含原始数据或逐帧缓存。解压到安全源码根目录后可先检查：

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-solution.txt
.venv/bin/python scripts/verify_r6_artifacts.py
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 LOKY_MAX_CPU_COUNT=2 .venv/bin/python -m unittest discover -s tests -v
```

模型注册表列出 30 个保存文件（18 个 R6 bundle/头、12 个历史 V3 bundle/校准器）；这不等于30次训练。R6实际新拟合17个HGB，M3和F2的部分bundle复用既有头。仅加载本团队来源已校验的 joblib。

## IDSSE 从公开原始文件到训练

下载与独立 parser 的可选依赖安装到隔离目录，不改动基础数值环境：

```sh
.venv/bin/python -m pip install --no-deps --target local_artifacts/r6/tools -r requirements-idsse-tools.txt
export PYTHONPATH="$PWD/local_artifacts/r6/tools"
export MPLCONFIGDIR="$PWD/local_artifacts/r6/matplotlib"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 LOKY_MAX_CPU_COUNT=2

for match in J03WPY J03WOH J03WQQ; do
  .venv/bin/python scripts/download_idsse.py --match "$match" || exit 1
  .venv/bin/python scripts/prepare_idsse.py \
    --positions "local_artifacts/r6/idsse-hf/DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-${match}.xml" \
    --matchinfo "local_artifacts/r6/idsse-hf/DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-${match}.xml" \
    --out "local_artifacts/r6/prepared/${match}" || exit 1
  .venv/bin/python scripts/crosscheck_idsse.py --match "$match" || exit 1
done
.venv/bin/python scripts/train_idsse_r6.py
.venv/bin/python scripts/ablate_r6_decoder.py
```

训练需要预先恢复 `local_artifacts/v3/1874553/baseline.joblib`（模型 ZIP 已提供），用于固定V3对照。仅复制模型到重跑目录；不要把原 R6 输出目录复制过去，否则训练器会拒绝覆盖。输入哈希必须匹配 `R6_IDSSE_INPUTS.json`；失败则检查源版本、依赖和解析代码，不能为了运行成功擅改冻结哈希。输出模型二进制可能因运行 Git/依赖元信息不同而有不同哈希；比较数值指标和预测，并为复现结果另建注册表，不能覆盖原注册表假装同一产物。

保留现有本机预测缓存时，可以核对九个保存bundle在两个域的精确回放：

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 LOKY_MAX_CPU_COUNT=2 .venv/bin/python scripts/verify_r6_artifacts.py --replay
```

三场共425,770帧；每次流式prepare与floodlight核对需读取完整公开XML。独立parser核对会拒绝覆盖，复现也应使用新目录。训练实际预算为12个60轮小HGB；M3直接复用M2概率。封存J03WMX/J03WN1不在下载CLI允许列表中。

## 可选：从头复现 SkillCorner/Metrica 退路

只有在新的空缓存目录运行以下步骤；下载器会写公开其他比赛缓存。无需重跑V2旧参数网格。

```sh
.venv/bin/python scripts/fetch_external_v2.py
.venv/bin/python -c 'from scripts.experiment_v2 import prepare; prepare()'
.venv/bin/python scripts/audit_evaluation_v3.py
```

从模型包恢复原四折V3后运行 `.venv/bin/python scripts/train_r6.py`。若还要从头拟合V3，改用 `scripts/research_v3.py` 和 `scripts/summarize_v3.py`，并遵循 `candidate_runbook_v3.md` 的冻结版本核验；不要在已恢复的V3目录中这样做。F1/F2没有完整连续真值，完整spell保持null。

## 来源与修改

IDSSE：DFL；Bassek等（2025），[Scientific Data论文](https://doi.org/10.1038/s41597-025-04505-y)，[Figshare原始发布](https://doi.org/10.6084/m9.figshare.28196177)，CC BY 4.0。实际采用[固定Hugging Face镜像修订](https://huggingface.co/datasets/pysport/idsse-data/tree/a715a38dfbaf5f58e431727c2b78d174101a703c)。只取三场metadata与observed positions；修改包括半场/帧对齐、H/A/D编码映射、坐标与位置差分特征、固定原始球遮挡。原始文件未修改，大小/MD5/SHA及独立核验见来源记录和输入清单。

SkillCorner与Metrica的固定版本、用途及许可说明沿用 `external_sources_v2.json` 和 `external_data_audit.md`。模型与源码的交接不包含赛事目标资料。目标资产操作只允许团队另行按 `R6_OFFLINE_RUNBOOK.md` 在隔离离线环境亲自执行。
