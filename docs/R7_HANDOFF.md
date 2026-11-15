# R7 研发交接

分支 `codex/r7-domain-robust-possession-20261115`，起点真实R6 `7e87ea3`，保留V3/R6全部模型、原协议和历史结果。PR将堆叠在PR2分支`codex/pr1-v2-integration-20261114`之上，main/PR1/PR2不合并不覆盖。最终source SHA与PR/CI由发布回执记录；此文件避免自引用提交哈希。

## 已完成与决定

- E01从可信本地模型重新算三场完整指标，并与R6逐帧输出一致。
- E02/E04完成18个真实HGB拟合，9个bundle；M4直接运动特征消融、M5同样增强消融、M6弱运动学几何对照。M5/M6均未通过预声明门槛，作为替代版本REVERT；模型和失败结果保留。
- E03/E05两套标签盲原始遮球、新运动学重算及完整错误归因。保留natural、原dropout、新short、新long各比赛分母和数据hash。
- E07另供应商已知H/A诊断发现：M2/M3把195个长缺球已知活球帧全部判D；不能凭DFL合成压力分数宣布迁移成功。
- 官方scorer未变，R7边界指标改用其正式支持的typed='pair'；R6历史报告不重写。
- 离线入口 `run_offline_baseline.command`沿用安全R6实现，合成测试覆盖，无真实赛事素材操作。

## 恢复与复现

在本安全代码根目录，Python3.12.5，`requirements-solution.txt`。已有产物禁止覆盖；复现用新的白名单源码副本和新缓存。原始输入与参数固定，重新运行的Git/代码/模型哈希另记，不伪装原产物。通过`requirements-idsse-tools.txt`安装隔离工具；需要下载/floodlight时设置`PYTHONPATH=local_artifacts/r6/tools`和`MPLCONFIGDIR=local_artifacts/r6/matplotlib`。

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 LOKY_MAX_CPU_COUNT=2 .venv/bin/python scripts/verify_r7_artifacts.py --replay
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 LOKY_MAX_CPU_COUNT=2 .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m pip check
```

原始开发数据的重建见R6_REPRODUCE；R7冻结输入核验在research_r7中强制执行。新目录依次运行：

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 LOKY_MAX_CPU_COUNT=2 .venv/bin/python scripts/research_r7.py core
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 LOKY_MAX_CPU_COUNT=2 .venv/bin/python scripts/research_r7.py prepare-stress
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 LOKY_MAX_CPU_COUNT=2 .venv/bin/python scripts/research_r7.py stress
```

源代码携带的历史汇总须保留，复现的新结果以新运行目录results.json为准。封存测试是全研究链一次性使用；**不能在新checkout里把已经看过的J03WMX/J03WN1再称为新盲测**。下载或parser故障发生在评分标记前可修复继续；评分开始后不自动重跑。

## 三人可执行TODO

|优先级/owner|任务与依赖|精确入口|验收标准|证据/阻碍与替代|
|---|---|---|---|---|
|P0 A 工程|审核stacked Draft PR，确认原PR2未被覆盖|`python scripts/audit_public_release.py --out /tmp/r7-review-unique.json`；CI workflow|公开增量无目标数据/密钥，官方scorer不变，测试通过|R7发布回执、CI；若网络受限用安全本地包审查，不强推|
|P0 A 授权成员|在隔离Mac确认Tactical ATD是否齐备，不能回传素材|`scripts/official_assets_preflight.command`；然后`run_offline_baseline.command`|授权成员本地文件/格式校验通过；不要上传回执或目标CSV|R7_OFFLINE_TEAM、R6_OFFLINE_RUNBOOK；缺件则人工用模板联系主办方|
|P0 B 方法|复核反冻结消融与漏D代价，保持M3系数固定|`scripts/verify_r7_artifacts.py --replay`；读CORE/STRESS JSON|逐场分母、5个IoU、D漏检、短控球及边界一致；承认新候选未通过|R7_MODEL_CARD、R7_DECISIONS；回退R6-M1，不开参数网格|
|P0 C 数据|审查一次封存证据和源定义，确认不再用它调参|只读R7_HOLDOUT_RESULTS.json及已冻结协议|两场逐半场独立parser一致；报告联赛/供应商混杂|无新数据时保持研究状态，不循环测试|
|P1 B+C|设计下一份独立公开供应商/真实缺球研究协议|先新建并提交新的protocol；再用公开原始位置建立schema3/4|增加自然缺球已知H/A支持；若无完整D则full_spell=null|R7_TRANSFER_RESULTS已有单场195帧证据不足；不得用目标GT补足|
|P1 A|核对官方截止精确时区与团队交付|由授权成员在原报名邮件中自行核对|人类决定最终baseline/提交；Codex不提交|离线保底始终可恢复；无合法ATD则明确不能推断|

## 模型与回退路径

本地 `local_artifacts/r7/core-v1/{match}-M4/M5/M6.joblib`；实际文件名示例`J03WPY-M4.joblib`。注册表给每个哈希。原始对照与预测在r6目录，R7新域在r7/prepared与stress-v1。固定交接折J03WPY是事先指定（训练J03WOH/J03WQQ），没有按成绩选折。回退 `local_artifacts/r6/idsse-m1-v1/J03WPY-M1.joblib`；V3路径及哈希见R6注册表。

所有候选 `competition_promoted=false`。合并、真实赛事运行及提交仍需团队授权；本轮仅获新Draft PR发布授权。
