# R6 持续交接

## 2026-11-14T12:32:56.935814+00:00 — R6 启动

PR #2 远端 SHA 093361a3d02ffa9ce195e65ccbb4ad5159624dad 已重新核对，仍为 Draft/Open；本地安全代码 SHA 1d62c7219868dfcbf98921fd0c7464f10dbed55b，是该整合提交的已核对来源。沿用该代码，不从 main 重写。分支 codex/r6-train-idsse-20261114；工作树启动时干净，43 项已有测试通过。V3 模型和报告只读保留。

立即任务：Figshare 实际下载→独立镜像故障转移→IDSSE 解析及真实训练；同时开发离线合成资产预检。目标受限资产与五分钟 GT 不读取。

## 2026-11-14T15:35:17.638787+00:00 — 真实训练完成、可恢复冻结

当前分支 `codex/r6-train-idsse-20261114`。训练基于 `8d9f14c`，M3基于 `1f166da`；最终交付提交由源码包 `SOURCE_REVISION.txt` 和 `git rev-parse HEAD` 给出，避免提交自引用。PR2仍为原Draft；R6未对外发布、未合并。

已完成三场公开IDSSE下载、425,770帧解析和floodlight六个半场独立核对；17个HGB真实新拟合，M1/M2三场留一验证，M3固定概率缺球消融。自然完整源标签spell均值V3适配0.3569、M1 0.5610、M2/M3 0.5646；缺球M1 0.4071、M2 0.4361、M3 0.4611。F1回退，F2/M2/M3仅实验，M1研发基线，无赛事晋级。77项测试通过。原12个V3文件SHA不变。

- 结果/失败原因：`docs/R6_RESULTS.md`；完整各比赛指标：`R6_IDSSE_RESULTS.json`、`R6_DECODER_RESULTS.json`、`R6_FALLBACK_RESULTS.json`。
- 代码入口/依赖/命令：`docs/R6_REPRODUCE.md`；模型哈希与回退：`docs/R6_MODEL_REGISTRY.json`；每轮协议、输入、Git SHA：追加式ledger。
- 原始公开数据：`local_artifacts/r6/idsse-hf/`；已准备输入：`local_artifacts/r6/prepared/`；模型/概率：`idsse-m1-v1/`、`missing-stability-v1/`、`fallback-v1/`（均位于local_artifacts/r6）。不移动、不覆盖。
- 团队离线保底：`docs/R6_OFFLINE_RUNBOOK.md`，两份.command已完成合成测试与shell语法检查。未查看实际资产，未运行目标baseline；只有视频时会明确报缺Tactical XML/TXT。

下一项优先任务：冻结独立压力验证协议后，测试此前未用的公开比赛及供应商观测差异；目前三场共同主队、同联赛且死球坐标冻结，不能继续据此调参数或宣布目标泛化。J03WMX/J03WN1仍未下载、未评分。不要扩展网络模型来替代独立证据。

发布待办：代码、结果和离线入口打包后，按R6第12节对外发布前由团队确认，再创建独立Draft PR；PR2/PR1/main均不覆盖。

### 2026-11-14T15:38:40.478147+00:00 — 交付前回归修复

九个IDSSE模型两个域的保存预测逐帧精确回放，六个M1/M2概率精确回放；30个模型/校准器文件SHA核验通过。发现离线shell的`${!CODEX_@}`在多个变量时展开成多个参数，现改为单参数展开并加入shell级回归测试，先失败后通过。最终测试数78。该修复不影响任何训练特征、模型或预测。
