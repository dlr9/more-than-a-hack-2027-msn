# R9 V2 精确恢复

安全工作树：`/Users/dailuren/Documents/ChatGPT/barca/review-pr1`；分支 `codex/r9-v2-spell-events-20261115`；基底R8/PR4为 `d73ca0f059f71d74a298236f8aed512a65eafad1`。PR4 open、draft=false、未合并；旧branch/ref不改。已发布PR4不重复上传。

冻结代码：D `fda7cd3`；E1错误合同 `4ce5f45`；E1fix `eb5bced`；E2 `720325a`；A `e45d3f3`；A2 `3da37b0`。每个结果同时保存协议SHA、实际运行代码hash与Git SHA；不单凭Git HEAD推断源码。E2的测试fixture修订在3da37b0，不影响推断代码。所有权重复用，不需要重新训练。

只读恢复（不会拟合或覆盖实验）：

```bash
cd /Users/dailuren/Documents/ChatGPT/barca/review-pr1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 LOKY_MAX_CPU_COUNT=2
.venv/bin/python scripts/verify_r9_artifacts.py
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m pip check
.venv/bin/python scripts/check_public_tree.py
git diff --check
```

输出 `local_artifacts/r9-v2/e09d,e09e1,e09e1fix,e09e2,e09a,e09a2` 全部已完成。D是36全场/72半场汇总、无oracle数组；E1错误合同24条保留；有效E1fix/E2各24条、A48条、A2全场24条加8个部分跨源组合。对应公开RESULTS仅汇总，权重/原始数据/逐帧预测不进Git。总计128个有效新预测的只读核验入口见上；24个错误合同输出只核哈希，不冒充当前代码结果。

从合法公共源恢复原缓存按R6/R7/R7B/R8 runbook；不要重新下载已有文件。所有数据来源/固定版本沿用 `R7_DATA_PROVENANCE.md`、`R7B_MODEL_CARDS.md`。新候选输出独立exclusive文件；已有receipt必须匹配协议/代码/预测哈希才复用。当前根目录数据不得替换为目标赛事素材。

首次重建实验只在独立公开源码副本、空的新实验输出路径执行对应历史checkpoint的runner；原E1不能用修复代码重标，原E1fix runner代码hash也与加入E2后的脚本不同，已有结果应只读verify。禁止删除现有目录以绕过exclusive保护。

日志：`/tmp/r9-e09d.log`、`/tmp/r9-e09e1.log`、`/tmp/r9-e09e1fix.log`、`/tmp/r9-e09e2.log`、`/tmp/r9-e09a.log`、`/tmp/r9-e09a2.log`、`/tmp/r9-verification.log`。152项本地测试有实际stdout，最终干净副本/重放核验以release receipt为准。无新训练进程或未完成拟合；若会话中断，先读STATUS及receipt，勿重复启动writer。CPU线程上限2。

R8的126测试/96解码/12C1概率/36历史分数重放是已有历史证据，本轮不重复冒充运行；本轮核验的是新128预测与15个相关模型hash。
