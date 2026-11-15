# R8 增量研发状态

恢复锚点：本地与远端 R7B HEAD 均为 `ccc4b2a1382c20112624e86a4dcdf50921673560`；启动时工作树干净。新分支 `codex/r8-possession-centric-20261115`，未改变 R7B、PR2、main。旧模型、协议、结果和失败记录只读保留。R7B 的9个bundle哈希及30组保存概率/解码于本轮再次逐元素精确重放通过，未重新拟合。

E08A 协议与代码 checkpoint：`8086ec6`。两个新候选均使用已有G1概率，H1分层、H2额外有期限双端锚点；M2/M3/G1原解码精确回放对照。8项新增合成测试通过后开始全场评分；新输出独占写入 `local_artifacts/r8/e08a/`。模型训练次数0。进行中的结果尚不能称为候选通过。

恢复命令（未完成时核验并跳过已完成receipt；完整完成后拒绝覆盖）：

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 LOKY_MAX_CPU_COUNT=2 .venv/bin/python -u scripts/evaluate_r8_decoder.py
```

运行日志 `/tmp/r8-e08a.log`；只有一个评分writer。系统进程清单的 `ps` 被沙箱拒绝，未绕过；旧训练进程的工具session已退出，本轮核验进程已完成。输入/模型哈希见 `R8_PROTOCOL.json`，逐项成本和预测哈希见追加账本 `R8_EXPERIMENT_LEDGER.jsonl`。尚未读取新holdout。

R7B [Draft PR #3](https://github.com/dlr9/more-than-a-hack-2027-msn/pull/3) 已创建并核验 `draft=true, merged=false, head=ccc4b2a`。连接器此前403；本轮额度恢复后正常通过主机审批，由gh创建，没有绕过审批。PR3以PR2分支为base。GitHub报告mergeable不等于团队已批准合并。

PR2方向8个缺失提交只读核对：6个原官方历史提交、2个安全整合merge。比较返回新增文件仅data说明、公开GT/基线结果、starter notebooks；没有solution/scripts/tests的遗漏修复。仅查询路径/统计，未打开GT/目标衍生输出或notebook内容，不拉取这些文件。R7B领先21、落后8；由A在审核合并时保留两边历史，本轮不重写refs。

下一步：完整E08A评分和门槛判定；失败则冻结E08B短缺球物理插补协议，不扫描惩罚参数。正式交付P0独立：A/C在授权隔离离线环境确认ATD、baseline、199319行格式与报名邮件截止时区。当前没有正式资产齐全、目标baseline运行或提交成功的证据；Codex不访问这些素材或派生结果。建议团队11/16北京时间18:00前冻结合法可运行提交路径，最终截止时区由人核实。
