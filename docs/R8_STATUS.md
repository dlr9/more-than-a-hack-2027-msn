# R8 增量研发状态

恢复锚点：本地与远端 R7B HEAD 均为 `ccc4b2a1382c20112624e86a4dcdf50921673560`；启动时工作树干净。新分支 `codex/r8-possession-centric-20261115`，未改变 R7B、PR2、main。旧模型、协议、结果和失败记录只读保留。R7B 的9个bundle哈希及30组保存概率/解码于本轮再次逐元素精确重放通过，未重新拟合。

E08A协议/代码checkpoint `8086ec6`、E08B `7b36628`、锚点修复与E08C训练 `ffa9e3d`。完整完成：A60组、锚点修复12组、B12组、C12组DFL自然/原遮球/短/长全场评分；SC/Metrica仅部分源标签迁移。实际新增C1三个模型包、6次HGB拟合，旧成功模型未重训。输入和代码hash在receipt，评分恢复期间源码修改由code_hashes记录，不把Git SHA单独当作完整执行身份。

下一条精确恢复命令（只读核验，不再训练或覆盖结果）：

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 LOKY_MAX_CPU_COUNT=2 .venv/bin/python scripts/verify_r8_artifacts.py
```

运行日志：`/tmp/r8-e08a.log`、`/tmp/r8-anchor-fix.log`、`/tmp/r8-e08b.log`、`/tmp/r8-e08c.log`及`/tmp/r8-e08c-recovery.log`。所有训练/评分工具session已退出，只有一个账本writer串行写入。系统进程清单的ps曾被沙箱拒绝，未绕过。一次C评分进程exit1无traceback、原因未确定；WOH-dropout孤立NPZ经双头/解码逐元素核验后恢复receipt，文件未覆盖、模型未重训。输入/模型哈希见各PROTOCOL/注册表，逐项成本和失败记录见追加账本。未读取新holdout。

实测决定：H1原遮球mean .482090（M3 .461130），短遮球.456105低于M3 .466996；SC195帧仍全D、Metrica漏弱D3891→4967。H2首版因首个重现球观测后向速度缺失而从未触发，保留失败并修复；修复后2142帧锚点，仅短遮球WQQ有微小spell变化，未过门槛。B1短遮球.431498、原遮球.439044，未超过强对照。C1自然.583183（三场均提高），但真D漏检在缺球域明显增加；Metrica漏弱D25126/25619。全部REVERT替代资格，KEEP可复现研究组件；没有正式晋级或目标成绩声明。

验证：126项本地测试、126项干净无数据/模型副本测试通过；pip check通过，官方scorer未改动。R6/R7共39个旧模型哈希核验通过；R7B9个包/30组精确预测重放通过。本轮R8重放结果以`/tmp/r8-final-replay.log`和最终发布核验receipt为准。所有旧模型与失败结果保留。

R7B [Draft PR #3](https://github.com/dlr9/more-than-a-hack-2027-msn/pull/3) 已创建并核验 `draft=true, merged=false, head=ccc4b2a`。连接器此前403；本轮额度恢复后正常通过主机审批，由gh创建，没有绕过审批。PR3以PR2分支为base。GitHub报告mergeable不等于团队已批准合并。

R7B实际远端CI：[pull_request run 43697430999](https://github.com/dlr9/more-than-a-hack-2027-msn/actions/runs/43697430999) success，push run43676717375同样success。R8仅发布独立Draft、以R7B为base；最终远端SHA/PR/CI以`local_artifacts/r8/publication-receipt.json`和本轮最终交接为准，不移动R7B/PR2/main。

PR2方向8个缺失提交只读核对：6个原官方历史提交、2个安全整合merge。比较返回新增文件仅data说明、公开GT/基线结果、starter notebooks；没有solution/scripts/tests的遗漏修复。仅查询路径/统计，未打开GT/目标衍生输出或notebook内容，不拉取这些文件。R7B领先21、落后8；由A在审核合并时保留两边历史，本轮不重写refs。

下一步最高价值任务：A审查新Draft与整合风险；B冻结失败、核查真D/假D与重采样先验变化，不继续同三场调参，不无证据启动duration/deep模型；C在授权隔离离线环境确认ATD、baseline、199319行格式与报名邮件截止时区。当前没有正式资产齐全、目标baseline运行或提交成功的证据，Codex不访问这些素材或派生结果。建议团队11/16北京时间18:00前冻结合法可运行提交路径，最终截止时区由人核实。研究回退为冻结M2/M3或R6-M1，正式交付保底由人按现有官方reference/smooth50入口决定。
