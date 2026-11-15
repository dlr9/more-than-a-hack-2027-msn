# R9 V2 状态

独立分支 `codex/r9-v2-spell-events-20261115`，起点远端/本地 `d73ca0f059f71d74a298236f8aed512a65eafad1`。PR4已核实open、draft=false、未合并，CI43723774465成功（无数据合成测试）。未重复上传PR4。旧模型和失败记录不动。

已验证R8注册模型哈希，复用历史126测试/96解码等receipt，不把历史记录冒充新重跑。新诊断6项合成测试通过；E09-D协议冻结后运行，无新增拟合。

恢复：`OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python scripts/diagnose_r9_spell_loss.py`。exclusive新输出；已完成同行记录按协议/代码哈希复用。单writer；不保存oracle标签。WOY/WR9未打开，WMX/WN1已消费不重命名盲测。

A核验公开增量与新Draft；B执行预算→物理机制→完整对照；C由真人核实独立标签/许可与正式离线通道。目标资产、实际CSV验证、提交均无证据：TARGET_ASSETS=UNKNOWN；LOCAL_CSV_VALIDATED=UNKNOWN；SUBMITTED=UNKNOWN。建议11/16北京18:00冻结；11/17准确时区由人查邮件。正式运行保底不等研发。
