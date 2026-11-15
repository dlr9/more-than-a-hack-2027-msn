# R9 V2 当前恢复点

分支 `codex/r9-v2-spell-events-20261115`，从远端/本地核实一致的R8 `d73ca0f059f71d74a298236f8aed512a65eafad1` 接续。PR4已核实open、draft=false、未合并，原CI43723774465成功（公开无数据合成测试）。未改旧远端ref、未重新上传PR4。最新本地/远端发布SHA见 `local_artifacts/r9-v2/release-receipt.json`；没有该receipt则不能声称发布完成。

实际完成D36全场/72半场预算；E1错误合同24结果保留、E1fix24、E2机制修正24；A固定2×2共48；A2单次固定分层消融24+部分跨源8。新增拟合0、权重0；所有历史模型及失败结果保存。相关冻结模型15个hash通过；有效预测128个已逐元素精确重放、官方完整spell评分一致；152项本地及152项干净无数据/模型副本测试均通过。pip check、路径闸门、diff check通过，不把历史R8重放算成本轮。

决定：HC_M3（M2 Alive+C1 Home+M3）KEEP_RESEARCH_COMPONENT：自然均值.584981 vs M3 .564637；压力域非全面超过H1。事件E1/E2 REVERT替代；C1 Alive及本轮分层解码REVERT。Metrica HC_M3弱D漏1846/25619，对照M3 1895；分层25380。SC195帧仍全D。TRANSFER_NOT_ESTABLISHED；competition_promoted=false。

下条无重训恢复命令：

```bash
cd /Users/dailuren/Documents/ChatGPT/barca/review-pr1
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 LOKY_MAX_CPU_COUNT=2 .venv/bin/python scripts/verify_r9_artifacts.py
```

各实验完整results/receipt均在 `local_artifacts/r9-v2/`，没有待恢复拟合。日志、checkpoint及中断处理见REPRODUCE；若verify正在运行，先查看`/tmp/r9-verification.log`，不要再启第二个。新增结果与账本仅单writer；oracle只留汇总。WOY/WR9未打开；WMX/WN1已消费，不当盲测。磁盘空闲曾降到约200MB，不删除历史、不新增大下载。

A：审查新Draft、远端SHA、CI和stacked历史，人工决定合并；B：冻结本轮参数与Pareto卡，不再用三场搜索，下一步需要独立事件时刻或完整跨供应商HAD；C：核验ELASTIC事件时间监督的使用方案/重叠限制，PathCRF现成输入合同不适合直接替换，赛后研究。正式目标资产/离线运行/验证/提交由授权真人独立完成，见人类清单。

TARGET_ASSETS=UNKNOWN；LOCAL_CSV_VALIDATED=UNKNOWN；SUBMITTED=UNKNOWN。建议11/16北京18:00冻结可用保底；11/17准确官方时区由人查报名邮件。研发不阻塞保底。
