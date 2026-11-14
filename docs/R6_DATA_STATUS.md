# R6 公开数据状态

## 2026-11-14T12:32:56.935814+00:00 — R6 启动

PR #2 远端 SHA 093361a3d02ffa9ce195e65ccbb4ad5159624dad 已重新核对，仍为 Draft/Open；本地安全代码 SHA 1d62c7219868dfcbf98921fd0c7464f10dbed55b，是该整合提交的已核对来源。沿用该代码，不从 main 重写。分支 codex/r6-train-idsse-20261114；工作树启动时干净，43 项已有测试通过。V3 模型和报告只读保留。

立即任务：Figshare 实际下载→独立镜像故障转移→IDSSE 解析及真实训练；同时开发离线合成资产预检。目标受限资产与五分钟 GT 不读取。

## 2026-11-14T15:28:34.425715+00:00 — 实际完成

IDSSE J03WPY/J03WOH/J03WQQ六XML已真实下载并核对size/MD5/SHA与完整XML。Figshare HTTP403；HF固定revision成功；floodlight下载同源403，解析API成功。三场425770帧标签全部已知，独立parser逐帧一致。原始数据仅在忽略目录。
