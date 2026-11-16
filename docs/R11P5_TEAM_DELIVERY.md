# 独立正式交付矩阵

科研REVERT与正式提交分离。用户最后反馈“不知道，应该还没提交”；不能改写为已提交或已验证。

|负责人|今天必须独立确认|真实状态|
|---|---|---|
|A|报名邮件截止准确时间/时区、团队名、Form及回执、公开代码和技术文档|UNKNOWN|
|B|隔离本地Tactical XML+TXT、实际runner命令、精确199319行CSV验证|ASSETS=UNKNOWN; RUNNER_VALIDATED=UNKNOWN; CSV_VALIDATED=UNKNOWN|
|C/Codex|外部科研源码、完整评分、恢复、DraftPR|本轮实验完成，候选REVERT；发布状态见PR及本地release receipt|
|A|正式Form提交|SUBMITTED=UNKNOWN|

B沿用授权本地Tactical运行链和通用解析器；不能把五分钟GT选参配置自动视为合法候选。云端没有运行正式资产预检、Tactical推断或验证最终CSV，科研CI不能代替这些动作。先完成可再生、合规的保底。队员只回复ASSETS/CSV_VALIDATED/SUBMITTED的YES/NO，不发送任何目标材料、哈希或派生统计。A手动审核PR栈，不自动合并。没有授权云端发送邮件/消息或代交Form。
