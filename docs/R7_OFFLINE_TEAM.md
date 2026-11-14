# 授权团队的离线保底

沿用 `R6_OFFLINE_RUNBOOK.md` 的隔离、断网及目录要求。R7入口 `scripts/run_offline_baseline.command` 只委托既有经合成测试的官方smooth50运行器，不新增模型选择、联网、上传或评分。先运行 `scripts/official_assets_preflight.command`。两份入口均必须由获授权成员在Codex工作区之外亲自启动。

若只有两个MP4，缺少 `tactical/BAR-ATH_tactical_atd.xml` 与 `tactical/BAR-ATH_tactical_atd.txt`，无法运行ATD baseline。这里只陈述条件，未检查团队实际文件。不要将预检输出、目标文件、结果CSV或回执发回Codex。运行器拒绝覆盖旧结果，校验帧编号/数量、标签、半场外X并生成仅限本地的哈希回执。

## 可由团队自行发送的补件邮件模板（尚未发送）

Subject: More than a Hack 2027 — request to confirm authorized ATD package access

Hello organizers,

We are an enrolled three-person team. Could you please confirm the authorized download location for the Tactical ATD metadata XML and raw tracking TXT, and whether our package should also include Broadcast ATD, Tactical Smart Tagging and homography assets? Our local preflight will require the Tactical XML/TXT to run the supplied baseline. Please also confirm the exact Phase 1 submission deadline and timezone.

Team name / registration reference: [team member fills in locally]

Thank you.

不得附目标视频、目标数据或实验结果。团队确认缺件后通过原参赛授权联系渠道发送；Codex没有代发邮件。
