# R9→正式提交：人类隔离离线通路

TARGET_ASSETS=UNKNOWN；LOCAL_CSV_VALIDATED=UNKNOWN；SUBMITTED=UNKNOWN。云端没有运行目标预检/推断/验证。只需团队口头反馈YES/NO，不回传素材、目标CSV、哈希回执或私人效果报告。

1. C（授权真人）：在无云同步、断网、关闭Codex的独立目录确认Tactical ATD XML+TXT。仅两个视频不够；缺ATD自行查官方子包/联系主办方。确认报名邮件准确截止时间/时区、Google账号、注册队名。
2. A：审查stacked PR2→PR3→PR4→本轮新Draft，人工决定整合；保留双方历史。用公开依赖锁 `requirements-solution.txt` 和既有 `docs/R6_OFFLINE_RUNBOOK.md` / `docs/R7_OFFLINE_TEAM.md` 准备隔离环境。正式运行用已审查 `scripts/official_assets_preflight.command`、`scripts/run_offline_reference.command`；这些入口默认拒绝Codex环境。R9没有自动替换官方reference/smooth50。
3. C：先生成可提交保底，然后在同一个准备上传的文件上运行 `evaluation/validate_submission.py`：199319数据行，frame0..199318，H/A/D/X，半场/时基正确。由公开源码及所发资产精确再生同一CSV，文件SHA仅记录在离线本地；不要上传任何目标派生物给Codex。
4. B+A：技术MD/PDF填写真实注册队名、三人职责、训练来源与许可、实际采用模型、版本/依赖/运行命令、有效控制处理和局限。未采用的R9组件不能当正式算法成绩。官方五分钟GT不得训练/选型。若拟用目标算法伪标签，先由人取得规则适用澄清，否则延期。
5. C+A：建议11/16北京时间18:00冻结已验证保底；11/17官方邮件时区优先。手工Google Form上传以队名命名CSV与技术文档、公开代码URL，独立确认提交成功。合并和最终提交均由团队决定。

研发失败不影响保底。HC_M3仍为研究组件，无完整跨供应商HAD，不能以DFL成绩承诺目标得分；研究更新须经团队明确决定，截止前不冒险替换已验证提交。
