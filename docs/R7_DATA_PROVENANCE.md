# R7 公共其他比赛来源

DFL IDSSE：DFL、Bassek等，Scientific Data (2025)，[论文](https://doi.org/10.1038/s41597-025-04505-y)，[Figshare](https://doi.org/10.6084/m9.figshare.28196177)，CC BY4.0。[公开镜像](https://huggingface.co/datasets/pysport/idsse-data/tree/a715a38dfbaf5f58e431727c2b78d174101a703c)固定修订a715a38。开发三场J03WPY/J03WOH/J03WQQ复用R6已校验完整缓存，没有重新下载。两场封存J03WMX/J03WN1在协议提交后按固定名字下载；小型metadata按Git blob摘要核验，大型position按LFS SHA256/大小核验，再完整流式解析和floodlight双parser核对。未下载事件或视频。

SkillCorner：原已使用的公开1874553比赛，固定仓库修订4340d274572876239c154c90bc507a9b3250a656。源文件与许可记录沿用 `external_sources_v2.json` / `external_data_audit.md`。重新从原始坐标建立schema3，仅检测球员；已有严格H/A监督原样保留，phase缺口不变成D。该单场不是新盲测，也不能支持跨供应商完整spell。R7没有使用Metrica事件训练新模型；R6弱监督历史单独保留。

修改：半场/帧重排、坐标单位归一化、位置差分、标签盲原始遮挡、特征消融。来源标签本身没有填补或修改。输入/模型/代码SHA见R7_INPUTS、各协议/结果和模型注册表；所有原始大文件、NPZ、joblib、逐帧输出留在忽略目录。

证据等级：开发集是真正执行的跨比赛验证，但同联赛且共主队；新压力域是合成观测退化；封存结果是一次性的同供应商跨联赛诊断；SkillCorner为部分已知H/A诊断；目标比赛成绩仍未知。任何一级都不能替代目标环境授权离线运行及团队批准。
