# R6：授权成员的 Mac 离线资产预检与官方参考运行器

本入口只能由获授权成员在独立本地环境亲自启动。Codex 本轮只用程序生成的临时夹具运行了测试，**未查看、扫描或处理真实比赛资产，未运行目标比赛 baseline**。下面的路径都是示例；不要将真实目录、输出 CSV、日志或校验回执粘贴回 Codex、上传 GitHub 或加入公共交接包。

## 隔离准备

1. 在读取资产之前，将本 PR 的公共代码复制到不受 Codex 管理、也不受 iCloud/Dropbox/OneDrive 等同步的独立目录，例如 `/Users/Shared/mtah-offline/code`。只复制公开源码和必要依赖，不复制工作区的数据缓存或目标资料。目标资产与结果分别放在独立目录，例如 `/Users/Shared/mtah-offline/assets` 和一个尚不存在的 `/Users/Shared/mtah-offline/run-001`；二者不能在源码目录中，不能互相包含。
2. **尚未放入受限资产时**准备 Python 和 `requirements-solution.txt` 依赖，检查 `numpy` 可导入。运行器不会安装依赖、联网下载、调用模型、发送遥测或上传任何结果。建议独立创建本地 `.venv`；复制的虚拟环境可能包含旧的绝对路径，不能直接假设可用。
3. 关闭 Codex、浏览器上传页、云同步和任何会读取该目录的代理；断开 Wi-Fi、有线网络及 VPN。授权成员从独立 Terminal/Finder 启动。启动脚本还强制使用 macOS `sandbox-exec` 的 `(deny network*)`，无法创建沙箱则停止，没有无沙箱回退。该工具在某些 macOS/企业策略下不可用，应先在**不含资产的合成环境**确认；不可用时不要删除保护继续执行。
4. 代码拒绝在 `$HOME/Documents/ChatGPT`、`$HOME/.codex` 或 `MTAH_FORBIDDEN_WORKSPACE` 指定目录及其子目录处理资料，也拒绝继承 `CODEX_*` 环境的启动。若自己的 Codex 工作区另有路径，执行前设置 `MTAH_FORBIDDEN_WORKSPACE`。这些检查不能发现所有自定义工作区、同步目录、后台程序或远程挂载；网络沙箱只约束本进程，**不证明其他进程不会上传文件**。人工断网、关闭同步、目录隔离仍是必要条件。不存在一个 `--ack` 参数可替代这些条件。

## 预期资产与预检

文件名直接来自公开 `games.py::asset_paths`，不会扫描全盘、递归枚举目录、解压归档或打开视频/ATD 内容。只对用户显式指定根目录下的以下九个路径检查存在性、普通文件类型和字节数；拒绝资产符号链接或目录逃逸。

| 相对路径 | 官方 smooth50 是否必需 |
|---|---|
| `tactical/BAR-ATH_tactical_atd.xml` | 是，且需非空 |
| `tactical/BAR-ATH_tactical_atd.txt` | 是，且需非空 |
| `tactical/BAR-ATH_tactical.mp4` | 否 |
| `tactical/BAR-ATH_tactical_homography.csv` | 否 |
| `tactical/BAR-ATH_tactical_smart_tagging.xml` | 否 |
| `broadcast/BAR-ATH_broadcast.mp4` | 否 |
| `broadcast/BAR-ATH_broadcast_atd.xml` | 否 |
| `broadcast/BAR-ATH_broadcast_atd.txt` | 否 |
| `broadcast/BAR-ATH_broadcast_homography.csv` | 否 |

在 Finder 双击 `scripts/official_assets_preflight.command`，按提示输入绝对目录。若没有可执行权限，先在独立 Terminal 对两个 `.command` 文件执行 `chmod +x`。也可明确传参：

```sh
export MTAH_OFFLINE_PYTHON=/Users/Shared/mtah-offline/code/.venv/bin/python
/Users/Shared/mtah-offline/code/scripts/official_assets_preflight.command \
  --data-dir /Users/Shared/mtah-offline/assets
```

Finder 启动默认使用源码根目录 `.venv/bin/python`；外部 Python 可通过 Terminal 设置 `MTAH_OFFLINE_PYTHON`。路径含空格时请加引号。若只有两个视频，返回非零状态并显示：**“缺少 Tactical ATD XML/TXT，不可运行官方 baseline，需取得赛事完整授权资产。”** 需由团队通过正式授权渠道补齐，不能用公共 GT 或视频推导来冒充 ATD。若资料在压缩包中，请获授权成员离线自行确认与解压到上述布局；本预检器不检查归档内容。

`reference_ready=true` 只表示两份 Tactical ATD 文件存在且非空，不证明格式正确、目录隔离成立或 baseline 已运行。

## 运行固定官方参考

在同样的隔离条件下，双击 `scripts/run_offline_reference.command`，输入资产目录和**尚不存在**的结果目录；也可执行：

```sh
/Users/Shared/mtah-offline/code/scripts/run_offline_reference.command \
  --data-dir /Users/Shared/mtah-offline/assets \
  --output-dir /Users/Shared/mtah-offline/run-001
```

运行器只打开 Tactical XML/TXT，先严格验证两队映射、track slot 数、逐行 `frame_id=1..199319`、坐标三元组和缺失值格式，避免官方宽松 loader 静默跳过坏行。随后直接调用 `starter.baseline.nearest_player_labels` 和 `smooth(window=50)`，不训练、不选择窗口、不读 GT，不打开视频、Smart Tagging、Homography 或 Broadcast 数据。官方半场外区间由 `games.py::half_video_blocks` 决定，并强制写 X；半场内预测保持官方 smooth50 结果。此 X 掩码是唯一输出变化，不能宣称已复现目标成绩。

输出均只保存在隔离目录：

- `baseline_smooth50.csv`：精确 `frame,label` 表头、199319 行、frame 从 0 到 199318、仅 H/A/D/X、半场外 X。写入 `.part` 后完整检查成功才改名；格式失败不会留下貌似成功的 CSV。
- `LOCAL_ONLY_receipt.json`：输入 XML/TXT、输出 CSV、关键代码 SHA256，以及格式验证结果。回执含目标衍生统计，**同样禁止传入 Codex 或公开仓库**。

结果目录已存在时拒绝覆盖；失败现场保留，修正输入后使用新的结果目录。输出通过格式验证不等于标签正确或达到官方分数。程序不评分、不提交、不合并 PR，团队另行审核是否采用。

## 本轮实际验证

```sh
.venv/bin/python -m unittest discover -s tests -p test_offline_assets.py -v
bash -n scripts/official_assets_preflight.command
bash -n scripts/run_offline_reference.command
```

8 个合成测试通过：只有视频的缺资产反馈且禁止打开内容、符号链接逃逸拒绝、官方 smooth50 半场内逐帧等价、坏时间轴拒绝、CSV 顺序/数量/标签/半场外 X 检查、CLI 隔离拒绝、合成运行 SHA256 回执与禁止覆盖，以及两份shell入口在继承多个CODEX环境变量时立即拒绝。这里的测试绕过 CLI 沙箱仅调用纯函数处理新建临时合成数据；没有启动真实资产入口。Mac OS 沙箱是否能在团队机器启动，需授权成员在无真实资产的隔离环境独立验证。

KEEP：保留本离线保底入口。下一步由授权成员先确认 Tactical 两份 ATD 是否齐备，再按以上步骤运行。官方 smooth50 公布成绩属于官方参考说明，不是本轮重新实测结果。
