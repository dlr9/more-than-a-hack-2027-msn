# R11 PathCRF 实物、合同与CPU执行回执

## 结论

**NO_GO_PATHCRF_ENVIRONMENT_DISK_RUNTIME；保留可恢复检查点与源码，转H2。**
确实下载了一个原作者权重、逐项核对模型代码/输入合同，并在真实公开J03WPY窗口运行适配及扰动测试。
CPU加载尝试在`import torch`报`ModuleNotFoundError`；**checkpoint CPU load、神经网络forward、CRF decode均未完成**。
未以此断言PathCRF架构不可用。没有模型成绩、新训练、完整spell评分或正式候选晋级。

## 唯一选定的模型与来源

- 源码：[hyunsungkim-ds/pathcrf](https://github.com/hyunsungkim-ds/pathcrf/tree/1b7c6f85a6b78c92e4543fde72ca2b86a933f997)，HEAD `1b7c6f85a6b78c92e4543fde72ca2b86a933f997`。
- 权重：`saved/140/model/state_dict_best.pt`，Git blob `e68a9e21d727203637fa58ad4df23c4656d9c747`，实际下载 **19,322,810 bytes**。
- 权重SHA256：`c82235a17416354b5544f85721ca850304765737a2d9d5435dcdefd52a0d3700`。
- `args.json` blob `5ec32047d303c6cbfdbc17d59f81ee08c635b939`；SetTransformer/SAB + dynamic sparse CRF，8维、11/队+4场外节点、5FPS、10秒、声明3,431,031参数。实际参数总量需成功load后核实。
- 已读取并下载原始`LICENSE`，MPL-2.0；权重与必要源码只在忽略目录保存。代码许可不授权任何目标比赛资产。
- ZIP格式静态检查：576个成员，展开19,208,933 bytes；pickle只反汇编未执行。GLOBAL仅OrderedDict、Torch存储类型及`_rebuild_tensor_v2`，这并非任意pickle的安全证明；恢复使用`weights_only=True`且固定SHA。
- 原checkpoint中存储设备含`cuda:0`，恢复明确`map_location='cpu'`，不得因此误称必须GPU。
- 原作者README提供7场Sportec；`args`为LOMOCV fold6。`train.py`排序文件、前一场作val；按仓库7场文件名推断，WPY/WOH属于训练、WQQ为验证、WR9为留出。所有本次WPY检查明确`DIAGNOSTIC_ONLY`，没有读取封存WR9/WOY数据。

## HF镜像核实（未下载第二个权重）

[HF仓库](https://huggingface.co/Youssef16Ahmed/nova-pathcrf-finetuned) API确认public、ungated、模型卡MPL-2.0，revision `c86163f1c40c5f937879a9f9ed2b1b5031212ccb`。
`round3/final/model/state_dict_best_team.pt`只是19,325,599 bytes，LFS SHA256 `719cb853446431948a8879d97ea98019357f04bc32801e241740b32b77aa5586`；无须下载580MB全库。
已只读模型卡与`round3/FINAL.json`：14train/1val/5test，val2017461；test2010085/2016236/2006229/1959846/1899585。
卡片推荐的1秒换队延迟未迁移；供应商team accuracy不是完整HAD spellF1。
14场训练ID并未在所读卡片中列全，不能宣称和PR1的20场独立；其分组也未经本地重现。
只选择原140是用户指定的canonical版本及最少依赖路线；没有看本地评估成绩选择权重。最多一个checkpoint进入执行尝试。

## 实际运行环境与有界恢复失败

原研发venv Python3.12.5、Intel x86_64；Codex bundled Python3.12.14也没有torch或torch_geometric。
初查两次磁盘约1.4GiB空闲，下载预算的2倍余量通过。随后共享盘骤降，安装前保护触发：

| 实物 | 实测值 |
|---|---:|
| Torch 2.2.2 cp312 Intel Mac wheel下载大小 | 150,797,270 bytes |
| Torch wheel展开总量 | 543,754,206 bytes |
| 展开前最低2倍余量（还未加其余依赖） | 1,087,508,412 bytes |
| 首次下载结束空闲 | 264,519,680 bytes |
| 真实探针执行时空闲 | 273,530,880 bytes |

wheel SHA256 `eb4d6e9d3663e26cd27dc3ad266b34445a16b54908e74725adb241aa56987533`。
未安装、未修改冻结venv、未清理历史或团队文件。没有重新下载第二套大依赖。
第一次HF目录请求因zsh未引用URL问号失败，随后用Python固定URL读取成功；网络故障没有被当成模型失败。

## 已真实运行的输入合同

新增`r11/pathcrf_adapter.py`接受已解析公开观测，仅玩家几何/ID/队伍，不接受球/标签。
半场内连续10秒、10/25→5Hz、H=0/A=1、显式GK、105×68角点坐标、左队先排、[-1/+1队伍,GK,x,y,vx,vy,speed,accel]。
重建运动学采用原代码SG15速度/SG9加速度公式；局部窗口端点和不做原始outlier填补是明确差异，**尚未证明与上游全半场预处理逐元素一致**。
未知GK、缺玩家、缺帧、跨半场或不稳定ID均拒绝；不将46可交换槽位伪装成22个人，不盲补NaN，不以未知当D。
原模型是ball-free，遮球本身不改输入；玩家一起缺失则本接口拒绝，而非保证真实缺球域可用。

实际J03WPY公开raw来源SHA256：`673abd10e7bb5bbab867595da7d663f967d3df531d7aa12d05f5f6a24d95bf0c`。
按不读标签的固定规则取首个兼容窗口，frame0..249。GK由窗口均值x最外侧推断，只是待核实角色，不是metadata验证。
输出[50,26,8]，tensor SHA256 `58840a4519b22b30de43e39a86590ef7dde0f4a52d75f2c44a3fb4ea33b5651e`。

| 真实公开窗口扰动 | 实际结果 |
|---|---|
| 每帧随机slot permutation，ID随观测移动 | 适配张量逐元素不变 |
| H/A编码互换、GK映射互换 | 几何张量不变，left_team映射反转 |
| 全玩家缺失 | 拒绝 |
| 连续1秒全玩家缺失 | 拒绝 |
| GK未知 | 拒绝 |
| 帧号缺口 | 拒绝 |

以上是**输入合同结果，不是模型预测鲁棒性**。模型置信度、自然/缺球HA改善、旧错误纠正、完整spell均`NOT_RUN/null`。
5FPS只有200ms分辨率；未上采样为虚假的25FPS精确换手。

8项合成测试实际通过：10/25Hz、单位、GK、slot permutation、NaN、缺帧、半场、镜像、队伍交换、拒收球字段。
`r11/pathcrf_runtime.py`保留精确CPU恢复入口：复制固定源码后只去掉`models.utils`未使用的PyG graph-collator导入并延迟类型注解，神经算子/参数/CRF不改；严格state_dict load。此分支尚未运行，需独立验证。

## 精确恢复与回退

回执：`local_artifacts/r11/pathcrf/attempt-v1.json`；源码清单`upstream_manifest.json`；唯一候选协议`selection-protocol-v1.json`。
协议在合同诊断后记录，明确`first_scoring_has_not_occurred=true`；不把它伪称此前已评分预注册。

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python -m unittest discover -s tests -p test_r11_pathcrf.py -v
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python -m scripts.probe_r11_pathcrf --receipt local_artifacts/r11/pathcrf/attempt-v2.json
```

第二条现在只会再次记录缺torch，**不要在条件没改变时重复运行**。
若授权环境空闲至少2GiB且足够覆盖所有下载+展开2倍余量，再建立`local_artifacts/r11/pathcrf/runtime/`隔离依赖（Torch2.2.2、NumPy1.26.4及其明确依赖）；以`PYTHONPATH`指向该目录运行上述命令，不修改旧venv。
`torch`旧版与NumPy2存在ABI风险，隔离NumPy1.26而不是降级R6–R10环境。

回退：M3/HC_M3及其哈希不变，禁接生产；PathCRF只保留恢复材料。当前继续H2合法新监督，不用新几何阈值代替独立信息。
