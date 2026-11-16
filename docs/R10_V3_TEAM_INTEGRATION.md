# R10-v3 三线整合与隔离（一次性代码审查）

基线 R9 `9f701a4f193d776515a6c2129d5809b6e736b8b2`；B线固定 `a54219dbff8948507b112234127816c133f7c08b`。
独立R10工作树中仅重构通用语义；没有合并、cherry-pick或修改B分支。没有读取目标素材、配置、逐帧结果或其诊断报告。

| 研发线 | 输入/监督合同与算法 | 有价值的增量 | 风险/处理 |
|---|---|---|---|
| PR1 `83b262a` | 44维、双HGB、Viterbi；已处理20场SkillCorner | 历史公开数据、工程入口 | 20场不是新盲测；phase空白不是完整D。保留PR2纠正，禁止回退旧标签语义；本次不重复下载 |
| R9 `9f701a4` | M2 Alive + C1 Home + 原M3；三场DFL完整源标签、部分SC/Metrica | 已冻结模型/概率/真实失败、完整spell预算 | 已反复开发三场；缺完整跨供应商HAD；不重训/不覆盖 |
| B V2 | ATD几何/速度、自动进攻防守Team、set-piece、半场Viterbi | 自动Team语义是不同观测来源；缓存SHA与测试有工程价值 | 标签来源有潜在增量但目标效果不可审查；选参污染配置不能继承 |
| B V3 | 可见停球、球范围、低速球员、动态D转移 | 可见运动与缺失分离 | 几何与旧模型同源；冻结坐标仍可成捷径；不直接移植参数 |
| B V4 | 保留V2 H/A发射，仅加可见D证据 | 可做增量机制消融 | BASE与选参V2相连；正D增强仍会吞短spell；不直接晋级 |

## 已实际读取的通用源码与Git blob

均来自上述B固定commit，仅源代码；未运行其loader、优化脚本或正式runner。

| 文件/原签名 | blob SHA | 本次迁移 |
|---|---|---|
| `models/fusion_v2.py`: `features(data_dir, cache=None)`, `predict(f,cfg,blocks=None)` | `6a49720993fd9c015e05f45d148480567c04bd25` | 队伍/速度/事件语义，重构为纯内存接口 |
| `models/fusion_v3.py`: `prepare(f,gap=0)`, `observations(f,p,c)`, `predict(f,c,prepared=None)` | `bfd98345685742e86219cb58676cef654f50bc24` | 审查；未迁移其运动阈值/默认参数 |
| `models/fusion_v4.py`: `prepare(f)`, `predict(f,c,prepared=None)` | `b21f587e73d0362ec649c5a692b292e3dc1c1225` | 审查；未迁移BASE/DEFAULT |
| `models/possession_detector.py`: `build_advanced_model(feats_path,data_dir=None)` | `86275b5f49d70f35a192a46c19eab8335a6c3f15` | 三个明确防守类别；不复制队名、硬D或最短片段规则 |
| `run_pipeline.py`: `main()` | `37b174cf6cee85f43369cd9309e7369a86ba7f62` | 分离predict_only概念；原入口未调用 |
| `models/optimize_v2.py` | `abda566553d00faf1c8c4b370feec09dbbabe0e8` | 仅审查选参来源，禁止迁移 |
| `models/optimize_v3.py` | `ce9562807ea9e851bc31c949a17ddcdafba66ade` | 仅审查选参来源，禁止迁移 |
| `models/optimize_v4.py` | `84ae53a62a8de9570a484f776a85186f26df8ae0` | 仅审查选参来源，禁止迁移 |
| `models/test_fusion_v2.py` | `5c6269fc9d20090f577d6e9df6b77ced07dd746a` | 参考非有限值、缺球、半场测试思想 |
| `models/test_fusion_v3.py` | `64111a7ef391dc63aff36577f280efbdc8272a63` | 参考gap/reset、缺队中性化、输入验证 |
| `models/test_fusion_v4.py` | `c99420463cc822b48cf4bfc6560f0fe301c12427` | 参考zero-change identity与nonmutation；不迁移选参测试 |
| `LICENSE` | `d9330daa5cb9bd8c01412b938e5c15539cb5d799` | MIT源码许可；原版权和授权条款在根LICENSE完整保留，赛事资产明确不在范围内 |

## 污染来源（来自源代码，不来自目标指标/报告）

- optimize_v2 源码41行读取public GT，79–88行用其窗口选择配置，112行写入fusion_v2_config。
- optimize_v3 源码53–54行读取public GT，81–91行选配置，102行写配置。
- optimize_v4 `select(trials,baseline,'full')`选择全public窗口最优，再由`publish_candidate`控制发布；bootstrap/leave-block不能消除历史曝光。
- run_pipeline 默认auto读取default_model，再推断，末尾自动subprocess执行public评分。该入口不用于R10研究。
- B的DEFAULT/BASE也不能当作干净物理先验；本次未导入任何其数值配置。
- 疑似目标派生文件仅路径层面识别，不读取或传播内容；由获授权本人处理，本文不对其内容作事实断言。

## 实际交付与验证

`r10/adapters/B_to_R9_contract.md`给出逐项映射。新增tracking纯数组适配、identity-linked速度、软Team interval证据与predict_only。
R9 `restart_proposals`复用；set-piece不当作D，N/A不当队伍；标签方向、重复、矛盾、时间和半场均有测试。
无任何源资产发现/读取/上传、模型fit或GT评分路径。关闭事件时同一decoder输入/输出完全保持。

实际测试命令：

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python -m unittest discover -s tests -p test_r10_adapters.py -v
```

首次10项测试有1失败：非restart坏行同时被restart与tactical parser计为两次拒绝。
修复为仅将set-piece行送入R9 parser，并将拒绝行号映射回原行号；不是改变比赛模型/阈值。
修复后10项全部通过，未运行真实赛事数据；完整研究/回归由主线程另记。

**决定：KEEP_CODE_ONLY_INTERFACE；事件效果NO_INDEPENDENT_COMPLETE_LABELS / TRANSFER_NOT_ESTABLISHED。**
可用的现有Metrica弱事件并非同构自动战术tag + 完整H/A/D；不能因此宣称自动Team事件已提高spell。
当前有判别力的完整外部实验应使用已存在DFL原轨迹的控制确立信息，由研究协议单独决定是否有增量。

回退：不调用r10接口即可保持原R9；无旧模型或配置被替换。
A：确认正式提交状态/截止；B：隔离核实资产/合法配置，提供YES/NO/UNKNOWN而非受限内容；C：确认已有20场公开缓存schema/hash/曝光，不重复下载。
