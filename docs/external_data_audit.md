# 外部数据审计、清洗与训练样本报告

> 本报告由 `python -m solution.audit_external` 从本地清洗后 NPZ 自动生成。
> 报告只包含公开外部数据的聚合统计，不包含赛事受限比赛的逐帧派生数据。

## 1. 结论摘要

- SkillCorner：20 场，合计 1,241,620 个视频帧；半场内 1,182,250 帧。
- 清洗后 SkillCorner NPZ 总大小：161.1 MiB。
- alive/dead 训练样本：alive 459,714，dead 508,731。
- 高置信 H/A 训练样本：home 182,811，away 173,457，合计 356,268。
- 可比较的 SkillCorner possession 与 phase 球队帧中，一致率为 83.74%。
- 清洗后 H/A 保留率为 58.19%（相对可比较帧）。
- 比赛级拆分泄漏：否；拆分是否完整覆盖全部比赛：是。
- Metrica Game 3 高置信 H/A 帧：72,606；高置信 D 帧：27,360。

## 2. 数据来源与冻结版本

- SkillCorner：`https://github.com/SkillCorner/opendata`，commit `4340d274572876239c154c90bc507a9b3250a656`。
- SkillCorner tracking：20 个文件，1,823,448,331 字节，均按 LFS SHA256 校验。
- Metrica：`https://github.com/metrica-sports/sample-data`，commit `e706dd506b360d69d9d123d5b8026e7294b13996`。
- SkillCorner archive SHA256：`bdf4ce561ee3ec070ef37c12ca1482398ea776716fd2fd73f153d8b81452d4e2`。
- Metrica archive SHA256：`8ba1674fe5e87d2c5a174b6077580ba5d444b964d8bba16d4dd2d8d62a064f9f`。

## 3. 标签语义差异与清洗原则

赛事要求活球时始终属于 H/A；争抢或 loose ball 延续上一支控球队，只有对手建立 controlled possession 才换权。死球期间 tracking 即使提供球队，赛事标签仍必须是 D。SkillCorner 的 `possession.group` 是供应商逐帧控球队估计，公开文档没有承诺使用同一 controlled-possession 定义，因此本项目将其视为弱监督。

清洗规则：

1. SkillCorner phase 区间按 `[frame_start, frame_end)` 半开区间解释，因为 duration 等于 `(end-start)/10`。
2. phase 覆盖帧是 raw alive；半场内未覆盖帧是 raw dead 候选。
3. 每个 phase 起止边界前后 1 秒全部置为 unknown，不训练 alive/dead，也不训练 H/A。
4. H/A 必须同时满足：clean alive、`possession.group` 已知、前后 1 秒均为同一球队、并与 `team_in_possession_id` 一致。
5. 任意未知、来源冲突、球队切换附近或 phase 边界帧都置为 `-1`，不强行贴标签。
6. Metrica 仅保留 PASS/CARRY 去掉首尾 2 帧后的 H/A，以及 BALL OUT 结束到下一次 SET PIECE 开始前的 D；冲突帧排除。
7. NPZ 同时保存严格训练标签 `y_alive/y_home` 与完整弱监督时间线 `eval_y_alive/eval_y_home`；后者由 phase 是否存在和 phase 球队构造，只用于粗粒度 spell/Viterbi 验证。

## 4. SkillCorner 聚合审计

- phase 边界排除帧（半场内）：213,833。
- raw alive 且 possession 已知：612,224。
- possession 与 phase 球队均已知：612,224。
- 两来源一致：512,659（83.74%）。
- 一致且前后 1 秒稳定：446,160。
- 最终 H/A 样本：356,268。
- clean dead 中仍有 SkillCorner 球队名的帧：12,230；这些球队标签已丢弃。
- phase 球队重叠冲突帧：0；这些帧不进入 H/A。

### 4.1 每场比赛

| match | frames | in-half | alive | dead | unknown(in-half) | H | A | source agreement | H/A retained |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1874553 | 60,301 | 60,282 | 24,537 | 25,352 | 10,393 | 6,918 | 12,589 | 86.49% | 61.94% |
| 1886347 | 59,061 | 59,042 | 23,265 | 24,851 | 10,926 | 11,122 | 7,331 | 85.74% | 59.50% |
| 1899585 | 60,531 | 60,512 | 21,177 | 28,610 | 10,726 | 8,445 | 7,348 | 82.16% | 55.74% |
| 1925299 | 61,301 | 61,282 | 26,351 | 23,485 | 11,446 | 13,709 | 6,996 | 85.20% | 60.32% |
| 1927964 | 59,071 | 59,052 | 28,752 | 20,282 | 10,027 | 12,841 | 11,144 | 87.62% | 65.84% |
| 1953632 | 59,251 | 59,232 | 24,744 | 24,414 | 10,074 | 5,405 | 14,365 | 85.39% | 61.54% |
| 1959846 | 58,541 | 58,532 | 21,211 | 26,030 | 11,291 | 7,971 | 8,707 | 83.85% | 57.03% |
| 1986691 | 59,091 | 59,072 | 23,400 | 25,495 | 10,186 | 8,218 | 10,251 | 84.81% | 59.94% |
| 1996435 | 57,621 | 57,602 | 23,520 | 23,495 | 10,587 | 9,994 | 8,000 | 84.06% | 57.35% |
| 1996436 | 60,281 | 60,262 | 19,470 | 30,776 | 10,016 | 8,186 | 6,318 | 81.89% | 55.59% |
| 2006229 | 59,271 | 59,252 | 24,004 | 24,635 | 10,613 | 11,723 | 7,214 | 85.79% | 60.46% |
| 2006363 | 59,091 | 59,072 | 24,079 | 24,605 | 10,388 | 10,145 | 8,372 | 84.30% | 58.14% |
| 2007448 | 58,081 | 58,062 | 21,087 | 26,891 | 10,084 | 7,890 | 8,664 | 85.77% | 58.87% |
| 2007721 | 57,561 | 57,542 | 25,059 | 22,089 | 10,394 | 9,906 | 10,168 | 85.46% | 61.27% |
| 2010085 | 58,991 | 58,972 | 24,156 | 24,633 | 10,192 | 12,595 | 6,697 | 84.26% | 60.60% |
| 2011166 | 71,391 | 59,292 | 18,697 | 29,823 | 10,772 | 5,360 | 8,309 | 79.09% | 52.12% |
| 2013725 | 69,861 | 56,792 | 21,893 | 23,389 | 11,510 | 7,735 | 7,834 | 78.55% | 51.88% |
| 2015213 | 71,351 | 59,902 | 20,960 | 27,003 | 11,939 | 6,810 | 9,228 | 82.72% | 54.22% |
| 2016236 | 70,551 | 60,042 | 23,180 | 25,233 | 11,629 | 8,310 | 9,385 | 81.95% | 55.95% |
| 2017461 | 70,421 | 58,452 | 20,172 | 27,640 | 10,640 | 9,528 | 4,537 | 76.67% | 50.86% |

## 5. 比赛级拆分

- 固定随机种子：`2027`。
- train（14）：`1874553, 1886347, 1899585, 1927964, 1953632, 1959846, 1986691, 1996436, 2006229, 2006363, 2007448, 2007721, 2010085, 2016236`。
- validation（3）：`1925299, 2013725, 2017461`。
- test（3）：`1996435, 2011166, 2015213`。
- 集合交叉：无。

### 5.1 各拆分训练样本量

| split | alive | dead | H | A |
|---|---:|---:|---:|---:|
| train | 328,121 | 353,896 | 129,675 | 128,553 |
| validation | 68,416 | 74,514 | 30,972 | 19,367 |
| test | 63,177 | 80,321 | 22,164 | 25,537 |

## 6. 特征完整性

- 共同特征数：44。
- SkillCorner 所有比赛 schema 一致：是。
- Metrica 与 SkillCorner schema 一致：是。
- 缺失值由 HistGradientBoostingClassifier 原生处理；下表用于识别跨供应商或字段级异常。

| feature | in-half missing | alive-model missing | H/A-model missing |
|---|---:|---:|---:|
| `away_d1_m` | 25.66% | 30.57% | 1.16% |
| `away_d1_mean_0p2s` | 25.55% | 30.43% | 1.15% |
| `away_d1_mean_0p5s` | 25.22% | 30.03% | 1.12% |
| `away_d1_mean_1p0s` | 24.67% | 29.37% | 1.07% |
| `away_d2_m` | 25.66% | 30.57% | 1.16% |
| `away_d3_m` | 25.66% | 30.57% | 1.16% |
| `away_d3_mean_m` | 25.66% | 30.57% | 1.16% |
| `away_local_speed_mean_mps` | 54.65% | 58.66% | 37.12% |
| `away_within_10p0m` | 0.00% | 0.00% | 0.00% |
| `away_within_1p0m` | 0.00% | 0.00% | 0.00% |
| `away_within_2p0m` | 0.00% | 0.00% | 0.00% |
| `away_within_3p0m` | 0.00% | 0.00% | 0.00% |
| `away_within_5p0m` | 0.00% | 0.00% | 0.00% |
| `ball_displacement_0p2s_m` | 25.89% | 30.73% | 1.19% |
| `ball_displacement_0p5s_m` | 26.22% | 30.99% | 1.24% |
| `ball_displacement_1p0s_m` | 26.78% | 31.42% | 1.37% |
| `ball_speed_missing` | 0.00% | 0.00% | 0.00% |
| `ball_speed_mps` | 25.77% | 30.65% | 1.18% |
| `ball_to_x0_m` | 25.66% | 30.57% | 1.16% |
| `ball_to_x1_m` | 25.66% | 30.57% | 1.16% |
| `ball_to_y0_m` | 25.66% | 30.57% | 1.16% |
| `ball_to_y1_m` | 25.66% | 30.57% | 1.16% |
| `ball_visible` | 0.00% | 0.00% | 0.00% |
| `ball_x` | 25.66% | 30.57% | 1.16% |
| `ball_y` | 25.66% | 30.57% | 1.16% |
| `home_d1_m` | 25.66% | 30.57% | 1.16% |
| `home_d1_mean_0p2s` | 25.55% | 30.43% | 1.15% |
| `home_d1_mean_0p5s` | 25.22% | 30.03% | 1.12% |
| `home_d1_mean_1p0s` | 24.67% | 29.37% | 1.07% |
| `home_d2_m` | 25.66% | 30.57% | 1.16% |
| `home_d3_m` | 25.66% | 30.57% | 1.16% |
| `home_d3_mean_m` | 25.66% | 30.57% | 1.16% |
| `home_local_speed_mean_mps` | 54.73% | 58.62% | 36.69% |
| `home_within_10p0m` | 0.00% | 0.00% | 0.00% |
| `home_within_1p0m` | 0.00% | 0.00% | 0.00% |
| `home_within_2p0m` | 0.00% | 0.00% | 0.00% |
| `home_within_3p0m` | 0.00% | 0.00% | 0.00% |
| `home_within_5p0m` | 0.00% | 0.00% | 0.00% |
| `n_away_visible` | 0.00% | 0.00% | 0.00% |
| `n_home_visible` | 0.00% | 0.00% | 0.00% |
| `nearest_distance_margin_m` | 25.66% | 30.57% | 1.16% |
| `visibility_rate_0p2s` | 0.00% | 0.00% | 0.00% |
| `visibility_rate_0p5s` | 0.00% | 0.00% | 0.00% |
| `visibility_rate_1p0s` | 0.00% | 0.00% | 0.00% |

## 7. Metrica Game 3 高置信样本

- 总帧数：143,761。
- H 高置信帧：37,512。
- A 高置信帧：35,094。
- D 高置信帧：27,360。
- 事件冲突排除帧：4,889。
- 用途：只做跨供应商 H/A 退化检查，不针对目标比赛编写常数。

## 8. 能否映射为赛事所需标签

### 8.1 可以高置信映射的部分

- **稳定 H/A 内部帧**：phase 明确活球、两个 SkillCorner 来源球队一致、前后 1 秒不换队。这些帧适合训练 `P(home | alive)` 的状态特征。
- **长停顿内部 D 帧**：处于正式半场、没有 phase、并远离任一 phase 边界至少 1 秒。这些帧适合训练 `P(alive)` 的 dead 类。
- **phase 内部 alive 帧**：远离所有 phase 边界，可作为 alive 正类。

### 8.2 不能直接映射的部分

- possession.group 的切换帧：SkillCorner 未公开保证与赛事 controlled-possession 的换权定义一致。
- 争抢、折射、头球和 loose-ball 帧：赛事要求延续上一队，SkillCorner 的具体规则未知。
- 死球期间的球队名：赛事明确丢弃 dead-ball ownership。
- phase 很短的缺口和所有 phase 边界：可能是时序误差或供应商阶段切分，不应硬标。

### 8.3 推荐用法

SkillCorner 不应被当作赛事真值的逐帧替代品。它适合训练“稳定活球/死球和稳定主客控球时的几何状态”，不适合监督精确换权帧。最终边界应由目标 ATD 的逐帧概率、Smart Tagging 重开球证据以及半场内 Viterbi 共同确定。

### 8.4 完整弱监督时间线的边界

`eval_y_alive/eval_y_home` 可以把 SkillCorner 映射成完整 H/A/D 代理时间线：有 phase 时取 `team_in_possession_id`，半场内无 phase 时取 D。它适合比较整体 fragmentation、粗粒度 spellF1 和候选 Viterbi 参数，但不是赛事 corrected tracking。特别是 controlled-possession 的精确换权帧仍可能偏移，因此严格训练继续使用清洗后的高置信内部帧，最终晋级还必须通过 Metrica 跨供应商检查。

## 9. 生成命令

```powershell
python -m solution.train prepare-skillcorner `
  --root local_artifacts\external\skillcorner\data\matches `
  --out local_artifacts\skillcorner_clean

python -m solution.train prepare-metrica `
  --meta local_artifacts\external\metrica\data\Sample_Game_3\Sample_Game_3_metadata.xml `
  --raw local_artifacts\external\metrica\data\Sample_Game_3\Sample_Game_3_tracking.txt `
  --events local_artifacts\external\metrica\data\Sample_Game_3\Sample_Game_3_events.json `
  --out local_artifacts\metrica_game3.npz

python -m solution.audit_external `
  --prepared local_artifacts\skillcorner_clean `
  --split local_artifacts\skillcorner_clean\split.json `
  --metrica local_artifacts\metrica_game3.npz `
  --source-manifest local_artifacts\external\download_manifest.json `
  --out docs\external_data_audit.md
```
