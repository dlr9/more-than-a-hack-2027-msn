"""Audit cleaned external samples and write a reproducible Markdown report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _count(values: np.ndarray, value: int) -> int:
    return int(np.sum(values == value))


def _percent(numerator: int | float, denominator: int | float) -> str:
    if not denominator:
        return "n/a"
    return f"{100.0 * numerator / denominator:.2f}%"


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def build_report(
    prepared_dir: str | Path,
    split_path: str | Path,
    metrica_path: str | Path,
    source_manifest_path: str | Path,
    output_path: str | Path,
) -> Path:
    prepared = Path(prepared_dir)
    paths = sorted(prepared.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no prepared NPZ files in {prepared}")
    split = json.loads(Path(split_path).read_text(encoding="utf-8"))
    source = json.loads(Path(source_manifest_path).read_text(encoding="utf-8-sig"))

    per_match: list[dict[str, int | float | str]] = []
    schema: list[str] | None = None
    context_totals = {name: 0 for name in ("in_half", "alive_model", "home_model")}
    finite: dict[str, np.ndarray] = {}
    aggregate_keys = (
        "frames", "in_half", "clean_alive", "clean_dead", "clean_unknown_in_half",
        "home", "away", "raw_live_team_known", "phase_comparable", "phase_agree",
        "stable", "selected_home_labels", "dead_with_named_team", "boundary_excluded",
        "phase_conflicts",
    )
    aggregate = {name: 0 for name in aggregate_keys}
    prepared_bytes = 0

    for path in paths:
        batch = _load(path)
        prepared_bytes += path.stat().st_size
        n_frames = len(batch["frame"])
        if str(batch["match_id"]) != path.stem:
            raise ValueError(f"match_id does not match filename: {path}")
        if not np.array_equal(batch["frame"], np.arange(n_frames)):
            raise ValueError(f"non-contiguous frame index: {path}")
        for label_name in ("y_alive", "y_home", "raw_y_alive", "raw_y_home", "phase_y_home",
                           "eval_y_alive", "eval_y_home"):
            if not set(np.unique(batch[label_name])).issubset({-1, 0, 1}):
                raise ValueError(f"invalid {label_name} codes: {path}")
        if batch["X"].shape[0] != n_frames or len(batch["period"]) != n_frames:
            raise ValueError(f"row count mismatch: {path}")
        names = batch["feature_names"].astype(str).tolist()
        if schema is None:
            schema = names
            finite = {name: np.zeros(3, dtype=np.int64) for name in schema}
        elif names != schema:
            raise ValueError(f"feature schema mismatch: {path}")
        y_alive = batch["y_alive"]
        y_home = batch["y_home"]
        raw_alive = batch["raw_y_alive"]
        raw_home = batch["raw_y_home"]
        phase_home = batch["phase_y_home"]
        period = batch["period"]
        stable = batch["home_stable"] > 0
        in_half = period > 0
        comparable = (raw_alive == 1) & (raw_home >= 0) & (phase_home >= 0)
        agree = comparable & (raw_home == phase_home)
        contexts = (in_half, y_alive >= 0, y_home >= 0)
        for context_index, (context_name, mask) in enumerate(zip(context_totals, contexts)):
            context_totals[context_name] += int(np.sum(mask))
            if np.any(mask):
                finite_matrix = np.isfinite(batch["X"][mask]).sum(axis=0)
                for feature_index, name in enumerate(schema):
                    finite[name][context_index] += int(finite_matrix[feature_index])

        row = {
            "match": str(batch["match_id"]),
            "frames": len(y_alive),
            "in_half": int(np.sum(in_half)),
            "clean_alive": _count(y_alive, 1),
            "clean_dead": _count(y_alive, 0),
            "clean_unknown_in_half": int(np.sum(in_half & (y_alive < 0))),
            "home": _count(y_home, 1),
            "away": _count(y_home, 0),
            "raw_live_team_known": int(np.sum((raw_alive == 1) & (raw_home >= 0))),
            "phase_comparable": int(np.sum(comparable)),
            "phase_agree": int(np.sum(agree)),
            "stable": int(np.sum(agree & stable)),
            "selected_home_labels": int(np.sum(y_home >= 0)),
            "dead_with_named_team": int(np.sum((y_alive == 0) & (raw_home >= 0))),
            "boundary_excluded": int(np.sum(in_half & (raw_alive >= 0) & (y_alive < 0))),
            "phase_conflicts": int(np.sum(batch["phase_team_conflict"] > 0)),
        }
        per_match.append(row)
        for key in aggregate:
            aggregate[key] += int(row[key])

    ids = {str(row["match"]) for row in per_match}
    split_sets = {name: set(map(str, split[name])) for name in ("train", "validation", "test")}
    leakage = bool(
        split_sets["train"] & split_sets["validation"]
        or split_sets["train"] & split_sets["test"]
        or split_sets["validation"] & split_sets["test"]
    )
    split_complete = set.union(*split_sets.values()) == ids
    by_match = {str(row["match"]): row for row in per_match}
    split_summary = {}
    for group in ("train", "validation", "test"):
        split_summary[group] = {
            key: sum(int(by_match[match][key]) for match in split[group])
            for key in ("clean_alive", "clean_dead", "home", "away")
        }

    metrica = _load(Path(metrica_path))
    metrica_names = metrica["feature_names"].astype(str).tolist()
    metrica_ha = (metrica["y_alive"] == 1) & (metrica["y_home"] >= 0)
    metrica_conflicts = int(np.sum(metrica.get("event_conflict", np.zeros(1)) > 0))

    lines = [
        "# 外部数据审计、清洗与训练样本报告",
        "",
        "> 本报告由 `python -m solution.audit_external` 从本地清洗后 NPZ 自动生成。",
        "> 报告只包含公开外部数据的聚合统计，不包含赛事受限比赛的逐帧派生数据。",
        "",
        "## 1. 结论摘要",
        "",
        f"- SkillCorner：{len(per_match)} 场，合计 {aggregate['frames']:,} 个视频帧；半场内 {aggregate['in_half']:,} 帧。",
        f"- 清洗后 SkillCorner NPZ 总大小：{prepared_bytes / (1024 ** 2):.1f} MiB。",
        f"- alive/dead 训练样本：alive {aggregate['clean_alive']:,}，dead {aggregate['clean_dead']:,}。",
        f"- 高置信 H/A 训练样本：home {aggregate['home']:,}，away {aggregate['away']:,}，合计 {aggregate['selected_home_labels']:,}。",
        f"- 可比较的 SkillCorner possession 与 phase 球队帧中，一致率为 {_percent(aggregate['phase_agree'], aggregate['phase_comparable'])}。",
        f"- 清洗后 H/A 保留率为 {_percent(aggregate['selected_home_labels'], aggregate['phase_comparable'])}（相对可比较帧）。",
        f"- 比赛级拆分泄漏：{'是' if leakage else '否'}；拆分是否完整覆盖全部比赛：{'是' if split_complete else '否'}。",
        f"- Metrica Game 3 高置信 H/A 帧：{int(np.sum(metrica_ha)):,}；高置信 D 帧：{_count(metrica['y_alive'], 0):,}。",
        "",
        "## 2. 数据来源与冻结版本",
        "",
        f"- SkillCorner：`{source['skillcorner']['url']}`，commit `{source['skillcorner']['commit']}`。",
        f"- SkillCorner tracking：{source['skillcorner']['tracking_files']} 个文件，{source['skillcorner']['tracking_bytes']:,} 字节，均按 LFS SHA256 校验。",
        f"- Metrica：`{source['metrica']['url']}`，commit `{source['metrica']['commit']}`。",
        f"- SkillCorner archive SHA256：`{source['skillcorner']['archive_sha256']}`。",
        f"- Metrica archive SHA256：`{source['metrica']['archive_sha256']}`。",
        "",
        "## 3. 标签语义差异与清洗原则",
        "",
        "赛事要求活球时始终属于 H/A；争抢或 loose ball 延续上一支控球队，只有对手建立 controlled possession 才换权。死球期间 tracking 即使提供球队，赛事标签仍必须是 D。SkillCorner 的 `possession.group` 是供应商逐帧控球队估计，公开文档没有承诺使用同一 controlled-possession 定义，因此本项目将其视为弱监督。",
        "",
        "清洗规则：",
        "",
        "1. SkillCorner phase 区间按 `[frame_start, frame_end)` 半开区间解释，因为 duration 等于 `(end-start)/10`。",
        "2. phase 覆盖帧是 raw alive；半场内未覆盖帧是 raw dead 候选。",
        "3. 每个 phase 起止边界前后 1 秒全部置为 unknown，不训练 alive/dead，也不训练 H/A。",
        "4. H/A 必须同时满足：clean alive、`possession.group` 已知、前后 1 秒均为同一球队、并与 `team_in_possession_id` 一致。",
        "5. 任意未知、来源冲突、球队切换附近或 phase 边界帧都置为 `-1`，不强行贴标签。",
        "6. Metrica 仅保留 PASS/CARRY 去掉首尾 2 帧后的 H/A，以及 BALL OUT 结束到下一次 SET PIECE 开始前的 D；冲突帧排除。",
        "7. NPZ 同时保存严格训练标签 `y_alive/y_home` 与完整弱监督时间线 `eval_y_alive/eval_y_home`；后者由 phase 是否存在和 phase 球队构造，只用于粗粒度 spell/Viterbi 验证。",
        "",
        "## 4. SkillCorner 聚合审计",
        "",
        f"- phase 边界排除帧（半场内）：{aggregate['boundary_excluded']:,}。",
        f"- raw alive 且 possession 已知：{aggregate['raw_live_team_known']:,}。",
        f"- possession 与 phase 球队均已知：{aggregate['phase_comparable']:,}。",
        f"- 两来源一致：{aggregate['phase_agree']:,}（{_percent(aggregate['phase_agree'], aggregate['phase_comparable'])}）。",
        f"- 一致且前后 1 秒稳定：{aggregate['stable']:,}。",
        f"- 最终 H/A 样本：{aggregate['selected_home_labels']:,}。",
        f"- clean dead 中仍有 SkillCorner 球队名的帧：{aggregate['dead_with_named_team']:,}；这些球队标签已丢弃。",
        f"- phase 球队重叠冲突帧：{aggregate['phase_conflicts']:,}；这些帧不进入 H/A。",
        "",
        "### 4.1 每场比赛",
        "",
        "| match | frames | in-half | alive | dead | unknown(in-half) | H | A | source agreement | H/A retained |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in per_match:
        lines.append(
            f"| {row['match']} | {row['frames']:,} | {row['in_half']:,} | "
            f"{row['clean_alive']:,} | {row['clean_dead']:,} | {row['clean_unknown_in_half']:,} | "
            f"{row['home']:,} | {row['away']:,} | "
            f"{_percent(row['phase_agree'], row['phase_comparable'])} | "
            f"{_percent(row['selected_home_labels'], row['phase_comparable'])} |"
        )

    lines.extend([
        "",
        "## 5. 比赛级拆分",
        "",
        f"- 固定随机种子：`{split.get('seed', 2027)}`。",
        f"- train（{len(split['train'])}）：`{', '.join(map(str, split['train']))}`。",
        f"- validation（{len(split['validation'])}）：`{', '.join(map(str, split['validation']))}`。",
        f"- test（{len(split['test'])}）：`{', '.join(map(str, split['test']))}`。",
        f"- 集合交叉：{'存在，需修复' if leakage else '无'}。",
        "",
        "### 5.1 各拆分训练样本量",
        "",
        "| split | alive | dead | H | A |",
        "|---|---:|---:|---:|---:|",
        *[
            f"| {group} | {split_summary[group]['clean_alive']:,} | "
            f"{split_summary[group]['clean_dead']:,} | {split_summary[group]['home']:,} | "
            f"{split_summary[group]['away']:,} |"
            for group in ("train", "validation", "test")
        ],
        "",
        "## 6. 特征完整性",
        "",
        f"- 共同特征数：{len(schema or [])}。",
        f"- SkillCorner 所有比赛 schema 一致：是。",
        f"- Metrica 与 SkillCorner schema 一致：{'是' if metrica_names == schema else '否'}。",
        "- 缺失值由 HistGradientBoostingClassifier 原生处理；下表用于识别跨供应商或字段级异常。",
        "",
        "| feature | in-half missing | alive-model missing | H/A-model missing |",
        "|---|---:|---:|---:|",
    ])
    for name in schema or []:
        values = finite[name]
        missing = [context_totals[key] - int(values[index])
                   for index, key in enumerate(context_totals)]
        lines.append(
            f"| `{name}` | {_percent(missing[0], context_totals['in_half'])} | "
            f"{_percent(missing[1], context_totals['alive_model'])} | "
            f"{_percent(missing[2], context_totals['home_model'])} |"
        )

    lines.extend([
        "",
        "## 7. Metrica Game 3 高置信样本",
        "",
        f"- 总帧数：{len(metrica['y_alive']):,}。",
        f"- H 高置信帧：{int(np.sum(metrica_ha & (metrica['y_home'] == 1))):,}。",
        f"- A 高置信帧：{int(np.sum(metrica_ha & (metrica['y_home'] == 0))):,}。",
        f"- D 高置信帧：{_count(metrica['y_alive'], 0):,}。",
        f"- 事件冲突排除帧：{metrica_conflicts:,}。",
        "- 用途：只做跨供应商 H/A 退化检查，不针对目标比赛编写常数。",
        "",
        "## 8. 能否映射为赛事所需标签",
        "",
        "### 8.1 可以高置信映射的部分",
        "",
        "- **稳定 H/A 内部帧**：phase 明确活球、两个 SkillCorner 来源球队一致、前后 1 秒不换队。这些帧适合训练 `P(home | alive)` 的状态特征。",
        "- **长停顿内部 D 帧**：处于正式半场、没有 phase、并远离任一 phase 边界至少 1 秒。这些帧适合训练 `P(alive)` 的 dead 类。",
        "- **phase 内部 alive 帧**：远离所有 phase 边界，可作为 alive 正类。",
        "",
        "### 8.2 不能直接映射的部分",
        "",
        "- possession.group 的切换帧：SkillCorner 未公开保证与赛事 controlled-possession 的换权定义一致。",
        "- 争抢、折射、头球和 loose-ball 帧：赛事要求延续上一队，SkillCorner 的具体规则未知。",
        "- 死球期间的球队名：赛事明确丢弃 dead-ball ownership。",
        "- phase 很短的缺口和所有 phase 边界：可能是时序误差或供应商阶段切分，不应硬标。",
        "",
        "### 8.3 推荐用法",
        "",
        "SkillCorner 不应被当作赛事真值的逐帧替代品。它适合训练“稳定活球/死球和稳定主客控球时的几何状态”，不适合监督精确换权帧。最终边界应由目标 ATD 的逐帧概率、Smart Tagging 重开球证据以及半场内 Viterbi 共同确定。",
        "",
        "### 8.4 完整弱监督时间线的边界",
        "",
        "`eval_y_alive/eval_y_home` 可以把 SkillCorner 映射成完整 H/A/D 代理时间线：有 phase 时取 `team_in_possession_id`，半场内无 phase 时取 D。它适合比较整体 fragmentation、粗粒度 spellF1 和候选 Viterbi 参数，但不是赛事 corrected tracking。特别是 controlled-possession 的精确换权帧仍可能偏移，因此严格训练继续使用清洗后的高置信内部帧，最终晋级还必须通过 Metrica 跨供应商检查。",
        "",
        "## 9. 生成命令",
        "",
        "```powershell",
        "python -m solution.train prepare-skillcorner `",
        "  --root local_artifacts\\external\\skillcorner\\data\\matches `",
        "  --out local_artifacts\\skillcorner_clean",
        "",
        "python -m solution.train prepare-metrica `",
        "  --meta local_artifacts\\external\\metrica\\data\\Sample_Game_3\\Sample_Game_3_metadata.xml `",
        "  --raw local_artifacts\\external\\metrica\\data\\Sample_Game_3\\Sample_Game_3_tracking.txt `",
        "  --events local_artifacts\\external\\metrica\\data\\Sample_Game_3\\Sample_Game_3_events.json `",
        "  --out local_artifacts\\metrica_game3.npz",
        "",
        "python -m solution.audit_external `",
        "  --prepared local_artifacts\\skillcorner_clean `",
        "  --split local_artifacts\\skillcorner_clean\\split.json `",
        "  --metrica local_artifacts\\metrica_game3.npz `",
        "  --source-manifest local_artifacts\\external\\download_manifest.json `",
        "  --out docs\\external_data_audit.md",
        "```",
        "",
    ])

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--metrica", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(build_report(args.prepared, args.split, args.metrica,
                       args.source_manifest, args.out))


if __name__ == "__main__":
    main()
