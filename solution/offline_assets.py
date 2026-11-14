"""Human-operated offline assets check and official reference; never reads ground truth.

The CLI is launched by the macOS .command wrappers from an isolated public-code copy.
Importable helpers support synthetic tests; importing this module reads no assets.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import stat

from games import GAME, GAMES, SOURCES, asset_paths, half_video_blocks


def _within(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def _safe_regular(path: Path, root: Path) -> bool:
    # Resolve prevents a known asset name from redirecting to a workspace or arbitrary file.
    if not _within(path.resolve(), root.resolve()) or path.is_symlink():
        raise ValueError(f"asset symlink/path escape rejected: {path.name}")
    try:
        return stat.S_ISREG(path.stat().st_mode)
    except FileNotFoundError:
        return False


def preflight(data_dir: Path) -> dict:
    """Stat only the nine official names. No glob, recursive scan, open, or hashes."""
    root = Path(data_dir).resolve()
    rows = []
    for source in SOURCES:
        for asset, filename in asset_paths(data_dir=str(root), source=source).items():
            path = Path(filename)
            exists = _safe_regular(path, root)
            rows.append({"source": source, "asset": asset,
                         "expected": str(path.relative_to(root)), "exists": exists,
                         "bytes": path.stat().st_size if exists else None})
    missing = [row["expected"] for row in rows
               if row["source"] == "tactical" and row["asset"] in ("atd_meta", "atd_raw")
               and not (row["exists"] and row["bytes"] > 0)]
    return {"assets": rows, "reference_ready": not missing,
            "missing_required": missing,
            "message": ("Tactical ATD XML/TXT 存在且非空；内容尚未验证。" if not missing else
                        "缺少 Tactical ATD XML/TXT，不可运行官方 baseline，需取得赛事完整授权资产。"),
            "contents_read": False, "baseline_executed": False}


def isolation_guard(data_dir: Path, output_dir: Path | None = None) -> None:
    """Fail closed for this development workspace and inherited Codex sessions.

    These checks supplement an OS network sandbox and human disconnection; they cannot
    prove that another process, cloud sync, or a custom workspace is absent.
    """
    if platform.system() != "Darwin" or os.environ.get("MTAH_OFFLINE_SANDBOX") != "macos-deny-network":
        raise ValueError("Use the macOS .command wrapper with sandbox-exec; direct CLI refused")
    if any(name.startswith("CODEX_") for name in os.environ):
        raise ValueError("Codex environment detected; use a separate human-operated Terminal")
    source = Path(__file__).resolve().parents[1]
    home = Path.home()
    forbidden = [home / ".codex", home / "Documents" / "ChatGPT"]
    custom = os.environ.get("MTAH_FORBIDDEN_WORKSPACE")
    if custom:
        forbidden.append(Path(custom).expanduser().resolve())
    paths = [source, Path(data_dir).resolve()]
    if output_dir is not None:
        paths.append(Path(output_dir).resolve())
    if any(_within(path, root.resolve()) for path in paths for root in forbidden):
        raise ValueError("Code/assets/output must be outside Codex workspaces and .codex")
    if _within(paths[1], source) or (output_dir is not None and _within(paths[2], source)):
        raise ValueError("Assets and output must be outside the public-code checkout")
    if output_dir is not None and (_within(paths[2], paths[1]) or _within(paths[1], paths[2])):
        raise ValueError("Keep output separate from the asset directory")


def strict_csv(path: Path, n_frames: int, blocks: list[tuple[int, int]]) -> dict:
    """No GT: exact header/order/count/labels and X outside official half boundaries."""
    counts = {label: 0 for label in "HADX"}
    count = 0
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        if next(reader, None) != ["frame", "label"]:
            raise ValueError("expected exact frame,label header")
        for frame, row in enumerate(reader):
            if len(row) != 2 or row[0] != str(frame) or row[1] not in counts:
                raise ValueError(f"invalid CSV record at frame {frame}")
            inside = any(start <= frame <= stop for start, stop in blocks)
            if not inside and row[1] != "X":
                raise ValueError(f"non-X outside halves at frame {frame}")
            counts[row[1]] += 1
            count += 1
    if count != n_frames:
        raise ValueError(f"expected {n_frames} rows, got {count}")
    return {"rows": count, "label_counts": counts, "format_valid": True,
            "accuracy_evaluated": False}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reference_labels(meta: Path, raw: Path, n_frames: int, pitch: tuple,
                     blocks: list[tuple[int, int]]):
    """Official reference implementation with explicit X outside the two halves.

    Strict streaming validation prevents the official permissive loader from silently
    ignoring malformed, missing, duplicated, or out-of-order input rows.
    """
    import math
    import numpy as np
    from solution.atd import team_metadata
    from starter.baseline import nearest_player_labels, smooth, team_of_each_track
    mapping, _ = team_metadata(meta)
    official_mapping = team_of_each_track(meta)
    if not np.array_equal(mapping, official_mapping):
        raise ValueError("team mapping differs from the official reference")
    count = 0
    with raw.open(encoding="utf-8") as handle:
        for count, line in enumerate(handle, 1):
            fields = line.rstrip("\r\n").split(":")
            if len(fields) != 3 or fields[0] != str(count) or count > n_frames:
                raise ValueError(f"invalid ATD timeline at row {count}")
            groups = fields[1].split(";")
            if len(groups) != len(mapping):
                raise ValueError(f"slot count differs at row {count}")
            for group in groups + [fields[2]]:
                values = group.split(",")
                if len(values) != 3:
                    raise ValueError(f"expected x,y,speed at row {count}")
                try:
                    x, y, speed = map(float, values)
                except ValueError:
                    raise ValueError(f"nonnumeric observation at row {count}") from None
                if any(math.isinf(v) for v in (x, y, speed)):
                    raise ValueError(f"infinite observation at row {count}")
                if math.isnan(x) != math.isnan(y):
                    raise ValueError(f"incomplete coordinate at row {count}")
                # Official parser uses uppercase N as its missing-value sentinel.
                if math.isnan(x) and not group.startswith("N"):
                    raise ValueError(f"unsupported NaN spelling at row {count}")
    if count != n_frames:
        raise ValueError("incomplete ATD timeline")
    codes = nearest_player_labels(str(raw), official_mapping, n_frames, pitch)
    labels = smooth(codes, 50)  # Preserve the published reference, no parameter search.
    inside = np.zeros(n_frames, dtype=bool)
    for start, stop in blocks:
        inside[start:stop + 1] = True
    labels[~inside] = "X"
    return labels


def run_reference(data_dir: Path, output_dir: Path) -> Path:
    status = preflight(data_dir)
    if not status["reference_ready"]:
        raise ValueError(status["message"] + " " + ", ".join(status["missing_required"]))
    # A new directory makes overwrite, historical result loss and hidden stale files impossible.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    paths = {k: Path(v) for k, v in asset_paths(data_dir=str(data_dir)).items()}
    blocks = [(start, stop) for _, start, stop in half_video_blocks(GAME)]
    labels = reference_labels(paths["atd_meta"], paths["atd_raw"],
                              GAMES[GAME]["n_video_frames"], GAMES[GAME]["pitch"], blocks)
    from solution.submission import write_submission
    temporary = output_dir / "baseline_smooth50.csv.part"
    write_submission(temporary, labels)
    checked = strict_csv(temporary, len(labels), blocks)
    target = temporary.with_suffix("")
    temporary.rename(target)
    receipt = {"method": "official starter.baseline nearest_player_labels + smooth(50)",
               "modification": "X outside games.py half_video_blocks", "game": GAME,
               "validation": checked, "output_sha256": file_sha256(target),
               "inputs_sha256": {key: file_sha256(paths[key]) for key in ("atd_meta", "atd_raw")},
               "code_sha256": {name: file_sha256(Path(__file__).resolve().parents[1] / name)
                               for name in ("games.py", "starter/baseline.py", "solution/offline_assets.py")},
               "gt_read": False, "submission_performed": False,
               "privacy": "LOCAL ONLY: do not upload this receipt or CSV to Codex/GitHub"}
    (output_dir / "LOCAL_ONLY_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("preflight", "reference"))
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        isolation_guard(args.data_dir, args.output_dir)
        if args.mode == "preflight":
            result = preflight(args.data_dir)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result["reference_ready"] else 2
        if args.output_dir is None:
            raise ValueError("reference requires --output-dir (a new isolated directory)")
        target = run_reference(args.data_dir, args.output_dir)
        print(f"LOCAL ONLY: reference CSV and validation receipt saved to {target.parent}")
        print("Format validated; accuracy not evaluated. No GT read and nothing submitted.")
        return 0
    except (OSError, ValueError) as error:
        print(f"STOP: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
