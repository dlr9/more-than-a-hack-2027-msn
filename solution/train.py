"""Prepare external data and train the two-head model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .external import (deterministic_match_split, prepare_metrica_game3,
                       prepare_skillcorner_match, write_split)
from .modeling import train_from_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config")
    sub = parser.add_subparsers(dest="command", required=True)

    skill = sub.add_parser("prepare-skillcorner")
    skill.add_argument("--root", required=True)
    skill.add_argument("--out", required=True)

    metrica = sub.add_parser("prepare-metrica")
    metrica.add_argument("--meta", required=True)
    metrica.add_argument("--raw", required=True)
    metrica.add_argument("--events", required=True)
    metrica.add_argument("--out", required=True)

    train = sub.add_parser("fit")
    train.add_argument("--prepared", required=True)
    train.add_argument("--split", required=True)
    train.add_argument("--out", required=True)
    train.add_argument("--metrica", help="prepared Metrica Game 3 NPZ for the cross-provider gate")

    args = parser.parse_args()
    config = load_config(args.config)
    radii = [float(value) for value in config["distance_radii_m"]]
    windows = [float(value) for value in config["temporal_windows_s"]]
    if args.command == "prepare-skillcorner":
        root, output = Path(args.root), Path(args.out)
        matches = sorted(path for path in root.iterdir() if path.is_dir())
        for match in matches:
            print(prepare_skillcorner_match(match, output / f"{match.name}.npz", radii, windows))
        split = deterministic_match_split([path.name for path in matches], config["random_seed"])
        print(write_split(output / "split.json", split, config["random_seed"]))
    elif args.command == "prepare-metrica":
        print(prepare_metrica_game3(args.meta, args.raw, args.events, args.out, radii, windows))
    else:
        report = train_from_split(args.prepared, args.split, args.out, config, args.metrica)
        print(json.dumps({
            "selected_viterbi": report["selected_viterbi"],
            "validation_gate": report["validation_gate"],
            "metrica_gate": report["metrica_gate"],
            "promoted": report["promoted"],
        }, indent=2))


if __name__ == "__main__":
    main()

