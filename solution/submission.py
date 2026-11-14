"""Submission writing and reproducibility helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np


VALID_LABELS = frozenset("HADX")


def write_submission(path: str | Path, labels: Iterable[str]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(list(labels))
    if values.ndim != 1:
        raise ValueError("labels must be a one-dimensional sequence")
    invalid = sorted(set(values.tolist()) - VALID_LABELS)
    if invalid:
        raise ValueError(f"invalid submission labels: {invalid}")
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines(f"{frame},{label}\n" for frame, label in enumerate(values))
    return output


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
