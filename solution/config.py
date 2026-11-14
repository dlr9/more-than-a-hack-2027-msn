"""Versioned configuration shared by inference and training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


CONFIG_PATH = Path(__file__).with_name("config.json")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    selected = Path(path) if path else CONFIG_PATH
    with selected.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("version") != 1:
        raise ValueError(f"unsupported config version: {config.get('version')!r}")
    return config



def timing_frames(config: dict, frame_key: str, fps: float) -> int:
    """Legacy frame counts are specified at 25 FPS; optional seconds take precedence."""
    from .evidence import seconds_to_frames
    seconds_key = frame_key.removesuffix("_frames") + "_s"
    seconds = config[seconds_key] if seconds_key in config else config[frame_key] / 25.
    return seconds_to_frames(float(seconds), fps, minimum=1)
