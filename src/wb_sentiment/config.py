"""Validated TOML settings with paths relative to the project directory."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REQUIRED = {
    "train": ("source_path", "data_path", "versions_dir", "artifact_dir", "report_dir",
              "experiment_name"),
    "inference": ("model_path",),
}


def load_config(path: Path | None = None) -> dict:
    config_path = path or Path(os.getenv("WB_CONFIG_PATH", ROOT / "config.toml"))
    with config_path.open("rb") as stream:
        config = tomllib.load(stream)
    for section, keys in REQUIRED.items():
        values = config.get(section, {})
        for key in keys:
            if not isinstance(values.get(key), str) or not values[key].strip():
                raise ValueError(f"Обязательная настройка {section}.{key} не заполнена")
    seed = config["train"].get("seed")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("train.seed должен быть целым числом от 0 до 2**32 - 1")
    return config


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path
