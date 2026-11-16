"""Dataset loading, versioned preprocessing and group-safe data splits."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

TEXT_COLUMNS = ("text", "pros", "cons")
RATING_TO_SENTIMENT = {1: "negative", 2: "negative", 3: "neutral", 4: "positive", 5: "positive"}
CLASS_ORDER = ("negative", "neutral", "positive")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_source(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    required = {"review_id", "product_id", "rating", *TEXT_COLUMNS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"В исходном файле отсутствуют столбцы: {', '.join(sorted(missing))}")
    return frame


def prepare_version(source: pd.DataFrame, version: str) -> pd.DataFrame:
    """Build v1 (all usable reviews) or v2 (deduplicated text/rating pairs)."""
    if version not in {"v1", "v2"}:
        raise ValueError("Поддерживаются только версии данных v1 и v2")
    frame = source.copy()
    for column in TEXT_COLUMNS:
        frame[column] = frame[column].fillna("").astype(str)
    frame["review_text"] = frame[list(TEXT_COLUMNS)].agg(" ".join, axis=1)
    frame["review_text"] = frame["review_text"].map(lambda value: re.sub(r"\s+", " ", value).strip())
    frame = frame.loc[frame["review_text"].str.len() > 0].copy()
    frame["sentiment"] = frame["rating"].map(RATING_TO_SENTIMENT)
    frame = frame.dropna(subset=["sentiment"])
    if version == "v2":
        frame["review_text"] = frame["review_text"].str.casefold()
        frame = frame.drop_duplicates(subset=["review_text", "rating"], keep="first")
    return frame.reset_index(drop=True)


def save_versions(source_path: Path, output_dir: Path) -> dict:
    source = read_source(source_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": {"path": source_path.name, "sha256": sha256(source_path), "rows": len(source)},
        "versions": {},
        "preprocessing": "join text/pros/cons, normalize whitespace, drop empty text; v2 lowercases text and removes duplicate text+rating pairs",
    }
    for version in ("v1", "v2"):
        output_path = output_dir / f"{version}.csv"
        prepared = prepare_version(source, version)
        prepared.to_csv(output_path, index=False)
        manifest["versions"][version] = {
            "path": output_path.name,
            "sha256": sha256(output_path),
            "rows": len(prepared),
        }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def grouped_split(frame: pd.DataFrame, seed: int = 42) -> dict[str, pd.DataFrame]:
    """Make approximately 70/10/20 train/validation/test split without product leakage."""
    indices = frame.index.to_numpy()
    groups = frame["product_id"].to_numpy()
    outer = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    train_val_pos, test_pos = next(outer.split(indices, frame["sentiment"], groups))
    train_val = frame.iloc[train_val_pos]
    test = frame.iloc[test_pos]
    inner = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=seed)
    train_pos, val_pos = next(
        inner.split(train_val, train_val["sentiment"], train_val["product_id"])
    )
    result = {"train": train_val.iloc[train_pos], "validation": train_val.iloc[val_pos], "test": test}
    group_sets = {name: set(part["product_id"]) for name, part in result.items()}
    assert not (group_sets["train"] & group_sets["validation"])
    assert not (group_sets["train"] & group_sets["test"])
    assert not (group_sets["validation"] & group_sets["test"])
    return result
