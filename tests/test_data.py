import pandas as pd
import pytest

from wb_sentiment.data import grouped_split, prepare_version, read_source, save_versions


def test_text_preprocessing_and_deduplicated_version():
    frame = pd.DataFrame({
        "review_id": [1, 2, 3], "product_id": [10, 10, 11], "rating": [5, 5, 1],
        "text": [" Хорошо  ", "Хорошо", "Плохо"], "pros": [None, "", ""], "cons": ["", "", ""],
    })
    v1 = prepare_version(frame, "v1")
    v2 = prepare_version(frame, "v2")
    assert v1.loc[0, "review_text"] == "Хорошо"
    assert v2.loc[0, "review_text"] == "хорошо"
    assert len(v1) == 3
    assert len(v2) == 2


def test_grouped_split_has_no_product_overlap():
    rows = []
    for product in range(60):
        for review in range(5):
            rows.append({"product_id": product, "sentiment": ("negative", "neutral", "positive")[product % 3],
                         "review_text": f"review {product} {review}"})
    splits = grouped_split(pd.DataFrame(rows))
    ids = {name: set(part.product_id) for name, part in splits.items()}
    assert not ids["train"] & ids["validation"]
    assert not ids["train"] & ids["test"]
    assert not ids["validation"] & ids["test"]


def test_save_versions_creates_distinct_hashed_snapshots(tmp_path):
    source_path = tmp_path / "filtered.csv"
    pd.DataFrame({
        "review_id": [1, 2], "product_id": [10, 11], "rating": [5, 1],
        "text": ["Хорошо", "Плохо"], "pros": ["", ""], "cons": ["", ""],
    }).to_csv(source_path, index=False)
    manifest = save_versions(source_path, tmp_path / "versions")
    assert manifest["source"]["rows"] == 2
    assert manifest["versions"]["v1"]["sha256"] != manifest["versions"]["v2"]["sha256"]
    assert read_source(source_path).shape[0] == 2


def test_unknown_version_and_missing_columns_fail_clearly(tmp_path):
    with pytest.raises(ValueError, match="версии данных"):
        prepare_version(pd.DataFrame(), "v3")
    invalid = tmp_path / "invalid.csv"
    pd.DataFrame({"rating": [5]}).to_csv(invalid, index=False)
    with pytest.raises(ValueError, match="отсутствуют столбцы"):
        read_source(invalid)
