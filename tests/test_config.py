from pathlib import Path

import pytest

from wb_sentiment.config import ROOT, load_config, resolve_path


def test_default_paths_are_portable():
    config = load_config()
    assert config["train"]["seed"] == 42
    assert resolve_path(config["inference"]["model_path"]) == ROOT / "artifacts/final_model.joblib"
    assert resolve_path("/tmp/model.joblib") == Path("/tmp/model.joblib")


def test_config_rejects_missing_required_value(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[train]\nseed=42\n')
    with pytest.raises(ValueError, match="train.source_path"):
        load_config(path)


def test_config_rejects_invalid_seed(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text((ROOT / "config.toml").read_text().replace("seed = 42", "seed = -1"))
    with pytest.raises(ValueError, match="train.seed"):
        load_config(path)


def test_config_can_be_selected_with_environment(monkeypatch, tmp_path):
    path = tmp_path / "custom.toml"
    path.write_text((ROOT / "config.toml").read_text().replace("seed = 42", "seed = 7"))
    monkeypatch.setenv("WB_CONFIG_PATH", str(path))
    assert load_config()["train"]["seed"] == 7
