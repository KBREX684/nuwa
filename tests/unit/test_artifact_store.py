"""Unit tests for nuwa.persistence.artifact_store and nuwa.config.store."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nuwa.config.schema import NuwaConfig
from nuwa.config.store import ConfigStore
from nuwa.core.exceptions import ConfigError
from nuwa.persistence.artifact_store import ArtifactStore

# ---------------------------------------------------------------------------
# ArtifactStore tests
# ---------------------------------------------------------------------------


class TestArtifactStore:
    """Tests for the file-based ArtifactStore."""

    def test_save_and_load_config(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path / "artifacts")
        config = {"model": "gpt-4o", "temperature": 0.7}
        store.save_config_snapshot(1, config)

        loaded = store.load_config_snapshot(1)
        assert loaded == config

    def test_load_missing_returns_none(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path / "artifacts")
        assert store.load_config_snapshot(99) is None

    def test_save_and_load_prompt(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path / "artifacts")
        prompt = "You are a helpful assistant."
        store.save_prompt_snapshot(3, prompt)

        loaded = store.load_prompt_snapshot(3)
        assert loaded == prompt

    def test_list_snapshots(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path / "artifacts")
        store.save_config_snapshot(1, {"a": 1})
        store.save_config_snapshot(3, {"a": 3})
        store.save_config_snapshot(2, {"a": 2})

        rounds = store.list_snapshots()
        assert rounds == [1, 2, 3]

    def test_list_snapshots_empty(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path / "artifacts")
        assert store.list_snapshots() == []

    def test_get_diff(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path / "artifacts")
        store.save_config_snapshot(1, {"model": "v1", "temp": 0.5})
        store.save_config_snapshot(2, {"model": "v2", "temp": 0.7})

        diff = store.get_diff(1, 2)
        assert "v1" in diff
        assert "v2" in diff

    def test_get_diff_identical(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path / "artifacts")
        config = {"model": "same", "temp": 0.5}
        store.save_config_snapshot(1, config)
        store.save_config_snapshot(2, config)

        diff = store.get_diff(1, 2)
        assert diff.strip() == ""

    def test_get_diff_missing_snapshot_raises(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path / "artifacts")
        store.save_config_snapshot(1, {"a": 1})
        with pytest.raises(FileNotFoundError, match="round 99"):
            store.get_diff(1, 99)

    def test_store_dir_created(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "c"
        ArtifactStore(nested)
        assert nested.is_dir()

    def test_repr(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path / "artifacts")
        assert "ArtifactStore" in repr(store)

    def test_unicode_config(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path / "artifacts")
        config = {"prompt": "你好世界", "model": "gpt-4o"}
        store.save_config_snapshot(1, config)
        loaded = store.load_config_snapshot(1)
        assert loaded["prompt"] == "你好世界"


# ---------------------------------------------------------------------------
# ConfigStore tests
# ---------------------------------------------------------------------------


class TestConfigStore:
    """Tests for the ConfigStore YAML/JSON load/save."""

    def test_load_yaml(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "nuwa.yaml"
        cfg_path.write_text(
            "llm_model: openai/gpt-4o\ntraining_direction: Improve quality\nmax_rounds: 5\n",
            encoding="utf-8",
        )
        config = ConfigStore.load(cfg_path)
        assert config.llm_model == "openai/gpt-4o"
        assert config.max_rounds == 5

    def test_load_json(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "nuwa.json"
        cfg_path.write_text(
            json.dumps({"llm_model": "anthropic/claude", "max_rounds": 3}),
            encoding="utf-8",
        )
        config = ConfigStore.load(cfg_path)
        assert config.llm_model == "anthropic/claude"
        assert config.max_rounds == 3

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="not found"):
            ConfigStore.load(tmp_path / "nonexistent.yaml")

    def test_load_invalid_yaml_raises(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "bad.yaml"
        cfg_path.write_text("not: a mapping\n- just\n- a list\n", encoding="utf-8")
        with pytest.raises(ConfigError):
            ConfigStore.load(cfg_path)

    def test_save_yaml(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "output.yaml"
        config = NuwaConfig(training_direction="Test", max_rounds=7)
        ConfigStore.save(config, cfg_path)

        loaded = ConfigStore.load(cfg_path)
        assert loaded.training_direction == "Test"
        assert loaded.max_rounds == 7

    def test_save_json(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "output.json"
        config = NuwaConfig(training_direction="Test JSON", max_rounds=3)
        ConfigStore.save(config, cfg_path)

        loaded = ConfigStore.load(cfg_path)
        assert loaded.training_direction == "Test JSON"

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "deep" / "nested" / "config.yaml"
        config = NuwaConfig()
        ConfigStore.save(config, cfg_path)
        assert cfg_path.is_file()

    def test_get_default_config(self) -> None:
        config = ConfigStore.get_default_config()
        assert isinstance(config, NuwaConfig)

    def test_load_or_default_with_file(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "nuwa.yaml"
        cfg_path.write_text(
            "training_direction: Loaded\nmax_rounds: 15\n",
            encoding="utf-8",
        )
        config = ConfigStore.load_or_default(cfg_path)
        assert config.training_direction == "Loaded"
        assert config.max_rounds == 15

    def test_load_or_default_without_file(self, tmp_path: Path) -> None:
        config = ConfigStore.load_or_default(tmp_path / "nonexistent.yaml")
        assert isinstance(config, NuwaConfig)
        # Should have default values
        assert config.max_rounds == 10

    def test_resolve_config_path_explicit(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "explicit.yaml"
        cfg_path.write_text("training_direction: X\n", encoding="utf-8")
        result = ConfigStore.resolve_config_path(explicit=cfg_path)
        assert result == cfg_path

    def test_resolve_config_path_fallback(self, tmp_path: Path) -> None:
        result = ConfigStore.resolve_config_path(
            explicit=tmp_path / "missing.yaml",
            project_dir=tmp_path,
        )
        # Returns the explicit path even if it doesn't exist
        assert result == tmp_path / "missing.yaml"

    def test_repr(self) -> None:
        assert "ConfigStore" in repr(ConfigStore())
