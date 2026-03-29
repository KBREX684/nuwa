"""Unit tests for nuwa.config.schema — NuwaConfig validation."""

from __future__ import annotations

import pytest

from nuwa.config.schema import NuwaConfig


class TestNuwaConfigDefaults:
    def test_default_model(self):
        cfg = NuwaConfig()
        assert cfg.llm_model == "openai/gpt-4o"

    def test_default_rounds(self):
        cfg = NuwaConfig()
        assert cfg.max_rounds == 10

    def test_default_project_dir(self):
        cfg = NuwaConfig()
        assert cfg.project_dir is not None

    def test_default_resume_and_distributed(self):
        cfg = NuwaConfig()
        assert cfg.distributed_workers == 1
        assert cfg.resume is False
        assert cfg.plugin_modules == []


class TestNuwaConfigValidation:
    def test_max_rounds_minimum(self):
        with pytest.raises(Exception):
            NuwaConfig(max_rounds=0)

    def test_max_rounds_negative(self):
        with pytest.raises(Exception):
            NuwaConfig(max_rounds=-1)

    def test_samples_per_round_minimum(self):
        with pytest.raises(Exception):
            NuwaConfig(samples_per_round=0)

    def test_distributed_workers_minimum(self):
        with pytest.raises(Exception):
            NuwaConfig(distributed_workers=0)

    def test_train_val_split_bounds(self):
        with pytest.raises(Exception):
            NuwaConfig(train_val_split=0.0)
        with pytest.raises(Exception):
            NuwaConfig(train_val_split=1.0)
        # Valid values
        NuwaConfig(train_val_split=0.5)
        NuwaConfig(train_val_split=0.9)

    def test_overfitting_threshold_non_negative(self):
        with pytest.raises(Exception):
            NuwaConfig(overfitting_threshold=-0.1)

    def test_regression_tolerance_non_negative(self):
        with pytest.raises(Exception):
            NuwaConfig(regression_tolerance=-0.1)

    def test_consistency_threshold_bounds(self):
        with pytest.raises(Exception):
            NuwaConfig(consistency_threshold=1.5)
        with pytest.raises(Exception):
            NuwaConfig(consistency_threshold=-0.1)

    def test_connector_type_valid(self):
        for ct in ("http", "cli", "function"):
            cfg = NuwaConfig(connector_type=ct)
            assert cfg.connector_type == ct

    def test_connector_type_invalid(self):
        with pytest.raises(Exception):
            NuwaConfig(connector_type="grpc")


class TestNuwaConfigBuildMethods:
    def test_build_training_config(self):
        cfg = NuwaConfig(
            training_direction="improve accuracy",
            max_rounds=5,
            samples_per_round=15,
        )
        tc = cfg.build_training_config()
        assert tc.training_direction == "improve accuracy"
        assert tc.max_rounds == 5
        assert tc.samples_per_round == 15

    def test_build_llm_kwargs_includes_model(self):
        cfg = NuwaConfig(llm_model="deepseek/deepseek-chat")
        kwargs = cfg.build_llm_kwargs()
        assert kwargs["model"] == "deepseek/deepseek-chat"

    def test_build_llm_kwargs_with_api_key(self):
        cfg = NuwaConfig(llm_api_key="sk-test-key")
        kwargs = cfg.build_llm_kwargs()
        assert kwargs.get("api_key") is not None

    def test_build_llm_kwargs_with_base_url(self):
        cfg = NuwaConfig(llm_base_url="https://api.custom.com/v1")
        kwargs = cfg.build_llm_kwargs()
        assert kwargs.get("base_url") == "https://api.custom.com/v1"

    def test_build_connector_http(self):
        cfg = NuwaConfig(
            connector_type="http",
            connector_params={"url": "http://localhost:8000/chat"},
        )
        connector = cfg.build_connector()
        assert hasattr(connector, "invoke")

    def test_build_connector_function(self):
        def my_agent(user_input: str, config: dict | None = None) -> str:
            return "response"

        cfg = NuwaConfig(
            connector_type="function",
            connector_params={"func": my_agent},
        )
        connector = cfg.build_connector()
        assert hasattr(connector, "invoke")

    def test_build_connector_cli(self):
        cfg = NuwaConfig(
            connector_type="cli",
            connector_params={"command": "echo"},
        )
        connector = cfg.build_connector()
        assert hasattr(connector, "invoke")


class TestNuwaConfigYaml:
    def test_roundtrip_yaml(self, tmp_path):
        cfg = NuwaConfig(
            training_direction="test roundtrip",
            max_rounds=3,
        )
        path = tmp_path / "config.yaml"
        cfg.to_yaml(path)

        loaded = NuwaConfig.from_yaml(path)
        assert loaded.training_direction == "test roundtrip"
        assert loaded.max_rounds == 3
