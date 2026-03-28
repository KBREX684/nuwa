"""Main Pydantic configuration schema for the Nuwa AI Trainer framework."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from pydantic import BaseModel, Field, SecretStr, model_validator

from nuwa.core.defaults import DEFAULT_LLM_MODEL, DEFAULT_LLM_TEMPERATURE
from nuwa.core.types import TrainingConfig

logger = logging.getLogger(__name__)


class NuwaConfig(BaseModel):
    """Top-level configuration for a Nuwa training session.

    Provides helpers to hydrate the object from / serialise to YAML, and
    convenience factories for the main runtime objects (training config,
    connector, LLM backend).
    """

    model_config = {"arbitrary_types_allowed": True}

    # -- LLM settings --------------------------------------------------
    llm_model: str = Field(
        default=DEFAULT_LLM_MODEL,
        description="Model identifier in ``provider/model`` format.",
    )
    llm_api_key: SecretStr | None = Field(
        default=None,
        description="API key for the LLM provider (kept as SecretStr).",
    )
    llm_base_url: str | None = Field(
        default=None,
        description="Optional base URL override for the LLM provider.",
    )
    llm_temperature: float = Field(
        default=DEFAULT_LLM_TEMPERATURE,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for LLM calls.",
    )

    # -- Connector settings --------------------------------------------
    connector_type: Literal["http", "cli", "function"] = Field(
        default="http",
        description="Which connector adapter to use.",
    )
    connector_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Keyword arguments forwarded to the connector constructor.",
    )

    # -- Training settings ---------------------------------------------
    training_direction: str = Field(
        default="Improve the target agent's response quality.",
        description="Natural-language description of the desired training goal.",
    )
    max_rounds: int = Field(default=10, ge=1)
    samples_per_round: int = Field(default=20, ge=1)
    train_val_split: float = Field(default=0.7, gt=0.0, lt=1.0)
    overfitting_threshold: float = Field(default=0.15, ge=0.0)
    consistency_runs: int = Field(default=3, ge=1)
    consistency_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    regression_tolerance: float = Field(default=0.05, ge=0.0)

    # -- Misc ----------------------------------------------------------
    project_dir: Path = Field(
        default=Path(".nuwa"),
        description="Directory for persisted artefacts (runs, logs, configs).",
    )
    verbose: bool = False

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def _validate_connector_params(self) -> NuwaConfig:
        """Ensure required connector params are present for the chosen type."""
        if self.connector_type == "http" and "url" not in self.connector_params:
            logger.warning("connector_type is 'http' but 'url' is missing from connector_params.")
        if self.connector_type == "cli" and "command" not in self.connector_params:
            logger.warning(
                "connector_type is 'cli' but 'command' is missing from connector_params."
            )
        if self.connector_type == "function" and "func" not in self.connector_params:
            logger.warning(
                "connector_type is 'function' but 'func' is missing from connector_params."
            )
        return self

    # ------------------------------------------------------------------
    # YAML I/O
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: Path) -> NuwaConfig:
        """Deserialise a :class:`NuwaConfig` from a YAML file.

        Parameters
        ----------
        path:
            Filesystem path to a ``.yaml`` / ``.yml`` file.

        Returns
        -------
        NuwaConfig
        """
        text = Path(path).read_text(encoding="utf-8")
        loaded = yaml.safe_load(text)
        if not isinstance(loaded, dict):
            raise ValueError(f"Expected a YAML mapping in {path}, got {type(loaded).__name__}")
        data = cast(dict[str, Any], loaded)
        return cls.model_validate(data)

    def to_yaml(self, path: Path) -> None:
        """Serialise the configuration to a YAML file.

        :class:`~pydantic.SecretStr` fields are written as ``"***"`` to avoid
        leaking secrets to disk.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self._serialisable_dict()
        path.write_text(
            yaml.dump(data, default_flow_style=False, sort_keys=False),
            encoding="utf-8",
        )
        logger.info("Wrote config to %s", path)

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    def build_training_config(self) -> TrainingConfig:
        """Create a :class:`~nuwa.core.types.TrainingConfig` from this config."""
        return TrainingConfig(
            training_direction=self.training_direction,
            max_rounds=self.max_rounds,
            samples_per_round=self.samples_per_round,
            train_val_split=self.train_val_split,
            overfitting_threshold=self.overfitting_threshold,
            consistency_runs=self.consistency_runs,
            consistency_threshold=self.consistency_threshold,
            regression_tolerance=self.regression_tolerance,
        )

    def build_connector(self) -> Any:
        """Instantiate the connector specified by *connector_type*.

        Uses :func:`nuwa.connectors.registry.create_connector`.
        """
        from nuwa.connectors.registry import create_connector

        return create_connector(self.connector_type, **self.connector_params)

    def build_llm_kwargs(self) -> dict[str, Any]:
        """Return a dict of keyword arguments suitable for an LLM backend.

        This is intentionally backend-agnostic: callers can unpack it into
        whatever LLM client they use.
        """
        kwargs: dict[str, Any] = {
            "model": self.llm_model,
            "temperature": self.llm_temperature,
        }
        if self.llm_api_key:
            kwargs["api_key"] = self.llm_api_key.get_secret_value()
        if self.llm_base_url:
            kwargs["base_url"] = self.llm_base_url
        return kwargs

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _serialisable_dict(self) -> dict[str, Any]:
        """Produce a JSON-safe dict, masking secrets."""
        data = cast(dict[str, Any], json.loads(self.model_dump_json()))
        # Normalise Path objects to strings for YAML readability.
        if "project_dir" in data:
            data["project_dir"] = str(self.project_dir)
        return data

    def __repr__(self) -> str:
        return (
            f"NuwaConfig(llm_model={self.llm_model!r}, "
            f"connector_type={self.connector_type!r}, "
            f"max_rounds={self.max_rounds})"
        )
