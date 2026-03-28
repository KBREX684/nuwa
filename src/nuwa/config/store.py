"""Persistent configuration store for Nuwa projects."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import yaml

from nuwa.config.schema import NuwaConfig
from nuwa.core.exceptions import ConfigError

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_NAME = "nuwa.yaml"


class ConfigStore:
    """Load, save, and manage :class:`~nuwa.config.schema.NuwaConfig` instances.

    The store is stateless -- every operation takes or returns explicit paths
    and config objects, making it easy to test and compose.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def load(path: Path | str) -> NuwaConfig:
        """Load a :class:`NuwaConfig` from a YAML or JSON file.

        Parameters
        ----------
        path:
            Path to the configuration file.  Both ``.yaml`` / ``.yml`` and
            ``.json`` extensions are supported.

        Raises
        ------
        ConfigError
            If the file does not exist, cannot be parsed, or fails
            Pydantic validation.
        """
        path = Path(path)
        if not path.is_file():
            raise ConfigError(f"Configuration file not found: {path}")

        try:
            text = path.read_text(encoding="utf-8")
            if path.suffix == ".json":
                data = json.loads(text)
            else:
                data = yaml.safe_load(text)

            if not isinstance(data, dict):
                raise ConfigError(
                    f"Expected a mapping in {path}, got {type(data).__name__}"
                )

            config = NuwaConfig.model_validate(data)
            logger.info("Loaded configuration from %s", path)
            return config

        except ConfigError:
            raise
        except Exception as exc:
            raise ConfigError(f"Failed to load config from {path}: {exc}") from exc

    @staticmethod
    def save(config: NuwaConfig, path: Path | str) -> None:
        """Persist *config* to disk.

        The format is chosen by the file extension: ``.json`` writes JSON,
        everything else writes YAML.

        Parameters
        ----------
        config:
            The configuration to serialise.
        path:
            Destination file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if path.suffix == ".json":
                text = config.model_dump_json(indent=2) + "\n"
                path.write_text(text, encoding="utf-8")
            else:
                config.to_yaml(path)
            logger.info("Saved configuration to %s", path)
        except Exception as exc:
            raise ConfigError(f"Failed to save config to {path}: {exc}") from exc

    @staticmethod
    def get_default_config() -> NuwaConfig:
        """Return a :class:`NuwaConfig` populated with sensible defaults.

        Useful for bootstrapping a new project or running with zero
        configuration.
        """
        return NuwaConfig()

    @staticmethod
    def resolve_config_path(
        explicit: Path | str | None = None,
        project_dir: Path | str | None = None,
    ) -> Path:
        """Determine the config file path using a fall-through strategy.

        1. *explicit* -- if provided and the file exists, use it.
        2. ``<project_dir>/nuwa.yaml`` -- if *project_dir* is given.
        3. ``./nuwa.yaml`` in the current working directory.

        Returns the first candidate that exists on disk; if none do, returns
        the best candidate path (so the caller can create it).
        """
        candidates: list[Path] = []

        if explicit is not None:
            candidates.append(Path(explicit))

        if project_dir is not None:
            candidates.append(Path(project_dir) / _DEFAULT_CONFIG_NAME)

        candidates.append(Path.cwd() / _DEFAULT_CONFIG_NAME)

        for candidate in candidates:
            if candidate.is_file():
                return candidate

        # Nothing found -- return the best-guess path
        return candidates[0]

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @classmethod
    def load_or_default(cls, path: Path | str | None = None) -> NuwaConfig:
        """Load from *path* if it exists, otherwise return defaults."""
        if path is not None and Path(path).is_file():
            return cls.load(path)
        return cls.get_default_config()

    def __repr__(self) -> str:
        return "ConfigStore()"
