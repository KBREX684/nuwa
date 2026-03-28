"""Snapshot storage for prompt and configuration artefacts.

Each training round can persist its active configuration and prompt so that
users can inspect, diff, and roll back changes across the optimisation
history.
"""

from __future__ import annotations

import difflib
import json
import logging
import re
from pathlib import Path
from typing import Any, cast

logger = logging.getLogger(__name__)

_CONFIG_PATTERN = "round_{round_num}_config.json"
_PROMPT_PATTERN = "round_{round_num}_prompt.txt"
_ROUND_RE = re.compile(r"^round_(\d+)_config\.json$")


class ArtifactStore:
    """File-based snapshot store for per-round configs and prompts.

    Parameters
    ----------
    store_dir:
        Root directory for stored artefacts.  Created on first write.
    """

    def __init__(self, store_dir: Path) -> None:
        self._store_dir = Path(store_dir)
        self._store_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Config snapshots
    # ------------------------------------------------------------------

    def save_config_snapshot(self, round_num: int, config: dict[str, Any]) -> None:
        """Persist *config* as ``round_<N>_config.json``.

        Parameters
        ----------
        round_num:
            Training round number (>= 0).
        config:
            Arbitrary JSON-serialisable configuration dictionary.
        """
        path = self._config_path(round_num)
        path.write_text(
            json.dumps(config, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        logger.debug("Saved config snapshot for round %d at %s", round_num, path)

    def load_config_snapshot(self, round_num: int) -> dict[str, Any] | None:
        """Load the config snapshot for *round_num*, or ``None`` if absent."""
        path = self._config_path(round_num)
        if not path.exists():
            return None
        text = path.read_text(encoding="utf-8")
        loaded = json.loads(text)
        return cast(dict[str, Any], loaded)

    # ------------------------------------------------------------------
    # Prompt snapshots
    # ------------------------------------------------------------------

    def save_prompt_snapshot(self, round_num: int, prompt: str) -> None:
        """Persist *prompt* as ``round_<N>_prompt.txt``.

        Parameters
        ----------
        round_num:
            Training round number (>= 0).
        prompt:
            The full prompt string active during this round.
        """
        path = self._prompt_path(round_num)
        path.write_text(prompt, encoding="utf-8")
        logger.debug("Saved prompt snapshot for round %d at %s", round_num, path)

    def load_prompt_snapshot(self, round_num: int) -> str | None:
        """Load the prompt snapshot for *round_num*, or ``None`` if absent."""
        path = self._prompt_path(round_num)
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def list_snapshots(self) -> list[int]:
        """Return sorted list of round numbers that have config snapshots."""
        rounds: list[int] = []
        for entry in self._store_dir.iterdir():
            match = _ROUND_RE.match(entry.name)
            if match:
                rounds.append(int(match.group(1)))
        rounds.sort()
        return rounds

    # ------------------------------------------------------------------
    # Diffing
    # ------------------------------------------------------------------

    def get_diff(self, round_a: int, round_b: int) -> str:
        """Return a unified text diff between two config snapshots.

        Parameters
        ----------
        round_a, round_b:
            Round numbers to compare.

        Returns
        -------
        str
            Unified diff string.  Empty when the configs are identical or
            when either snapshot is missing.

        Raises
        ------
        FileNotFoundError
            If either snapshot does not exist on disk.
        """
        config_a = self.load_config_snapshot(round_a)
        config_b = self.load_config_snapshot(round_b)

        if config_a is None:
            raise FileNotFoundError(
                f"Config snapshot for round {round_a} not found in {self._store_dir}"
            )
        if config_b is None:
            raise FileNotFoundError(
                f"Config snapshot for round {round_b} not found in {self._store_dir}"
            )

        lines_a = (json.dumps(config_a, indent=2, ensure_ascii=False) + "\n").splitlines(
            keepends=True
        )
        lines_b = (json.dumps(config_b, indent=2, ensure_ascii=False) + "\n").splitlines(
            keepends=True
        )

        diff = difflib.unified_diff(
            lines_a,
            lines_b,
            fromfile=f"round_{round_a}_config.json",
            tofile=f"round_{round_b}_config.json",
        )
        return "".join(diff)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _config_path(self, round_num: int) -> Path:
        return self._store_dir / _CONFIG_PATTERN.format(round_num=round_num)

    def _prompt_path(self, round_num: int) -> Path:
        return self._store_dir / _PROMPT_PATTERN.format(round_num=round_num)

    @property
    def store_dir(self) -> Path:
        """Absolute path to the artefact store directory."""
        return self._store_dir

    def __repr__(self) -> str:
        return f"ArtifactStore(store_dir={self._store_dir!r})"
