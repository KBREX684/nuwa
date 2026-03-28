"""Sandboxed agent wrapper -- all mutations stay in the sandbox."""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from nuwa.core.types import AgentResponse
from nuwa.sandbox.diff import DiffEntry, deep_diff

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Snapshot dataclass
# ---------------------------------------------------------------------------


@dataclass
class ConfigSnapshot:
    """An immutable, versioned snapshot of a sandbox configuration.

    Attributes:
        version: Monotonically increasing version number starting at 1.
        config: Deep-copied configuration dict at the time of the snapshot.
        timestamp: UTC timestamp when the snapshot was created.
        description: Human-readable note describing why this snapshot was taken.
        round_num: Training round number that triggered the snapshot, if any.
    """

    version: int
    config: dict[str, Any]
    timestamp: datetime
    description: str
    round_num: int | None = None


# ---------------------------------------------------------------------------
# SandboxedAgent
# ---------------------------------------------------------------------------


class SandboxedAgent:
    """A sandboxed wrapper around a real ``TargetAgent``.

    Implements the ``TargetAgent`` protocol but ensures that:

    * :meth:`invoke` calls the real agent with the **sandbox** config injected
      via the ``config`` parameter -- the real agent's internal state is never
      touched.
    * :meth:`apply_config` only modifies the sandbox copy, **never** the real
      agent.
    * :meth:`get_current_config` returns the sandbox config.
    * Every :meth:`apply_config` call creates a versioned
      :class:`ConfigSnapshot` persisted to disk.
    * :meth:`rollback` restores any previous snapshot version.

    Parameters:
        real_agent: The underlying ``TargetAgent`` whose ``invoke`` is called.
        original_config: The real agent's config at sandbox-creation time.
        session_id: Unique identifier for this sandbox session.
        snapshot_dir: Directory where JSON snapshot files are written.
    """

    def __init__(
        self,
        real_agent: Any,  # TargetAgent protocol
        original_config: dict[str, Any],
        session_id: str,
        snapshot_dir: Path,
    ) -> None:
        self._real_agent = real_agent
        self._original_config: dict[str, Any] = copy.deepcopy(original_config)
        self._sandbox_config: dict[str, Any] = copy.deepcopy(original_config)
        self._snapshots: list[ConfigSnapshot] = []
        self._session_id = session_id
        self._snapshot_dir = snapshot_dir
        self._mutation_count: int = 0

        # Ensure snapshot directory exists.
        self._snapshot_dir.mkdir(parents=True, exist_ok=True)

        # Persist the initial snapshot (version 0 = original).
        self._persist_snapshot(
            ConfigSnapshot(
                version=0,
                config=copy.deepcopy(original_config),
                timestamp=datetime.now(UTC),
                description="initial snapshot (original config)",
                round_num=None,
            )
        )

    # ------------------------------------------------------------------
    # TargetAgent protocol methods
    # ------------------------------------------------------------------

    async def invoke(
        self,
        input_text: str,
        config: dict[str, Any] | None = None,
    ) -> AgentResponse:
        """Invoke the real agent with the sandbox config injected.

        The real agent receives the sandbox config via the ``config``
        keyword argument -- its internal state (via ``apply_config``) is
        **never** modified.

        Args:
            input_text: The prompt to send to the agent.
            config: Optional per-call overrides merged on top of the sandbox
                config.

        Returns:
            The :class:`AgentResponse` from the real agent.
        """
        effective_config = {**self._sandbox_config, **(config or {})}
        response = await self._real_agent.invoke(input_text, config=effective_config)
        return cast(AgentResponse, response)

    def apply_config(self, config: dict[str, Any]) -> None:
        """Apply *config* to the **sandbox only**. Creates a versioned snapshot.

        The real agent is **never** touched by this method.

        Args:
            config: The new configuration dict to adopt in the sandbox.
        """
        self._sandbox_config = copy.deepcopy(config)
        self._mutation_count += 1
        version = self._mutation_count

        snapshot = ConfigSnapshot(
            version=version,
            config=copy.deepcopy(config),
            timestamp=datetime.now(UTC),
            description=f"apply_config (mutation #{version})",
            round_num=None,
        )
        self._snapshots.append(snapshot)
        self._persist_snapshot(snapshot)

        logger.debug(
            "Sandbox %s: applied config v%d (mutation #%d)",
            self._session_id,
            version,
            self._mutation_count,
        )

    def get_current_config(self) -> dict[str, Any]:
        """Return a deep copy of the sandbox's current configuration.

        This is the sandbox config, **not** the real agent's config.
        """
        return copy.deepcopy(self._sandbox_config)

    # ------------------------------------------------------------------
    # Sandbox-specific methods
    # ------------------------------------------------------------------

    def get_original_config(self) -> dict[str, Any]:
        """Return a deep copy of the immutable original config snapshot."""
        return copy.deepcopy(self._original_config)

    def rollback(self, version: int | None = None) -> dict[str, Any]:
        """Rollback the sandbox config to a specific snapshot version.

        Args:
            version: The snapshot version to restore.  ``None`` means the
                most recent snapshot before the current one.  ``0`` restores
                the original config.

        Returns:
            The restored configuration dict (deep copy).

        Raises:
            ValueError: If the requested version does not exist.
        """
        if version == 0 or (version is None and not self._snapshots):
            self._sandbox_config = copy.deepcopy(self._original_config)
            logger.info("Sandbox %s: rolled back to original config", self._session_id)
            return copy.deepcopy(self._sandbox_config)

        if version is None:
            # Roll back to the previous snapshot (second-to-last if it exists).
            if len(self._snapshots) >= 2:
                target = self._snapshots[-2]
            else:
                # Only one snapshot -- roll back to original.
                self._sandbox_config = copy.deepcopy(self._original_config)
                logger.info(
                    "Sandbox %s: rolled back to original config", self._session_id
                )
                return copy.deepcopy(self._sandbox_config)
        else:
            target = None
            for snap in self._snapshots:
                if snap.version == version:
                    target = snap
                    break
            if target is None:
                raise ValueError(
                    f"Snapshot version {version} not found. "
                    f"Available: {[s.version for s in self._snapshots]}"
                )

        assert target is not None
        self._sandbox_config = copy.deepcopy(target.config)
        logger.info(
            "Sandbox %s: rolled back to snapshot v%d",
            self._session_id,
            target.version,
        )
        return copy.deepcopy(self._sandbox_config)

    def get_diff(self) -> dict[str, Any]:
        """Return a structured diff between the original and current sandbox config.

        Returns:
            A dict with ``entries`` (list of serialised :class:`DiffEntry`) and
            ``summary`` (human-readable change count).
        """
        entries: list[DiffEntry] = deep_diff(self._original_config, self._sandbox_config)
        return {
            "entries": [
                {
                    "path": e.path,
                    "change_type": e.change_type,
                    "old_value": e.old_value,
                    "new_value": e.new_value,
                }
                for e in entries
            ],
            "summary": f"{len(entries)} change(s) from original config",
        }

    @property
    def mutation_count(self) -> int:
        """Number of config mutations applied in this sandbox session."""
        return self._mutation_count

    @property
    def session_id(self) -> str:
        """Unique identifier for this sandbox session."""
        return self._session_id

    @property
    def snapshots(self) -> list[ConfigSnapshot]:
        """Return the list of all snapshots (read-only view)."""
        return list(self._snapshots)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _persist_snapshot(self, snapshot: ConfigSnapshot) -> None:
        """Write a snapshot to disk as a JSON file for crash recovery.

        The filename encodes the version number for easy ordering.
        """
        filename = f"snapshot_v{snapshot.version:04d}.json"
        filepath = self._snapshot_dir / filename
        try:
            data = {
                "version": snapshot.version,
                "config": snapshot.config,
                "timestamp": snapshot.timestamp.isoformat(),
                "description": snapshot.description,
                "round_num": snapshot.round_num,
                "session_id": self._session_id,
            }
            filepath.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
            logger.debug("Persisted snapshot to %s", filepath)
        except OSError:
            logger.exception("Failed to persist snapshot v%d to %s", snapshot.version, filepath)

    def __repr__(self) -> str:
        return (
            f"SandboxedAgent(session={self._session_id!r}, "
            f"mutations={self._mutation_count}, "
            f"snapshots={len(self._snapshots)})"
        )
