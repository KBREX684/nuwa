"""Sandbox manager -- orchestrates isolated training environments."""

from __future__ import annotations

import copy
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nuwa.sandbox.agent import ConfigSnapshot, SandboxedAgent

logger = logging.getLogger(__name__)


class SandboxManager:
    """Manages isolated training environments.

    The sandbox intercepts all mutations to the target agent, ensuring:

    1. The real agent's config is **never** modified during training.
    2. All experiments run against a deep-copied config snapshot.
    3. A full rollback chain is maintained for every round.
    4. Changes are only promoted to the real agent after explicit human
       approval via :meth:`promote`.

    Parameters:
        real_agent: The ``TargetAgent`` whose behaviour is being improved.
        project_dir: Root directory for sandbox persistence
            (defaults to ``.nuwa``).
    """

    def __init__(
        self,
        real_agent: Any,  # TargetAgent protocol
        project_dir: Path = Path(".nuwa"),
    ) -> None:
        self._real_agent = real_agent
        self._project_dir = project_dir
        self._sessions: dict[str, SandboxedAgent] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def enter(self) -> SandboxedAgent:
        """Create a sandboxed copy of the target agent.

        Snapshots the current config and returns a :class:`SandboxedAgent`
        wrapper that satisfies the ``TargetAgent`` protocol.

        Returns:
            A new :class:`SandboxedAgent` ready for training.
        """
        session_id = uuid.uuid4().hex[:12]
        original_config = self._real_agent.get_current_config()
        snapshot_dir = self._project_dir / "sandbox" / session_id

        sandboxed = SandboxedAgent(
            real_agent=self._real_agent,
            original_config=copy.deepcopy(original_config),
            session_id=session_id,
            snapshot_dir=snapshot_dir,
        )

        self._sessions[session_id] = sandboxed
        self._write_session_metadata(session_id, original_config)

        logger.info(
            "Sandbox session %s created. Original config snapshotted to %s",
            session_id,
            snapshot_dir,
        )
        return sandboxed

    async def promote(self, sandboxed: SandboxedAgent) -> dict[str, Any]:
        """Apply the sandbox's current config to the real agent.

        This is the **only** code path through which changes reach the real
        agent.  Intended to be called after explicit human approval.

        Args:
            sandboxed: The :class:`SandboxedAgent` whose config should be
                promoted.

        Returns:
            The promoted configuration dict.

        Raises:
            ValueError: If the sandboxed agent does not belong to this manager.
        """
        self._validate_session(sandboxed)

        promoted_config = sandboxed.get_current_config()
        self._real_agent.apply_config(promoted_config)

        # Record the promotion event on disk.
        self._write_promotion_record(sandboxed, promoted_config)

        logger.info(
            "Sandbox session %s: promoted config to real agent (%d mutations applied)",
            sandboxed.session_id,
            sandboxed.mutation_count,
        )
        return promoted_config

    async def discard(self, sandboxed: SandboxedAgent) -> dict[str, Any]:
        """Discard all sandbox changes and return the original config.

        The real agent is **not** modified (it was never touched in the first
        place).

        Args:
            sandboxed: The :class:`SandboxedAgent` to discard.

        Returns:
            The original configuration that was snapshotted at sandbox
            creation time.
        """
        self._validate_session(sandboxed)
        original = sandboxed.get_original_config()

        # Record the discard event on disk.
        self._write_discard_record(sandboxed)

        logger.info(
            "Sandbox session %s: discarded (%d mutations thrown away)",
            sandboxed.session_id,
            sandboxed.mutation_count,
        )
        return original

    def get_snapshot_history(self) -> list[dict[str, Any]]:
        """Return all config snapshots taken across active sandbox sessions.

        Returns:
            A list of serialised snapshot dicts, ordered by session then
            version.
        """
        history: list[dict[str, Any]] = []
        for session_id, sandboxed in self._sessions.items():
            for snap in sandboxed.snapshots:
                history.append(
                    {
                        "session_id": session_id,
                        "version": snap.version,
                        "config": snap.config,
                        "timestamp": snap.timestamp.isoformat(),
                        "description": snap.description,
                        "round_num": snap.round_num,
                    }
                )
        return history

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_session(self, sandboxed: SandboxedAgent) -> None:
        """Raise ``ValueError`` if the sandboxed agent is not tracked."""
        if sandboxed.session_id not in self._sessions:
            raise ValueError(
                f"Unknown sandbox session {sandboxed.session_id!r}. "
                "Was it created by this SandboxManager?"
            )

    def _write_session_metadata(
        self,
        session_id: str,
        original_config: dict[str, Any],
    ) -> None:
        """Persist session metadata to disk."""
        meta_dir = self._project_dir / "sandbox" / session_id
        meta_dir.mkdir(parents=True, exist_ok=True)
        meta_path = meta_dir / "session.json"
        try:
            data = {
                "session_id": session_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "original_config": original_config,
                "status": "active",
            }
            meta_path.write_text(
                json.dumps(data, indent=2, default=str), encoding="utf-8"
            )
        except OSError:
            logger.exception("Failed to write session metadata for %s", session_id)

    def _write_promotion_record(
        self,
        sandboxed: SandboxedAgent,
        promoted_config: dict[str, Any],
    ) -> None:
        """Persist a promotion event to disk."""
        record_path = (
            self._project_dir
            / "sandbox"
            / sandboxed.session_id
            / "promotion.json"
        )
        try:
            data = {
                "session_id": sandboxed.session_id,
                "promoted_at": datetime.now(timezone.utc).isoformat(),
                "mutation_count": sandboxed.mutation_count,
                "promoted_config": promoted_config,
                "diff": sandboxed.get_diff(),
            }
            record_path.write_text(
                json.dumps(data, indent=2, default=str), encoding="utf-8"
            )
        except OSError:
            logger.exception(
                "Failed to write promotion record for session %s",
                sandboxed.session_id,
            )

    def _write_discard_record(self, sandboxed: SandboxedAgent) -> None:
        """Persist a discard event to disk."""
        record_path = (
            self._project_dir
            / "sandbox"
            / sandboxed.session_id
            / "discard.json"
        )
        try:
            data = {
                "session_id": sandboxed.session_id,
                "discarded_at": datetime.now(timezone.utc).isoformat(),
                "mutation_count": sandboxed.mutation_count,
                "diff_at_discard": sandboxed.get_diff(),
            }
            record_path.write_text(
                json.dumps(data, indent=2, default=str), encoding="utf-8"
            )
        except OSError:
            logger.exception(
                "Failed to write discard record for session %s",
                sandboxed.session_id,
            )

    def __repr__(self) -> str:
        return (
            f"SandboxManager(project_dir={self._project_dir!r}, "
            f"active_sessions={len(self._sessions)})"
        )
