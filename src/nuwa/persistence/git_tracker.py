"""Lightweight git-based experiment tracker.

Inspired by autoresearch's *git-as-database* approach, this module provides
optional version control of experiment configurations.  Every training round
can be committed as a snapshot, enabling rollback and a clear improvement
chain.

The tracker gracefully no-ops when git is unavailable or ``enabled=False``.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CONFIG_FILENAME = "config_snapshot.json"

# A safe branch / ref name: alphanumerics, hyphens, underscores, dots, slashes.
_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9._/\-]+$")


class GitTracker:
    """Optional git-backed experiment version control.

    Parameters
    ----------
    project_dir:
        Working directory where the git repository lives (or will be
        initialised).
    enabled:
        Master switch.  When *False* every public method is a silent no-op.
    """

    def __init__(self, project_dir: Path, *, enabled: bool = False) -> None:
        self._project_dir = Path(project_dir)
        self._enabled = enabled
        self._git_available: bool | None = None  # lazily detected

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_branch_name(name: str) -> None:
        """Raise ``ValueError`` if *name* looks like a git injection attempt."""
        if not name:
            raise ValueError("Branch name must not be empty.")
        if not _SAFE_NAME_RE.match(name):
            raise ValueError(
                f"Invalid branch name {name!r}: "
                "only alphanumerics, hyphens, underscores, dots, and slashes allowed."
            )
        if name.startswith("-"):
            raise ValueError(f"Branch name {name!r} must not start with a dash.")

    @staticmethod
    def _sanitize_message(msg: str) -> str:
        """Strip control characters from a commit message."""
        # Remove newlines and null bytes that could confuse git.
        return msg.replace("\x00", "").replace("\n", " ").replace("\r", "")[:500]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def init_branch(self, name: str) -> bool:
        """Create (and switch to) an experiment branch.

        Returns *True* if the branch was created successfully, *False*
        otherwise (including when the tracker is disabled).
        """
        if not self._should_run():
            return False

        self._validate_branch_name(name)

        # Ensure there is a git repo.
        if not (self._project_dir / ".git").is_dir():
            ok = self._run_git("init")
            if not ok:
                return False
            # Make an initial commit so branches work.
            self._run_git("commit", "--allow-empty", "-m", "Initial commit (nuwa tracker)")

        ok = self._run_git("checkout", "-B", name)
        if ok:
            logger.info("GitTracker: switched to branch %r", name)
        return ok

    def commit_round(
        self,
        round_num: int,
        config: dict[str, Any],
        message: str = "",
    ) -> bool:
        """Commit a config snapshot for the given round.

        The configuration is written to ``config_snapshot.json`` and committed
        with a message that includes the round number.

        Returns *True* on success.
        """
        if not self._should_run():
            return False

        config_path = self._project_dir / _CONFIG_FILENAME
        config_path.write_text(
            json.dumps(config, indent=2, default=str),
            encoding="utf-8",
        )

        commit_msg = self._sanitize_message(message) or f"Round {round_num} config snapshot"
        self._run_git("add", _CONFIG_FILENAME)
        ok = self._run_git("commit", "-m", f"[round-{round_num}] {commit_msg}")
        if ok:
            logger.info("GitTracker: committed round %d", round_num)
        return ok

    def rollback_to(self, round_num: int) -> bool:
        """Reset the current branch to the commit for *round_num*.

        Uses ``git log --grep`` to find the matching commit hash, then
        performs a hard reset.

        Returns *True* on success.
        """
        if not self._should_run():
            return False

        commit_hash = self._find_round_commit(round_num)
        if not commit_hash:
            logger.warning(
                "GitTracker: no commit found for round %d; rollback skipped.",
                round_num,
            )
            return False

        ok = self._run_git("reset", "--hard", commit_hash)
        if ok:
            logger.info("GitTracker: rolled back to round %d (%s)", round_num, commit_hash[:8])
        return ok

    def get_improvement_chain(self) -> list[dict[str, Any]]:
        """Return the chain of kept improvements as a list of dicts.

        Each entry contains ``round_num``, ``commit_hash``, and ``message``.
        """
        if not self._should_run():
            return []

        result = self._run_git_output(
            "log", "--oneline", "--grep=\\[round-", "--format=%H %s"
        )
        if result is None:
            return []

        chain: list[dict[str, Any]] = []
        for line in result.strip().splitlines():
            if not line.strip():
                continue
            parts = line.split(" ", 1)
            commit_hash = parts[0]
            message = parts[1] if len(parts) > 1 else ""

            # Extract round number from "[round-N]" prefix.
            round_num: int | None = None
            if "[round-" in message:
                try:
                    token = message.split("[round-")[1].split("]")[0]
                    round_num = int(token)
                except (IndexError, ValueError):
                    pass

            chain.append(
                {
                    "commit_hash": commit_hash,
                    "round_num": round_num,
                    "message": message,
                }
            )

        # Reverse so oldest first.
        chain.reverse()
        return chain

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _should_run(self) -> bool:
        """Return True only when tracking is enabled and git is available."""
        if not self._enabled:
            return False
        if self._git_available is None:
            self._git_available = shutil.which("git") is not None
            if not self._git_available:
                logger.info("GitTracker: git binary not found; tracker disabled.")
        return self._git_available

    def _run_git(self, *args: str) -> bool:
        """Run a git command; return True on success."""
        try:
            subprocess.run(
                ["git", *args],
                cwd=self._project_dir,
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
            logger.debug("GitTracker: git %s failed: %s", " ".join(args), exc)
            return False

    def _run_git_output(self, *args: str) -> str | None:
        """Run a git command and return its stdout, or None on failure."""
        try:
            proc = subprocess.run(
                ["git", *args],
                cwd=self._project_dir,
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )
            return proc.stdout
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
            logger.debug("GitTracker: git %s failed: %s", " ".join(args), exc)
            return None

    def _find_round_commit(self, round_num: int) -> str | None:
        """Find the commit hash tagged with ``[round-N]``."""
        output = self._run_git_output(
            "log", "--all", f"--grep=[round-{round_num}]", "--format=%H", "-1"
        )
        if output and output.strip():
            return output.strip()
        return None

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"GitTracker(project_dir={self._project_dir!r}, "
            f"enabled={self._enabled})"
        )
