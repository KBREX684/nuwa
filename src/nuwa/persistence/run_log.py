"""JSONL append-only run log for persisting training round results.

Each round result is serialised as a single JSON line and appended to
``runs.jsonl`` inside the configured log directory.  This provides a
crash-safe, human-readable audit trail of every training round.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import UTC, datetime
from pathlib import Path

from nuwa.core.types import RoundResult

logger = logging.getLogger(__name__)

_LOG_FILENAME = "runs.jsonl"
_ARCHIVE_PREFIX = "runs_archived_"
_TSV_FILENAME = "results.tsv"
_TSV_HEADER = "round_num\ttrain_score\tval_score\tstatus\tdescription\ttimestamp\n"


class RunLog:
    """Append-only JSONL log of :class:`~nuwa.core.types.RoundResult` objects.

    Parameters
    ----------
    log_dir:
        Directory where the ``runs.jsonl`` file is stored.  Created
        automatically if it does not exist.
    """

    def __init__(self, log_dir: Path) -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self._log_dir / _LOG_FILENAME
        self._tsv_path = self._log_dir / _TSV_FILENAME

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def append_round(self, round_result: RoundResult) -> None:
        """Serialise *round_result* to JSON and append it as a single line.

        The file is opened in append mode so concurrent readers never see
        a partially written line (on POSIX systems with reasonably sized
        writes).
        """
        line = round_result.model_dump_json()
        with self._log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        logger.debug("Appended round %d to %s", round_result.round_num, self._log_path)

    def load_history(self) -> list[RoundResult]:
        """Read every round result from the JSONL file.

        Returns an empty list when the log file does not yet exist.
        Malformed lines are skipped with a warning.
        """
        if not self._log_path.exists():
            return []

        results: list[RoundResult] = []
        with self._log_path.open("r", encoding="utf-8") as fh:
            for line_num, raw_line in enumerate(fh, start=1):
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    results.append(RoundResult.model_validate_json(raw_line))
                except (json.JSONDecodeError, ValueError) as exc:
                    logger.warning(
                        "Skipping malformed line %d in %s: %s",
                        line_num,
                        self._log_path,
                        exc,
                    )
        return results

    def get_latest_run(self) -> RoundResult | None:
        """Return the most recently appended round result, or ``None``."""
        history = self.load_history()
        return history[-1] if history else None

    def get_best_round(self) -> RoundResult | None:
        """Return the round with the highest validation mean score.

        If no round has validation scores, falls back to train scores.
        Returns ``None`` when the log is empty.
        """
        history = self.load_history()
        if not history:
            return None

        def _val_score(rr: RoundResult) -> float:
            if rr.val_scores is not None:
                return rr.val_scores.mean_score
            return rr.train_scores.mean_score

        return max(history, key=_val_score)

    def clear(self) -> None:
        """Archive the current log file and start fresh.

        The existing file is renamed with a UTC timestamp suffix so that
        no data is lost.  If the log file does not exist this is a no-op.
        """
        if not self._log_path.exists():
            logger.info("No log file to archive at %s", self._log_path)
            return

        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        archive_name = f"{_ARCHIVE_PREFIX}{ts}.jsonl"
        archive_path = self._log_dir / archive_name
        shutil.move(str(self._log_path), str(archive_path))
        logger.info("Archived %s -> %s", self._log_path, archive_path)

    # ------------------------------------------------------------------
    # Results TSV (append-only, survives rollbacks)
    # ------------------------------------------------------------------

    def append_tsv_line(
        self,
        round_num: int,
        train_score: float,
        val_score: float | None,
        status: str,
        description: str = "",
    ) -> None:
        """Append a single experiment result line to ``results.tsv``.

        The TSV file is a lightweight, human-readable experiment log inspired
        by autoresearch's ``results.tsv`` pattern.  It is **never deleted** by
        :meth:`clear` so that it survives config rollbacks and provides a
        permanent audit trail.

        A header row is written automatically when the file is first created.

        Parameters
        ----------
        round_num:
            The training round number.
        train_score:
            Mean training score for the round.
        val_score:
            Mean validation score (``None`` if validation was skipped).
        status:
            Short status token, e.g. ``"kept"``, ``"reverted"``, ``"error"``.
        description:
            Free-form description of what happened in this round.
        """
        # Sanitise description: collapse tabs/newlines to spaces.
        safe_desc = description.replace("\t", " ").replace("\n", " ").strip()

        needs_header = not self._tsv_path.exists() or self._tsv_path.stat().st_size == 0

        with self._tsv_path.open("a", encoding="utf-8") as fh:
            if needs_header:
                fh.write(_TSV_HEADER)

            val_str = f"{val_score:.4f}" if val_score is not None else "N/A"
            ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
            fh.write(f"{round_num}\t{train_score:.4f}\t{val_str}\t{status}\t{safe_desc}\t{ts}\n")

        logger.debug("Appended round %d to %s", round_num, self._tsv_path)

    @property
    def tsv_path(self) -> Path:
        """Absolute path to the append-only results TSV file."""
        return self._tsv_path

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def log_path(self) -> Path:
        """Absolute path to the JSONL log file."""
        return self._log_path

    def __repr__(self) -> str:
        return f"RunLog(log_dir={self._log_dir!r})"
