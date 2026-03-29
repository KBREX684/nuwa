"""Running phase -- live display callbacks for the training loop."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from nuwa.conversation.renderer import NuwaRenderer
from nuwa.core.types import RoundResult

__all__ = ["RunningPhase"]


class RunningPhase:
    """Provides a live-updating display during training loop execution.

    This phase does not own the training loop itself; instead it creates a
    callback that the :class:`~nuwa.engine.TrainingLoop` invokes at round end
    to refresh the console display.
    """

    def __init__(self, renderer: NuwaRenderer) -> None:
        self._renderer = renderer
        self._console = renderer.console
        self._scores: list[float] = []
        self._current_round: int = 0
        self._max_rounds: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_callback(self) -> Callable[..., None]:
        """Return a callback matching ``TrainingLoop`` contract.

        Callback signature:
            ``fn(round_result: RoundResult, context: Any) -> None``
        """

        def _callback(round_result: RoundResult, context: Any) -> None:  # noqa: ANN401
            if not self._scores:
                self._max_rounds = int(getattr(getattr(context, "config", None), "max_rounds", 0))
                self._renderer.status(f"训练开始，最多 {self._max_rounds} 轮。")

            self._current_round = round_result.round_num
            self._console.print(f"\n[bold cyan]{'─' * 50}[/bold cyan]")
            self._console.print(
                f"[bold cyan]轮次 {self._current_round} / {self._max_rounds}[/bold cyan]"
            )

            score = round_result.train_scores.mean_score
            self._scores.append(score)
            self._renderer.round_summary(round_result)
            self._print_trend()

        return _callback

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _print_trend(self) -> None:
        """Print a compact text-based score trend line."""
        if not self._scores:
            return
        self._renderer._spark_chart(self._scores, label="训练均分趋势")
