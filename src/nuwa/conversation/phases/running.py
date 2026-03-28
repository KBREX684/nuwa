"""Running phase -- live display callbacks for the training loop."""

from __future__ import annotations

from typing import Any, Callable

from rich.live import Live
from rich.table import Table
from rich.text import Text

from nuwa.conversation.renderer import NuwaRenderer
from nuwa.core.types import LoopContext, RoundResult

__all__ = ["RunningPhase"]


class RunningPhase:
    """Provides a live-updating display during training loop execution.

    This phase does not own the training loop itself; instead it creates a
    callback that the :class:`~nuwa.engine.TrainingLoop` can invoke after
    each stage or round to refresh the console display.
    """

    def __init__(self, renderer: NuwaRenderer) -> None:
        self._renderer = renderer
        self._console = renderer.console
        self._scores: list[float] = []
        self._current_stage: str = ""
        self._current_round: int = 0
        self._max_rounds: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_callback(self) -> Callable[..., None]:
        """Return a callback ``fn(event, **kwargs)`` suitable for the loop.

        Recognised events:

        * ``"loop_start"``  -- kwargs: ``max_rounds``
        * ``"round_start"`` -- kwargs: ``round_num``
        * ``"stage"``       -- kwargs: ``stage_name``
        * ``"round_end"``   -- kwargs: ``round_result`` (:class:`RoundResult`)
        * ``"loop_end"``    -- (no extra kwargs)
        """

        def _callback(event: str, **kwargs: Any) -> None:  # noqa: ANN401
            if event == "loop_start":
                self._max_rounds = kwargs.get("max_rounds", 0)
                self._scores.clear()
                self._renderer.status(
                    f"训练开始，最多 {self._max_rounds} 轮。"
                )

            elif event == "round_start":
                self._current_round = kwargs.get("round_num", 0)
                self._console.print(
                    f"\n[bold cyan]{'─' * 50}[/bold cyan]"
                )
                self._console.print(
                    f"[bold cyan]轮次 {self._current_round} / "
                    f"{self._max_rounds}[/bold cyan]"
                )

            elif event == "stage":
                stage_name = kwargs.get("stage_name", "")
                self._current_stage = stage_name
                stage_labels: dict[str, str] = {
                    "generate": "生成评估样本",
                    "evaluate": "评估目标 Agent",
                    "score": "打分与分析",
                    "reflect": "反思与诊断",
                    "mutate": "变异与优化",
                    "guardrail": "护栏检查",
                }
                label = stage_labels.get(stage_name, stage_name)
                self._console.print(
                    f"  [dim]▸ {label}...[/dim]"
                )

            elif event == "round_end":
                rr: RoundResult | None = kwargs.get("round_result")
                if rr is not None:
                    score = rr.train_scores.mean_score
                    self._scores.append(score)
                    self._renderer.round_summary(rr)
                    self._print_trend()

            elif event == "loop_end":
                self._console.print()
                self._renderer.success("训练循环已完成。")

        return _callback

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _print_trend(self) -> None:
        """Print a compact text-based score trend line."""
        if not self._scores:
            return
        self._renderer._spark_chart(self._scores, label="训练均分趋势")
