"""Rich-based console rendering utilities for the Nuwa conversation UI."""

from __future__ import annotations

import difflib
from typing import Any

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from nuwa.core.types import LoopContext, RoundResult, TrainingResult

__all__ = ["NuwaRenderer"]

_VERSION = "0.1.0"

_BANNER = r"""
   ╔══════════════════════════════════════╗
   ║      女 娲  N U W A                  ║
   ║      AI Agent Trainer  v{ver}       ║
   ╚══════════════════════════════════════╝
"""


class NuwaRenderer:
    """Central rendering helper that owns a single :class:`rich.console.Console`."""

    def __init__(self, *, console: Console | None = None) -> None:
        self.console = console or Console()

    # ------------------------------------------------------------------
    # Banner / branding
    # ------------------------------------------------------------------

    def banner(self) -> None:
        """Display the Nuwa ASCII-art banner with the current version."""
        art = _BANNER.replace("{ver}", _VERSION)
        self.console.print(Text(art, style="bold cyan"))

    # ------------------------------------------------------------------
    # Simple styled messages
    # ------------------------------------------------------------------

    def status(self, msg: str) -> None:
        """Print an informational status line in cyan."""
        self.console.print(f"[cyan][bold]>[/bold] {escape(msg)}[/cyan]")

    def error(self, msg: str) -> None:
        """Print an error message in red."""
        self.console.print(f"[bold red]✗ 错误:[/bold red] [red]{escape(msg)}[/red]")

    def warning(self, msg: str) -> None:
        """Print a warning in amber/yellow."""
        self.console.print(
            f"[bold dark_orange]⚠ 警告:[/bold dark_orange] "
            f"[dark_orange]{escape(msg)}[/dark_orange]"
        )

    def success(self, msg: str) -> None:
        """Print a success message in green."""
        self.console.print(
            f"[bold green]✓ 成功:[/bold green] [green]{escape(msg)}[/green]"
        )

    # ------------------------------------------------------------------
    # Round summary table
    # ------------------------------------------------------------------

    def round_summary(self, round_result: RoundResult) -> None:
        """Render a compact table summarising a single training round."""
        table = Table(
            title=f"第 {round_result.round_num} 轮 结果",
            title_style="bold cyan",
            show_lines=True,
        )
        table.add_column("指标", style="bold")
        table.add_column("值", justify="right")

        train_mean = round_result.train_scores.mean_score
        train_pass = round_result.train_scores.pass_rate
        table.add_row("训练集均分", f"{train_mean:.3f}")
        table.add_row("训练集通过率", f"{train_pass:.1%}")

        if round_result.val_scores is not None:
            val_mean = round_result.val_scores.mean_score
            val_pass = round_result.val_scores.pass_rate
            table.add_row("验证集均分", f"{val_mean:.3f}")
            table.add_row("验证集通过率", f"{val_pass:.1%}")

        refl = round_result.reflection
        table.add_row("诊断", escape(refl.diagnosis[:120]))
        table.add_row("优先级", refl.priority)

        if refl.failure_patterns:
            patterns = "; ".join(refl.failure_patterns[:3])
            table.add_row("失败模式", escape(patterns))

        if round_result.mutation and round_result.applied:
            table.add_row(
                "已应用变更",
                escape(round_result.mutation.description[:100]),
            )

        self.console.print(table)

    # ------------------------------------------------------------------
    # Training progress (live)
    # ------------------------------------------------------------------

    def training_progress(self, context: LoopContext) -> None:
        """Print a compact progress snapshot including a text-based score chart."""
        header = (
            f"[bold cyan]轮次 {context.round_num} / "
            f"{context.config.max_rounds}[/bold cyan]  "
            f"最佳验证分: [green]{context.best_val_score:.3f}[/green]"
        )
        self.console.print(header)

        # Text-based sparkline of historical train scores
        if context.history:
            scores = [
                r.train_scores.mean_score for r in context.history
            ]
            self._spark_chart(scores, label="训练均分趋势")

        if context.reflection:
            self.console.print(
                f"  [dim]最新诊断:[/dim] "
                f"{escape(context.reflection.diagnosis[:100])}"
            )

    def _spark_chart(self, values: list[float], *, label: str = "") -> None:
        """Render a simple text-based bar chart of values in [0, 1]."""
        bar_width = 30
        lines: list[str] = []
        for i, v in enumerate(values):
            filled = int(v * bar_width)
            bar = "█" * filled + "░" * (bar_width - filled)
            lines.append(f"  R{i:>2d} [{bar}] {v:.3f}")
        chart_text = "\n".join(lines)
        if label:
            chart_text = f"  [bold]{escape(label)}[/bold]\n{chart_text}"
        self.console.print(chart_text)

    # ------------------------------------------------------------------
    # Config diff
    # ------------------------------------------------------------------

    def diff_display(
        self,
        original_config: dict[str, Any],
        proposed_config: dict[str, Any],
    ) -> None:
        """Show a unified diff between two configuration dicts."""
        import json as _json

        old_lines = _json.dumps(original_config, indent=2, ensure_ascii=False).splitlines(
            keepends=True
        )
        new_lines = _json.dumps(proposed_config, indent=2, ensure_ascii=False).splitlines(
            keepends=True
        )
        diff = difflib.unified_diff(
            old_lines, new_lines, fromfile="原始配置", tofile="建议配置", lineterm=""
        )
        diff_text = "\n".join(diff)
        if not diff_text:
            self.console.print("[dim]配置无变化。[/dim]")
            return

        panel = Panel(
            diff_text,
            title="配置差异",
            title_align="left",
            border_style="cyan",
            expand=False,
        )
        self.console.print(panel)

    # ------------------------------------------------------------------
    # Approval panel
    # ------------------------------------------------------------------

    def approval_panel(self, result: TrainingResult) -> None:
        """Render the final training results summary for human review."""
        table = Table(
            title="训练完成 - 最终报告",
            title_style="bold green",
            show_lines=True,
        )
        table.add_column("指标", style="bold")
        table.add_column("值", justify="right")

        table.add_row("总轮次", str(len(result.rounds)))
        table.add_row("最佳轮次", str(result.best_round))
        table.add_row("最佳验证分", f"{result.best_val_score:.3f}")
        table.add_row("停止原因", escape(result.stop_reason))
        table.add_row("总耗时", f"{result.total_duration_s:.1f} 秒")

        self.console.print(table)

        # Score trend
        if result.rounds:
            scores = [r.train_scores.mean_score for r in result.rounds]
            self._spark_chart(scores, label="训练分数趋势")

            first_score = result.rounds[0].train_scores.mean_score
            improvement = result.best_val_score - first_score
            colour = "green" if improvement > 0 else "red"
            self.console.print(
                f"\n  [bold]分数提升:[/bold] [{colour}]{improvement:+.3f}[/{colour}]"
            )
