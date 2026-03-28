"""Approval phase -- present results and collect human decision."""

from __future__ import annotations

from typing import Literal

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from nuwa.conversation.renderer import NuwaRenderer
from nuwa.core.types import TrainingResult

__all__ = ["ApprovalPhase"]


class ApprovalPhase:
    """Present final training results and ask the human for a decision."""

    async def run(
        self,
        renderer: NuwaRenderer,
        result: TrainingResult,
    ) -> Literal["accept", "reject", "extend"]:
        """Show the results and return the human's decision.

        Returns
        -------
        ``"accept"``
            Apply the optimised configuration.
        ``"reject"``
            Discard all changes and keep the original configuration.
        ``"extend"``
            Continue training for more rounds.
        """
        console = renderer.console

        console.print()
        console.print(f"[bold cyan]{'═' * 50}[/bold cyan]")
        renderer.approval_panel(result)
        console.print()

        # -- Before / after config diff ------------------------------------
        if result.rounds:
            original_cfg = result.rounds[0].mutation.original_config if (
                result.rounds[0].mutation
            ) else {}
            renderer.diff_display(original_cfg, result.final_config)

        # -- Score improvement ---------------------------------------------
        self._show_score_improvement(console, result)

        # -- Best sample outputs -------------------------------------------
        self._show_best_samples(console, result)

        # -- Collect decision ----------------------------------------------
        console.print()
        console.print("[bold cyan]请选择操作：[/bold cyan]")
        console.print("  [bold]1.[/bold] [green]接受[/green] -- 应用优化后的配置")
        console.print("  [bold]2.[/bold] [red]拒绝[/red] -- 保留原始配置")
        console.print("  [bold]3.[/bold] [dark_orange]继续训练[/dark_orange] -- 追加更多轮次")
        console.print()

        choice = Prompt.ask(
            "[cyan]请选择[/cyan]",
            choices=["1", "2", "3"],
            default="1",
            console=console,
        )

        decision_map: dict[str, Literal["accept", "reject", "extend"]] = {
            "1": "accept",
            "2": "reject",
            "3": "extend",
        }
        decision = decision_map[choice]

        if decision == "accept":
            renderer.success("已接受优化结果，新配置将被保存。")
        elif decision == "reject":
            renderer.warning("已拒绝优化结果，将保留原始配置。")
        else:
            renderer.status("将继续追加训练轮次。")

        return decision

    # ------------------------------------------------------------------
    # Internal display helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _show_score_improvement(console: Console, result: TrainingResult) -> None:
        """Show a compact before/after score comparison."""
        if not result.rounds:
            return

        first = result.rounds[0].train_scores.mean_score
        best = result.best_val_score
        delta = best - first
        colour = "green" if delta > 0 else ("red" if delta < 0 else "dim")

        table = Table(title="分数对比", title_style="bold cyan", show_lines=True)
        table.add_column("阶段", style="bold")
        table.add_column("均分", justify="right")
        table.add_row("初始 (第 0 轮)", f"{first:.3f}")
        table.add_row(f"最佳 (第 {result.best_round} 轮)", f"{best:.3f}")
        table.add_row("变化", f"[{colour}]{delta:+.3f}[/{colour}]")
        console.print(table)

    @staticmethod
    def _show_best_samples(console: Console, result: TrainingResult) -> None:
        """Display up to 3 of the highest-scored sample outputs."""
        if not result.rounds:
            return

        # Collect all scored results from the best round
        best_round = result.rounds[result.best_round] if (
            result.best_round < len(result.rounds)
        ) else result.rounds[-1]

        scored = sorted(
            best_round.train_scores.results,
            key=lambda r: r.score,
            reverse=True,
        )[:3]

        if not scored:
            return

        console.print()
        console.print("[bold cyan]最佳表现样本（前 3 个）：[/bold cyan]")
        for i, sr in enumerate(scored, 1):
            panel_content = (
                f"[bold]输入:[/bold] {escape(sr.sample.input_text[:200])}\n"
                f"[bold]输出:[/bold] {escape(sr.response.output_text[:200])}\n"
                f"[bold]得分:[/bold] [green]{sr.score:.3f}[/green]\n"
                f"[bold]评语:[/bold] {escape(sr.reasoning[:150])}"
            )
            console.print(
                Panel(
                    panel_content,
                    title=f"样本 {i}",
                    border_style="green",
                    expand=False,
                )
            )
