"""Direction phase -- collects training goals and parameters."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt

from nuwa.conversation.renderer import NuwaRenderer

__all__ = ["DirectionPhase"]


class DirectionPhase:
    """Interactive phase that gathers the training direction and scope."""

    async def run(self, renderer: NuwaRenderer) -> dict[str, Any]:
        """Drive the direction-setting dialogue.

        Returns
        -------
        dict
            Keys: ``training_direction``, ``max_rounds``, ``focus_areas``.
        """
        console = renderer.console

        console.print()
        renderer.status("现在让我们确定训练方向。")
        console.print()

        # -- Training direction -------------------------------------------
        direction = await self._collect_direction(console)

        # -- Focus areas --------------------------------------------------
        focus_areas = await self._collect_focus_areas(console)

        # -- Max rounds ---------------------------------------------------
        max_rounds = await self._collect_max_rounds(console)

        # -- Summary & confirmation ----------------------------------------
        console.print()
        focus_str = "、".join(focus_areas) if focus_areas else "无特定限制"
        summary = (
            f"[bold]训练目标:[/bold] {direction}\n"
            f"[bold]重点领域:[/bold] {focus_str}\n"
            f"[bold]最大轮数:[/bold] {max_rounds}\n"
        )
        console.print(
            Panel(summary, title="训练方向摘要", border_style="cyan", expand=False)
        )

        if not Confirm.ask("[cyan]以上方向是否正确？[/cyan]", default=True, console=console):
            renderer.warning("请重新输入训练方向。")
            return await self.run(renderer)

        return {
            "training_direction": direction,
            "max_rounds": max_rounds,
            "focus_areas": focus_areas,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _collect_direction(self, console: Console) -> str:
        console.print(
            "[cyan]请用自然语言描述您希望训练达到的目标。[/cyan]\n"
            "[dim]例如: \"提升客服机器人对退货流程的回答准确度和礼貌程度\"[/dim]\n"
            "[dim]例如: \"让代码助手生成更安全、更符合最佳实践的 Python 代码\"[/dim]"
        )
        console.print()

        direction = Prompt.ask("[cyan]训练目标[/cyan]", console=console)
        direction = direction.strip()

        if len(direction) < 10:
            console.print(
                "[dark_orange]目标描述较短，更详细的描述有助于获得更好的训练效果。[/dark_orange]"
            )
            elaboration = Prompt.ask(
                "[cyan]能否补充更多细节？(留空保持原样)[/cyan]",
                default="",
                console=console,
            )
            if elaboration.strip():
                direction = f"{direction}。{elaboration.strip()}"

        return direction

    async def _collect_focus_areas(self, console: Console) -> list[str]:
        console.print()
        console.print(
            "[cyan]是否有特定的关注领域？[/cyan]\n"
            "[dim]例如: 准确性、响应速度、语气风格、安全性[/dim]\n"
            "[dim]多个领域用逗号分隔，留空表示全面优化。[/dim]"
        )
        raw = Prompt.ask("[cyan]关注领域[/cyan]", default="", console=console)
        if not raw.strip():
            return []
        # Split on Chinese and ASCII commas
        areas: list[str] = []
        for part in raw.replace("，", ",").split(","):
            part = part.strip()
            if part:
                areas.append(part)
        return areas

    async def _collect_max_rounds(self, console: Console) -> int:
        console.print()
        console.print(
            "[cyan]设置最大训练轮数。[/cyan]\n"
            "[dim]更多轮次通常能获得更好结果，但耗时更长。建议 5-20 轮。[/dim]"
        )
        max_rounds = IntPrompt.ask(
            "[cyan]最大轮数[/cyan]", default=10, console=console
        )
        if max_rounds < 1:
            console.print("[dark_orange]轮数至少为 1，已自动修正。[/dark_orange]")
            max_rounds = 1
        elif max_rounds > 100:
            console.print(
                "[dark_orange]轮数较多 (>100)，训练可能耗时很长。[/dark_orange]"
            )
            if not Confirm.ask("[cyan]确认继续？[/cyan]", default=False, console=console):
                max_rounds = IntPrompt.ask(
                    "[cyan]请重新输入轮数[/cyan]", default=10, console=console
                )
        return max_rounds
