"""Onboarding phase -- collects LLM, API key, and connector settings."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt

from nuwa.conversation.renderer import NuwaRenderer

__all__ = ["OnboardingPhase"]

_LLM_OPTIONS = {
    "1": ("openai/gpt-4o", "OpenAI GPT-4o -- 最强综合能力"),
    "2": ("openai/gpt-4o-mini", "OpenAI GPT-4o-mini -- 性价比高"),
    "3": ("anthropic/claude-3-5-sonnet", "Anthropic Claude 3.5 Sonnet -- 优秀推理能力"),
    "4": ("deepseek/deepseek-chat", "DeepSeek Chat -- 中文能力突出"),
}

_CONNECTOR_OPTIONS = {
    "1": ("http", "HTTP API -- 通过 HTTP 端点连接目标 Agent"),
    "2": ("cli", "CLI 命令 -- 通过命令行调用目标 Agent"),
    "3": ("function", "Python 函数 -- 直接调用 Python 可调用对象"),
}


class OnboardingPhase:
    """Interactive onboarding: gather LLM choice, API key, and connector info."""

    async def run(self, renderer: NuwaRenderer) -> dict[str, Any]:
        """Drive the onboarding conversation and return collected settings.

        Returns
        -------
        dict
            Keys: ``llm_model``, ``llm_api_key``, ``connector_type``,
            ``connector_params``.
        """
        console = renderer.console

        renderer.status("欢迎使用女娲 (Nuwa) AI Agent 训练系统！")
        console.print()
        console.print(
            "[cyan]让我们先完成基础设置。整个过程只需要几分钟。[/cyan]\n"
        )

        # -- LLM selection -------------------------------------------------
        llm_model = await self._select_llm(console)
        api_key = await self._collect_api_key(console, llm_model)

        # -- Connector selection -------------------------------------------
        connector_type, connector_params = await self._select_connector(console)

        # -- Summary -------------------------------------------------------
        console.print()
        summary = (
            f"[bold]模型:[/bold] {llm_model}\n"
            f"[bold]API 密钥:[/bold] {'已设置' if api_key else '未提供'}\n"
            f"[bold]连接器:[/bold] {connector_type}\n"
        )
        console.print(
            Panel(summary, title="基础设置摘要", border_style="cyan", expand=False)
        )

        if not Confirm.ask("[cyan]以上设置是否正确？[/cyan]", default=True, console=console):
            renderer.warning("请重新输入设置。")
            return await self.run(renderer)  # recurse to re-do

        return {
            "llm_model": llm_model,
            "llm_api_key": api_key,
            "connector_type": connector_type,
            "connector_params": connector_params,
        }

    # ------------------------------------------------------------------
    # LLM
    # ------------------------------------------------------------------

    async def _select_llm(self, console: Console) -> str:
        console.print("[bold cyan]第一步：选择 LLM 模型[/bold cyan]\n")
        for key, (model, desc) in _LLM_OPTIONS.items():
            console.print(f"  [bold]{key}.[/bold] {desc}")
        console.print(f"  [bold]5.[/bold] 自定义 (手动输入 provider/model)")
        console.print()

        choice = Prompt.ask(
            "[cyan]请选择[/cyan]",
            choices=["1", "2", "3", "4", "5"],
            default="1",
            console=console,
        )

        if choice == "5":
            custom = Prompt.ask(
                "[cyan]请输入模型标识 (格式: provider/model)[/cyan]",
                console=console,
            )
            if "/" not in custom:
                console.print("[dark_orange]提示：建议使用 provider/model 格式。[/dark_orange]")
            return custom.strip()

        return _LLM_OPTIONS[choice][0]

    # ------------------------------------------------------------------
    # API key
    # ------------------------------------------------------------------

    async def _collect_api_key(self, console: Console, llm_model: str) -> str | None:
        console.print()
        provider = llm_model.split("/")[0] if "/" in llm_model else llm_model
        console.print(
            f"[cyan]需要为 [bold]{provider}[/bold] 提供 API 密钥。[/cyan]"
        )
        console.print("[dim]（也可以稍后通过环境变量或配置文件设置）[/dim]")

        api_key = Prompt.ask(
            "[cyan]API 密钥 (留空跳过)[/cyan]",
            default="",
            console=console,
            password=True,
        )
        return api_key.strip() or None

    # ------------------------------------------------------------------
    # Connector
    # ------------------------------------------------------------------

    async def _select_connector(
        self, console: Console
    ) -> tuple[str, dict[str, Any]]:
        console.print()
        console.print("[bold cyan]第二步：选择目标 Agent 连接方式[/bold cyan]\n")
        for key, (ctype, desc) in _CONNECTOR_OPTIONS.items():
            console.print(f"  [bold]{key}.[/bold] {desc}")
        console.print()

        choice = Prompt.ask(
            "[cyan]请选择[/cyan]",
            choices=["1", "2", "3"],
            default="1",
            console=console,
        )
        connector_type = _CONNECTOR_OPTIONS[choice][0]
        params: dict[str, Any] = {}

        if connector_type == "http":
            url = Prompt.ask(
                "[cyan]请输入目标 Agent 的 HTTP URL[/cyan]",
                default="http://localhost:8000/chat",
                console=console,
            )
            params["url"] = url.strip()

            method = Prompt.ask(
                "[cyan]HTTP 方法[/cyan]",
                choices=["POST", "GET"],
                default="POST",
                console=console,
            )
            params["method"] = method

        elif connector_type == "cli":
            command = Prompt.ask(
                "[cyan]请输入调用目标 Agent 的命令[/cyan]",
                console=console,
            )
            params["command"] = command.strip()

        elif connector_type == "function":
            func_path = Prompt.ask(
                "[cyan]请输入 Python 可调用对象路径 (例: my_module.agent_fn)[/cyan]",
                console=console,
            )
            params["func"] = func_path.strip()

        return connector_type, params
