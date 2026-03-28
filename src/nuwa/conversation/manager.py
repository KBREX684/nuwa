"""State-machine conversation manager that drives the interactive Nuwa session."""

from __future__ import annotations

import enum
from typing import Any

from pydantic import SecretStr
from rich.prompt import Confirm

from nuwa.config.schema import NuwaConfig
from nuwa.conversation.phases.approval import ApprovalPhase
from nuwa.conversation.phases.direction import DirectionPhase
from nuwa.conversation.phases.onboarding import OnboardingPhase
from nuwa.conversation.phases.running import RunningPhase
from nuwa.conversation.renderer import NuwaRenderer
from nuwa.core.types import TrainingResult

__all__ = ["ConversationPhase", "ConversationManager"]


class ConversationPhase(enum.Enum):
    """Phases of the interactive conversation flow."""

    ONBOARDING = "onboarding"
    DIRECTION = "direction"
    CONFIGURING = "configuring"
    RUNNING = "running"
    APPROVAL = "approval"
    COMPLETED = "completed"


class ConversationManager:
    """Orchestrates the full interactive training conversation.

    Transitions through :class:`ConversationPhase` stages, delegating each
    to the appropriate phase handler.  The output is a fully-populated
    :class:`~nuwa.config.schema.NuwaConfig`.
    """

    def __init__(self, renderer: NuwaRenderer | None = None) -> None:
        self.renderer = renderer or NuwaRenderer()
        self.phase = ConversationPhase.ONBOARDING
        self._collected: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run_interactive(self) -> NuwaConfig:
        """Drive the full interactive conversation and return a config.

        The conversation proceeds through these phases in order:

        1. **Onboarding** -- LLM, API key, connector
        2. **Direction**  -- training goal, rounds, focus areas
        3. **Configuring** -- build and confirm the final config
        4. (Optionally) **Running** / **Approval** happen outside this
           method, driven by the CLI.

        Returns
        -------
        NuwaConfig
            A fully validated configuration ready for a training run.
        """
        self.renderer.banner()

        # Phase 1: Onboarding
        self.phase = ConversationPhase.ONBOARDING
        onboarding_data = await OnboardingPhase().run(self.renderer)
        self._collected.update(onboarding_data)

        # Phase 2: Direction
        self.phase = ConversationPhase.DIRECTION
        direction_data = await DirectionPhase().run(self.renderer)
        self._collected.update(direction_data)

        # Phase 3: Configuring -- build and confirm
        self.phase = ConversationPhase.CONFIGURING
        config = self._build_config()

        self.renderer.console.print()
        self.renderer.status("配置已生成！以下是完整训练配置：")
        self.renderer.console.print()
        self._display_config_summary(config)

        if not Confirm.ask(
            "[cyan]确认开始训练？[/cyan]",
            default=True,
            console=self.renderer.console,
        ):
            self.renderer.warning("已取消。您可以重新运行以修改配置。")
            raise KeyboardInterrupt("用户取消")

        self.phase = ConversationPhase.RUNNING
        return config

    # ------------------------------------------------------------------
    # Running & Approval (called externally by CLI)
    # ------------------------------------------------------------------

    def create_running_phase(self) -> RunningPhase:
        """Create a :class:`RunningPhase` bound to this manager's renderer."""
        self.phase = ConversationPhase.RUNNING
        return RunningPhase(self.renderer)

    async def run_approval(self, result: TrainingResult) -> str:
        """Run the approval phase and return the decision string.

        Returns one of ``"accept"``, ``"reject"``, ``"extend"``.
        """
        self.phase = ConversationPhase.APPROVAL
        decision = await ApprovalPhase().run(self.renderer, result)
        if decision != "extend":
            self.phase = ConversationPhase.COMPLETED
        return decision

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_config(self) -> NuwaConfig:
        """Assemble a :class:`NuwaConfig` from collected data."""
        api_key_raw = self._collected.get("llm_api_key")
        api_key = SecretStr(api_key_raw) if api_key_raw else None

        # Embed focus areas into the training direction
        direction = self._collected.get("training_direction", "")
        focus_areas: list[str] = self._collected.get("focus_areas", [])
        if focus_areas:
            direction = f"{direction} [重点: {', '.join(focus_areas)}]"

        return NuwaConfig(
            llm_model=self._collected.get("llm_model", "openai/gpt-4o"),
            llm_api_key=api_key,
            connector_type=self._collected.get("connector_type", "http"),
            connector_params=self._collected.get("connector_params", {}),
            training_direction=direction,
            max_rounds=self._collected.get("max_rounds", 10),
        )

    def _display_config_summary(self, config: NuwaConfig) -> None:
        """Print a human-readable summary of the final config."""
        from rich.table import Table

        table = Table(title="训练配置", title_style="bold cyan", show_lines=True)
        table.add_column("参数", style="bold")
        table.add_column("值")

        table.add_row("LLM 模型", config.llm_model)
        table.add_row(
            "API 密钥",
            "已设置" if config.llm_api_key else "[dim]未设置[/dim]",
        )
        table.add_row("连接器类型", config.connector_type)
        table.add_row("连接器参数", str(config.connector_params) or "-")
        table.add_row("训练目标", config.training_direction)
        table.add_row("最大轮数", str(config.max_rounds))
        table.add_row("每轮样本数", str(config.samples_per_round))
        table.add_row("训练/验证拆分", f"{config.train_val_split:.0%}")
        table.add_row("过拟合阈值", str(config.overfitting_threshold))

        self.renderer.console.print(table)
