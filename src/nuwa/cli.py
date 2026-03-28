"""Typer-based CLI for the Nuwa AI Agent Trainer.

Entry points
------------
- ``nuwa train``   -- interactive training session
- ``nuwa run``     -- headless training from a config file
- ``nuwa web``     -- launch the web dashboard server
- ``nuwa status``  -- show last training run results
- ``nuwa init``    -- generate a default config file
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer

__all__ = ["app"]

import nuwa

_VERSION = nuwa.__version__

app = typer.Typer(
    name="nuwa",
    help="女娲 Nuwa -- AI Agent Trainer",
    add_completion=False,
    no_args_is_help=True,
)


# ------------------------------------------------------------------
# Version callback
# ------------------------------------------------------------------

def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"女娲 Nuwa v{_VERSION}")
        raise typer.Exit()


@app.callback()
def _main(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="显示版本号并退出。",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """女娲 Nuwa -- AI Agent Trainer CLI."""


# ------------------------------------------------------------------
# train (interactive)
# ------------------------------------------------------------------

@app.command("train")
def train_cmd(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="启用详细日志输出。"),
) -> None:
    """启动交互式训练会话。

    通过对话引导您完成模型选择、目标设定和训练执行。
    """
    from nuwa.conversation.manager import ConversationManager
    from nuwa.conversation.renderer import NuwaRenderer

    renderer = NuwaRenderer()

    async def _run() -> None:
        manager = ConversationManager(renderer=renderer)
        try:
            config = await manager.run_interactive()
        except KeyboardInterrupt:
            renderer.warning("训练会话已取消。")
            raise typer.Exit(code=1)

        if verbose:
            config_obj = config.model_dump(exclude={"llm_api_key"})
            renderer.console.print_json(json.dumps(config_obj, default=str))

        # Save config for the run
        project_dir = config.project_dir
        project_dir.mkdir(parents=True, exist_ok=True)
        config.to_yaml(project_dir / "config.yaml")
        renderer.success(f"配置已保存到 {project_dir / 'config.yaml'}")

        # Run the training pipeline
        from nuwa.conversation.phases.approval import ApprovalPhase
        from nuwa.conversation.phases.running import RunningPhase
        from nuwa.engine.loop import TrainingLoop
        from nuwa.guardrails.consistency import ConsistencyGuardrail
        from nuwa.guardrails.overfitting import OverfittingGuardrail
        from nuwa.guardrails.regression import RegressionGuardrail
        from nuwa.llm.backend import LiteLLMBackend
        from nuwa.persistence.artifact_store import ArtifactStore
        from nuwa.persistence.run_log import RunLog

        llm_kwargs = config.build_llm_kwargs()
        backend = LiteLLMBackend(**llm_kwargs)
        connector = config.build_connector()
        training_config = config.build_training_config()

        guardrails = [
            OverfittingGuardrail(threshold=config.overfitting_threshold),
            RegressionGuardrail(tolerance=config.regression_tolerance),
            ConsistencyGuardrail(threshold=config.consistency_threshold),
        ]

        running_phase = RunningPhase(renderer)
        callback = running_phase.build_callback()

        loop = TrainingLoop(
            config=training_config,
            backend=backend,
            target=connector,
            guardrails=guardrails,
            callbacks=[callback],
        )

        run_log = RunLog(log_dir=project_dir)
        artifact_store = ArtifactStore(store_dir=project_dir / "artifacts")

        renderer.banner()
        renderer.status("训练循环启动中...")

        result = await loop.run()

        # Persist round results
        for rr in result.rounds:
            run_log.append_round(rr)

        # Save final result as JSON for `nuwa status`
        results_file = project_dir / "last_result.json"
        results_file.write_text(
            result.model_dump_json(indent=2), encoding="utf-8"
        )

        # Save final config snapshot
        if result.final_config:
            artifact_store.save_config_snapshot(
                result.best_round, result.final_config
            )

        # Show approval panel
        approval = ApprovalPhase()
        await approval.run(renderer, result)

    asyncio.run(_run())


# ------------------------------------------------------------------
# run (headless)
# ------------------------------------------------------------------

@app.command("run")
def run_cmd(
    config_path: Path = typer.Option(
        ...,
        "--config",
        "-c",
        help="YAML 配置文件路径。",
        exists=True,
        readable=True,
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="启用详细日志输出。"),
) -> None:
    """从配置文件启动无头训练。

    适合 CI/CD 或脚本化场景。
    """
    from nuwa.config.schema import NuwaConfig
    from nuwa.conversation.renderer import NuwaRenderer

    renderer = NuwaRenderer()
    renderer.banner()

    try:
        config = NuwaConfig.from_yaml(config_path)
    except Exception as exc:
        renderer.error(f"配置文件加载失败: {exc}")
        raise typer.Exit(code=1)

    renderer.success(f"已加载配置: {config_path}")
    if verbose:
        config_obj = config.model_dump(exclude={"llm_api_key"})
        renderer.console.print_json(json.dumps(config_obj, default=str))

    renderer.status(f"训练目标: {config.training_direction}")
    renderer.status(f"最大轮数: {config.max_rounds}")

    # Build runtime objects from config
    from nuwa.engine.loop import TrainingLoop
    from nuwa.guardrails.consistency import ConsistencyGuardrail
    from nuwa.guardrails.overfitting import OverfittingGuardrail
    from nuwa.guardrails.regression import RegressionGuardrail
    from nuwa.llm.backend import LiteLLMBackend
    from nuwa.persistence.artifact_store import ArtifactStore
    from nuwa.persistence.run_log import RunLog

    llm_kwargs = config.build_llm_kwargs()
    backend = LiteLLMBackend(**llm_kwargs)
    connector = config.build_connector()
    training_config = config.build_training_config()

    guardrails = [
        OverfittingGuardrail(threshold=config.overfitting_threshold),
        RegressionGuardrail(tolerance=config.regression_tolerance),
        ConsistencyGuardrail(threshold=config.consistency_threshold),
    ]

    loop = TrainingLoop(
        config=training_config,
        backend=backend,
        target=connector,
        guardrails=guardrails,
    )

    project_dir = config.project_dir
    project_dir.mkdir(parents=True, exist_ok=True)
    run_log = RunLog(log_dir=project_dir)
    artifact_store = ArtifactStore(store_dir=project_dir / "artifacts")

    renderer.status("训练循环启动中...")

    try:
        result = asyncio.run(loop.run())
    except Exception as exc:
        renderer.error(f"训练循环执行失败: {exc}")
        raise typer.Exit(code=1)

    # Persist round results to run log
    for rr in result.rounds:
        run_log.append_round(rr)

    # Save final result as JSON for `nuwa status`
    results_file = project_dir / "last_result.json"
    results_file.write_text(result.model_dump_json(indent=2), encoding="utf-8")

    # Save final config snapshot
    if result.final_config:
        artifact_store.save_config_snapshot(result.best_round, result.final_config)

    # Display final results
    renderer.approval_panel(result)


# ------------------------------------------------------------------
# web
# ------------------------------------------------------------------

@app.command("web")
def web_cmd(
    host: str = typer.Option("0.0.0.0", "--host", help="Web 服务监听地址。"),
    port: int = typer.Option(8080, "--port", "-p", help="Web 服务端口。"),
    reload: bool = typer.Option(False, "--reload", help="启用自动重载（开发模式）。"),
) -> None:
    """启动 Nuwa Web 控制台。"""
    try:
        import uvicorn
    except ImportError:
        typer.echo(
            "缺少 Web 依赖，请先安装: pip install 'nuwa-trainer[web]'",
            err=True,
        )
        raise typer.Exit(code=1)

    uvicorn.run("nuwa.web.server:app", host=host, port=port, reload=reload)


# ------------------------------------------------------------------
# status
# ------------------------------------------------------------------

@app.command("status")
def status_cmd(
    project_dir: Path = typer.Option(
        Path(".nuwa"),
        "--dir",
        "-d",
        help="Nuwa 项目目录路径。",
    ),
) -> None:
    """显示上一次训练运行的结果。"""
    from nuwa.conversation.renderer import NuwaRenderer
    from nuwa.persistence.run_log import RunLog

    renderer = NuwaRenderer()

    results_file = project_dir / "last_result.json"
    config_file = project_dir / "config.yaml"

    if not project_dir.exists():
        renderer.error(f"项目目录不存在: {project_dir}")
        renderer.status("运行 'nuwa init' 创建默认配置，或运行 'nuwa train' 开始训练。")
        raise typer.Exit(code=1)

    # Show config if available
    if config_file.exists():
        renderer.status(f"配置文件: {config_file}")
        from nuwa.config.schema import NuwaConfig

        try:
            config = NuwaConfig.from_yaml(config_file)
            renderer.console.print(f"  模型: [cyan]{config.llm_model}[/cyan]")
            renderer.console.print(f"  目标: [cyan]{config.training_direction}[/cyan]")
            renderer.console.print(f"  轮数: [cyan]{config.max_rounds}[/cyan]")
        except Exception as exc:
            renderer.warning(f"配置文件读取异常: {exc}")
    else:
        renderer.warning("未找到配置文件。")

    # Show run log summary
    run_log = RunLog(log_dir=project_dir)
    history = run_log.load_history()
    if history:
        renderer.console.print()
        renderer.status(f"运行日志: 共 {len(history)} 轮记录")
        best = run_log.get_best_round()
        if best and best.val_scores is not None:
            renderer.console.print(
                f"  最佳轮次: [cyan]{best.round_num}[/cyan]  "
                f"验证分: [green]{best.val_scores.mean_score:.3f}[/green]"
            )
        latest = run_log.get_latest_run()
        if latest:
            renderer.console.print(
                f"  最近轮次: [cyan]{latest.round_num}[/cyan]  "
                f"训练分: [green]{latest.train_scores.mean_score:.3f}[/green]"
            )

    # Show last result if available
    if results_file.exists():
        renderer.console.print()
        renderer.status("上次训练结果:")
        try:
            data = json.loads(results_file.read_text(encoding="utf-8"))
            from nuwa.core.types import TrainingResult

            result = TrainingResult.model_validate(data)
            renderer.approval_panel(result)
        except Exception as exc:
            renderer.warning(f"结果文件读取异常: {exc}")
    elif not history:
        renderer.status("暂无训练结果记录。运行 'nuwa train' 开始首次训练。")


# ------------------------------------------------------------------
# init
# ------------------------------------------------------------------

@app.command("init")
def init_cmd(
    output: Path = typer.Option(
        Path("nuwa-config.yaml"),
        "--output",
        "-o",
        help="输出配置文件路径。",
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="覆盖已存在的文件。"
    ),
) -> None:
    """生成默认配置文件。"""
    from nuwa.config.schema import NuwaConfig
    from nuwa.conversation.renderer import NuwaRenderer

    renderer = NuwaRenderer()

    if output.exists() and not force:
        renderer.error(f"文件已存在: {output}")
        renderer.status("使用 --force 覆盖，或指定其他路径。")
        raise typer.Exit(code=1)

    config = NuwaConfig()
    config.to_yaml(output)
    renderer.success(f"默认配置已生成: {output}")
    renderer.status(f"编辑配置文件后，运行 'nuwa run --config {output}' 开始训练。")
