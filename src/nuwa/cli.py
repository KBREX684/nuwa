"""Typer-based CLI for the Nuwa AI Agent Trainer.

Entry points
------------
- ``nuwa train``   -- interactive training session
- ``nuwa run``     -- headless training from a config file
- ``nuwa benchmark`` -- benchmark suite listing/runs
- ``nuwa web``     -- launch the web dashboard server
- ``nuwa status``  -- show last training run results
- ``nuwa init``    -- generate a default config file
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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
benchmark_app = typer.Typer(help="运行 Nuwa 基准测试套件。")
app.add_typer(benchmark_app, name="benchmark")


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
        results_file.write_text(result.model_dump_json(indent=2), encoding="utf-8")

        # Save final config snapshot
        if result.final_config:
            artifact_store.save_config_snapshot(result.best_round, result.final_config)

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
    resume: bool | None = typer.Option(
        None,
        "--resume/--no-resume",
        help="是否启用断点续训（默认跟随配置文件 resume 字段）。",
    ),
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
    renderer.status(f"分布式 worker: {config.distributed_workers}")

    # Build runtime objects from config
    from nuwa.core.types import TrainingResult
    from nuwa.engine.distributed.coordinator import DistributedTrainingCoordinator
    from nuwa.engine.loop import TrainingLoop
    from nuwa.guardrails.consistency import ConsistencyGuardrail
    from nuwa.guardrails.overfitting import OverfittingGuardrail
    from nuwa.guardrails.regression import RegressionGuardrail
    from nuwa.llm.backend import LiteLLMBackend
    from nuwa.persistence.artifact_store import ArtifactStore
    from nuwa.persistence.run_log import RunLog
    from nuwa.sandbox.manager import SandboxManager

    llm_kwargs = config.build_llm_kwargs()
    connector = config.build_connector()
    training_config = config.build_training_config()

    guardrails = [
        OverfittingGuardrail(threshold=config.overfitting_threshold),
        RegressionGuardrail(tolerance=config.regression_tolerance),
        ConsistencyGuardrail(threshold=config.consistency_threshold),
    ]

    project_dir = config.project_dir
    project_dir.mkdir(parents=True, exist_ok=True)
    run_log = RunLog(log_dir=project_dir)
    artifact_store = ArtifactStore(store_dir=project_dir / "artifacts")
    results_file = project_dir / "last_result.json"

    resume_enabled = config.resume if resume is None else resume
    start_round = 1
    initial_history: list[Any] = []
    initial_best_config: dict[str, Any] | None = None
    initial_best_val_score = 0.0

    if resume_enabled:
        initial_history = run_log.load_history()
        if initial_history:
            start_round = max(rr.round_num for rr in initial_history) + 1
            renderer.status(
                f"Resume 模式开启：已有 {len(initial_history)} 条历史记录，"
                f"从第 {start_round} 轮继续。"
            )

            if results_file.exists():
                try:
                    previous = TrainingResult.model_validate_json(
                        results_file.read_text(encoding="utf-8")
                    )
                    initial_best_config = dict(previous.final_config)
                    initial_best_val_score = previous.best_val_score
                except Exception as exc:
                    renderer.warning(f"读取 last_result.json 失败，回退到日志推断: {exc}")

            if initial_best_config is None:
                best_round = run_log.get_best_round()
                if best_round is not None:
                    if best_round.val_scores is not None:
                        initial_best_val_score = best_round.val_scores.mean_score
                    else:
                        initial_best_val_score = best_round.train_scores.mean_score

                snapshots = artifact_store.list_snapshots()
                if snapshots:
                    initial_best_config = artifact_store.load_config_snapshot(snapshots[-1])
        else:
            renderer.warning("Resume 已启用，但未检测到历史记录，将从第 1 轮开始。")

    renderer.status("训练循环启动中...")

    try:
        if config.distributed_workers > 1:
            workers = config.distributed_workers
            coordinator = DistributedTrainingCoordinator()

            async def _run_distributed() -> TrainingResult:
                worker_runs = []
                for worker_idx in range(workers):
                    worker_backend = LiteLLMBackend(**llm_kwargs)
                    worker_sandbox = SandboxManager(
                        connector,
                        project_dir=project_dir / "sandbox_workers" / f"worker_{worker_idx + 1}",
                    )
                    worker_loop = TrainingLoop(
                        config=training_config,
                        backend=worker_backend,
                        target=connector,
                        guardrails=guardrails,
                        sandbox=worker_sandbox,
                        start_round=start_round,
                        initial_history=initial_history,
                        initial_best_config=initial_best_config,
                        initial_best_val_score=initial_best_val_score,
                    )
                    worker_runs.append(worker_loop.run)

                winner, all_results, winner_idx = await coordinator.run_workers(worker_runs)
                renderer.success(
                    f"分布式训练完成：worker {winner_idx + 1}/{workers} 胜出，"
                    f"最佳验证分={winner.best_val_score:.3f}"
                )
                renderer.status(
                    "各 worker 验证分: "
                    + ", ".join(
                        f"{idx + 1}:{r.best_val_score:.3f}" for idx, r in enumerate(all_results)
                    )
                )
                return winner

            result = asyncio.run(_run_distributed())
        else:
            backend = LiteLLMBackend(**llm_kwargs)
            loop = TrainingLoop(
                config=training_config,
                backend=backend,
                target=connector,
                guardrails=guardrails,
                start_round=start_round,
                initial_history=initial_history,
                initial_best_config=initial_best_config,
                initial_best_val_score=initial_best_val_score,
            )
            result = asyncio.run(loop.run())
    except Exception as exc:
        renderer.error(f"训练循环执行失败: {exc}")
        raise typer.Exit(code=1)

    # Persist only newly executed rounds when running in resume mode.
    rounds_to_persist = [rr for rr in result.rounds if rr.round_num >= start_round]
    for rr in rounds_to_persist:
        run_log.append_round(rr)

    # Save final result as JSON for `nuwa status`
    results_file.write_text(result.model_dump_json(indent=2), encoding="utf-8")

    # Save final config snapshot
    if result.final_config:
        artifact_store.save_config_snapshot(result.best_round, result.final_config)

    # Display final results
    renderer.approval_panel(result)


# ------------------------------------------------------------------
# benchmark
# ------------------------------------------------------------------


@benchmark_app.command("list")
def benchmark_list_cmd(
    config_path: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="可选：加载配置并注册插件后再列出 benchmark。",
    ),
) -> None:
    """列出可用 benchmark 套件。"""
    from nuwa.benchmarks.registry import get_benchmark, list_benchmarks
    from nuwa.config.schema import NuwaConfig
    from nuwa.conversation.renderer import NuwaRenderer
    from nuwa.plugins.loader import load_plugins

    renderer = NuwaRenderer()

    if config_path is not None:
        if not config_path.exists():
            renderer.error(f"配置文件不存在: {config_path}")
            raise typer.Exit(code=1)
        try:
            cfg = NuwaConfig.from_yaml(config_path)
            if cfg.plugin_modules:
                load_plugins(cfg.plugin_modules)
        except Exception as exc:
            renderer.error(f"配置文件加载失败: {exc}")
            raise typer.Exit(code=1)

    suites = list_benchmarks()
    if not suites:
        renderer.warning("当前未注册任何 benchmark 套件。")
        return

    renderer.success(f"可用 benchmark 套件: {len(suites)}")
    for suite_name in suites:
        suite = get_benchmark(suite_name)
        renderer.console.print(
            f"- [cyan]{suite.name}[/cyan] ({len(suite.cases)} cases): {suite.description}"
        )


@benchmark_app.command("run")
def benchmark_run_cmd(
    config_path: Path = typer.Option(
        ...,
        "--config",
        "-c",
        help="YAML 配置文件路径（用于构建目标连接器）。",
        exists=True,
        readable=True,
    ),
    suite: str = typer.Option(
        ...,
        "--suite",
        "-s",
        help="benchmark 套件名称，可通过 `nuwa benchmark list` 查看。",
    ),
    max_concurrency: int = typer.Option(5, "--max-concurrency", help="最大并发执行数。"),
) -> None:
    """运行指定 benchmark 套件并输出评分结果。"""
    from nuwa.benchmarks.runner import run_benchmark
    from nuwa.config.schema import NuwaConfig
    from nuwa.conversation.renderer import NuwaRenderer

    renderer = NuwaRenderer()
    renderer.banner()

    try:
        cfg = NuwaConfig.from_yaml(config_path)
    except Exception as exc:
        renderer.error(f"配置文件加载失败: {exc}")
        raise typer.Exit(code=1)

    connector = cfg.build_connector()
    runtime_config = connector.get_current_config()

    try:
        result = asyncio.run(
            run_benchmark(
                connector,
                suite_name=suite,
                config=runtime_config,
                max_concurrency=max_concurrency,
            )
        )
    except KeyError as exc:
        renderer.error(str(exc))
        raise typer.Exit(code=1)
    except Exception as exc:
        renderer.error(f"Benchmark 执行失败: {exc}")
        raise typer.Exit(code=1)

    renderer.success(
        f"Benchmark 完成: {result.suite_name} | "
        f"mean_score={result.mean_score:.3f} | pass_rate={result.pass_rate:.1%}"
    )
    for case in result.cases:
        renderer.console.print(
            f"- {case.case_id}: score=[green]{case.score:.3f}[/green], "
            f"latency={case.latency_ms:.1f}ms"
        )

    output_dir = cfg.project_dir / "benchmarks"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = output_dir / f"{result.suite_name}_{timestamp}.json"
    out_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    renderer.status(f"结果已保存: {out_path}")


# ------------------------------------------------------------------
# web
# ------------------------------------------------------------------


@app.command("web")
def web_cmd(
    host: str = typer.Option("0.0.0.0", "--host", help="Web 服务监听地址。"),  # nosec B104
    port: int = typer.Option(9090, "--port", "-p", help="Web 服务端口。"),
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
    force: bool = typer.Option(False, "--force", "-f", help="覆盖已存在的文件。"),
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
