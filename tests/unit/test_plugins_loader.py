"""Unit tests for plugin loading and registration hooks."""

from __future__ import annotations

from uuid import uuid4

from nuwa.benchmarks.registry import get_benchmark
from nuwa.connectors.registry import CONNECTOR_MAP
from nuwa.plugins.loader import load_plugin, loaded_plugins, reset_loaded_plugins


def test_load_plugin_registers_connector_and_benchmark(tmp_path, monkeypatch) -> None:
    suffix = uuid4().hex[:8]
    module_name = f"test_plugin_{suffix}"
    connector_name = f"connector_{suffix}"
    benchmark_name = f"benchmark_{suffix}"

    module_path = tmp_path / f"{module_name}.py"
    module_path.write_text(
        "\n".join(
            [
                "from nuwa.benchmarks.models import BenchmarkCase, BenchmarkSuite",
                "",
                "def register(context):",
                f"    context.register_connector('{connector_name}', "
                "'nuwa.connectors.function_call:FunctionCallAdapter')",
                f"    context.register_benchmark('{benchmark_name}', BenchmarkSuite(",
                f"        name='{benchmark_name}',",
                "        description='plugin benchmark',",
                "        cases=[BenchmarkCase(id='c1', input_text='hello')],",
                "    ))",
                "",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    reset_loaded_plugins()
    try:
        module = load_plugin(module_name)
        assert module.__name__ == module_name
        assert module_name in loaded_plugins()
        assert connector_name in CONNECTOR_MAP
        suite = get_benchmark(benchmark_name)
        assert suite.name == benchmark_name
        assert len(suite.cases) == 1
    finally:
        CONNECTOR_MAP.pop(connector_name, None)
        from nuwa.benchmarks import registry as bench_registry

        bench_registry._REGISTRY.pop(benchmark_name, None)  # type: ignore[attr-defined]
        reset_loaded_plugins()


def test_load_plugin_supports_zero_arg_register(tmp_path, monkeypatch) -> None:
    suffix = uuid4().hex[:8]
    module_name = f"test_plugin_zero_{suffix}"

    module_path = tmp_path / f"{module_name}.py"
    module_path.write_text(
        "\n".join(
            [
                "CALLED = False",
                "",
                "def register():",
                "    global CALLED",
                "    CALLED = True",
                "",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    reset_loaded_plugins()
    module = load_plugin(module_name)
    assert bool(getattr(module, "CALLED", False)) is True
    reset_loaded_plugins()
