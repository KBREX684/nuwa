"""Plugin context helpers for Nuwa extension modules."""

from __future__ import annotations

from typing import Any


class PluginContext:
    """Context object passed to plugin registration hooks."""

    def register_connector(self, name: str, dotted_path: str) -> None:
        """Register a custom connector implementation."""
        from nuwa.connectors.registry import register_connector

        register_connector(name, dotted_path)

    def register_benchmark(self, name: str, suite: Any) -> None:
        """Register a custom benchmark suite."""
        from nuwa.benchmarks.registry import register_benchmark

        register_benchmark(name, suite)
