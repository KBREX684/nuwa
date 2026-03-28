"""Sandbox isolation layer for safe, non-destructive agent training."""

from __future__ import annotations

from nuwa.sandbox.agent import SandboxedAgent
from nuwa.sandbox.manager import SandboxManager

__all__ = ["SandboxManager", "SandboxedAgent"]
