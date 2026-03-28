"""CLI subprocess adapter for invoking target agents as local commands."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Literal

import yaml

from nuwa.core.exceptions import ConnectorError
from nuwa.core.types import AgentResponse

logger = logging.getLogger(__name__)


class _LatencyTimer:
    """Tiny context-manager for measuring wall-clock milliseconds."""

    __slots__ = ("_start", "elapsed_ms")

    def __enter__(self) -> _LatencyTimer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: object) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0


class CliAdapter:
    """Runs a local CLI programme as if it were a target agent.

    The agent binary is invoked once per :meth:`invoke` call.  Input can be
    delivered either via *stdin* or as an extra positional argument, controlled
    by *input_mode*.

    Implements the ``TargetAgent`` protocol defined in
    :mod:`nuwa.connectors.http_api`.

    Parameters
    ----------
    command:
        Path or name of the executable (resolved via ``$PATH``).
    args:
        Static arguments prepended to every invocation.
    input_mode:
        ``"stdin"`` -- pipe the input text to the process's standard input.
        ``"arg"``   -- append the input text as the last positional argument.
    timeout:
        Maximum wall-clock seconds before the subprocess is killed.
    config_file:
        Optional path to a YAML or JSON file that stores the agent's
        configuration.  Used by :meth:`get_current_config` and
        :meth:`apply_config`.
    """

    def __init__(
        self,
        command: str,
        args: list[str] | None = None,
        input_mode: Literal["stdin", "arg"] = "stdin",
        timeout: int = 60,
        config_file: str | None = None,
    ) -> None:
        self._command = command
        self._args = list(args) if args else []
        self._input_mode = input_mode
        self._timeout = timeout
        self._config_file = Path(config_file) if config_file else None
        self._cached_config: dict[str, Any] = {}

        # Eagerly load config file if it exists
        if self._config_file and self._config_file.is_file():
            self._cached_config = self._read_config_file(self._config_file)

    # ------------------------------------------------------------------
    # TargetAgent protocol
    # ------------------------------------------------------------------

    async def invoke(
        self,
        input_text: str,
        config: dict[str, Any] | None = None,
    ) -> AgentResponse:
        """Execute the CLI command and return the agent's stdout as output.

        If *config* is supplied it is written to the config file (if one is
        configured) before the subprocess starts so the agent picks up the
        changes.
        """
        if config:
            self.apply_config(config)

        cmd_parts = [self._command, *self._args]
        stdin_data: bytes | None = None

        if self._input_mode == "arg":
            cmd_parts.append(input_text)
        else:
            stdin_data = input_text.encode("utf-8")

        try:
            with _LatencyTimer() as timer:
                proc = await asyncio.create_subprocess_exec(
                    *cmd_parts,
                    stdin=asyncio.subprocess.PIPE if stdin_data else None,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(input=stdin_data),
                    timeout=self._timeout,
                )

            stdout_text = stdout_bytes.decode("utf-8", errors="replace").strip()
            stderr_text = stderr_bytes.decode("utf-8", errors="replace").strip()

            if proc.returncode != 0:
                error_msg = (
                    f"Process exited with code {proc.returncode}"
                    + (f": {stderr_text}" if stderr_text else "")
                )
                logger.warning("CLI adapter: %s", error_msg)
                return AgentResponse(
                    output_text=stdout_text or error_msg,
                    latency_ms=timer.elapsed_ms,
                    raw_metadata={
                        "returncode": proc.returncode,
                        "stderr": stderr_text,
                        "error": error_msg,
                    },
                )

            return AgentResponse(
                output_text=stdout_text,
                latency_ms=timer.elapsed_ms,
                raw_metadata={"returncode": 0, "stderr": stderr_text},
            )

        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise ConnectorError(
                f"CLI command timed out after {self._timeout}s: {' '.join(cmd_parts)}"
            )
        except FileNotFoundError as exc:
            raise ConnectorError(f"Command not found: {self._command}") from exc
        except Exception as exc:
            logger.error("CLI adapter error: %s", exc)
            raise ConnectorError(f"Subprocess error: {exc}") from exc

    def get_current_config(self) -> dict[str, Any]:
        """Return the current agent configuration.

        If a config file is set, it is re-read from disk so we always reflect
        the latest state (the agent may modify its own config).
        """
        if self._config_file and self._config_file.is_file():
            self._cached_config = self._read_config_file(self._config_file)
        return dict(self._cached_config)

    def apply_config(self, config: dict[str, Any]) -> None:
        """Write *config* to the config file (and update the local cache)."""
        self._cached_config = dict(config)
        if self._config_file:
            self._write_config_file(self._config_file, config)
            logger.debug("Wrote config to %s", self._config_file)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_config_file(path: Path) -> dict[str, Any]:
        """Read a JSON or YAML config file and return a dict."""
        text = path.read_text(encoding="utf-8")
        if path.suffix in (".yaml", ".yml"):
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        if not isinstance(data, dict):
            raise ConnectorError(f"Config file {path} does not contain a mapping")
        return data

    @staticmethod
    def _write_config_file(path: Path, config: dict[str, Any]) -> None:
        """Write *config* to a JSON or YAML file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix in (".yaml", ".yml"):
            path.write_text(
                yaml.dump(config, default_flow_style=False, sort_keys=False),
                encoding="utf-8",
            )
        else:
            path.write_text(
                json.dumps(config, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

    def __repr__(self) -> str:
        return (
            f"CliAdapter(command={self._command!r}, "
            f"input_mode={self._input_mode!r}, timeout={self._timeout}s)"
        )
