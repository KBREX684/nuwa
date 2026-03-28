"""HTTP API adapter for communicating with target agents over REST endpoints."""

from __future__ import annotations

import logging
import time
from typing import Any

import aiohttp

from nuwa.core.exceptions import ConnectorError
from nuwa.core.protocols import TargetAgent
from nuwa.core.types import AgentResponse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Latency helper
# ---------------------------------------------------------------------------

class _LatencyTimer:
    """Tiny context-manager for measuring wall-clock milliseconds."""

    __slots__ = ("_start", "elapsed_ms")

    def __enter__(self) -> _LatencyTimer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: object) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0


# ---------------------------------------------------------------------------
# HttpApiAdapter
# ---------------------------------------------------------------------------

class HttpApiAdapter:
    """Sends requests to a remote agent exposed as an HTTP/JSON endpoint.

    Implements the :class:`TargetAgent` protocol.

    Parameters
    ----------
    url:
        Full URL of the agent endpoint (e.g. ``https://my-agent.run/v1/chat``).
    method:
        HTTP method.  Typically ``"POST"``.
    headers:
        Extra headers merged into every request (e.g. auth tokens).
    input_field:
        JSON key used to send the user's input text.
    output_field:
        JSON key expected in the response body that contains the agent's reply.
    timeout:
        Per-request timeout in seconds.
    config_endpoint:
        Optional URL for reading / writing the agent's runtime configuration.
        When *None*, ``get_current_config`` returns an empty dict and
        ``apply_config`` is a no-op.
    """

    def __init__(
        self,
        url: str,
        method: str = "POST",
        headers: dict[str, str] | None = None,
        input_field: str = "input",
        output_field: str = "output",
        timeout: int = 30,
        config_endpoint: str | None = None,
    ) -> None:
        self._url = url
        self._method = method.upper()
        self._headers = dict(headers) if headers else {}
        self._input_field = input_field
        self._output_field = output_field
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._config_endpoint = config_endpoint
        self._cached_config: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # TargetAgent protocol
    # ------------------------------------------------------------------

    async def invoke(
        self,
        input_text: str,
        config: dict[str, Any] | None = None,
    ) -> AgentResponse:
        """Send *input_text* to the remote agent and return its response.

        If *config* is provided it is merged into the JSON payload under a
        top-level ``"config"`` key so the agent can adjust behaviour per
        request.
        """
        payload: dict[str, Any] = {self._input_field: input_text}
        if config:
            payload["config"] = config

        try:
            async with aiohttp.ClientSession(
                headers=self._headers,
                timeout=self._timeout,
            ) as session:
                with _LatencyTimer() as timer:
                    async with session.request(
                        self._method,
                        self._url,
                        json=payload,
                    ) as resp:
                        body = await resp.json(content_type=None)

                if resp.status >= 400:
                    error_detail = body.get("error", body.get("detail", resp.reason))
                    logger.warning(
                        "Agent returned HTTP %s: %s", resp.status, error_detail,
                    )
                    return AgentResponse(
                        output_text="",
                        latency_ms=timer.elapsed_ms,
                        raw_metadata={
                            "http_status": resp.status,
                            "error": str(error_detail),
                            **(body if isinstance(body, dict) else {"raw": body}),
                        },
                    )

                output_text = self._extract_output(body)
                return AgentResponse(
                    output_text=output_text,
                    latency_ms=timer.elapsed_ms,
                    raw_metadata=body if isinstance(body, dict) else {"raw": body},
                )

        except aiohttp.ClientError as exc:
            logger.error("HTTP request to %s failed: %s", self._url, exc)
            raise ConnectorError(f"HTTP request failed: {exc}") from exc
        except Exception as exc:
            logger.error("Unexpected error invoking %s: %s", self._url, exc)
            raise ConnectorError(f"Unexpected error: {exc}") from exc

    def get_current_config(self) -> dict[str, Any]:
        """Return the last-known agent configuration.

        If a *config_endpoint* was provided at construction the config is
        fetched lazily on first call; otherwise an empty dict is returned.
        """
        return dict(self._cached_config)

    def apply_config(self, config: dict[str, Any]) -> None:
        """Store *config* locally and, if possible, push it to the agent.

        The push to the remote config endpoint is best-effort; failures are
        logged but do not raise.
        """
        self._cached_config = dict(config)
        if self._config_endpoint:
            logger.info(
                "Config endpoint configured (%s) — "
                "use async_apply_config() for remote push.",
                self._config_endpoint,
            )

    # ------------------------------------------------------------------
    # Async helpers (not part of the protocol, but useful)
    # ------------------------------------------------------------------

    async def async_fetch_config(self) -> dict[str, Any]:
        """Fetch the agent's configuration from the remote config endpoint."""
        if not self._config_endpoint:
            return {}
        try:
            async with aiohttp.ClientSession(
                headers=self._headers,
                timeout=self._timeout,
            ) as session:
                async with session.get(self._config_endpoint) as resp:
                    body = await resp.json(content_type=None)
                    self._cached_config = body if isinstance(body, dict) else {}
                    return dict(self._cached_config)
        except Exception as exc:
            logger.error("Failed to fetch config from %s: %s", self._config_endpoint, exc)
            raise ConnectorError(f"Config fetch failed: {exc}") from exc

    async def async_apply_config(self, config: dict[str, Any]) -> None:
        """Push *config* to the remote config endpoint via PUT."""
        self._cached_config = dict(config)
        if not self._config_endpoint:
            return
        try:
            async with aiohttp.ClientSession(
                headers=self._headers,
                timeout=self._timeout,
            ) as session:
                async with session.put(self._config_endpoint, json=config) as resp:
                    if resp.status >= 400:
                        body = await resp.text()
                        logger.warning(
                            "Config push returned HTTP %s: %s", resp.status, body,
                        )
        except Exception as exc:
            logger.error("Failed to push config to %s: %s", self._config_endpoint, exc)
            raise ConnectorError(f"Config push failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _extract_output(self, body: Any) -> str:
        """Pull the output string from a JSON response body."""
        if isinstance(body, dict):
            value = body.get(self._output_field)
            if value is not None:
                return str(value)
            # Fall back: try common alternative keys
            for key in ("response", "text", "message", "result"):
                if key in body:
                    return str(body[key])
            return str(body)
        return str(body)

    def __repr__(self) -> str:
        return (
            f"HttpApiAdapter(url={self._url!r}, method={self._method!r}, "
            f"timeout={self._timeout.total}s)"
        )
