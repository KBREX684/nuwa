"""Connector factory and registry for the Nuwa AI Trainer framework."""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from typing import Any, cast

from nuwa.core.exceptions import ConfigError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Lazy imports are used so that optional heavy dependencies (e.g. aiohttp)
# are only loaded when the corresponding connector type is actually requested.

CONNECTOR_MAP: dict[str, str] = {
    "http": "nuwa.connectors.http_api:HttpApiAdapter",
    "cli": "nuwa.connectors.cli_adapter:CliAdapter",
    "function": "nuwa.connectors.function_call:FunctionCallAdapter",
}
"""Maps short type names to fully-qualified ``module:class`` references."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _import_class(dotted: str) -> type[Any]:
    """Import a class given a ``module.path:ClassName`` string."""
    module_path, _, class_name = dotted.rpartition(":")
    if not module_path or not class_name:
        raise ConfigError(
            f"Invalid connector reference {dotted!r}; "
            "expected 'module.path:ClassName'."
        )
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name, None)
    if cls is None:
        raise ConfigError(
            f"Class {class_name!r} not found in module {module_path!r}."
        )
    return cast(type[Any], cls)


def _resolve_callable_ref(ref: str | Callable[..., Any]) -> Callable[..., Any]:
    """Resolve a ``module.path:callable`` reference into a Python callable."""
    if callable(ref):
        return ref
    if not isinstance(ref, str) or ":" not in ref:
        raise ConfigError(
            "Function connector expects a callable or 'module.path:callable' string."
        )
    module_path, _, attr_name = ref.rpartition(":")
    if not module_path or not attr_name:
        raise ConfigError(f"Invalid callable reference: {ref!r}")
    module = importlib.import_module(module_path)
    obj = getattr(module, attr_name, None)
    if obj is None:
        raise ConfigError(
            f"Callable {attr_name!r} not found in module {module_path!r}."
        )
    if not callable(obj):
        raise ConfigError(
            f"Resolved object {module_path}:{attr_name} is not callable."
        )
    return cast(Callable[..., Any], obj)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_connector(connector_type: str, **params: Any) -> Any:
    """Instantiate a connector by its short type name.

    Parameters
    ----------
    connector_type:
        One of the keys in :data:`CONNECTOR_MAP` (``"http"``, ``"cli"``,
        ``"function"``), or a fully-qualified ``module:Class`` reference for
        third-party connectors.
    **params:
        Keyword arguments forwarded to the connector constructor.

    Returns
    -------
    TargetAgent
        A freshly constructed connector instance that satisfies the
        ``TargetAgent`` protocol.

    Raises
    ------
    ConfigError
        If the requested type is unknown or the constructor rejects the
        supplied parameters.
    """
    dotted = CONNECTOR_MAP.get(connector_type, connector_type)

    # Allow fully-qualified references to pass through directly
    if ":" not in dotted:
        raise ConfigError(
            f"Unknown connector type {connector_type!r}. "
            f"Available types: {list_connectors()}"
        )

    cls = _import_class(dotted)
    if connector_type == "function":
        # Web UI and YAML configs commonly provide a module reference string;
        # normalise to the callable expected by FunctionCallAdapter(func=...).
        if "func" not in params:
            if "module" in params:
                params = dict(params)
                params["func"] = _resolve_callable_ref(params.pop("module"))
            elif "callable" in params:
                params = dict(params)
                params["func"] = _resolve_callable_ref(params.pop("callable"))

    try:
        instance = cls(**params)
    except TypeError as exc:
        raise ConfigError(
            f"Failed to create connector {connector_type!r} with "
            f"params {params}: {exc}"
        ) from exc

    logger.info("Created connector %s (%s)", connector_type, cls.__name__)
    return instance


def list_connectors() -> list[str]:
    """Return the short names of all registered connector types."""
    return sorted(CONNECTOR_MAP)


def register_connector(name: str, dotted_path: str) -> None:
    """Register a custom connector type at runtime.

    Parameters
    ----------
    name:
        Short name to use when creating the connector (e.g. ``"grpc"``).
    dotted_path:
        Fully-qualified ``module.path:ClassName`` reference.
    """
    if ":" not in dotted_path:
        raise ConfigError(
            f"dotted_path must be 'module.path:ClassName', got {dotted_path!r}"
        )
    CONNECTOR_MAP[name] = dotted_path
    logger.info("Registered connector %r -> %r", name, dotted_path)
