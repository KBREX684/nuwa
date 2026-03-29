"""Plugin loading utilities."""

from __future__ import annotations

import importlib
from types import ModuleType

from nuwa.plugins.context import PluginContext

_REGISTER_FN_NAMES = ("register", "register_plugin", "setup_plugin")
_LOADED: set[str] = set()


def _call_register(module: ModuleType, context: PluginContext) -> bool:
    for fn_name in _REGISTER_FN_NAMES:
        fn = getattr(module, fn_name, None)
        if fn is None:
            continue
        if callable(fn):
            try:
                fn(context)
            except TypeError:
                # Backward-compatible plugin signature: register()
                fn()
            return True
    return False


def load_plugin(module_path: str, *, reload: bool = False) -> ModuleType:
    """Import and initialise one plugin module."""
    if not reload and module_path in _LOADED:
        return importlib.import_module(module_path)

    module = importlib.import_module(module_path)
    _call_register(module, PluginContext())
    _LOADED.add(module_path)
    return module


def load_plugins(module_paths: list[str], *, reload: bool = False) -> list[ModuleType]:
    """Load multiple plugins and return imported modules."""
    loaded: list[ModuleType] = []
    for module_path in module_paths:
        loaded.append(load_plugin(module_path, reload=reload))
    return loaded


def loaded_plugins() -> list[str]:
    """Return sorted list of loaded plugin module paths."""
    return sorted(_LOADED)


def reset_loaded_plugins() -> None:
    """Reset plugin-loaded cache (mainly for tests)."""
    _LOADED.clear()
