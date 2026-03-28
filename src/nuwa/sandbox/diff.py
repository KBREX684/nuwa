"""Config diff utilities for comparing agent configurations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DiffEntry:
    """A single difference between two configuration values.

    Attributes:
        path: Dot-notation path to the changed key, e.g. ``"llm.temperature"``.
        change_type: One of ``"added"``, ``"removed"``, or ``"modified"``.
        old_value: The value before the change (``None`` for additions).
        new_value: The value after the change (``None`` for removals).
    """

    path: str
    change_type: str  # "added" | "removed" | "modified"
    old_value: Any
    new_value: Any


def deep_diff(
    original: dict[str, Any],
    modified: dict[str, Any],
    path: str = "",
) -> list[DiffEntry]:
    """Recursively compare two config dicts and return a list of changes.

    Args:
        original: The baseline configuration.
        modified: The changed configuration.
        path: Internal accumulator for the dot-notation key path.

    Returns:
        A list of :class:`DiffEntry` objects describing every difference.
    """
    entries: list[DiffEntry] = []
    all_keys = set(original.keys()) | set(modified.keys())

    for key in sorted(all_keys):
        current_path = f"{path}.{key}" if path else key

        if key not in original:
            entries.append(
                DiffEntry(
                    path=current_path,
                    change_type="added",
                    old_value=None,
                    new_value=modified[key],
                )
            )
        elif key not in modified:
            entries.append(
                DiffEntry(
                    path=current_path,
                    change_type="removed",
                    old_value=original[key],
                    new_value=None,
                )
            )
        elif isinstance(original[key], dict) and isinstance(modified[key], dict):
            entries.extend(deep_diff(original[key], modified[key], current_path))
        elif original[key] != modified[key]:
            entries.append(
                DiffEntry(
                    path=current_path,
                    change_type="modified",
                    old_value=original[key],
                    new_value=modified[key],
                )
            )

    return entries


def format_diff_text(entries: list[DiffEntry]) -> str:
    """Format diff entries as human-readable text with terminal colours.

    Uses ANSI escape codes: green for additions, red for removals, yellow for
    modifications.

    Args:
        entries: The diff entries to format.

    Returns:
        A multi-line string suitable for terminal display.
    """
    if not entries:
        return "No differences."

    _GREEN = "\033[32m"
    _RED = "\033[31m"
    _YELLOW = "\033[33m"
    _RESET = "\033[0m"

    lines: list[str] = []
    for entry in entries:
        if entry.change_type == "added":
            lines.append(
                f"{_GREEN}+ {entry.path}: {entry.new_value!r}{_RESET}"
            )
        elif entry.change_type == "removed":
            lines.append(
                f"{_RED}- {entry.path}: {entry.old_value!r}{_RESET}"
            )
        elif entry.change_type == "modified":
            lines.append(
                f"{_YELLOW}~ {entry.path}: {entry.old_value!r} -> {entry.new_value!r}{_RESET}"
            )
    return "\n".join(lines)


def format_diff_html(entries: list[DiffEntry]) -> str:
    """Format diff entries as an HTML snippet for the web UI.

    Args:
        entries: The diff entries to format.

    Returns:
        An HTML string using ``<div>`` elements with inline colour styles.
    """
    if not entries:
        return '<div class="diff-empty">No differences.</div>'

    _STYLES = {
        "added": "color: #22c55e;",
        "removed": "color: #ef4444; text-decoration: line-through;",
        "modified": "color: #eab308;",
    }
    _PREFIXES = {"added": "+", "removed": "-", "modified": "~"}

    parts: list[str] = ['<div class="diff">']
    for entry in entries:
        style = _STYLES.get(entry.change_type, "")
        prefix = _PREFIXES.get(entry.change_type, "?")
        if entry.change_type == "added":
            text = f"{prefix} {entry.path}: {entry.new_value!r}"
        elif entry.change_type == "removed":
            text = f"{prefix} {entry.path}: {entry.old_value!r}"
        else:
            text = (
                f"{prefix} {entry.path}: {entry.old_value!r} &rarr; "
                f"{entry.new_value!r}"
            )
        parts.append(f'  <div class="diff-entry" style="{style}">{text}</div>')
    parts.append("</div>")
    return "\n".join(parts)
