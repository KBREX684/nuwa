"""Nuwa SDK -- high-level integration layer for agent training.

Exports the key symbols for SDK usage::

    from nuwa.sdk import trainable, train, train_sync, NuwaTrainer
"""

from __future__ import annotations

from nuwa.sdk.decorator import NuwaMeta, trainable
from nuwa.sdk.quick import train, train_sync
from nuwa.sdk.trainer import NuwaTrainer

__all__ = [
    "NuwaMeta",
    "NuwaTrainer",
    "train",
    "train_sync",
    "trainable",
]
