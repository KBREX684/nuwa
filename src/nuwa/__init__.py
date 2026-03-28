"""女娲 Nuwa - AI Agent Trainer.

Optimise your AI agents through automated training loops powered by
LLM-driven evaluation, reflection, and mutation.
"""

__version__ = "0.1.0"
__project__ = "女娲 Nuwa"

# Re-export key public classes for convenient top-level imports.
from nuwa.config.schema import NuwaConfig
from nuwa.connectors.cli_adapter import CliAdapter
from nuwa.connectors.function_call import FunctionCallAdapter
from nuwa.connectors.http_api import HttpApiAdapter
from nuwa.conversation.manager import ConversationManager
from nuwa.engine.loop import TrainingLoop
from nuwa.llm.backend import LiteLLMBackend
from nuwa.sandbox.agent import SandboxedAgent

# Sandbox isolation
from nuwa.sandbox.manager import SandboxManager

# SDK: decorator, one-liner, trainer
from nuwa.sdk.decorator import trainable
from nuwa.sdk.quick import train, train_sync
from nuwa.sdk.trainer import NuwaTrainer

__all__ = [
    "__version__",
    "__project__",
    # Core
    "NuwaConfig",
    "TrainingLoop",
    "LiteLLMBackend",
    "HttpApiAdapter",
    "CliAdapter",
    "FunctionCallAdapter",
    "ConversationManager",
    # SDK
    "trainable",
    "train",
    "train_sync",
    "NuwaTrainer",
    # Sandbox
    "SandboxManager",
    "SandboxedAgent",
]
