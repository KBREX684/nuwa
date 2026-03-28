"""Interactive conversation UI for Nuwa agent training."""

from __future__ import annotations

from nuwa.conversation.manager import ConversationManager, ConversationPhase
from nuwa.conversation.phases.approval import ApprovalPhase
from nuwa.conversation.phases.direction import DirectionPhase
from nuwa.conversation.phases.onboarding import OnboardingPhase
from nuwa.conversation.phases.running import RunningPhase
from nuwa.conversation.renderer import NuwaRenderer

__all__ = [
    "ConversationManager",
    "ConversationPhase",
    "NuwaRenderer",
    "OnboardingPhase",
    "DirectionPhase",
    "RunningPhase",
    "ApprovalPhase",
]
