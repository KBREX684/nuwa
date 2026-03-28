"""Base guardrail classes for the Nuwa AI Trainer framework.

Provides :class:`BaseGuardrail`, a concrete foundation for all guardrails,
and :class:`CompositeGuardrail`, which aggregates multiple guardrails into a
single check that short-circuits on the first failure.
"""

from __future__ import annotations

import logging
from typing import Sequence

from nuwa.core.types import GuardrailVerdict, RoundResult

logger = logging.getLogger(__name__)


class BaseGuardrail:
    """Minimal base class satisfying the :class:`~nuwa.core.protocols.Guardrail` protocol.

    Subclasses **must** override :meth:`check`; the :attr:`name` property
    defaults to the class name but can be overridden via *guardrail_name*.

    Args:
        guardrail_name: Optional human-readable name.  Falls back to the
            concrete class's ``__name__`` when omitted.
    """

    def __init__(self, guardrail_name: str | None = None) -> None:
        self._name = guardrail_name or type(self).__name__

    @property
    def name(self) -> str:
        """Human-readable identifier for this guardrail."""
        return self._name

    def check(self, history: list[RoundResult]) -> GuardrailVerdict:
        """Evaluate the training history and return a verdict.

        The default implementation unconditionally passes.  Subclasses should
        override this with domain-specific logic.

        Args:
            history: All round results accumulated so far.

        Returns:
            A :class:`GuardrailVerdict` indicating whether to continue.
        """
        return GuardrailVerdict(
            passed=True,
            guardrail_name=self.name,
            reason="No check implemented; default pass.",
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


class CompositeGuardrail(BaseGuardrail):
    """Aggregates multiple guardrails and runs them all in sequence.

    Evaluation order matches the insertion order of *guardrails*.  The
    composite returns the **first failing** verdict encountered, or a
    synthetic all-passed verdict if every child passes.

    If any child verdict sets ``should_stop=True``, that flag is propagated
    even when the composite ultimately passes (an unlikely but defensive
    scenario).

    Args:
        guardrails: Ordered sequence of guardrail instances to evaluate.
        guardrail_name: Optional override for :attr:`name`.
    """

    def __init__(
        self,
        guardrails: Sequence[BaseGuardrail],
        guardrail_name: str | None = None,
    ) -> None:
        super().__init__(guardrail_name=guardrail_name or "CompositeGuardrail")
        self._guardrails: tuple[BaseGuardrail, ...] = tuple(guardrails)

    @property
    def guardrails(self) -> tuple[BaseGuardrail, ...]:
        """The child guardrails managed by this composite."""
        return self._guardrails

    def check(self, history: list[RoundResult]) -> GuardrailVerdict:
        """Run every child guardrail and return the first failure.

        Args:
            history: All round results accumulated so far.

        Returns:
            The first failing :class:`GuardrailVerdict`, or an all-passed
            verdict when every child passes.
        """
        if not self._guardrails:
            logger.debug("%s: no child guardrails configured; passing.", self.name)
            return GuardrailVerdict(
                passed=True,
                guardrail_name=self.name,
                reason="No child guardrails configured.",
            )

        should_stop = False
        for guardrail in self._guardrails:
            logger.debug("%s: running child guardrail %s", self.name, guardrail.name)
            verdict = guardrail.check(history)

            # Track the most severe stop signal across all children.
            should_stop = should_stop or verdict.should_stop

            if not verdict.passed:
                logger.warning(
                    "%s: child guardrail %s FAILED — %s",
                    self.name,
                    guardrail.name,
                    verdict.reason,
                )
                # Ensure the stop flag is at least as severe as what we've seen.
                if should_stop and not verdict.should_stop:
                    verdict = verdict.model_copy(update={"should_stop": True})
                return verdict

        logger.debug("%s: all %d child guardrails passed.", self.name, len(self._guardrails))
        return GuardrailVerdict(
            passed=True,
            should_stop=should_stop,
            guardrail_name=self.name,
            reason=f"All {len(self._guardrails)} guardrails passed.",
        )
