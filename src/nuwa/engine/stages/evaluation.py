"""Evaluation stage -- scores agent responses via the LLM judge.

Combines subjective LLM-based scoring with deterministic
:class:`~nuwa.engine.stages.immutable_scorer.ImmutableScorer` metrics.
The final score is a weighted blend (default 60 % LLM, 40 % immutable).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from jinja2 import Template

from nuwa.core.exceptions import LLMError
from nuwa.core.types import LoopContext, ScoreCard, ScoredResult
from nuwa.engine.stages.immutable_scorer import ImmutableScorer
from nuwa.llm.prompts import OUTPUT_SCORING
from nuwa.llm.response_parser import parse_json_response

logger = logging.getLogger(__name__)

_SCORING_TEMPLATE = Template(OUTPUT_SCORING)

_PASS_THRESHOLD = 0.7
_MAX_CONCURRENCY = 5

# Weight split between LLM (subjective) and immutable (objective) scores.
_LLM_WEIGHT = 0.6
_IMMUTABLE_WEIGHT = 0.4


class EvaluationStage:
    """Score every (input, expected, actual) triple using the LLM backend.

    An :class:`ImmutableScorer` is used alongside the LLM judge so that
    deterministic, rule-based metrics (format compliance, length, latency,
    keyword presence) are always factored in.  These cannot be gamed by
    prompt optimisation.

    Parameters
    ----------
    immutable_scorer:
        Pre-configured scorer instance.  When *None* a default scorer is
        created automatically.
    llm_weight:
        Weight for the LLM-based score in ``[0, 1]``.
    immutable_weight:
        Weight for the immutable score in ``[0, 1]``.
    """

    def __init__(
        self,
        *,
        immutable_scorer: ImmutableScorer | None = None,
        llm_weight: float = _LLM_WEIGHT,
        immutable_weight: float = _IMMUTABLE_WEIGHT,
    ) -> None:
        self._immutable_scorer = immutable_scorer or ImmutableScorer()
        self._llm_weight = llm_weight
        self._immutable_weight = immutable_weight

    @property
    def name(self) -> str:
        return "evaluation"

    async def execute(self, context: LoopContext) -> LoopContext:
        backend = context.backend_ref

        if not context.train_results:
            logger.warning("Round %d: no train_results to evaluate.", context.round_num)
            context.train_scores = ScoreCard(results=[], failure_analysis="")
            return context

        sem = asyncio.Semaphore(_MAX_CONCURRENCY)
        immutable_scorer = self._immutable_scorer
        llm_w = self._llm_weight
        imm_w = self._immutable_weight

        async def _score_with_sem(sr: ScoredResult) -> ScoredResult:
            async with sem:
                llm_score, reasoning = await self._score_one(
                    backend,
                    sr.sample.input_text,
                    sr.sample.expected_behavior,
                    sr.response.output_text,
                )

                # Deterministic immutable metrics
                immutable_metrics = immutable_scorer.score(sr.sample, sr.response)
                immutable_agg = immutable_scorer.aggregate(immutable_metrics)

                # Blended final score
                score = llm_w * llm_score + imm_w * immutable_agg
                score = max(0.0, min(1.0, score))

                # Append immutable breakdown to reasoning for transparency
                metrics_summary = " | ".join(
                    f"{k}={v:.2f}" for k, v in immutable_metrics.items()
                )
                full_reasoning = (
                    f"{reasoning} [immutable: {metrics_summary} "
                    f"-> agg={immutable_agg:.2f}]"
                )

                return ScoredResult(
                    sample=sr.sample,
                    response=sr.response,
                    score=score,
                    reasoning=full_reasoning,
                )

        scored = list(await asyncio.gather(*(_score_with_sem(sr) for sr in context.train_results)))

        # Build failure analysis summary for items below threshold.
        failures = [s for s in scored if s.score < _PASS_THRESHOLD]
        failure_analysis = ""
        if failures:
            lines: list[str] = []
            for f in failures:
                lines.append(
                    f"- [{f.score:.2f}] input={f.sample.input_text[:80]!r} | "
                    f"reason={f.reasoning[:120]}"
                )
            failure_analysis = (
                f"{len(failures)}/{len(scored)} samples scored below "
                f"{_PASS_THRESHOLD}:\n" + "\n".join(lines)
            )

        card = ScoreCard(results=scored, failure_analysis=failure_analysis)
        context.train_scores = card
        context.train_results = scored  # update with real scores

        logger.info(
            "Round %d: train mean_score=%.3f  pass_rate=%.2f  failures=%d/%d",
            context.round_num,
            card.mean_score,
            card.pass_rate,
            len(failures),
            len(scored),
        )
        return context

    # ------------------------------------------------------------------

    @staticmethod
    async def _score_one(
        backend: Any,
        input_text: str,
        expected_behavior: str,
        actual_output: str,
    ) -> tuple[float, str]:
        """Score a single evaluation triple. Returns (score, reasoning)."""
        prompt = _SCORING_TEMPLATE.render(
            input_text=input_text,
            expected_behavior=expected_behavior,
            actual_output=actual_output,
            scoring_rubric=None,
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "You are an objective evaluation judge."},
            {"role": "user", "content": prompt},
        ]

        try:
            raw = await backend.complete(messages, temperature=0.1)
            data = parse_json_response(raw)
            score = float(data.get("score", 0.0))  # type: ignore[union-attr]
            score = max(0.0, min(1.0, score))
            reasoning_en = data.get("reasoning_en", "")  # type: ignore[union-attr]
            reasoning_zh = data.get("reasoning_zh", "")  # type: ignore[union-attr]
            reasoning = reasoning_en or reasoning_zh or "(no reasoning provided)"
            return score, str(reasoning)
        except (LLMError, KeyError, TypeError, ValueError) as exc:
            logger.warning("Scoring failed for a sample: %s", exc)
            return 0.0, f"Scoring error: {exc}"
