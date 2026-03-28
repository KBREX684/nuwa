"""Multi-judge ensemble evaluator for distributed scoring.

Runs multiple LLM judges concurrently on the same agent outputs and
aggregates their scores using a configurable ensemble strategy.  This
produces more robust evaluations than a single judge while also being
faster (all judges score in parallel).
"""

from __future__ import annotations

import asyncio
import logging
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from jinja2 import Template

from nuwa.core.exceptions import LLMError
from nuwa.core.types import AgentResponse, EvalSample, ScoreCard, ScoredResult
from nuwa.engine.stages.immutable_scorer import ImmutableScorer
from nuwa.llm.prompts import OUTPUT_SCORING
from nuwa.llm.response_parser import parse_json_response

logger = logging.getLogger(__name__)

_SCORING_TEMPLATE = Template(OUTPUT_SCORING)
_PASS_THRESHOLD = 0.7
_JUDGE_CONCURRENCY = 5  # per-judge concurrency for scoring samples


# ---------------------------------------------------------------------------
# Ensemble strategy
# ---------------------------------------------------------------------------


class EnsembleStrategy(StrEnum):
    """Strategy used to combine scores from multiple judges."""

    MEAN = "mean"
    """Average all judge scores."""

    MEDIAN = "median"
    """Median score (robust to outlier judges)."""

    WEIGHTED = "weighted"
    """Weighted average using per-judge ``JudgeConfig.weight``."""

    MAJORITY = "majority"
    """Majority voting on pass/fail relative to the pass threshold."""

    MIN = "min"
    """Conservative: use lowest score (strictest judge)."""


# ---------------------------------------------------------------------------
# Judge configuration
# ---------------------------------------------------------------------------


@dataclass
class JudgeConfig:
    """Configuration for a single judge in the ensemble.

    Parameters
    ----------
    name:
        Human-readable identifier (e.g. ``"gpt4o-judge"``).
    backend:
        A ``ModelBackend`` instance used by this judge for LLM calls.
    weight:
        Relative weight when using :attr:`EnsembleStrategy.WEIGHTED`.
    scoring_prompt:
        Optional custom Jinja2 scoring prompt.  When ``None`` the
        framework default ``OUTPUT_SCORING`` template is used.
    """

    name: str
    backend: Any  # ModelBackend protocol
    weight: float = 1.0
    scoring_prompt: str | None = None


# ---------------------------------------------------------------------------
# Parallel evaluator
# ---------------------------------------------------------------------------


class ParallelEvaluator:
    """Multi-judge ensemble evaluator.

    Runs multiple LLM judges concurrently on the same results.  Each
    judge independently scores every ``(input, expected, output)`` triple.
    Scores are then aggregated using the chosen ensemble strategy.

    Benefits
    --------
    * More robust scoring (reduces single-judge bias).
    * Faster overall evaluation (judges run in parallel).
    * Supports mixing different models (e.g. GPT-4o + Claude as judges).
    * Transparent per-judge breakdown in scoring reasoning.

    Parameters
    ----------
    judges:
        List of judge configurations.  At least one is required.
    strategy:
        Ensemble aggregation strategy.
    immutable_scorer:
        Optional deterministic scorer.  When ``None`` a default instance
        is created.
    immutable_weight:
        Weight applied to the immutable score in the final blend.
    llm_weight:
        Weight applied to the aggregated LLM ensemble score.
    """

    def __init__(
        self,
        judges: list[JudgeConfig],
        strategy: EnsembleStrategy = EnsembleStrategy.MEAN,
        immutable_scorer: ImmutableScorer | None = None,
        immutable_weight: float = 0.3,
        llm_weight: float = 0.7,
    ) -> None:
        if not judges:
            raise ValueError("At least one JudgeConfig is required.")
        self._judges = judges
        self._strategy = strategy
        self._immutable_scorer = immutable_scorer or ImmutableScorer()
        self._immutable_weight = immutable_weight
        self._llm_weight = llm_weight

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def evaluate_batch(
        self,
        results: list[tuple[EvalSample, AgentResponse]],
        on_progress: Callable[[int, int], Any] | None = None,
    ) -> ScoreCard:
        """Score all *results* using every judge in parallel.

        Parameters
        ----------
        results:
            ``(sample, response)`` pairs to evaluate.
        on_progress:
            Optional callback ``on_progress(completed_judges, total_judges)``.

        Returns
        -------
        ScoreCard
            Aggregated score card with per-sample ensemble scores and a
            failure analysis summary.
        """
        if not results:
            return ScoreCard(results=[], failure_analysis="")

        num_judges = len(self._judges)
        completed_judges = 0
        lock = asyncio.Lock()

        async def _judge_with_progress(
            judge: JudgeConfig,
        ) -> list[float]:
            nonlocal completed_judges
            scores = await self._run_single_judge(judge, results)
            async with lock:
                completed_judges += 1
                if on_progress is not None:
                    try:
                        on_progress(completed_judges, num_judges)
                    except Exception:  # noqa: BLE001
                        pass
            return scores

        # Run all judges concurrently -- each judge returns a list of
        # per-sample scores.
        logger.info(
            "ParallelEvaluator: scoring %d samples with %d judges "
            "(strategy=%s)",
            len(results),
            num_judges,
            self._strategy.value,
        )

        judge_scores: list[list[float]] = list(
            await asyncio.gather(
                *(_judge_with_progress(j) for j in self._judges)
            )
        )

        # --- Aggregate column-wise (per sample) --------------------------
        num_samples = len(results)
        scored_results: list[ScoredResult] = []

        for idx in range(num_samples):
            sample, response = results[idx]

            # Collect this sample's scores from every judge.
            per_judge: list[float] = [
                judge_scores[j][idx] for j in range(num_judges)
            ]

            # Aggregate LLM ensemble score.
            ensemble_score = self._aggregate(per_judge)

            # Deterministic immutable metrics (applied once, not per judge).
            immutable_metrics = self._immutable_scorer.score(sample, response)
            immutable_agg = self._immutable_scorer.aggregate(immutable_metrics)

            # Blend LLM ensemble with immutable score.
            final_score = (
                self._llm_weight * ensemble_score
                + self._immutable_weight * immutable_agg
            )
            final_score = max(0.0, min(1.0, final_score))

            # Build transparent reasoning with per-judge breakdown.
            judge_parts = ", ".join(
                f"{self._judges[j].name}: {per_judge[j]:.2f}"
                for j in range(num_judges)
            )
            metrics_summary = " | ".join(
                f"{k}={v:.2f}" for k, v in immutable_metrics.items()
            )
            reasoning = (
                f"{judge_parts}, ensemble({self._strategy.value}): "
                f"{ensemble_score:.2f} "
                f"[immutable: {metrics_summary} -> agg={immutable_agg:.2f}] "
                f"final={final_score:.2f}"
            )

            scored_results.append(
                ScoredResult(
                    sample=sample,
                    response=response,
                    score=final_score,
                    reasoning=reasoning,
                )
            )

        # --- Build failure analysis --------------------------------------
        failures = [s for s in scored_results if s.score < _PASS_THRESHOLD]
        failure_analysis = ""
        if failures:
            lines: list[str] = []
            for f in failures:
                lines.append(
                    f"- [{f.score:.2f}] input={f.sample.input_text[:80]!r} | "
                    f"reason={f.reasoning[:120]}"
                )
            failure_analysis = (
                f"{len(failures)}/{len(scored_results)} samples scored below "
                f"{_PASS_THRESHOLD}:\n" + "\n".join(lines)
            )

        logger.info(
            "ParallelEvaluator: mean=%.3f  pass_rate=%.2f  failures=%d/%d",
            (
                sum(s.score for s in scored_results) / len(scored_results)
                if scored_results
                else 0.0
            ),
            (
                sum(1 for s in scored_results if s.score >= _PASS_THRESHOLD)
                / len(scored_results)
                if scored_results
                else 0.0
            ),
            len(failures),
            len(scored_results),
        )

        return ScoreCard(results=scored_results, failure_analysis=failure_analysis)

    # ------------------------------------------------------------------
    # Single-judge execution
    # ------------------------------------------------------------------

    async def _run_single_judge(
        self,
        judge: JudgeConfig,
        results: list[tuple[EvalSample, AgentResponse]],
    ) -> list[float]:
        """Run one judge across all results.

        Scores are computed concurrently within the judge (bounded by a
        semaphore of ``_JUDGE_CONCURRENCY``).

        Returns
        -------
        list[float]
            Per-sample scores in the same order as *results*.
        """
        sem = asyncio.Semaphore(_JUDGE_CONCURRENCY)
        template = (
            Template(judge.scoring_prompt)
            if judge.scoring_prompt
            else _SCORING_TEMPLATE
        )

        async def _score_one(
            sample: EvalSample, response: AgentResponse
        ) -> float:
            async with sem:
                return await self._call_judge(
                    judge, template, sample, response
                )

        scores: list[float] = list(
            await asyncio.gather(
                *(
                    _score_one(sample, response)
                    for sample, response in results
                )
            )
        )

        logger.debug(
            "Judge '%s' finished: mean=%.3f across %d samples",
            judge.name,
            sum(scores) / len(scores) if scores else 0.0,
            len(scores),
        )
        return scores

    # ------------------------------------------------------------------
    # LLM call for a single judge + sample
    # ------------------------------------------------------------------

    @staticmethod
    async def _call_judge(
        judge: JudgeConfig,
        template: Template,
        sample: EvalSample,
        response: AgentResponse,
    ) -> float:
        """Invoke the judge backend for a single sample and return a score."""
        prompt = template.render(
            input_text=sample.input_text,
            expected_behavior=sample.expected_behavior,
            actual_output=response.output_text,
            scoring_rubric=None,
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "You are an objective evaluation judge."},
            {"role": "user", "content": prompt},
        ]

        try:
            raw = await judge.backend.complete(messages, temperature=0.1)
            data = parse_json_response(raw)
            if not isinstance(data, dict):
                raise LLMError("Scoring response must be a JSON object.")
            score = float(data.get("score", 0.0))
            return max(0.0, min(1.0, score))
        except (LLMError, KeyError, TypeError, ValueError) as exc:
            logger.warning(
                "Judge '%s' scoring failed for sample %s: %s",
                judge.name,
                sample.id,
                exc,
            )
            return 0.0

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def _aggregate(self, scores: list[float]) -> float:
        """Aggregate per-judge scores for a single sample."""
        if not scores:
            return 0.0

        if self._strategy == EnsembleStrategy.MEAN:
            return sum(scores) / len(scores)

        if self._strategy == EnsembleStrategy.MEDIAN:
            return statistics.median(scores)

        if self._strategy == EnsembleStrategy.WEIGHTED:
            total_weight = sum(j.weight for j in self._judges)
            if total_weight == 0:
                return 0.0
            weighted_sum = sum(
                s * j.weight for s, j in zip(scores, self._judges)
            )
            return weighted_sum / total_weight

        if self._strategy == EnsembleStrategy.MAJORITY:
            passed = sum(1 for s in scores if s >= _PASS_THRESHOLD)
            return 1.0 if passed > len(scores) / 2 else 0.0

        if self._strategy == EnsembleStrategy.MIN:
            return min(scores)

        # Fallback (should not happen with enum).
        return sum(scores) / len(scores)
