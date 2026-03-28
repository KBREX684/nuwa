"""Multi-objective scorer -- rates agent outputs across multiple axes.

Uses a single LLM call per sample with a multi-axis scoring prompt so that
all objectives are evaluated together rather than requiring N separate calls.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from nuwa.core.types import AgentResponse, EvalSample
from nuwa.engine.objectives.types import (
    MultiObjectiveScore,
    MultiObjectiveScoreCard,
    ObjectiveSet,
)
from nuwa.llm.response_parser import parse_json_response

logger = logging.getLogger(__name__)


class MultiObjectiveScorer:
    """Scores agent outputs across multiple objectives.

    Uses a single LLM call with a multi-axis scoring prompt to obtain
    per-objective scores efficiently (instead of *N* separate calls).
    """

    def __init__(self, objectives: ObjectiveSet, backend: Any) -> None:
        self._objectives = objectives
        self._backend = backend

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def score_one(
        self,
        sample: EvalSample,
        response: AgentResponse,
    ) -> MultiObjectiveScore:
        """Score a single (sample, response) pair across all objectives."""
        messages = self._build_scoring_prompt(sample, response)

        try:
            raw = await self._backend.complete(messages, temperature=0.1)
            data = parse_json_response(raw)
            if not isinstance(data, dict):
                raise ValueError("Expected a JSON object from the scorer LLM.")

            raw_scores: dict[str, float] = {}
            scores_block = data.get("scores", {})
            for name in self._objectives.names():
                val = float(scores_block.get(name, 0.0))
                raw_scores[name] = max(0.0, min(1.0, val))

            weighted = self._compute_weighted(raw_scores)
            return MultiObjectiveScore(scores=raw_scores, weighted_aggregate=weighted)

        except Exception as exc:
            logger.warning("Multi-objective scoring failed for sample %s: %s", sample.id, exc)
            # Fall back to zeros so the pipeline does not break.
            fallback = {n: 0.0 for n in self._objectives.names()}
            return MultiObjectiveScore(scores=fallback, weighted_aggregate=0.0)

    async def score_batch(
        self,
        results: list[tuple[EvalSample, AgentResponse]],
        max_concurrency: int = 5,
    ) -> MultiObjectiveScoreCard:
        """Score a batch of results and return a :class:`MultiObjectiveScoreCard`."""
        sem = asyncio.Semaphore(max_concurrency)

        async def _bounded(sample: EvalSample, resp: AgentResponse) -> MultiObjectiveScore:
            async with sem:
                return await self.score_one(sample, resp)

        scores = list(await asyncio.gather(*(_bounded(s, r) for s, r in results)))
        return MultiObjectiveScoreCard.from_scored_results(scores, self._objectives)

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_scoring_prompt(
        self,
        sample: EvalSample,
        response: AgentResponse,
    ) -> list[dict[str, Any]]:
        """Build a multi-axis scoring prompt.

        The prompt asks the LLM to rate each objective separately and return
        structured JSON with per-objective scores and reasoning.
        """
        objectives_block = "\n".join(
            f"- **{obj.name}** ({obj.description or obj.name}): "
            f"score from 0.0 to 1.0 (direction: {obj.direction})"
            for obj in self._objectives.objectives
        )

        obj_keys_example = ", ".join(f'"{obj.name}": 0.85' for obj in self._objectives.objectives)
        reasoning_keys_example = ", ".join(
            f'"{obj.name}": "..."' for obj in self._objectives.objectives
        )

        user_prompt = (
            "You are a multi-objective evaluation judge for the Nuwa AI Trainer "
            "(女娲 AI 训练框架).  Score the agent's output on EACH of the "
            "following objectives independently.\n\n"
            "## Objectives / 评估维度\n"
            f"{objectives_block}\n\n"
            "## Evaluation Pair / 评估对\n"
            f"**Input / 输入**:\n{sample.input_text}\n\n"
            f"**Expected behaviour / 期望行为**:\n{sample.expected_behavior}\n\n"
            f"**Actual output / 实际输出**:\n{response.output_text}\n\n"
            "## Scoring Instructions / 评分指南\n"
            "1. Evaluate the actual output against the expected behaviour on "
            "each objective independently.\n"
            "2. Assign a score from 0.0 (worst) to 1.0 (best) for each objective.\n"
            "3. Provide concise reasoning for each score.\n\n"
            "## Output format / 输出格式\n"
            "Return ONLY JSON (no markdown fences):\n"
            "{\n"
            f'  "scores": {{{obj_keys_example}}},\n'
            f'  "reasoning": {{{reasoning_keys_example}}}\n'
            "}"
        )

        return [
            {"role": "system", "content": "You are an objective multi-axis evaluation judge."},
            {"role": "user", "content": user_prompt},
        ]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _compute_weighted(self, scores: dict[str, float]) -> float:
        """Compute a weighted aggregate from per-objective scores."""
        weights = self._objectives.weights_dict()
        total_weight = sum(weights.get(n, 1.0) for n in self._objectives.names()) or 1.0
        weighted = (
            sum(scores.get(n, 0.0) * weights.get(n, 1.0) for n in self._objectives.names())
            / total_weight
        )
        return max(0.0, min(1.0, weighted))
