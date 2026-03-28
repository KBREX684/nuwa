"""Reflection stage -- diagnoses failures and proposes improvements.

Applies *context window discipline* inspired by autoresearch: instead of
dumping all failure details into the LLM prompt, we select the top-N worst
failures, truncate long outputs, and prepend a compact summary statistics
block.  This keeps the prompt focused and avoids overwhelming the model.
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any, Literal, cast

from jinja2 import Template

from nuwa.core.defaults import (
    FAILURE_SCORE_THRESHOLD,
    MAX_FAILURES_FOR_PROMPT,
    MAX_OUTPUT_CHARS,
    MAX_PASSES_FOR_PROMPT,
    TEMPERATURE_REFLECTION,
)
from nuwa.core.exceptions import LLMError
from nuwa.core.types import LoopContext, Reflection
from nuwa.llm.prompts import FAILURE_REFLECTION
from nuwa.llm.response_parser import parse_json_response

logger = logging.getLogger(__name__)

_REFLECTION_TEMPLATE = Template(FAILURE_REFLECTION)

_FAILURE_SCORE_THRESHOLD = FAILURE_SCORE_THRESHOLD

# Context-window discipline constants
_MAX_FAILURES_FOR_PROMPT = MAX_FAILURES_FOR_PROMPT
_MAX_PASSES_FOR_PROMPT = MAX_PASSES_FOR_PROMPT
_MAX_OUTPUT_CHARS = MAX_OUTPUT_CHARS


class ReflectionStage:
    """Analyse failures and produce an actionable Reflection."""

    @property
    def name(self) -> str:
        return "reflection"

    async def execute(self, context: LoopContext) -> LoopContext:
        backend = context.backend_ref
        score_card = context.train_scores

        if score_card is None or not score_card.results:
            context.reflection = Reflection(
                round_num=context.round_num,
                diagnosis="No scored results available for reflection.",
                failure_patterns=[],
                proposed_changes=[],
                priority="low",
            )
            return context

        # Focus on failures (score < threshold) but include a sample of
        # passing items for contrast -- up to 30 items total.
        failures = [r for r in score_card.results if r.score < _FAILURE_SCORE_THRESHOLD]
        passes = [r for r in score_card.results if r.score >= _FAILURE_SCORE_THRESHOLD]

        # If everything passed, still reflect but at low urgency.
        if not failures:
            context.reflection = Reflection(
                round_num=context.round_num,
                diagnosis=(
                    f"All {len(score_card.results)} samples scored above "
                    f"{_FAILURE_SCORE_THRESHOLD}. Mean score: {score_card.mean_score:.3f}."
                ),
                failure_patterns=[],
                proposed_changes=[],
                priority="low",
            )
            logger.info("Round %d: no failures to reflect on.", context.round_num)
            return context

        # --- Context-window discipline: summarise before prompting ---------
        # Sort failures by score ascending (worst first) and cap at top N.
        failures_sorted = sorted(failures, key=lambda r: r.score)
        top_failures = failures_sorted[:_MAX_FAILURES_FOR_PROMPT]

        # Build compact summary statistics block.
        all_scores = [r.score for r in score_card.results]
        mean_score = sum(all_scores) / len(all_scores)
        pass_rate = sum(1 for s in all_scores if s >= _FAILURE_SCORE_THRESHOLD) / len(all_scores)

        # Simple score distribution (quartile buckets).
        buckets = {"0.0-0.25": 0, "0.25-0.50": 0, "0.50-0.75": 0, "0.75-1.0": 0}
        for s in all_scores:
            if s < 0.25:
                buckets["0.0-0.25"] += 1
            elif s < 0.50:
                buckets["0.25-0.50"] += 1
            elif s < 0.75:
                buckets["0.50-0.75"] += 1
            else:
                buckets["0.75-1.0"] += 1

        std_dev = (
            math.sqrt(sum((s - mean_score) ** 2 for s in all_scores) / len(all_scores))
            if len(all_scores) > 1
            else 0.0
        )

        summary_stats = (
            f"=== Summary Statistics ===\n"
            f"Total samples: {len(score_card.results)}\n"
            f"Failures: {len(failures)} ({len(failures) / len(score_card.results) * 100:.1f}%)\n"
            f"Pass rate: {pass_rate:.2%}\n"
            f"Mean score: {mean_score:.3f} (std: {std_dev:.3f})\n"
            f"Score distribution: {buckets}\n"
            f"Showing top {len(top_failures)} worst failures (of {len(failures)} total).\n"
            f"========================="
        )

        # Build scored_results payload -- truncate long outputs.
        def _truncate(text: str, max_chars: int = _MAX_OUTPUT_CHARS) -> str:
            if len(text) <= max_chars:
                return text
            return text[:max_chars] + f"... [{len(text) - max_chars} chars truncated]"

        items_for_prompt: list[dict[str, Any]] = []
        for r in top_failures:
            items_for_prompt.append(
                {
                    "score": r.score,
                    "input_text": _truncate(r.sample.input_text),
                    "expected_behavior": _truncate(r.sample.expected_behavior),
                    "actual_output": _truncate(r.response.output_text),
                    "reasoning_en": _truncate(r.reasoning),
                }
            )
        # Add a few passing items for contrast (also truncated).
        for r in passes[:_MAX_PASSES_FOR_PROMPT]:
            items_for_prompt.append(
                {
                    "score": r.score,
                    "input_text": _truncate(r.sample.input_text),
                    "expected_behavior": _truncate(r.sample.expected_behavior),
                    "actual_output": _truncate(r.response.output_text),
                    "reasoning_en": _truncate(r.reasoning),
                }
            )

        config_snapshot = json.dumps(context.current_config, indent=2, default=str)

        prompt = _REFLECTION_TEMPLATE.render(
            scored_results=items_for_prompt,
            current_config=config_snapshot,
        )
        # Prepend summary statistics so the LLM sees the big picture first.
        prompt = summary_stats + "\n\n" + prompt

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "You are a senior AI diagnostics analyst."},
            {"role": "user", "content": prompt},
        ]

        logger.info(
            "Round %d: reflecting on %d failures (%d total scored)",
            context.round_num,
            len(failures),
            len(score_card.results),
        )

        try:
            raw = await backend.complete(messages, temperature=TEMPERATURE_REFLECTION)
            data = parse_json_response(raw)
            if not isinstance(data, dict):
                raise LLMError("Reflection response is not a JSON object.")

            diagnosis = (
                data.get("diagnosis_summary_en", "")
                or data.get("diagnosis_summary_zh", "")
                or "No diagnosis provided."
            )

            failure_patterns: list[str] = []
            for fp in data.get("failure_patterns", []):
                label = fp.get("label_en", "") or fp.get("label_zh", "")
                root_cause = fp.get("root_cause", "")
                if label or root_cause:
                    failure_patterns.append(
                        f"{label}: {root_cause}" if label and root_cause else label or root_cause
                    )

            proposed_changes: list[str] = []
            for pc in data.get("proposed_changes", []):
                desc = pc.get("description_en", "") or pc.get("description_zh", "")
                if desc:
                    proposed_changes.append(desc)

            # Determine priority based on failure ratio.
            failure_ratio = len(failures) / len(score_card.results)
            if failure_ratio > 0.7:
                priority = "critical"
            elif failure_ratio > 0.4:
                priority = "high"
            elif failure_ratio > 0.15:
                priority = "medium"
            else:
                priority = "low"

            context.reflection = Reflection(
                round_num=context.round_num,
                diagnosis=str(diagnosis),
                failure_patterns=failure_patterns,
                proposed_changes=proposed_changes,
                priority=cast(Literal["low", "medium", "high", "critical"], priority),
            )

        except (LLMError, KeyError, TypeError, ValueError) as exc:
            logger.error("Reflection LLM call failed: %s", exc)
            context.reflection = Reflection(
                round_num=context.round_num,
                diagnosis=f"Reflection failed due to LLM error: {exc}",
                failure_patterns=["LLM reflection unavailable"],
                proposed_changes=["Retry reflection in next round"],
                priority="medium",
            )

        logger.info(
            "Round %d: reflection complete -- %d patterns, %d proposed changes",
            context.round_num,
            len(context.reflection.failure_patterns),
            len(context.reflection.proposed_changes),
        )
        return context
