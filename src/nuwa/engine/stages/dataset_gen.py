"""Dataset generation stage -- synthesises eval samples via the LLM."""

from __future__ import annotations

import logging
import math
from typing import Any

from jinja2 import Template

from nuwa.core.exceptions import LLMError
from nuwa.core.types import EvalSample, LoopContext
from nuwa.llm.prompts import DATASET_GENERATION
from nuwa.llm.response_parser import parse_json_response

logger = logging.getLogger(__name__)

_TEMPLATE = Template(DATASET_GENERATION)


class DatasetGenStage:
    """Generate evaluation samples using the LLM backend and split them."""

    @property
    def name(self) -> str:
        return "dataset_generation"

    async def execute(self, context: LoopContext) -> LoopContext:
        backend = context.backend_ref
        config = context.config

        # Collect failure patterns from prior rounds for harder samples.
        previous_failures: list[str] = []
        if context.round_num > 1 and context.history:
            for rr in context.history:
                if rr.reflection:
                    previous_failures.extend(rr.reflection.failure_patterns)
            # Keep only the most recent / unique patterns (cap at 15).
            seen: set[str] = set()
            deduped: list[str] = []
            for fp in reversed(previous_failures):
                key = fp.strip().lower()
                if key not in seen:
                    seen.add(key)
                    deduped.append(fp)
            previous_failures = list(reversed(deduped))[:15]

        prompt_text = _TEMPLATE.render(
            training_direction=config.training_direction,
            round_num=context.round_num,
            total_rounds=config.max_rounds,
            num_samples=config.samples_per_round,
            previous_failures=previous_failures if previous_failures else None,
            tags_hint=None,
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "You are a dataset generation assistant."},
            {"role": "user", "content": prompt_text},
        ]

        logger.info(
            "Round %d: generating %d eval samples via LLM",
            context.round_num,
            config.samples_per_round,
        )

        raw_response = await backend.complete(messages, temperature=0.9)
        raw_data = parse_json_response(raw_response)

        if not isinstance(raw_data, list):
            raise LLMError(
                f"Expected a JSON array of samples, got {type(raw_data).__name__}."
            )

        samples: list[EvalSample] = []
        for idx, item in enumerate(raw_data):
            try:
                # Normalise difficulty values the LLM may return.
                difficulty = str(item.get("difficulty", "medium")).lower()
                if difficulty not in ("easy", "medium", "hard"):
                    difficulty = "medium"
                samples.append(
                    EvalSample(
                        input_text=str(item["input_text"]),
                        expected_behavior=str(item["expected_behavior"]),
                        difficulty=difficulty,  # type: ignore[arg-type]
                        tags=list(item.get("tags", [])),
                    )
                )
            except (KeyError, TypeError) as exc:
                logger.warning(
                    "Skipping malformed sample at index %d: %s", idx, exc
                )

        if not samples:
            raise LLMError("LLM produced zero valid eval samples.")

        # --- split into train / val ----------------------------------------
        split_idx = max(1, math.floor(len(samples) * config.train_val_split))
        context.train_set = samples[:split_idx]
        context.val_set = samples[split_idx:] if split_idx < len(samples) else []

        logger.info(
            "Round %d: %d train / %d val samples ready",
            context.round_num,
            len(context.train_set),
            len(context.val_set),
        )
        return context
