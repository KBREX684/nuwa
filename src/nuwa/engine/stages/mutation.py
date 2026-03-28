"""Mutation stage -- proposes and applies config/prompt changes."""

from __future__ import annotations

import json
import logging
from typing import Any

from jinja2 import Template

from nuwa.core.defaults import MAX_MUTATIONS_PER_PROPOSAL, TEMPERATURE_MUTATION
from nuwa.core.exceptions import LLMError
from nuwa.core.types import LoopContext, Mutation
from nuwa.llm.prompts import MUTATION_PROPOSAL
from nuwa.llm.response_parser import parse_json_response

logger = logging.getLogger(__name__)

_MUTATION_TEMPLATE = Template(MUTATION_PROPOSAL)


class MutationStage:
    """Generate targeted config/prompt mutations and apply them."""

    @property
    def name(self) -> str:
        return "mutation"

    async def execute(self, context: LoopContext) -> LoopContext:
        backend = context.backend_ref
        reflection = context.reflection

        if reflection is None:
            logger.warning(
                "Round %d: no reflection available, skipping mutation.", context.round_num
            )
            context.proposed_mutation = None
            return context

        # If the reflection reports no issues, skip mutation.
        if reflection.priority == "low" and not reflection.failure_patterns:
            logger.info(
                "Round %d: reflection is low priority with no failures, skipping mutation.",
                context.round_num,
            )
            context.proposed_mutation = None
            return context

        original_config = dict(context.current_config)
        current_prompt = original_config.get("system_prompt", "")
        config_json = json.dumps(original_config, indent=2, default=str)
        reflection_json = json.dumps(reflection.model_dump(), indent=2, default=str)

        prompt = _MUTATION_TEMPLATE.render(
            reflection_json=reflection_json,
            current_prompt=current_prompt,
            current_config=config_json,
            max_mutations=MAX_MUTATIONS_PER_PROPOSAL,
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "You are a prompt-engineering specialist."},
            {"role": "user", "content": prompt},
        ]

        logger.info("Round %d: generating mutation proposal", context.round_num)

        try:
            raw = await backend.complete(messages, temperature=TEMPERATURE_MUTATION)
            data = parse_json_response(raw)
            if not isinstance(data, dict):
                raise LLMError("Mutation response is not a JSON object.")

            mutations_list = data.get("mutations", [])
            if not mutations_list:
                logger.info("Round %d: LLM proposed no mutations.", context.round_num)
                context.proposed_mutation = None
                return context

            # Build the proposed config by applying mutations sequentially.
            proposed_config = dict(original_config)
            descriptions: list[str] = []
            rationales: list[str] = []

            for mut in mutations_list:
                mut_type = mut.get("type", "")
                desc = mut.get("description_en", "") or mut.get("description_zh", "")
                rationale = mut.get("rationale_en", "") or mut.get("rationale_zh", "")

                if desc:
                    descriptions.append(desc)
                if rationale:
                    rationales.append(rationale)

                if mut_type == "config_change":
                    config_path = mut.get("config_path", "")
                    config_value = mut.get("config_value")
                    if config_path and config_value is not None:
                        self._apply_config_path(proposed_config, config_path, config_value)

                elif mut_type in ("prompt_replace", "prompt_insert", "prompt_delete"):
                    current_sys_prompt = proposed_config.get("system_prompt", "")
                    old_text = mut.get("old_text", "")
                    new_text = mut.get("new_text", "")

                    if mut_type == "prompt_replace" and old_text:
                        proposed_config["system_prompt"] = current_sys_prompt.replace(
                            old_text, new_text, 1
                        )
                    elif mut_type == "prompt_insert" and new_text:
                        target_section = mut.get("target_section", "")
                        if target_section and target_section in current_sys_prompt:
                            idx = current_sys_prompt.index(target_section) + len(target_section)
                            proposed_config["system_prompt"] = (
                                current_sys_prompt[:idx]
                                + "\n"
                                + new_text
                                + current_sys_prompt[idx:]
                            )
                        else:
                            proposed_config["system_prompt"] = current_sys_prompt + "\n" + new_text
                    elif mut_type == "prompt_delete" and old_text:
                        proposed_config["system_prompt"] = current_sys_prompt.replace(
                            old_text, "", 1
                        )

            mutation = Mutation(
                description=" | ".join(descriptions) if descriptions else "LLM-proposed mutation",
                original_config=original_config,
                proposed_config=proposed_config,
                reasoning=" | ".join(rationales) if rationales else "See reflection for details.",
            )
            context.proposed_mutation = mutation

            logger.info(
                "Round %d: mutation proposed -- %d changes: %s",
                context.round_num,
                len(mutations_list),
                mutation.description[:200],
            )

        except (LLMError, KeyError, TypeError, ValueError) as exc:
            logger.error("Mutation stage failed: %s", exc)
            context.proposed_mutation = None

        return context

    # ------------------------------------------------------------------

    @staticmethod
    def _apply_config_path(config: dict[str, Any], path: str, value: Any) -> None:
        """Set a value in a nested dict using a dot-separated path.

        Raises ``ValueError`` if the path would overwrite a non-dict value
        at an intermediate key (which would destroy existing config data).
        """
        keys = path.split(".")
        obj = config
        for key in keys[:-1]:
            if key not in obj:
                obj[key] = {}
                obj = obj[key]
            elif isinstance(obj[key], dict):
                obj = obj[key]
            else:
                raise ValueError(
                    f"Cannot set '{path}': intermediate key '{key}' holds a "
                    f"{type(obj[key]).__name__}, not a dict. "
                    f"Refusing to overwrite."
                )
        obj[keys[-1]] = value
