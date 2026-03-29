"""Customer Service Agent — a trainable chatbot for the Nuwa demo.

This agent uses DeepSeek (via LiteLLM) to answer customer service questions.
It starts with a deliberately mediocre system prompt so Nuwa has room to
improve it through training.

Usage:
    This module is imported by train.py — you don't run it directly.
"""

from __future__ import annotations

import os
from typing import Any

# ---------------------------------------------------------------------------
# Mutable config — Nuwa mutates this via the config parameter
# ---------------------------------------------------------------------------
_current_config: dict[str, Any] = {
    "system_prompt": (
        "你是一个客服助手。请回答用户的问题。"
        # Intentionally vague — Nuwa should improve this
    ),
    "temperature": 0.7,
}

# ---------------------------------------------------------------------------
# Embedded knowledge base (so the agent can give factual answers)
# ---------------------------------------------------------------------------
KNOWLEDGE_BASE = """
退货政策: 商品签收后7天内可无理由退货，需保持商品完好且不影响二次销售。退货运费由买家承担。
换货政策: 商品签收后15天内可申请换货，仅支持同款不同规格换货。换货运费由卖家承担。
保修政策: 电子产品保修1年，家电保修3年，其他商品保修6个月。保修期内非人为损坏免费维修。
物流查询: 用户可通过订单详情页查看物流信息，或拨打物流客服电话 400-123-4567 查询。
支付方式: 支持支付宝、微信支付、银行卡、花呗分期。订单金额满200元可享花呗6期免息。
营业时间: 在线客服 9:00-22:00（含周末），电话客服 9:00-18:00（工作日）。
投诉渠道: 如对服务不满意可发送邮件至 complaint@example.com 或致电 400-765-4321。
会员权益: 金牌会员享95折优惠、免运费、优先发货。银牌会员享98折优惠。
"""


async def _call_llm(system_prompt: str, user_input: str, temperature: float = 0.7) -> str:
    """Call DeepSeek API via LiteLLM."""
    import litellm

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

    response = await litellm.acompletion(
        model="deepseek/deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
        api_key=api_key,
        api_base=base_url,
        temperature=temperature,
        max_tokens=512,
    )
    return response.choices[0].message.content or ""


async def customer_service_agent(user_input: str, config: dict[str, Any] | None = None) -> str:
    """Trainable customer service agent.

    Nuwa calls this function with different configs to evaluate and improve
    the agent's response quality.
    """
    global _current_config
    if config:
        _current_config = config

    prompt = _current_config.get(
        "system_prompt",
        "你是一个客服助手。请回答用户的问题。",
    )
    temperature = _current_config.get("temperature", 0.7)

    # Inject the knowledge base into the prompt if not already present
    if "退货政策" not in prompt:
        prompt = prompt + "\n\n以下是公司政策信息，供你参考回答用户问题:\n" + KNOWLEDGE_BASE.strip()

    return await _call_llm(prompt, user_input, temperature)


def get_current_config() -> dict[str, Any]:
    """Return the agent's current config (used by Nuwa)."""
    return dict(_current_config)


def apply_config(config: dict[str, Any]) -> None:
    """Apply a new config to the agent (used by Nuwa)."""
    global _current_config
    _current_config = dict(config)
