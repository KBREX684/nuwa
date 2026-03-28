"""示例：一个简单的客服聊天机器人 (Flask Mock)

运行方式:
    pip install flask openai
    export OPENAI_API_KEY="sk-..."
    python target_agent.py

此脚本启动一个本地 Flask 服务器，模拟一个可被 Nuwa 训练的客服聊天机器人。
Nuwa 会通过 HTTP API 与此 Agent 交互，评估其回答质量并自动优化。

连接 Nuwa 的方法:
    1. 启动本服务: python target_agent.py
    2. 在 nuwa_config.yaml 中设置:
       connector_type: "http"
       connector_params:
         url: "http://localhost:5000/chat"
    3. 运行: nuwa train --config nuwa_config.yaml
"""

from __future__ import annotations

import os
from typing import Any

from flask import Flask, jsonify, request

app = Flask(__name__)

# ---------------------------------------------------------------------------
# 系统提示词 -- Nuwa 可以通过 /config 端点动态修改此提示词来优化 Agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "你是一个友好、专业的客服聊天机器人。\n"
    "请用简洁、清晰的语言回答用户的问题。\n"
    "如果你不确定答案，请诚实地告知用户，并建议他们联系人工客服。\n"
    "始终保持礼貌和耐心的态度。"
)


def _get_chat_response(user_input: str) -> str:
    """Call OpenAI to generate a response using the current system prompt.

    Falls back to a simple echo response when the ``openai`` package is
    not installed or no API key is configured, so the example can still
    be used as a structural reference without a live LLM.
    """
    try:
        import openai

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return f"[Mock 回复] 收到您的消息：{user_input}。请设置 OPENAI_API_KEY 以启用 AI 回复。"

        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_input},
            ],
            temperature=0.7,
            max_tokens=512,
        )
        return response.choices[0].message.content or ""
    except ImportError:
        return f"[Mock 回复] 收到您的消息：{user_input}。请安装 openai 包以启用 AI 回复。"
    except Exception as exc:
        return f"[错误] 生成回复时出错：{exc}"


# ---------------------------------------------------------------------------
# HTTP 端点
# ---------------------------------------------------------------------------


@app.route("/chat", methods=["POST"])
def chat() -> tuple[Any, int]:
    """主聊天端点。

    Nuwa 会向此端点发送 POST 请求：
        请求体: {"input": "用户的问题"}
        响应体: {"output": "机器人的回答"}
    """
    data = request.get_json(silent=True) or {}
    user_input = data.get("input", "")

    if not user_input:
        return jsonify({"error": "缺少 'input' 字段"}), 400

    response_text = _get_chat_response(user_input)
    return jsonify({"output": response_text}), 200


@app.route("/config", methods=["POST"])
def update_config() -> tuple[Any, int]:
    """配置更新端点。

    Nuwa 可以通过此端点修改 Agent 的系统提示词：
        请求体: {"system_prompt": "新的系统提示词..."}
        响应体: {"status": "ok", "system_prompt": "..."}
    """
    global SYSTEM_PROMPT

    data = request.get_json(silent=True) or {}
    new_prompt = data.get("system_prompt")

    if new_prompt is not None:
        SYSTEM_PROMPT = new_prompt

    return jsonify({"status": "ok", "system_prompt": SYSTEM_PROMPT}), 200


@app.route("/config", methods=["GET"])
def get_config() -> tuple[Any, int]:
    """查询当前配置（调试用）。"""
    return jsonify({"system_prompt": SYSTEM_PROMPT}), 200


@app.route("/health", methods=["GET"])
def health() -> tuple[Any, int]:
    """健康检查端点。"""
    return jsonify({"status": "healthy"}), 200


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  客服聊天机器人 - Nuwa 训练目标 Agent")
    print("  POST /chat    - 聊天接口")
    print("  POST /config  - 更新系统提示词")
    print("  GET  /health  - 健康检查")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=True)
