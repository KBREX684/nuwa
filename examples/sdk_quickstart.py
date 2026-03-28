"""女娲 Nuwa SDK 快速入门示例

展示如何用最少的代码将 Nuwa 集成到你的智能体项目中。

用法:
    pip install nuwa-trainer
    python sdk_quickstart.py
"""

from __future__ import annotations

import asyncio

import nuwa

# ============================================================
# 第一步：定义你的智能体函数
# ============================================================
# 用 @nuwa.trainable 标记你的智能体。
# 函数必须接受一个 config 参数（Nuwa 会通过它注入优化后的配置）。

@nuwa.trainable(name="客服Bot", description="一个简单的客服对话机器人")
def customer_service_bot(user_input: str, config: dict | None = None) -> str:
    """你的智能体逻辑。替换为你的真实实现。"""
    # 从 config 中读取可训练的参数
    cfg = config or {}
    system_prompt = cfg.get("system_prompt", "你是一个有帮助的客服助手。")
    temperature = cfg.get("temperature", 0.7)
    style = cfg.get("style", "professional")

    # 这里替换为你的真实 LLM 调用
    # response = openai.chat.completions.create(
    #     model="gpt-4o",
    #     messages=[
    #         {"role": "system", "content": system_prompt},
    #         {"role": "user", "content": user_input},
    #     ],
    #     temperature=temperature,
    # )
    # return response.choices[0].message.content

    # 示例：模拟返回
    return f"[{style}] 针对您的问题「{user_input}」，{system_prompt[:20]}..."


# ============================================================
# 用法一：一行代码启动训练（最简模式）
# ============================================================
def example_one_liner():
    """最简单的方式——一个函数调用搞定。"""
    print("\n=== 用法一：一行代码训练 ===")

    result = nuwa.train_sync(
        customer_service_bot,
        direction="提升客服回答的准确率和用户满意度，遇到模糊问题时主动确认需求",
        model="openai/gpt-4o",        # 替换为你的模型
        # api_key="sk-...",            # 替换为你的 API Key
        max_rounds=3,
    )

    print(f"训练完成！最佳验证分: {result.best_val_score:.3f}")
    print(f"停止原因: {result.stop_reason}")


# ============================================================
# 用法二：完整控制模式（推荐生产使用）
# ============================================================
async def example_full_control():
    """完整控制——创建 Trainer 实例，手动决定是否采纳。"""
    print("\n=== 用法二：完整控制模式 ===")

    # 创建训练器（沙箱默认开启）
    trainer = nuwa.NuwaTrainer(
        agent=customer_service_bot,
        direction="提升回答质量：更准确、更友好、遇到歧义主动澄清",
        model="openai/gpt-4o",
        # api_key="sk-...",
        max_rounds=5,
        samples_per_round=15,
        sandbox=True,                 # 沙箱模式（默认开启）
        on_round_end=on_round_end,    # 可选：每轮回调
    )

    # 运行训练
    result = await trainer.run()

    # 查看结果
    print(f"\n训练完成！共 {len(result.rounds)} 轮")
    print(f"最佳轮次: R{result.best_round}  最佳分: {result.best_val_score:.3f}")
    print(f"停止原因: {result.stop_reason}")

    # 决定是否采纳
    if result.best_val_score > 0.7:
        config = trainer.promote()    # 将最佳配置应用到真实智能体
        print(f"已采纳优化配置: {list(config.keys())}")
    else:
        trainer.discard()             # 丢弃所有更改
        print("分数不够理想，已丢弃更改")


def on_round_end(event: dict):
    """每轮结束时的回调——可以用来记录日志或发通知。"""
    r = event.get("round", "?")
    ts = event.get("train_score", 0)
    vs = event.get("val_score", 0)
    print(f"  [回调] 轮次 {r} 完成: 训练分={ts:.3f} 验证分={vs:.3f}")


# ============================================================
# 用法三：不用装饰器——直接传普通函数
# ============================================================
def example_plain_function():
    """不用 @trainable 也能用。直接传函数即可。"""
    print("\n=== 用法三：不用装饰器 ===")

    def my_simple_bot(query: str, config: dict | None = None) -> str:
        prompt = config.get("prompt", "你好") if config else "你好"
        return f"{prompt}: {query}"

    # 直接传给 train_sync
    result = nuwa.train_sync(
        my_simple_bot,
        direction="让回答更详细",
        max_rounds=2,
    )
    print(f"训练完成！分数: {result.best_val_score:.3f}")


# ============================================================
# 主函数
# ============================================================
if __name__ == "__main__":
    print("=" * 50)
    print("女娲 Nuwa SDK 快速入门")
    print("=" * 50)

    # 注意：以下示例需要有效的 LLM API Key 才能真正运行。
    # 没有 Key 的情况下会报错。可以先通过 Web UI 的"演示模式"体验。
    print("\n提示：运行以下示例需要配置 LLM API Key。")
    print("无 Key 体验请使用: nuwa train (交互模式) 或 Web UI 的演示模式。")
    print("\n如需运行，请取消下方注释：")
    print("  example_one_liner()")
    print("  asyncio.run(example_full_control())")
    print("  example_plain_function()")

    # 取消注释以运行：
    # example_one_liner()
    # asyncio.run(example_full_control())
    # example_plain_function()
