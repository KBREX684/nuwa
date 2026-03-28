"""DeepSeek E2E Demo: 用女娲训练一个客服问答智能体

完整案例：展示如何用 Nuwa + DeepSeek API 自动优化智能体的 system_prompt。

运行方式:
    pip install -e .
    export DEEPSEEK_API_KEY="sk-..."
    python train_with_deepseek.py

预计耗时: 2-4 分钟（2 轮 x 5 样本）
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# 1. 定义目标智能体 — 一个调用 DeepSeek 的客服 Bot
# ---------------------------------------------------------------------------

# 可被 Nuwa 训练的配置参数
_current_config: dict[str, Any] = {
    "system_prompt": "你是一个客服助手，请回答用户的问题。",
}


async def call_deepseek(system_prompt: str, user_input: str) -> str:
    """通过 LiteLLM 调用 DeepSeek API 生成回复。"""
    import litellm

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    response = await litellm.acompletion(
        model="deepseek/deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
        api_key=api_key,
        temperature=0.7,
        max_tokens=512,
    )
    return response.choices[0].message.content or ""


async def customer_service_agent(user_input: str, config: dict[str, Any] | None = None) -> str:
    """可被 Nuwa 训练的客服智能体。

    Nuwa 会通过 config 参数注入优化后的配置（如 system_prompt）。
    """
    global _current_config
    if config:
        _current_config = config

    prompt = _current_config.get("system_prompt", "你是一个客服助手。")
    return await call_deepseek(prompt, user_input)


# ---------------------------------------------------------------------------
# 2. 训练主流程
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path(__file__).parent / "results"


async def main():
    print("=" * 60)
    print("  女娲 Nuwa × DeepSeek — 端到端训练演示")
    print("=" * 60)

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("错误: 请设置 DEEPSEEK_API_KEY 环境变量")
        print("  export DEEPSEEK_API_KEY='sk-...'")
        sys.exit(1)

    print(f"\nAPI Key: sk-...{api_key[-4:]}")
    print(f"时间: {datetime.now(UTC).isoformat()}")
    print(f"输出目录: {OUTPUT_DIR}")

    import nuwa

    # ── 训练配置 ──────────────────────────────────────────
    training_direction = (
        "优化这个客服智能体的回复质量。目标：\n"
        "1. 回答要准确、有条理，给出具体可操作的建议\n"
        "2. 语气要友好、专业，体现服务意识\n"
        "3. 遇到模糊问题时主动追问确认用户需求\n"
        "4. 对无法解决的问题给出替代方案或转人工建议"
    )

    start_time = time.perf_counter()

    # ── 使用完整控制模式 ──────────────────────────────────
    trainer = nuwa.NuwaTrainer(
        agent=customer_service_agent,
        training_direction=training_direction,
        model="deepseek/deepseek-chat",
        llm_api_key=api_key,
        llm_base_url="https://api.deepseek.com",
        max_rounds=2,
        samples_per_round=5,
        train_val_split=0.6,
        sandbox=True,
        verbose=True,
        on_round_end=_on_round_end,
    )

    print("\n[开始训练] max_rounds=2, samples_per_round=5\n")

    # 运行训练
    result = await trainer.run()

    elapsed = time.perf_counter() - start_time

    # ── 输出结果 ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  训练完成")
    print("=" * 60)
    total_rounds = len(result.rounds)
    print(f"  总轮次:   {total_rounds}")
    print(f"  最佳轮次: R{result.best_round}")
    print(f"  最佳验证分: {result.best_val_score:.4f}")
    print(f"  停止原因: {result.stop_reason}")
    print(f"  耗时:     {elapsed:.1f}s")

    # 每轮详情
    if result.rounds:
        print(f"\n  {'Round':<8} {'Train Score':<14} {'Val Score':<12} {'Mutation Applied'}")
        print("  " + "-" * 55)
        for r in result.rounds:
            ts = r.train_scores.mean_score if r.train_scores else 0.0
            vs = r.val_scores.mean_score if r.val_scores else 0.0
            mut = "Yes" if r.mutation else "None"
            print(f"  R{r.round_num:<7} {ts:<14.4f} {vs:<12.4f} {mut}")

    # 最终配置
    if result.final_config:
        print(f"\n  最终配置:")
        for k, v in result.final_config.items():
            val_str = str(v)[:120] + "..." if len(str(v)) > 120 else str(v)
            print(f"    {k}: {val_str}")

    # ── 决定是否采纳 ──────────────────────────────────────
    if result.best_val_score >= 0.6:
        promoted = trainer.promote()
        print(f"\n  ✓ 已采纳优化配置 (val_score={result.best_val_score:.4f} >= 0.6)")
    else:
        trainer.discard()
        print(f"\n  ✗ 分数不够理想 (val_score={result.best_val_score:.4f} < 0.6)，已丢弃")

    # ── 保存结果到文件 ────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = _build_report(result, elapsed)
    report_path = OUTPUT_DIR / f"run_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  结果已保存到: {report_path}")

    # Markdown 摘要
    md_path = OUTPUT_DIR / "README.md"
    md_path.write_text(_build_markdown(report), encoding="utf-8")
    print(f"  案例报告已保存到: {md_path}")

    return result


def _on_round_end(round_result: Any, context: Any) -> None:
    """每轮结束回调 — 打印进度。"""
    r = round_result.round_num
    ts = round_result.train_scores.mean_score if round_result.train_scores else 0.0
    vs = round_result.val_scores.mean_score if round_result.val_scores else 0.0
    print(f"  [回调] Round {r} done: train={ts:.3f} val={vs:.3f}")


def _build_report(result: Any, elapsed: float) -> dict[str, Any]:
    """构建 JSON 报告。"""
    rounds_data = []
    for r in (result.rounds or []):
        rd: dict[str, Any] = {
            "round_num": r.round_num,
            "train_mean": r.train_scores.mean_score if r.train_scores else 0.0,
            "val_mean": r.val_scores.mean_score if r.val_scores else 0.0,
            "train_pass_rate": r.train_scores.pass_rate if r.train_scores else 0.0,
            "val_pass_rate": r.val_scores.pass_rate if r.val_scores else 0.0,
        }
        if r.reflection:
            rd["reflection"] = {
                "diagnosis": r.reflection.diagnosis[:300],
                "failure_patterns": r.reflection.failure_patterns[:5],
                "proposed_changes": r.reflection.proposed_changes[:5],
                "priority": r.reflection.priority,
            }
        if r.mutation:
            rd["mutation"] = {
                "rationale": r.mutation.rationale[:200] if r.mutation.rationale else "",
                "changes": r.mutation.changes,
            }
        rounds_data.append(rd)

    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "model": "deepseek/deepseek-chat",
        "provider": "DeepSeek",
        "total_rounds": len(result.rounds),
        "best_round": result.best_round,
        "best_val_score": result.best_val_score,
        "stop_reason": result.stop_reason,
        "elapsed_seconds": round(elapsed, 1),
        "final_config": result.final_config,
        "rounds": rounds_data,
    }


def _build_markdown(report: dict[str, Any]) -> str:
    """构建 Markdown 案例报告。"""
    lines = [
        "# DeepSeek × Nuwa 训练案例报告\n",
        f"**时间**: {report['timestamp']}",
        f"**模型**: {report['model']} ({report['provider']})",
        f"**轮次**: {report['total_rounds']}",
        f"**最佳验证分**: {report['best_val_score']:.4f}",
        f"**耗时**: {report['elapsed_seconds']}s",
        f"**停止原因**: {report['stop_reason']}\n",
        "## 每轮详情\n",
        "| Round | Train Score | Val Score | Train Pass% | Val Pass% | Mutation |",
        "|-------|-------------|-----------|-------------|-----------|----------|",
    ]
    for r in report["rounds"]:
        mut = "Yes" if "mutation" in r else "None"
        lines.append(
            f"| R{r['round_num']} | {r['train_mean']:.4f} | {r['val_mean']:.4f} "
            f"| {r['train_pass_rate']:.2%} | {r['val_pass_rate']:.2%} | {mut} |"
        )

    # 反思详情
    for r in report["rounds"]:
        if "reflection" in r:
            ref = r["reflection"]
            lines.append(f"\n### R{r['round_num']} 反思\n")
            lines.append(f"**诊断**: {ref['diagnosis']}\n")
            if ref["failure_patterns"]:
                lines.append("**失败模式**:")
                for fp in ref["failure_patterns"]:
                    lines.append(f"- {fp}")
            if ref["proposed_changes"]:
                lines.append("\n**改进建议**:")
                for pc in ref["proposed_changes"]:
                    lines.append(f"- {pc}")
        if "mutation" in r:
            mut = r["mutation"]
            lines.append(f"\n### R{r['round_num']} 变异\n")
            lines.append(f"**理由**: {mut['rationale']}")
            if mut["changes"]:
                lines.append(f"\n**变更**:")
                lines.append(f"```json")
                lines.append(json.dumps(mut["changes"], ensure_ascii=False, indent=2)[:500])
                lines.append(f"```")

    # 最终配置
    if report["final_config"]:
        lines.append(f"\n## 最终配置\n")
        lines.append(f"```json")
        lines.append(json.dumps(report["final_config"], ensure_ascii=False, indent=2))
        lines.append(f"```")

    lines.append(f"\n---\n*Generated by Nuwa Trainer × DeepSeek*")
    return "\n".join(lines)


if __name__ == "__main__":
    asyncio.run(main())
