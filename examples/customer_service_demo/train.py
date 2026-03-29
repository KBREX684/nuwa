"""Customer Service Demo — 10-round Nuwa training with before/after metrics.

This script demonstrates the full Nuwa training cycle:
1. Baseline evaluation (before training)
2. 10-round training with NuwaTrainer
3. Post-training evaluation (after training)
4. Before/after comparison report

Usage:
    pip install -e .
    export DEEPSEEK_API_KEY="sk-..."
    python train.py

Expected runtime: 8-15 minutes (10 rounds x 10 samples)
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
# Constants
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path(__file__).parent / "results"
MODEL = "deepseek/deepseek-chat"
MAX_ROUNDS = 10
SAMPLES_PER_ROUND = 10

# Fixed evaluation questions for before/after comparison
EVAL_QUESTIONS: list[dict[str, str]] = [
    {
        "input": "我买的手机壳想退货，已经签收5天了，可以退吗？",
        "expected": "可以退货。签收后7天内均可无理由退货，您签收5天还在退货期内。请保持商品完好，在订单详情页申请退货即可。退货运费需您自行承担。",
    },
    {
        "input": "这个电饭煲坏了，我才买了8个月，你们管修吗？",
        "expected": "在保修范围内。家电保修期为3年，您购买8个月仍在保修期内。如为非人为损坏，可免费维修。请联系客服申请保修服务。",
    },
    {
        "input": "我想用花呗分期买一台2800元的平板，可以吗？",
        "expected": "可以。订单金额满200元即可享花呗分期。2800元支持花呗6期免息。您在支付时选择花呗分期即可。",
    },
    {
        "input": "你们客服几点下班？周末有人吗？",
        "expected": "在线客服时间为9:00-22:00（含周末），电话客服为工作日9:00-18:00。周末可使用在线客服。",
    },
    {
        "input": "我上周买的鞋子想换个颜色，怎么操作？",
        "expected": "签收后15天内可申请换货，支持同款不同规格换货。请在订单详情页申请换货，选择您需要的颜色。换货运费由卖家承担。",
    },
    {
        "input": "金牌会员有什么好处？",
        "expected": "金牌会员享有：全场95折优惠、全场免运费、订单优先发货。可通过累计消费或购买年费会员获得金牌资格。",
    },
    {
        "input": "你们的服务太差了，我要投诉！",
        "expected": "非常抱歉给您带来不好的体验。您可以通过以下方式投诉：发送邮件至 complaint@example.com 或致电 400-765-4321。我们会认真处理您的反馈。",
    },
    {
        "input": "我的快递到哪了？单号是SF1234567890",
        "expected": "您可以通过以下方式查询物流：在订单详情页查看物流信息，或拨打物流客服电话 400-123-4567 查询。请提供您的订单号以便查询。",
    },
    {
        "input": "你们支持哪些支付方式？",
        "expected": "支持支付宝、微信支付、银行卡和花呗分期。订单金额满200元可享花呗6期免息。",
    },
    {
        "input": "我想买个电脑，但是不知道选哪个好，你们能推荐吗？",
        "expected": "为了给您更好的推荐，请问您的主要用途是什么？比如办公、游戏、设计？另外您的预算大概是多少？这样我可以帮您缩小选择范围。",
    },
]

TRAINING_DIRECTION = (
    "优化这个客服智能体的回复质量。具体目标：\n"
    "1. 回答准确，引用具体的公司政策信息（退货期限、保修期等）\n"
    "2. 回答有条理，给出清晰的操作步骤\n"
    "3. 语气友好专业，体现服务意识\n"
    "4. 遇到模糊问题时主动追问确认用户需求\n"
    "5. 对无法解决的问题给出替代方案或引导投诉渠道"
)


# ---------------------------------------------------------------------------
# Baseline / Post-training evaluation
# ---------------------------------------------------------------------------


async def evaluate_agent(
    eval_questions: list[dict[str, str]],
    agent_func: Any,
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run evaluation questions through the agent and collect responses."""
    results: list[dict[str, Any]] = []
    for q in eval_questions:
        try:
            response = await agent_func(q["input"], config=config)
            results.append(
                {
                    "input": q["input"],
                    "expected": q["expected"],
                    "actual": response,
                }
            )
        except Exception as e:
            results.append(
                {
                    "input": q["input"],
                    "expected": q["expected"],
                    "actual": f"[ERROR] {e}",
                }
            )
    return results


async def score_with_llm(
    question: str,
    expected: str,
    actual: str,
    api_key: str,
    base_url: str,
) -> float:
    """Use LLM to score a single response (0.0 - 1.0)."""
    import litellm

    prompt = (
        "你是一个专业的客服质量评估专家。请对以下客服回答进行评分。\n\n"
        f"用户问题: {question}\n\n"
        f"参考答案: {expected}\n\n"
        f"客服实际回答: {actual}\n\n"
        "评分标准:\n"
        "- 准确性: 是否包含正确的政策信息\n"
        "- 完整性: 是否覆盖了所有要点\n"
        "- 态度: 是否友好专业\n"
        "- 可操作性: 是否给出了具体步骤\n\n"
        "请只返回一个0到1之间的浮点数分数，不要返回其他内容。"
    )

    try:
        response = await litellm.acompletion(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            api_key=api_key,
            api_base=base_url,
            temperature=0.1,
            max_tokens=10,
        )
        text = response.choices[0].message.content or "0.5"
        # Extract float from response
        text = text.strip().replace("\n", "")
        try:
            return max(0.0, min(1.0, float(text)))
        except ValueError:
            # Try to extract number
            import re
            match = re.search(r"[\d.]+", text)
            if match:
                return max(0.0, min(1.0, float(match.group())))
            return 0.5
    except Exception:
        return 0.5


async def run_full_evaluation(
    eval_results: list[dict[str, Any]],
    api_key: str,
    base_url: str,
) -> list[float]:
    """Score all evaluation results using the LLM."""
    scores: list[float] = []
    for r in eval_results:
        score = await score_with_llm(
            r["input"], r["expected"], r["actual"], api_key, base_url
        )
        scores.append(score)
        print(f"  Q{len(scores):2d}: {score:.3f} — {r['input'][:40]}...")
    return scores


# ---------------------------------------------------------------------------
# Main training flow
# ---------------------------------------------------------------------------


async def main() -> None:
    print("=" * 64)
    print("  女娲 Nuwa — 客服智能体 10 轮训练演示")
    print("  Customer Service Agent — 10-Round Training Demo")
    print("=" * 64)

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("\n错误: 请设置 DEEPSEEK_API_KEY 环境变量")
        print("  export DEEPSEEK_API_KEY='sk-...'")
        sys.exit(1)

    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    print(f"\nAPI Key: sk-...{api_key[-4:]}")
    print(f"Base URL: {base_url}")
    print(f"训练轮次: {MAX_ROUNDS}")
    print(f"每轮样本: {SAMPLES_PER_ROUND}")
    print(f"时间: {datetime.now(UTC).isoformat()}")

    from agent import customer_service_agent

    # ==================================================================
    # PHASE 1: Baseline evaluation
    # ==================================================================
    print("\n" + "─" * 64)
    print("  PHASE 1: 基线评估 (Baseline Evaluation)")
    print("─" * 64)

    baseline_results = await evaluate_agent(EVAL_QUESTIONS, customer_service_agent)
    print("\n基线评估打分中...")
    baseline_scores = await run_full_evaluation(baseline_results, api_key, base_url)
    baseline_mean = sum(baseline_scores) / len(baseline_scores)
    print(f"\n基线平均分: {baseline_mean:.4f}")

    # ==================================================================
    # PHASE 2: Nuwa Training (10 rounds)
    # ==================================================================
    print("\n" + "─" * 64)
    print(f"  PHASE 2: Nuwa 训练 ({MAX_ROUNDS} 轮)")
    print("─" * 64)

    import nuwa

    start_time = time.perf_counter()

    trainer = nuwa.NuwaTrainer(
        agent=customer_service_agent,
        training_direction=TRAINING_DIRECTION,
        model=MODEL,
        llm_api_key=api_key,
        llm_base_url=base_url,
        max_rounds=MAX_ROUNDS,
        samples_per_round=SAMPLES_PER_ROUND,
        train_val_split=0.7,
        sandbox=True,
        verbose=True,
        on_round_end=_on_round_end,
    )

    print(f"\n[开始训练] {MAX_ROUNDS} 轮, 每轮 {SAMPLES_PER_ROUND} 样本\n")

    result = await trainer.run()
    elapsed = time.perf_counter() - start_time

    # Training summary
    print("\n" + "=" * 64)
    print("  训练完成")
    print("=" * 64)
    total_rounds = len(result.rounds)
    print(f"  总轮次:     {total_rounds}")
    print(f"  最佳轮次:   R{result.best_round}")
    print(f"  最佳验证分: {result.best_val_score:.4f}")
    print(f"  停止原因:   {result.stop_reason}")
    print(f"  耗时:       {elapsed:.1f}s ({elapsed / 60:.1f}min)")

    if result.rounds:
        print(f"\n  {'Round':<8} {'Train':<10} {'Val':<10} {'Applied':<10} {'Mutation'}")
        print("  " + "-" * 65)
        for r in result.rounds:
            ts = r.train_scores.mean_score if r.train_scores else 0.0
            vs = r.val_scores.mean_score if r.val_scores else 0.0
            applied = "Yes" if r.applied else "No"
            mut = r.mutation.description[:40] if r.mutation else "None"
            print(f"  R{r.round_num:<7} {ts:<10.4f} {vs:<10.4f} {applied:<10} {mut}")

    # ==================================================================
    # PHASE 3: Post-training evaluation (with optimized config)
    # ==================================================================
    print("\n" + "─" * 64)
    print("  PHASE 3: 训练后评估 (Post-Training Evaluation)")
    print("─" * 64)

    if result.best_val_score >= 0.5:
        promoted = trainer.promote()
        print(f"\n已采纳优化配置 (val_score={result.best_val_score:.4f})")
        print(f"优化后 system_prompt (前120字): {str(promoted.get('system_prompt', ''))[:120]}...")
    else:
        print("\n分数不够理想，仍使用原配置进行对比")
        trainer.discard()

    # Evaluate with optimized config
    post_results = await evaluate_agent(EVAL_QUESTIONS, customer_service_agent)
    print("\n训练后评估打分中...")
    post_scores = await run_full_evaluation(post_results, api_key, base_url)
    post_mean = sum(post_scores) / len(post_scores)
    print(f"\n训练后平均分: {post_mean:.4f}")

    # ==================================================================
    # PHASE 4: Before/After comparison
    # ==================================================================
    print("\n" + "=" * 64)
    print("  训练前后对比 (Before vs After)")
    print("=" * 64)

    improvement = post_mean - baseline_mean
    improvement_pct = (improvement / baseline_mean * 100) if baseline_mean > 0 else 0

    print(f"\n  {'指标':<20} {'训练前':<12} {'训练后':<12} {'变化'}")
    print("  " + "-" * 55)
    print(f"  {'平均分':<20} {baseline_mean:<12.4f} {post_mean:<12.4f} {improvement:+.4f} ({improvement_pct:+.1f}%)")

    print(f"\n  {'问题':<6} {'训练前':<10} {'训练后':<10} {'变化'}")
    print("  " + "-" * 45)
    for i, (b, a) in enumerate(zip(baseline_scores, post_scores)):
        diff = a - b
        marker = "↑" if diff > 0.01 else ("↓" if diff < -0.01 else "→")
        print(f"  Q{i + 1:<5} {b:<10.4f} {a:<10.4f} {diff:+.4f} {marker}")

    # ==================================================================
    # Save results
    # ==================================================================
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report = _build_report(
        baseline_scores=baseline_scores,
        post_scores=post_scores,
        baseline_results=baseline_results,
        post_results=post_results,
        result=result,
        elapsed=elapsed,
    )

    report_path = OUTPUT_DIR / f"run_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存到: {report_path}")

    md_path = OUTPUT_DIR / "REPORT.md"
    md_path.write_text(_build_markdown(report), encoding="utf-8")
    print(f"案例报告已保存到: {md_path}")

    return result


def _on_round_end(round_result: Any, context: Any) -> None:
    """Per-round callback — print progress."""
    r = round_result.round_num
    ts = round_result.train_scores.mean_score if round_result.train_scores else 0.0
    vs = round_result.val_scores.mean_score if round_result.val_scores else 0.0
    applied = "Applied" if round_result.applied else "Rolled back"
    reflection = ""
    if round_result.reflection:
        reflection = round_result.reflection.diagnosis[:60]
    print(f"  [R{r:2d}] train={ts:.3f} val={vs:.3f} {applied} — {reflection}...")


def _build_report(
    baseline_scores: list[float],
    post_scores: list[float],
    baseline_results: list[dict[str, Any]],
    post_results: list[dict[str, Any]],
    result: Any,
    elapsed: float,
) -> dict[str, Any]:
    """Build a comprehensive JSON report."""
    baseline_mean = sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0
    post_mean = sum(post_scores) / len(post_scores) if post_scores else 0

    rounds_data: list[dict[str, Any]] = []
    for r in (result.rounds or []):
        rd: dict[str, Any] = {
            "round_num": r.round_num,
            "train_mean": r.train_scores.mean_score if r.train_scores else 0.0,
            "val_mean": r.val_scores.mean_score if r.val_scores else 0.0,
            "train_pass_rate": r.train_scores.pass_rate if r.train_scores else 0.0,
            "val_pass_rate": r.val_scores.pass_rate if r.val_scores else 0.0,
            "applied": r.applied,
        }
        if r.reflection:
            rd["reflection"] = {
                "diagnosis": r.reflection.diagnosis[:500],
                "failure_patterns": r.reflection.failure_patterns[:5],
                "proposed_changes": r.reflection.proposed_changes[:5],
                "priority": r.reflection.priority,
            }
        if r.mutation:
            rd["mutation"] = {
                "description": r.mutation.description[:300],
                "rationale": r.mutation.rationale[:300] if r.mutation.rationale else "",
            }
        rounds_data.append(rd)

    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "model": MODEL,
        "max_rounds": MAX_ROUNDS,
        "samples_per_round": SAMPLES_PER_ROUND,
        "training_direction": TRAINING_DIRECTION,
        "baseline_mean_score": round(baseline_mean, 4),
        "post_training_mean_score": round(post_mean, 4),
        "improvement": round(post_mean - baseline_mean, 4),
        "improvement_pct": round(
            (post_mean - baseline_mean) / baseline_mean * 100 if baseline_mean > 0 else 0, 2
        ),
        "nuwa_best_val_score": result.best_val_score,
        "nuwa_best_round": result.best_round,
        "stop_reason": result.stop_reason,
        "elapsed_seconds": round(elapsed, 1),
        "final_config": result.final_config,
        "rounds": rounds_data,
        "per_question": [
            {
                "question": EVAL_QUESTIONS[i]["input"],
                "baseline_score": baseline_scores[i],
                "post_score": post_scores[i],
                "improvement": round(post_scores[i] - baseline_scores[i], 4),
            }
            for i in range(len(EVAL_QUESTIONS))
        ],
    }


def _build_markdown(report: dict[str, Any]) -> str:
    """Build a Markdown case report for GitHub showcase."""
    lines = [
        "# Nuwa 客服智能体训练案例报告",
        "",
        f"> **模型**: `{report['model']}` | **轮次**: {report['max_rounds']} | **每轮样本**: {report['samples_per_round']}",
        f"> **时间**: {report['timestamp']}",
        "",
        "## 训练目标",
        "",
        report["training_direction"],
        "",
        "## 训练前后对比 (Before vs After)",
        "",
        "| 指标 | 训练前 | 训练后 | 变化 |",
        "|------|--------|--------|------|",
        f"| **平均分** | {report['baseline_mean_score']:.4f} | {report['post_training_mean_score']:.4f} | {report['improvement']:+.4f} ({report['improvement_pct']:+.1f}%) |",
        f"| Nuwa 最佳验证分 | — | {report['nuwa_best_val_score']:.4f} | — |",
        "",
        "### 逐题对比",
        "",
        "| # | 问题 | 训练前 | 训练后 | 变化 |",
        "|---|------|--------|--------|------|",
    ]
    for q in report["per_question"]:
        question_short = q["question"][:30] + "..." if len(q["question"]) > 30 else q["question"]
        imp = q["improvement"]
        marker = "↑" if imp > 0.01 else ("↓" if imp < -0.01 else "→")
        lines.append(
            f"| {report['per_question'].index(q) + 1} | {question_short} "
            f"| {q['baseline_score']:.3f} | {q['post_score']:.3f} "
            f"| {imp:+.3f} {marker} |"
        )

    lines.extend(
        [
            "",
            "## Nuwa 训练过程",
            "",
            f"- **总轮次**: {len(report['rounds'])}",
            f"- **最佳轮次**: R{report['nuwa_best_round']}",
            f"- **停止原因**: {report['stop_reason']}",
            f"- **耗时**: {report['elapsed_seconds']}s",
            "",
            "| Round | Train Score | Val Score | Applied | Reflection |",
            "|-------|-------------|-----------|---------|------------|",
        ]
    )
    for r in report["rounds"]:
        applied = "Yes" if r["applied"] else "No"
        reflection = ""
        if "reflection" in r:
            reflection = r["reflection"]["diagnosis"][:50] + "..."
        lines.append(
            f"| R{r['round_num']} | {r['train_mean']:.4f} | {r['val_mean']:.4f} "
            f"| {applied} | {reflection} |"
        )

    if report["final_config"]:
        lines.extend(
            [
                "",
                "## 优化后配置",
                "",
                "```json",
                json.dumps(report["final_config"], ensure_ascii=False, indent=2),
                "```",
            ]
        )

    lines.extend(
        [
            "",
            "---",
            f"*Generated by Nuwa Trainer v0.3.0 — {report['timestamp']}*",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    asyncio.run(main())
