# Customer Service Agent — Nuwa 10-Round Training Demo

This demo trains a customer service chatbot with Nuwa for 10 rounds and produces before/after comparison metrics.

## Prerequisites

- Python 3.11+
- Nuwa installed: `pip install -e .` (from project root)
- DeepSeek API key: [platform.deepseek.com](https://platform.deepseek.com/)

## Quick Start

```bash
# From project root
pip install -e .

# Set your DeepSeek API key
export DEEPSEEK_API_KEY="sk-..."

# Run the demo
cd examples/customer_service_demo
python train.py
```

## What It Does

1. **Baseline evaluation** — 10 fixed customer service questions evaluated against the initial (mediocre) system prompt
2. **10-round Nuwa training** — Nuwa automatically generates test data, evaluates, reflects on failures, and mutates the config
3. **Post-training evaluation** — Same 10 questions evaluated against the optimized config
4. **Before/after report** — Per-question score comparison saved to `results/`

## Expected Output

```
  Before vs After
  ──────────────────────────────────────────────────
  平均分        0.4500      0.7200      +0.2700 (+60.0%)

  Q1     0.3500      0.7800      +0.4300 ↑
  Q2     0.4000      0.8500      +0.4500 ↑
  ...
```

Results are saved to:
- `results/run_YYYYMMDD_HHMMSS.json` — Full JSON report
- `results/REPORT.md` — Markdown case report for GitHub showcase

## Runtime

Expect 8-15 minutes depending on API latency (10 rounds x 10 samples + 2 evaluations x 10 questions).
