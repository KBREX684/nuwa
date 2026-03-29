<div align="center">

# 女娲 Nuwa

**AI Agent Trainer — 让智能体自动进化**

[![CI](https://github.com/KBREX684/nuwa/actions/workflows/ci.yml/badge.svg)](https://github.com/KBREX684/nuwa/actions)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Version](https://img.shields.io/badge/version-0.2.1-brightgreen)](https://github.com/KBREX684/nuwa/releases)
[![License: BSL 1.1](https://img.shields.io/badge/License-BSL%201.1-blue.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-KBREX684%2Fnuwa-black?logo=github)](https://github.com/KBREX684/nuwa)

[中文](#什么是女娲) | [English](#what-is-nuwa) | [API 文档](API.md) | [故障排查](TROUBLESHOOTING.md) | [安全策略](SECURITY.md)

</div>

---

## 目录 / Table of Contents

- [什么是女娲 / What is Nuwa](#什么是女娲--what-is-nuwa)
- [核心特性 / Key Features](#核心特性--key-features)
- [架构 / Architecture](#架构--architecture)
- [快速开始 / Quick Start](#快速开始--quick-start)
- [沙箱隔离 / Sandbox Isolation](#沙箱隔离--sandbox-isolation)
- [配置 / Configuration](#配置--configuration)
- [接入你的智能体 / Connecting Your Agent](#接入你的智能体--connecting-your-agent)
- [工作原理 / How It Works](#工作原理--how-it-works)
- [安全护栏 / Guardrails](#安全护栏--guardrails)
- [项目结构 / Project Structure](#项目结构--project-structure)
- [文档 / Documentation](#文档--documentation)
- [开发 / Development](#开发--development)
- [路线图 / Roadmap](#路线图--roadmap)
- [许可证 / License](#许可证--license)

---

## 什么是女娲 / What is Nuwa

女娲（Nuwa）是一个 **AI 智能体训练器**——一个用来优化其他 AI 智能体的元智能体。它通过自动化的数据集生成、执行评估、反思总结和提示词变异循环，持续优化目标智能体的 Prompt 与配置参数，无需人工反复调试。

灵感来源于 [AutoResearch](https://github.com/autoresearch) 等自动化科研框架的理念：将"调优智能体"这件事本身交给智能体来完成。

<details><summary>English</summary>

Nuwa is an **AI Agent Trainer** — a meta-agent that optimizes other AI agents. It automatically generates evaluation datasets, runs your agent, scores the outputs, reflects on failure patterns, and mutates prompts and configurations to improve performance. No more manual prompt engineering.

Inspired by automated research frameworks: let an agent handle the work of tuning agents.

</details>

## 核心特性 / Key Features

- **Closed-loop training / 闭环训练**
  Dataset Gen → Execution → Evaluation → Reflection → Mutation → Validation
- **Multi-LLM support / 多模型支持**
  OpenAI, Anthropic, DeepSeek, Ollama, 以及任何 LiteLLM 兼容的模型提供商
- **Flexible connectors / 灵活连接器**
  HTTP API、CLI 子进程、Python 函数调用——三种方式接入你的智能体
- **Guardrails / 安全护栏**
  过拟合检测、回归预防、一致性校验，确保每一轮优化都是真实进步
- **Sandbox isolation / 沙箱隔离**
  训练全程在沙箱中运行，真实智能体的配置不会被修改，仅在人工审批后生效
- **Python SDK / 编程接口**
  `@nuwa.trainable` 装饰器 + `nuwa.train()` 一行代码启动训练，零配置文件集成
- **Web UI / 浏览器控制台**
  实时训练监控面板，SSE 推送，演示模式无需 API Key
- **Interactive CLI / 交互式命令行**
  对话式引导配置，开箱即用
- **Full audit trail / 完整审计日志**
  JSONL 日志 + 每轮配置快照，训练过程完全可追溯
- **Human-in-the-loop / 人机协同**
  全自动训练，最终人工审批确认
- **Parallel evaluation / 并行评估**
  多智能体并行执行 + 多评委集成评分（Mean/Median/Weighted/Majority/Min），可配置并发度
- **Multi-objective optimization / 多目标优化**
  Pareto 前沿跟踪，支持安全性、准确性、友好度等多维度联合优化
- **Circuit breaker / 熔断器**
  LLM API 连续失败时自动熔断，防止雪崩效应
- **Rate limiting / 速率限制**
  Web API 内置请求频率限制，防止滥用

## 架构 / Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        女娲 Nuwa Training Loop                  │
│                                                                 │
│   ┌───────────┐    ┌───────────┐    ┌───────────┐              │
│   │ 1. Dataset │───▶│ 2. Execute│───▶│ 3. Evaluate│             │
│   │    Gen     │    │   Agent   │    │   Output  │              │
│   └───────────┘    └───────────┘    └─────┬─────┘              │
│         ▲                                 │                     │
│         │                                 ▼                     │
│   ┌───────────┐    ┌───────────┐    ┌───────────┐              │
│   │ 6. Validate│◀──│ 5. Mutate │◀──│ 4. Reflect │              │
│   │  & Accept │    │  Prompt   │    │  on Errors│              │
│   └───────────┘    └───────────┘    └───────────┘              │
│                                                                 │
│   ╔═══════════════════════════════════════════════════════════╗ │
│   ║  Guardrails: Overfitting ∙ Regression ∙ Consistency      ║ │
│   ║  Circuit Breaker ∙ Rate Limiting                         ║ │
│   ╚═══════════════════════════════════════════════════════════╝ │
└─────────────────────────────────────────────────────────────────┘
```

## 快速开始 / Quick Start

### 30 秒体验 / 30-Second Demo

无需 API Key，直接在浏览器中体验完整训练流程：

```bash
pip install git+https://github.com/KBREX684/nuwa.git
nuwa web
# 浏览器打开 http://localhost:9090 → 点击「演示模式」
```

### 环境要求 / Requirements

- Python 3.11+
- 至少一个 LLM API Key（OpenAI / Anthropic / DeepSeek / Ollama 本地无需 Key）

### 安装 / Install

```bash
# 从 GitHub 安装
pip install git+https://github.com/KBREX684/nuwa.git

# 或克隆后本地安装
git clone https://github.com/KBREX684/nuwa.git
cd nuwa
pip install -e .
```

Web UI 需要额外依赖：

```bash
pip install 'nuwa-trainer[web]'
```

### 环境配置 / Environment

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env，至少配置一个 LLM API Key：
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...
# DEEPSEEK_API_KEY=sk-...
```

### 交互模式 / Interactive Mode

```bash
nuwa train
```

### 无头模式 / Headless Mode

```bash
nuwa run --config config.yaml
```

### Python SDK（推荐）

```python
import nuwa

@nuwa.trainable(name="客服Bot")
def my_agent(user_input: str, config: dict | None = None) -> str:
    """最简示例：返回配置中的 prompt + 用户输入"""
    prompt = config.get("system_prompt", "你是助手") if config else "你是助手"
    return f"[{prompt}] 收到: {user_input}"

result = nuwa.train_sync(
    my_agent,
    direction="提升回答准确率和用户满意度",
    model="deepseek/deepseek-chat",
    max_rounds=5,
)

print(f"最佳分数: {result.best_val_score:.3f}")
```

### 完整控制模式

```python
import asyncio
import nuwa

# my_agent 通过 @nuwa.trainable 定义（见上方 SDK 示例）

async def main():
    trainer = nuwa.NuwaTrainer(
        agent=my_agent,
        direction="让回答更准确、更友好",
        model="openai/gpt-4o",
        max_rounds=5,
        sandbox=True,
    )
    result = await trainer.run()

    if result.best_val_score > 0.8:
        trainer.promote()  # 应用最佳配置
    else:
        trainer.discard()  # 保持原样

asyncio.run(main())
```

### Web UI 控制台

```bash
nuwa web  # http://localhost:9090
```

详细 API 文档请参阅 [API.md](API.md)。

## 沙箱隔离 / Sandbox Isolation

```
┌──────────────────────────────────────────────┐
│  沙箱 (Sandbox)                               │
│  ┌─────────────┐    ┌──────────────────────┐ │
│  │ 配置副本     │───▶│  训练循环             │ │
│  │ (deep copy) │◀──│  所有变异在此执行       │ │
│  └─────────────┘    └──────────────────────┘ │
└──────────────────────┬───────────────────────┘
                       │ 人工审批通过后
                       ▼
┌──────────────────────────────────────────────┐
│  真实智能体 (训练期间完全不受影响)              │
│  promote() ──▶ 应用最佳配置                    │
│  discard() ──▶ 保持原样                        │
└──────────────────────────────────────────────┘
```

## 配置 / Configuration

```yaml
llm_model: deepseek/deepseek-chat
llm_api_key: ${DEEPSEEK_API_KEY}
llm_temperature: 0.7

connector_type: http
connector_params:
  url: "http://localhost:8000/chat"
  method: POST
  input_field: input
  output_field: output
  timeout: 30

training_direction: "优化客服智能体的回答质量"
max_rounds: 10
samples_per_round: 20
train_val_split: 0.7
overfitting_threshold: 0.15
regression_tolerance: 0.05
consistency_threshold: 0.8
consistency_runs: 3
project_dir: .nuwa
```

> **License note:** BSL 1.1 — free for development and evaluation. [Commercial license](LICENSE) required for production use.

## 接入你的智能体 / Connecting Your Agent

### HTTP API

```yaml
connector_type: http
connector_params:
  url: "http://localhost:8000/chat"
  method: POST
  headers:
    Authorization: "Bearer ${AGENT_API_KEY}"
  input_field: input
  output_field: reply
  timeout: 30
```

### CLI Subprocess

```yaml
connector_type: cli
connector_params:
  command: python
  args:
    - my_agent.py
  input_mode: stdin
  timeout: 60
  config_file: ./agent/config.yaml
```

### Python Function

```python
import nuwa

@nuwa.trainable(name="my_bot")
def my_agent(user_input: str, config: dict | None = None) -> str:
    return "response"
```

## 工作原理 / How It Works

| Stage | 阶段 | Description |
|-------|------|-------------|
| **1. Dataset Gen** | 数据集生成 | LLM 根据目标智能体的描述和历史表现，生成多样化的测试用例 |
| **2. Execute** | 执行 | 将测试用例发送给目标智能体，收集其输出 |
| **3. Evaluate** | 评估 | LLM 评审员对输出进行多维度打分 |
| **4. Reflect** | 反思 | 分析失败案例，总结出具体的改进方向和模式 |
| **5. Mutate** | 变异 | 基于反思结论，生成改进后的 Prompt / 配置参数候选方案 |
| **6. Validate** | 验证 | 在留出集上验证改进是否真实有效，通过护栏检查后接受变更 |

## 安全护栏 / Guardrails

**Overfitting Detection** — 训练集和留出集分别评分，分数差距过大则拒绝变更。

**Regression Prevention** — 每轮变更后重新验证，任何回归都触发回滚。

**Consistency Checking** — 同一输入多次运行，检查输出一致性。

**Circuit Breaker** — LLM API 连续失败 5 次后自动熔断 60 秒，防止雪崩。

## 项目结构 / Project Structure

```
nuwa/
├── core/                 # 数据模型、协议接口、异常、默认值
├── llm/                  # LLM 后端 (LiteLLM)、元提示词、解析器、熔断器
├── connectors/           # 目标智能体连接器 (HTTP, CLI, Python)
├── engine/               # 训练引擎
│   ├── loop.py           # 主循环编排器
│   ├── scheduler.py      # 收敛检测与早停
│   ├── stages/           # 六阶段流水线
│   ├── parallel/         # 并行执行与集成评估
│   └── objectives/       # 多目标优化与 Pareto 前沿
├── guardrails/           # 安全护栏 (过拟合/回归/一致性)
├── sandbox/              # 沙箱隔离层
├── sdk/                  # Python SDK (@trainable, NuwaTrainer, train())
├── web/                  # Web UI (FastAPI + SSE)
├── conversation/         # 交互式对话 UI (Rich)
├── config/               # 配置管理 (YAML)
├── persistence/          # 持久化 (JSONL + 快照 + Git)
└── cli.py                # CLI 入口
```

## 文档 / Documentation

| 文档 | 说明 |
|------|------|
| [API.md](API.md) | REST API 完整文档 |
| [SECURITY.md](SECURITY.md) | 安全策略与漏洞报告 |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | 常见问题与故障排查 |
| [CONTRIBUTING.md](CONTRIBUTING.md) | 贡献指南 |
| [CHANGELOG.md](CHANGELOG.md) | 版本变更记录 |

## 开发 / Development

```bash
git clone https://github.com/KBREX684/nuwa.git
cd nuwa
python3 -m venv .venv
source .venv/bin/activate
make install-dev
make test
make check
```

## 路线图 / Roadmap

- [x] Web UI — 浏览器端训练监控面板
- [x] Sandbox isolation — 沙箱隔离
- [x] Python SDK — @trainable + train() 编程接口
- [x] Multi-objective optimization — Pareto 前沿
- [x] Parallel evaluation — 多评委集成评分
- [x] Circuit breaker — LLM API 熔断器
- [x] Rate limiting — Web API 速率限制
- [x] Health check — /api/health 端点
- [ ] Plugin system — 可插拔的评估器、变异策略和连接器
- [ ] Benchmark suite — 内置标准化评测集
- [ ] Distributed training — 多机并行训练
- [ ] Training resume — 断点续训

## 许可证 / License

[Business Source License 1.1](LICENSE) — 非生产环境免费使用，2028-01-01 起自动转为 Apache 2.0。商业使用请联系维护者。

---

<div align="center">

**女娲造人，Nuwa 造智能体。**

</div>
