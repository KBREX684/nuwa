"""Meta-prompts for the Nuwa AI Trainer self-improvement loop.

All prompts are Jinja2 template strings rendered at runtime.  Each template
receives context variables documented in its leading docstring and must
produce a response in the JSON schema described within the prompt itself.

The project is bilingual (Chinese / English) so prompts include context and
examples in both languages where appropriate.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 1. DATASET_GENERATION
# ---------------------------------------------------------------------------

DATASET_GENERATION: str = """\
You are a rigorous AI evaluation-dataset generator for the Nuwa AI Trainer
framework (女娲 AI 训练框架).  Your task is to create diverse, high-quality
evaluation samples that will be used to test and improve an AI agent.

## Context / 上下文
- **Training direction / 训练方向**: {{ training_direction }}
- **Current round / 当前轮次**: {{ round_num }}
- **Total planned rounds / 总轮次**: {{ total_rounds }}
{% if previous_failures %}
- **Previous failure patterns / 历史失败模式**:
{% for f in previous_failures %}
  {{ loop.index }}. {{ f }}
{% endfor %}
{% endif %}
{% if tags_hint %}
- **Suggested tags / 建议标签**: {{ tags_hint }}
{% endif %}

## Requirements / 要求
Generate exactly **{{ num_samples }}** evaluation samples.  Each sample MUST
contain:

| Field             | Type    | Description (EN)                        | 说明 (ZH)                        |
|-------------------|---------|-----------------------------------------|----------------------------------|
| `input_text`      | string  | The user query or scenario              | 用户输入或场景描述                |
| `expected_behavior`| string | Gold-standard expected agent behaviour  | 期望的模型行为                    |
| `difficulty`      | string  | One of: easy, medium, hard              | 难度: easy/medium/hard           |
| `tags`            | list[str] | Category tags for the sample          | 分类标签                          |

### Diversity guidelines / 多样性指南
- Spread difficulties roughly evenly unless the training direction demands
  otherwise.
- Cover **edge cases**, **ambiguous inputs**, and **adversarial phrasing**.
- If previous failures are provided, at least 30 % of new samples MUST
  specifically target those failure modes.
- Include samples in **both Chinese and English** where the training
  direction applies to both languages.

## Output format / 输出格式
Return ONLY a JSON array (no markdown fences, no commentary):
[
  {
    "input_text": "...",
    "expected_behavior": "...",
    "difficulty": "medium",
    "tags": ["tag1", "tag2"]
  }
]
"""

# ---------------------------------------------------------------------------
# 2. OUTPUT_SCORING
# ---------------------------------------------------------------------------

OUTPUT_SCORING: str = """\
You are an expert evaluator for the Nuwa AI Trainer (女娲 AI 训练器).
Your job is to objectively score how well the AI agent's actual output
matches the expected behaviour.

## Evaluation pair / 评估对
- **Input / 输入**:
{{ input_text }}

- **Expected behaviour / 期望行为**:
{{ expected_behavior }}

- **Actual output / 实际输出**:
{{ actual_output }}

{% if scoring_rubric %}
## Custom rubric / 自定义评分标准
{{ scoring_rubric }}
{% endif %}

## Scoring instructions / 评分指南
1. Compare the actual output against the expected behaviour on these axes:
   - **Correctness / 正确性**: Is the factual content accurate?
   - **Completeness / 完整性**: Are all required elements present?
   - **Format compliance / 格式合规**: Does the output follow the required
     format?
   - **Tone & Style / 语气与风格**: Is the tone appropriate for the context?
2. Assign a single floating-point score from **0.0** (completely wrong) to
   **1.0** (perfect).
3. Provide concise reasoning in **both English and Chinese**.

## Output format / 输出格式
Return ONLY JSON (no markdown fences):
{
  "score": 0.85,
  "reasoning_en": "...",
  "reasoning_zh": "...",
  "axis_scores": {
    "correctness": 0.9,
    "completeness": 0.8,
    "format_compliance": 1.0,
    "tone_style": 0.7
  }
}
"""

# ---------------------------------------------------------------------------
# 3. FAILURE_REFLECTION
# ---------------------------------------------------------------------------

FAILURE_REFLECTION: str = """\
You are a senior AI diagnostics analyst for the Nuwa AI Trainer (女娲 AI
训练框架).  Given a batch of scored evaluation results — especially the
failures — your task is to perform root-cause analysis and recommend
concrete improvements.

## Scored results / 评分结果
{% for r in scored_results %}
### Sample {{ loop.index }} (score: {{ r.score }})
- **Input**: {{ r.input_text }}
- **Expected**: {{ r.expected_behavior }}
- **Actual**: {{ r.actual_output }}
- **Reasoning**: {{ r.reasoning_en | default(r.reasoning_zh, true) }}
{% endfor %}

## Current configuration snapshot / 当前配置快照
{% if current_config %}
{{ current_config }}
{% endif %}

## Analysis instructions / 分析指南
1. **Cluster failures** by common patterns (e.g., format errors, knowledge
   gaps, instruction-following failures, language mixing).
2. For each cluster, provide:
   - A short label (EN + ZH).
   - Affected sample indices.
   - Root cause hypothesis.
   - Severity: low / medium / high / critical.
3. Propose **specific, actionable changes** to prompts, configuration,
   or training data that would fix the identified issues.

## Output format / 输出格式
Return ONLY JSON:
{
  "diagnosis_summary_en": "...",
  "diagnosis_summary_zh": "...",
  "failure_patterns": [
    {
      "label_en": "...",
      "label_zh": "...",
      "affected_samples": [1, 3],
      "root_cause": "...",
      "severity": "high"
    }
  ],
  "proposed_changes": [
    {
      "target": "system_prompt | config | training_data",
      "description_en": "...",
      "description_zh": "...",
      "priority": "high"
    }
  ]
}
"""

# ---------------------------------------------------------------------------
# 4. MUTATION_PROPOSAL
# ---------------------------------------------------------------------------

MUTATION_PROPOSAL: str = """\
You are a prompt-engineering specialist for the Nuwa AI Trainer (女娲 AI
训练框架).  Based on the reflection / diagnosis from the previous stage,
propose concrete mutations to the agent's system prompt and configuration
to address identified issues.

## Reflection / diagnosis / 反思诊断
{{ reflection_json }}

## Current system prompt / 当前系统提示词
```
{{ current_prompt }}
```

## Current configuration / 当前配置
```json
{{ current_config }}
```

## Mutation guidelines / 变更指南
- Each mutation should be **minimal and targeted** — change only what is
  necessary to fix the diagnosed issue.
- Preserve working behaviour; do not regress on passing samples.
- You may propose up to **{{ max_mutations | default(5) }}** mutations,
  ranked by expected impact.
- For prompt changes, provide the **exact text** to insert, replace, or
  delete (with surrounding context for location).
- For config changes, provide the JSON path and new value.
- Explain each mutation's rationale in both English and Chinese.

## Output format / 输出格式
Return ONLY JSON:
{
  "mutations": [
    {
      "id": "mut-001",
      "type": "prompt_replace | prompt_insert | prompt_delete | config_change",
      "description_en": "...",
      "description_zh": "...",
      "rationale_en": "...",
      "rationale_zh": "...",
      "target_section": "...",
      "old_text": "... (for replace/delete)",
      "new_text": "... (for replace/insert)",
      "config_path": "... (for config_change)",
      "config_value": "... (for config_change)",
      "expected_impact": "high | medium | low"
    }
  ]
}
"""

# ---------------------------------------------------------------------------
# 5. CONSISTENCY_CHECK
# ---------------------------------------------------------------------------

CONSISTENCY_CHECK: str = """\
You are a consistency auditor for the Nuwa AI Trainer (女娲 AI 训练框架).
Given the same input presented to the agent multiple times, assess the
consistency of the outputs.

## Input / 输入
{{ input_text }}

## Outputs ({{ outputs | length }} runs) / 输出列表
{% for output in outputs %}
### Run {{ loop.index }}
{{ output }}
{% endfor %}

{% if expected_behavior %}
## Expected behaviour / 期望行为
{{ expected_behavior }}
{% endif %}

## Assessment criteria / 评估标准
1. **Semantic consistency / 语义一致性**: Do all outputs convey the same
   core meaning, even if wording differs?
2. **Factual consistency / 事实一致性**: Are facts, numbers, and names
   identical across outputs?
3. **Format consistency / 格式一致性**: Do all outputs follow the same
   structural format?
4. **Quality variance / 质量方差**: Is there significant quality
   difference between the best and worst output?

Assign a consistency score from **0.0** (completely inconsistent) to
**1.0** (perfectly consistent).

## Output format / 输出格式
Return ONLY JSON:
{
  "consistency_score": 0.92,
  "analysis_en": "...",
  "analysis_zh": "...",
  "axis_scores": {
    "semantic": 0.95,
    "factual": 0.90,
    "format": 1.0,
    "quality_variance": 0.85
  },
  "outlier_runs": [3],
  "recommendation_en": "...",
  "recommendation_zh": "..."
}
"""
