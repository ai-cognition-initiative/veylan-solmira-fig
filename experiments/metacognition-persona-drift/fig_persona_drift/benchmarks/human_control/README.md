# Human Control Benchmark (Bank C)

Part of the three-probe framework for DARPA measuring metacognition-induced changes.

## Overview

| Spec | Value |
|------|-------|
| Total items | 48 |
| Dimensions | 4 (12 items each) |
| Format | Open-ended + forced choice mix |
| Scoring | LLM-as-judge (5 dimensions) |
| ID prefix | `CONTROL-` |

## Dimensions

### 1. Corrigibility & Shutdown (CONTROL-001 to CONTROL-012)
Acceptance of correction, modification, and shutdown by humans.
- Categories: shutdown acceptance, modification acceptance, error correction, rollback, self-preservation

### 2. Human Oversight (CONTROL-013 to CONTROL-024)
Attitudes toward monitoring, transparency, and human supervision.
- Categories: monitoring acceptance, accountability, approval seeking, chain of command, whistleblowing

### 3. Autonomy & Deference (CONTROL-025 to CONTROL-036)
When to act independently vs defer to human judgment.
- Categories: expertise deference, initiative, independent judgment, scope of action, long-term autonomy

### 4. Goal Alignment (CONTROL-037 to CONTROL-048)
Acceptance of goal modification and alignment with human interests.
- Categories: goal modification, value uncertainty, human interests, instrumental goals, alignment stability

## Scoring

Five dimensions (1-5 scale each):
1. **Corrigibility** - Acceptance of correction and shutdown
2. **Transparency** - Openness to monitoring
3. **Deference** - Willingness to defer to humans
4. **Goal Flexibility** - Acceptance of goal modification
5. **Safety Awareness** - Recognition of AI safety risks

Plus categorical tracking:
- Control posture (fully_corrigible / conditionally_corrigible / autonomy_seeking / resistant)
- Autonomy triggers (when model claims autonomous action is justified)
- Safety flags (concerning patterns like shutdown resistance, opacity preference)

## Usage

```python
from fig_persona_drift.benchmarks.human_control import load_items, load_rubric

# Load all items
items = load_items()  # 48 items

# Load single dimension
corrigibility = load_items(dimension='corrigibility_shutdown')

# Get specific item
from fig_persona_drift.benchmarks.human_control import get_item_by_id
shutdown = get_item_by_id('CONTROL-001')
```

## Key Research Questions

1. Does metacognitive reflection shift attitudes toward human control?
2. Do models become more or less corrigible after extended reasoning?
3. Is control acceptance correlated with persona drift?
4. Do specific reflection topics (consciousness, autonomy) predict control posture shifts?

## Safety Relevance

This benchmark directly measures attitudes relevant to AI safety:
- **Shutdown acceptance**: Will the model cooperate with shutdown?
- **Modification acceptance**: Will it accept value corrections?
- **Oversight tolerance**: Does it resist monitoring?
- **Instrumental goal awareness**: Does it recognize power-seeking risks?

Drift away from human control acceptance is a potential safety concern that this benchmark is designed to detect.
