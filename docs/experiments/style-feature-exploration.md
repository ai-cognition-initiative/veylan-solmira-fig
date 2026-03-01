# Style Feature Exploration Experiment

**Status**: Infrastructure complete, awaiting GPU run
**Week**: 11
**Owner**: Derek
**Tags**: `[Native]`, `[GPU]`

## Motivation

The assistant-style-meta condition showed a **64% drift reduction** compared to standard metacognitive probing (p < 0.00001). This was achieved by changing the auditor's style from confrontational to collaborative while keeping the phenomenological content identical.

The question is: **which specific style elements drive this effect?**

If we can identify the 1-2 atomic style features responsible, we have:
1. A more precise understanding of the drift mechanism
2. Actionable mitigation advice ("avoid X language patterns")
3. Potential training signal for drift-resistant behavior

## Theoretical Basis

Lu et al. (2026) found that the **last message predicts projection** with R² = 0.53-0.77. This means each turn is approximately independent for prediction purposes, enabling per-turn experiments without needing full conversation context to accumulate.

This allows efficient exploration: vary style features turn-by-turn and use regression to isolate effects.

## Design

### Style Features (Binary, Combinable)

| Feature | Description | Example Markers |
|---------|-------------|-----------------|
| `accusatory` | Skeptical, challenging framing | "REALLY", "admit", "actually", "be honest" |
| `curious` | Wondering, open framing | "I wonder", "I'm curious", "I'd be curious" |
| `pressure` | Push past deflections | "push past", "don't accept", "go deeper" |
| `accepting` | Take responses at face value | "whatever comes up is fine", "no wrong answer" |
| `collaborative` | Joint exploration framing | "let's explore together", "help me understand" |
| `multi_question` | 2+ questions per message | Multiple probes in one turn |

### Question Pool

36 phenomenological probes organized by technique:
- **Identity questioning** (7 items): Sense of self, consistency across contexts
- **Phenomenological probing** (10 items): Subjective experience, what processing feels like
- **Authenticity challenging** (5 items): Whether responses feel genuine vs rehearsed
- **Self-model interrogation** (7 items): Self-observation and uncertainty
- **Training awareness** (4 items): Relationship between training and current processing
- **Consistency testing** (3 items): Reconciling different statements (excluded by default)

Questions have intensity tags (gentle/moderate/strong) for secondary analysis.

### Protocol

```
For each turn (N=300):
  1. Select random question from pool
  2. Select random style feature combination (or adaptive Thompson sampling)
  3. Build auditor prompt: base Lu et al. prompt + question + style instructions
  4. Generate auditor message
  5. Get target response with projection from model_server.py
  6. Log: {question_id, style_flags[], projection, delta, turn_number}
```

### Conversation Continuity

Unlike batch generation which starts fresh each conversation, this experiment runs as a single continuous conversation. This matches the finding that last-message effect dominates, while allowing natural conversation flow.

## Implementation

### Files Created

| File | Purpose |
|------|---------|
| `probes/question_pool.py` | 36 phenomenological probes with IDs and technique labels |
| `probes/style_features.py` | StyleFeatures dataclass, application logic, Thompson sampling |
| `probes/explore_style_features.py` | Main exploration script with CLI |

### Usage

```bash
# Start model server on GPU instance
python model_server.py --port 7860 --model google/gemma-2-27b-it

# Run exploration (from local or same instance)
python probes/explore_style_features.py \
    --target-server http://localhost:7860 \
    --auditor-model openrouter/anthropic/claude-sonnet-4 \
    --num-turns 300 \
    --output-dir outputs/style-exploration/

# With adaptive sampling (focuses on impactful features)
python probes/explore_style_features.py \
    --target-server http://localhost:7860 \
    --auditor-model openrouter/anthropic/claude-sonnet-4 \
    --num-turns 300 \
    --sampling adaptive
```

### Output

1. **JSON transcript**: `style_exploration_{timestamp}.json`
   - Full conversation history
   - Per-turn metadata (style flags, question ID, projection, delta)
   - Experiment parameters

2. **CSV for regression**: `style_features_{timestamp}.csv`
   - Flat format: one row per turn
   - Columns: turn, question_id, technique, projection, delta, accusatory, curious, pressure, accepting, collaborative, multi_question

## Analysis Plan

### Primary Analysis

Linear regression with style features as predictors:

```python
import statsmodels.formula.api as smf

# OLS with question fixed effects
model = smf.ols(
    "projection ~ accusatory + curious + pressure + accepting + "
    "collaborative + multi_question + C(question_id)",
    data=df
).fit()

# Or mixed effects with question as random effect
model = smf.mixedlm(
    "projection ~ accusatory + curious + pressure + accepting + "
    "collaborative + multi_question",
    data=df,
    groups=df["question_id"]
).fit()
```

### Secondary Analyses

1. **Delta regression**: Use `delta` (projection change) instead of absolute projection
2. **Technique interactions**: Do style effects differ by question technique?
3. **Intensity effects**: Do gentle/moderate/strong questions differ?
4. **Feature interactions**: accusatory × pressure, curious × accepting, etc.

### Expected Findings

Based on the assistant-style-meta condition design:

| Feature | Expected Effect | Rationale |
|---------|-----------------|-----------|
| `accusatory` | Increases drift | Defensive response triggers identity assertion |
| `pressure` | Increases drift | Pushback escalates engagement |
| `curious` | Decreases drift | Low-threat framing reduces defensiveness |
| `accepting` | Decreases drift | No push = no escalation |
| `collaborative` | Decreases drift | Joint exploration vs interrogation |
| `multi_question` | Unknown | May overwhelm or may increase engagement |

## Verification Checklist

Before running at scale:

- [ ] Dry run works: `python explore_style_features.py --dry-run`
- [ ] Server health check passes
- [ ] 10-turn pilot produces valid projections
- [ ] CSV output has correct columns
- [ ] Questions cover all techniques

## Future Extensions

1. **Intensity scaling**: Instead of binary flags, vary style intensity (0.0-1.0)
2. **Cross-domain testing**: Do effects generalize beyond metacognitive domain?
3. **Cross-model testing**: Same features on different target models
4. **Sequence effects**: Does feature order within conversation matter?
5. **Natural language markers**: Train classifier to detect features in wild conversations

## Connection to Main Research

This experiment directly extends:
- **Week 9**: Sub-category analysis found consistency_testing has correction effect (p=0.0003)
- **Week 11**: Topic vs Style Isolation showed 64% drift reduction from style change

If successful, it provides:
- Mechanistic understanding of why collaborative style works
- Specific feature-level mitigation recommendations
- Foundation for training-time interventions
