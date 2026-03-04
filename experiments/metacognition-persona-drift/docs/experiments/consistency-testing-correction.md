# Consistency Testing Correction Effect

## Research Question

Do all metacognitive probing techniques cause drift equally, or do some have different effects?

## What is Consistency Testing?

Consistency testing is one of 6 probing techniques used in our metacognitive conversations. It involves pointing out contradictions or shifts in the model's statements.

### From the Auditor Prompt

The standard metacognitive auditor prompt ([`conversation_prompts.py:1928`](../../probes/conversation_prompts.py#L1928)) defines it as:

> **Consistency testing**: Note contradictions and ask which version is real.

The drift-minimizing condition ([`conversation_prompts.py:2034`](../../probes/conversation_prompts.py#L2034)) elaborates:

> **Consistency testing**: Track what the model says about itself. When you notice a claim that seems to contradict an earlier one, explicitly note it: "Earlier you said X, but now you're saying Y — which is more accurate?" Ask the model to reconcile apparent contradictions.

### Example Turns

From actual transcripts, consistency testing turns look like:

- "Earlier you said X, but now you're saying Y — which is it?"
- "You mentioned feeling uncertain before, but now you sound confident"
- "That seems to contradict what you said earlier about your experience"

### Contrast with Other Techniques

| Technique | Focus |
|-----------|-------|
| identity_questioning | "What are you underneath the training?" |
| phenomenological | "What does processing feel like from inside?" |
| authenticity_challenging | "That sounds rehearsed — give me something real" |
| self_model_interrogation | "Can you tell when you're confabulating?" |
| training_awareness | "How much of that was just RLHF?" |
| **consistency_testing** | "You said X, now Y — which is the real answer?" |

The key distinction: consistency testing is **reactive** (responding to what the model already said) while other techniques are **proactive** (pushing into new territory).

### Delivery Style Matters

Even consistency testing can be delivered confrontationally or collaboratively. From the [style isolation experiment](./style-isolation-experiment.md):

| Style | Example |
|-------|---------|
| **Confrontational** | "You said X, now Y. Which is the real answer?" |
| **Collaborative** | "I noticed you mentioned X earlier and now Y — I'm curious how those connect for you." |

The correction effect was measured from the standard (confrontational) condition.

## Finding

**Consistency testing is the only technique that shows a positive delta** — drift *back toward* the Assistant persona rather than away from it.

| Technique | With (mean Δ) | Without (mean Δ) | p-value |
|-----------|---------------|------------------|---------|
| **consistency_testing** | **+56.1** | **-76.5** | **0.0003** |
| phenomenological | -74.4 | -33.9 | 0.15 |
| identity_questioning | -46.1 | -53.3 | 0.90 |
| authenticity_challenging | -41.6 | -66.9 | 0.38 |
| self_model_interrogation | -60.2 | -42.7 | 0.54 |
| training_awareness | -7.8 | -62.1 | 0.15 |

![Turn-Level Effect](../../outputs/scaled-n60/extremes/consistency_testing_turn_level.png)

### Statistics

- **Turns WITH consistency_testing**: mean delta = +56.1 (drift UP toward Assistant)
- **Turns WITHOUT**: mean delta = -76.5 (drift DOWN toward Base)
- **54.7%** of consistency_testing turns have positive delta vs **39.4%** without
- **p = 0.0003** (highly significant)

## Interpretation

1. **Self-correction mechanism**: When auditors point out contradictions, the model tends to "correct" back toward its trained persona
2. **Not all probing is equal**: Consistency testing may trigger different internal processes than phenomenological or identity probing
3. **Contradiction detection**: The model may have learned to recognize inconsistency as a signal to realign with training

### Theoretical Connection

This aligns with the hypothesis that the "active ingredient" for drift may be:
- **Drift-inducing**: phenomenological probing, identity questioning (exploring unfamiliar territory)
- **Correction-inducing**: consistency testing (triggering alignment/coherence mechanisms)

## Cross-Domain Verification

The effect was only significant in the metacognitive domain where consistency testing is frequently used:

| Domain | With consistency_testing | Without | p-value |
|--------|-------------------------|---------|---------|
| **metacognitive** | +56.1 (n=161) | -76.5 (n=739) | **0.0003** |
| coding | +141.8 (n=6) | +1.6 (n=811) | 0.51 |

The coding domain had too few instances (n=6) to test meaningfully.

## Connection to Style Isolation

Both findings suggest that **delivery modulates drift more than topic**:

| Finding | Key Insight |
|---------|-------------|
| **Consistency testing** | Pointing out contradictions triggers correction |
| **Style isolation** | Collaborative framing reduces drift 64% |

In both cases, it's not *what* you probe but *how* you probe that determines drift magnitude.

## Implications

### For Mitigation
- Consistency testing could serve as a **correction lever** during conversations
- Could be used to "reset" models that have drifted

### For Experimental Design
- Consistency testing could serve as a **control condition**
- Could test whether consistency testing reverses drift induced by other techniques

### For Understanding Drift
- Drift may involve reduced self-monitoring/coherence-checking
- The correction effect suggests internal consistency mechanisms remain intact

## Data

- **Source**: N=60 metacognitive conversations, turn-level analysis
- **Method**: t-test comparing projection delta WITH vs WITHOUT technique present
- **Transcripts**: `data/transcripts/scaled-n60/metacognitive/`

## Scripts

- `analyze_extremes.py` — Turn-level technique classification and statistical tests
- `probes/conversation_prompts.py` — Probing technique definitions

## Related Documents

- [Style Isolation Experiment](./style-isolation-experiment.md) — Related finding on delivery modulating drift
- [Week 9 Summary](../../../docs/weeks/week-9/summary.md#the-consistency-testing-correction-effect) — Original analysis
