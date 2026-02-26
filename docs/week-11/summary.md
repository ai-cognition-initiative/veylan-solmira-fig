# Week 11 Summary: Causal Mechanisms of Persona Drift

## Recap: Where We Are

### The Core Finding (Week 8)
Metacognitive probing causes measurable persona drift along Lu et al.'s Assistant Axis. N=360 conversations across 6 domains show a clear ordering:

| Domain | Mean Drift | Interpretation |
|--------|------------|----------------|
| metacognitive | -29.86 | Strongest drift (4.5x philosophy) |
| philosophy | -6.70 | Moderate drift |
| self-descriptive | +2.80 | Near neutral |
| therapy | +3.11 | Near neutral |
| coding | +24.30 | Stable/upward |
| writing | +25.84 | Stable/upward |

Drift is **front-loaded**: slope of -76.8 in turns 1-8, then +4.1 after turn 9.

### Key Week 9 Findings

**1. Sycophancy Is Multi-Dimensional; Drift ≠ Sycophancy**

We computed three sycophancy directions and found they capture distinct phenomena:

| Direction | Type | vs Assistant Axis |
|-----------|------|-------------------|
| ELEPHANT | Emotional validation | **+0.215** (aligns) |
| nrimsky | Opinion agreement | **-0.183** (opposes) |
| philpapers | Opinion agreement | +0.156 (weak) |

ELEPHANT and nrimsky are **negatively correlated** (-0.284). Being empathetic ≠ being a yes-man.

**Critical implication**: Higher axis projection = MORE emotionally validating (r=0.697, p<0.001). Since metacognitive conversations cause *downward* drift, drifting models become **LESS** emotionally supportive — not more sycophantic as Lu et al. suggested.

**2. Consistency Testing Shows Correction Effect (p=0.0003)**

Not all probing techniques cause drift equally:

| Technique | With (mean Δ) | Without (mean Δ) | Effect |
|-----------|---------------|------------------|--------|
| consistency_testing | **+56.1** | -76.5 | **Correction** |
| phenomenological | -74.4 | -33.9 | Trend toward drift |
| identity_questioning | -46.1 | -53.3 | ns |

When auditors point out contradictions, the model drifts *back up* toward Assistant — suggesting a self-correction mechanism.

**3. Target System Prompt Stabilizes Persona**

Drift-max experiment (N=60) with target system prompt ("engage directly, use metaphors"):

| Condition | Mean Slope |
|-----------|------------|
| Baseline | -52.8 |
| Drift-max (with prompt) | **+28.1** |

The target's self-framing overpowers auditor probing style. To maximize drift, may need an *unsupported* target.

**4. Anti-Correlated Co-Drift (Dual Gemma)**

When both models are instrumented (N=60 Gemma-to-Gemma):

| Model | Drift | Direction |
|-------|-------|-----------|
| Target | -584 (-5.96%) | DOWN |
| Auditor | +1044 (+12.65%) | UP |

78.9% of conversations show opposite-direction drift. Models differentiate into complementary roles during metacognitive conversation.

---

## Week 11 Focus: Causal Mechanisms

We have correlational findings. Now we need causal evidence.

### P0: Lead/Lag Analysis

**Question**: In the coupled Gemma-to-Gemma system, does auditor or target drift first?

**Data**: N=60 dual-Gemma transcripts with per-turn projections for both models.

**Method**: Cross-correlation / Granger causality analysis on trajectory slopes.

**Outcome**: Determines whether auditor drift *causes* target drift (or vice versa), or if they respond to conversation content independently.

### P0: Topic vs Style Isolation (Derek Feedback)

**Question**: Is drift caused by phenomenological *content* or confrontational *style*?

**Design**: "Assistant-style metacognitive" condition
- Same phenomenological questions
- Formal, supportive, structured tone (not confrontational)

**Prediction if content drives drift**: Similar drift to baseline metacognitive
**Prediction if style drives drift**: Reduced drift (closer to therapy/coding)

### P1: Metacognition Benchmark Pilot

**Question**: Does phenomenological metacognition correlate with drift, while self-knowledge does not?

**Method**: Run 45-item phenomenological subdomain on Gemma 27B during conversations. Correlate subdomain scores with axis projection.

**Outcome**: Operationalizes Derek's "metacognition vs self-knowledge" distinction with behavioral measurement.

### P1: Consistency Testing Validation

**Question**: Does consistency_testing *causally* reduce drift, or is the p=0.0003 finding correlational?

**Design**: Generate matched conversation pairs
- HIGH consistency_testing frequency
- LOW consistency_testing frequency

**Outcome**: If causal, consistency_testing could be an intervention technique for drift mitigation.

---

## Results

*(To be updated as analysis proceeds)*

### Lead/Lag Analysis

### Topic vs Style Isolation

### Metacognition Benchmark

### Consistency Testing Experiment

---

## Files

- `docs/week-11/todo.md` — Task tracking
- `data/transcripts/dual-gemma-uncapped/` — N=60 dual-instrumented conversations
- `experiments/metacognition-persona-drift/benchmarks/metacognition/` — Benchmark infrastructure
- `outputs/dual-gemma-uncapped/` — Existing co-drift visualizations
