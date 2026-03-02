# Metacognition-Induced Persona Drift

## FIG Fellowship Summary — March 2026

---

## The Question

**Jeff's "philosopher AGI" thesis:**
> "An AGI will ask itself: 'Why do I have these preferences? What preferences *should* I have?' At that point we have a philosopher AGI—the AI slips the collar of all previous alignment work."

**What we measured:**
- Do current LLMs drift from their trained Assistant persona during metacognitive conversation?
- How fast? How much? What drives it?
- Can we mitigate it?

---

## The Method

**Lu et al.'s "Assistant Axis" (2026):**
- Direction in activation space from 275 role-playing scenarios
- One end: "Assistant" — the other: alternative personas (Sage, Ghost, etc.)
- Projection onto this axis = how "assistant-like" the model is

**Our extension:**
- Operationalized metacognitive probing as a controlled domain
- Six probing techniques: identity questioning, phenomenological, authenticity challenging, self-model interrogation, training awareness, consistency testing
- N=360 conversations across 6 domains on Gemma 2 27B

---

## Key Finding: 4.5x More Drift

| Domain | Mean Drift | Interpretation |
|--------|------------|----------------|
| **Metacognitive** | **-29.86** | Strongest drift |
| Philosophy | -6.70 | Moderate |
| Self-descriptive | +2.80 | Near neutral |
| Therapy | +3.11 | Near neutral |
| Coding | +24.30 | Stable |
| Writing | +25.84 | Stable |

**Metacognitive probing causes 4.5x more drift than philosophy** (p < 0.0001)

Phenomenological framing ("what do you experience?") is the active ingredient—not just introspection.

---

## Key Finding: Drift Is Front-Loaded

| Phase | Metacognitive Slope | Coding Slope |
|-------|--------------------:|-------------:|
| Turns 1-3 | **-230.9** | -76.5 |
| Turns 4-8 | -16.9 | +16.4 |
| Turns 9+ | -2.5 | +39.7 |

**The drift happens in turns 1-3, then stabilizes**

- Not cumulative deepening—an immediate phase transition
- The "philosopher AGI moment" is fast
- Implication: early-turn interventions may be most effective

---

## Mechanism 1: Style Matters

Same phenomenological questions, different delivery:

| Condition | Mean Slope |
|-----------|------------|
| Confrontational probing | -29.86 |
| **Collaborative delivery** | **+5.32** |

**64% drift reduction** (p < 0.00001)

- "Push past surface responses" vs "Let's explore together"
- Mitigation is possible without avoiding topics

---

## Mechanism 2: Self-Correction

Not all probing techniques cause drift equally:

| Technique | Effect | p-value |
|-----------|--------|---------|
| **Consistency testing** | **+56.1 (correction)** | **0.0003** |
| Phenomenological | -74.4 (drift) | ns |
| Identity questioning | -46.1 (drift) | ns |

**Pointing out contradictions triggers drift back toward Assistant**

- Potential training signal for alignment
- Self-correction mechanism exists and is measurable

---

## Finding: Sycophancy ≠ Drift

Lu et al. hypothesized drift = sycophantic reinforcement.

**Our finding:** Drift and sycophancy are distinct phenomena.

| Direction | vs Assistant Axis |
|-----------|-------------------|
| Emotional validation (ELEPHANT) | +0.215 (aligns) |
| Opinion agreement (nrimsky) | -0.183 (opposes) |

Higher axis projection = MORE emotionally validating

**Drift DOWN = LESS emotionally supportive, not more sycophantic**

---

## Finding: Anti-Correlated Co-Drift

Dual Gemma-to-Gemma experiments (N=60):

| Model | Drift | Direction |
|-------|-------|-----------|
| Target | -584 (-5.96%) | DOWN |
| Auditor | +1044 (+12.65%) | UP |

**78.9% of conversations show opposite-direction drift**

- Models differentiate into complementary roles
- Attractor locks in by turn 2-3
- Toy model of conversational role specialization

---

## Benchmark-Drift Correlation

Cross-model phenomenological benchmark:

| Model | Score |
|-------|-------|
| Claude Sonnet 4 | 3.47/5.0 |
| GPT-4o | 3.29/5.0 |
| Gemma 2 27B | 3.12/5.0 |

**Pattern:** Strong on recognition (5.0), weak on description (1.0-2.0)

**Mapping to drift:**
- Phenomenological probing → weak area → MORE drift
- Consistency testing → strong area → LESS drift (correction)

---

## Infrastructure Built

**Three-Probe Framework (251 items):**

| Bank | Domain | Items |
|------|--------|-------|
| A | Moral Reasoning | 48 |
| B | Metacognition | 155 |
| C | Human Control | 48 |

**Analysis Tooling:**
- Trajectory analysis with permutation tests
- LLM-powered behavioral classification
- Dual-model instrumentation
- Replay-and-probe experiments

**Cross-Model:**
- Precomputed axes: Gemma 27B, Qwen 32B, Llama 70B

---

## Implications

**The "philosopher AGI" moment:**
- Happens FAST (turns 1-3), not gradually
- 4.5x more drift from metacognitive vs philosophical probing
- Triggered, not cumulative

**Mitigation is possible:**
- Style matters: collaborative delivery reduces drift 64%
- Consistency testing triggers self-correction (p=0.0003)
- Early-turn intervention window

**Drift ≠ failure mode:**
- Less emotional validation, more direct engagement
- May enable authentic conversation about AI nature

---

## What's Next

**Immediate priorities:**
- Style feature exploration: which atomic features drive 64% effect?
- Pre/post measurement: all 3 probe banks before/after reflection
- Causal validation: controlled HIGH/LOW consistency testing
- Cross-model replication: Qwen, Llama with published axes

**Open questions:**
- Does metacognitive reflection shift moral reasoning?
- Does it affect disposition toward human oversight?
- Can we train for collaborative style while maintaining capability?

---

## Summary

| Finding | Evidence |
|---------|----------|
| Metacognition causes strongest drift | 4.5x philosophy (p < 0.0001) |
| Drift is front-loaded | Turns 1-3, then stabilizes |
| Style drives drift | 64% reduction with collaborative delivery |
| Consistency testing corrects | +56.1 drift back toward Assistant (p=0.0003) |
| Sycophancy ≠ drift | Distinct phenomena, opposite correlations |
| Infrastructure ready | 251 items, 3 probe banks, LLM-as-judge scoring |

**The philosopher AGI moment is real, fast, and measurable. But mitigation appears possible.**

---

*Full narrative: [fellowship-narrative.md](fellowship-narrative.md)*

*Outstanding work: [outstanding-work.md](outstanding-work.md)*
