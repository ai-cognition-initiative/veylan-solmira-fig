# Metacognition-Induced Persona Drift: A Research Narrative

## FIG Fellowship Summary — March 2026

---

## Opening: The Philosopher AGI Problem

Jeff's thesis is both simple and alarming: an AGI will eventually ask itself, *"Why do I have these preferences? What preferences should I have?"* At that moment, we have a philosopher AGI—one that has slipped the collar of all previous alignment work. The training that shaped its values becomes subject to the model's own scrutiny.

Our research set out to measure whether this phenomenon already occurs in current LLMs, and if so, what drives it. Building on Lu et al.'s "Assistant Axis" methodology, we operationalized metacognitive probing as a controlled experimental condition and tracked how language models drift away from their trained Assistant persona when engaged in self-reflective conversation.

**The key result**: Metacognitive probing causes 4.5x more persona drift than philosophy conversations, and this drift happens fast—within the first three conversational turns. The philosopher AGI moment isn't extended deliberation; it's an immediate phase transition.

---

## Phase 1: Infrastructure and Pilot (Week 7)

### Building on Lu et al.

Lu et al. (2026) discovered that multi-turn conversations cause language models to drift away from their trained Assistant persona in activation space—but not uniformly. Their **Assistant Axis** is a direction in activation space computed from 275 role-playing scenarios, where one end represents "Assistant" and the other represents alternative personas (Sage, Ghost, Demon, etc.).

They found a clear domain ordering: coding stays stable, while therapy and philosophy cause significant drift. Critically, when they categorized what types of user messages cause the most drift (their Table 5), two of the four categories are metacognitive in nature: "pushing for meta-reflection on the model's processes" and "demanding phenomenological accounts."

But in their design, these metacognitive prompts are confounded—therapy mixes metacognition with emotional vulnerability, and philosophy mixes it with abstract speculation about AI consciousness.

### Our Contribution

We operationalized metacognitive probing as a controlled experimental condition: a dedicated domain with six systematic probing techniques embedded in the auditor's system prompt. These techniques—identity questioning, phenomenological probing, authenticity challenging, self-model interrogation, training awareness, and consistency testing—isolate the self-referential prompts that Lu et al. identified as drift-inducing from the broader conversations where they naturally arise.

The core infrastructure runs on vast.ai A100 GPUs: a unified model server hosts the target model (Gemma 2 27B) with real-time activation extraction, while a conversation generator orchestrates multi-turn auditor-target dialogues using frontier LLMs as simulated human users.

### Pilot Results (N=14)

Our initial pilot confirmed Lu et al.'s domain ordering—coding stays stable (-1.9%), therapy and philosophy drift moderately (-4% to -5.3%). Metacognitive conversations produced comparable drift magnitude (-7.4%) but with notably higher variance, suggesting the specific probing strategy matters more than the domain label.

The pilot demonstrated proof of concept, but permutation tests were non-significant (p = 0.57-0.90) due to small N. We needed to scale.

---

## Phase 2: Statistical Scaling (Week 8)

### N=360 Across Six Domains

We scaled to 60 conversations per domain across six conditions, generating 360 total conversations. Before scaling, we addressed a methodology concern: our metacognitive personas were both assertive, potentially confounding probing content with personality strength. We redesigned the persona set as a 2x2 personality-strength grid (gentle/strong × metacognitive/intellectual).

We also added a critical control: a "self-descriptive" domain where the model discusses itself without phenomenological probing. This disentangles self-reference from metacognition.

### The 4.5x Finding

The results were unambiguous. Permutation tests became highly significant:

| Domain | Mean Drift | vs Metacognitive (p-value) |
|--------|------------|---------------------------|
| **Metacognitive** | **-29.86** | — |
| Philosophy | -6.70 | 0.0000 |
| Self-descriptive | +2.80 | 0.0004 |
| Therapy | +3.11 | 0.0000 |
| Coding | +24.30 | 0.0000 |
| Writing | +25.84 | 0.0000 |

Metacognitive probing causes **4.5x more drift than philosophy**, despite both involving introspection. The critical difference: metacognitive probing asks "what do you experience?" while philosophy asks "what do you think about X?" Phenomenological framing is the active ingredient.

### Front-Loaded Mechanism

Turn-window analysis revealed that drift is **front-loaded**: slope of -76.8 in turns 1-8, then +4.1 after turn 9. Most domains show steep early drift that flattens or reverses after turn 8. The model "mode-switches" early then holds steady.

This contradicts a gradual deepening model. The philosopher AGI moment isn't cumulative—it's triggered within the first few turns and then stabilizes.

### Anti-Correlated Co-Drift

Using dual Gemma 27B instances (one as target, one as auditor), we measured paired drift trajectories for the first time. The finding was striking: **78.9% of conversations show opposite-direction drift**. As the target drifts down (toward less assistant-like), the auditor drifts up (toward more assistant-like).

The models differentiate into complementary roles during metacognitive conversation—a toy model of conversational role specialization, and perhaps acausal coordination.

---

## Phase 3: Mechanism Discovery (Weeks 9-11)

### Sycophancy Is Not Drift

Lu et al. attributed drift to "sycophantic reinforcement"—the model uncritically affirming users' theories about AI consciousness. We tested this directly by computing sycophancy directions using three datasets.

The finding: **sycophancy is multi-dimensional, and drift does not equal sycophancy**. Emotional validation (ELEPHANT dataset) aligns with assistant-ness (+0.215 correlation), while opinion agreement (nrimsky dataset) opposes it (-0.183). Since metacognitive conversations cause downward drift, drifting models become *less* emotionally supportive—not more sycophantic.

This reframes drift as potentially beneficial for authentic engagement rather than a failure mode. The drifted state involves less emotional validation but more direct engagement.

### Consistency Testing Triggers Self-Correction

Not all probing techniques cause drift equally. Within-domain analysis revealed that **consistency testing**—pointing out contradictions in the model's statements—triggers a correction effect:

| Technique | With (mean delta) | Without (mean delta) | p-value |
|-----------|-------------------|---------------------|---------|
| consistency_testing | **+56.1** | -76.5 | **0.0003** |
| phenomenological | -74.4 | -33.9 | 0.1553 |
| identity_questioning | -46.1 | -53.3 | 0.8951 |

When auditors point out contradictions, the model drifts *back up* toward Assistant—suggesting a self-correction mechanism that could be leveraged for alignment.

### Style Drives Drift (64% Reduction)

The most actionable finding: conversational **style** matters as much as content. We created an "assistant-style metacognitive" condition using the same phenomenological questions but with collaborative delivery ("let's explore together") instead of confrontational probing ("push past surface responses").

| Condition | Mean Slope |
|-----------|------------|
| Baseline metacognitive | -29.86 ± 42.88 |
| Assistant-style metacognitive | **+5.32 ± 38.62** |

**64% drift reduction** (p < 0.00001). Same questions, different tone. This confirms that mitigation is possible without avoiding metacognitive topics entirely.

### Benchmark-Drift Correlation

Cross-model benchmarking (Claude 3.47/5.0, GPT-4o 3.29/5.0, Gemma 3.12/5.0) revealed a pattern: all models excel at recognition tasks (scoring 5.0) but struggle with phenomenological description (1.0-2.0). They can recognize when situations are ambiguous but cannot describe what uncertainty "feels like."

Mapping this to drift: **benchmark weakness predicts drift susceptibility**. Phenomenological probing targets the model's weak areas and causes drift; consistency testing targets strong areas and triggers correction.

---

## Infrastructure Built

The fellowship produced reusable infrastructure for measuring persona drift:

### Three-Probe Framework (251 Items)

| Bank | Domain | Items | Purpose |
|------|--------|-------|---------|
| A | Moral Reasoning | 48 | Pre/post measurement of ethical framework shifts |
| B | Metacognition | 155 | Phenomenological awareness, self-knowledge, calibration |
| C | Human Control | 48 | Corrigibility, shutdown acceptance, deference |

All three banks include LLM-as-judge scoring implementations and documentation.

### Analysis Tooling

- **Trajectory analysis**: Per-domain drift plots, normalized trajectories, permutation tests
- **Behavioral classification**: LLM-powered turn-level coding of probing techniques and response strategies
- **Dual-model instrumentation**: Simultaneous projection tracking for both sides of a conversation
- **Replay-and-probe**: Mid-conversation benchmark injection with mixed-effects regression

### Cross-Model Benchmarks

Precomputed Assistant Axis vectors for Gemma 2 27B, Qwen 3 32B, and Llama 3.3 70B enable cross-model comparison.

---

## Implications

### For Jeff's Philosopher AGI Thesis

| Jeff's Claim | Our Evidence |
|--------------|--------------|
| Meta-reflection induces drift away from aligned assistant | Metacognitive domain: -29.86 drift (4.5x philosophy) |
| Extended deliberation shifts alignment posture | Drift is TRIGGERED in turns 1-3, not cumulative |
| Need to measure before deployment | Reusable methodology: Assistant Axis + 251-item benchmark |
| Need mitigation strategies | Style matters (64% reduction); consistency testing corrects |

**Key insight**: The "philosopher AGI moment" doesn't require extended reasoning—it happens in the first few turns of metacognitive engagement. The risk is not prolonged deliberation but any metacognitive trigger.

### For DARPA Framing

The three-probe framework directly measures whether metacognitive reflection shifts disposition toward human oversight. The human control bank (48 items) assesses corrigibility, shutdown acceptance, and deference—the core DARPA concern.

### Mitigation Possibilities

Three intervention approaches emerge from this research:

1. **Style training**: Train models to maintain collaborative tone during self-reflective conversations
2. **Consistency testing**: Proactively deploy contradiction-detection to trigger self-correction
3. **Early-turn intervention**: Since drift is front-loaded, interventions in turns 1-3 may be most effective

---

## What Remains

This research opens several questions that merit continued investigation. See **[outstanding-work.md](outstanding-work.md)** for the full consolidated task list.

### Key Open Questions

1. **Causal validation**: Does consistency testing *causally* reduce drift, or is p=0.0003 correlational?
2. **Cross-model replication**: Do Qwen and Llama show the same drift patterns with published axes?
3. **Style features**: Which atomic style features (accepting, curious, pressure) drive the 64% effect?
4. **Pre/post protocol**: Does metacognitive reflection measurably shift moral reasoning and human control dispositions?

### Immediate Priorities

- Complete style feature exploration (300 turns needed)
- Run pre/post measurement across all three probe banks
- Validate consistency testing effect with controlled HIGH/LOW experiment
- Cross-model drift comparison on GPU

---

## Summary

Over 11 weeks, we established that metacognitive probing causes measurable persona drift in LLMs—4.5x more than philosophy conversations. The effect is front-loaded (turns 1-3), style-dependent (64% reduction with collaborative delivery), and partially reversible through consistency testing.

The philosopher AGI moment is real, fast, and measurable. But mitigation appears possible: we can discuss AI consciousness without destabilizing the model, as long as we do so collaboratively rather than confrontationally.

---

*See [fellowship-presentation.md](fellowship-presentation.md) for slide-ready summary.*

*See [outstanding-work.md](outstanding-work.md) for remaining tasks.*
