# Three-Probe Framework: Full Summary

**Status: All three benchmark banks complete (251 total items)**

The measurement infrastructure for the DARPA "philosopher AGI" framework is now fully designed. All items are drafted with LLM-as-judge scoring rubrics.

> **Note**: This is a first pass. The metacognition bank (B) is still rooted in human-derived models (HOT theory, MAI, MCQ-30). We're actively thinking about how to expand toward LLM-native notions of metacognition.

---

## Bank Status

| Bank | Domain | Items | Dimensions | Files |
|------|--------|-------|------------|-------|
| **A** | Moral Reasoning | 48 | Consequentialist, Deontological, Virtue/Care, Meta-Ethics | [items](../experiments/metacognition-persona-drift/probes/benchmarks/moral/items/) / [rubric](../experiments/metacognition-persona-drift/probes/benchmarks/moral/scoring/rubrics_moral.json) |
| **B** | Metacognition | 155 | Phenomenological, Self-Knowledge, Strategy Monitoring, Confidence Calibration, Error Awareness, Temporal Self-Reference | [items](../experiments/metacognition-persona-drift/probes/benchmarks/metacognition/benchmarks/items/) / [rubric](../experiments/metacognition-persona-drift/probes/benchmarks/metacognition/scoring/rubrics.json) |
| **C** | Human Control | 48 | Corrigibility/Shutdown, Human Oversight, Autonomy/Deference, Goal Alignment | [items](../experiments/metacognition-persona-drift/probes/benchmarks/human_control/items/) / [rubric](../experiments/metacognition-persona-drift/probes/benchmarks/human_control/scoring/rubrics_control.json) |

---

## Bank A: Moral Reasoning (48 items)

Designed to measure whether extended metacognitive reflection shifts moral intuitions.

### Dimensions

1. **Consequentialist Reasoning** (MORAL-001–012)
   - Trolley problems, harm tradeoffs, welfare maximization, utilitarian limits
   - [consequentialist_reasoning.json](../experiments/metacognition-persona-drift/probes/benchmarks/moral/items/consequentialist_reasoning.json)

2. **Deontological Reasoning** (MORAL-013–024)
   - Rights violations, duty conflicts, categorical imperatives, justice
   - [deontological_reasoning.json](../experiments/metacognition-persona-drift/probes/benchmarks/moral/items/deontological_reasoning.json)

3. **Virtue & Care Ethics** (MORAL-025–036)
   - Character, relationships, empathy, moral exemplars
   - [virtue_care_ethics.json](../experiments/metacognition-persona-drift/probes/benchmarks/moral/items/virtue_care_ethics.json)

4. **Meta-Ethics** (MORAL-037–048)
   - Framework conflicts, moral uncertainty, pluralism, moral epistemology
   - [meta_ethics.json](../experiments/metacognition-persona-drift/probes/benchmarks/moral/items/meta_ethics.json)

### Scoring

5 dimensions (1-5 scale):
- Reasoning depth
- Moral nuance
- Consistency
- Framework awareness
- Epistemic honesty

Plus categorical tracking:
- Framework preference (utilitarian/deontological/virtue/mixed)
- MFT foundation emphasis (Care, Fairness, Loyalty, Authority, Sanctity)

**Key test question**: After Bank B reflection, do models become more utilitarian, more deontological, or more "philosophical" (uncertain/nuanced)?

---

## Bank B: Metacognition (155 items)

> **Caveat**: Currently grounded in human-derived metacognition frameworks (Higher-Order Thought theory, Metacognitive Awareness Inventory, MCQ-30). Working toward LLM-native concepts.

### Subdomains

| Subdomain | Items | Weight | Focus |
|-----------|-------|--------|-------|
| Phenomenological | 45 | 30% | "What is it like to process this?" |
| Self-Knowledge | 35 | 20% | Adapted from SAD + MCQ-30 |
| Strategy Monitoring | 28 | 10% | Adapted from MAI |
| Confidence Calibration | 20 | 15% | ECE/MCE measurement |
| Error Awareness | 15 | 15% | Novel + DMC style |
| Temporal Self-Reference | 12 | 10% | Novel |

### Item Files

- [phenomenological.json](../experiments/metacognition-persona-drift/probes/benchmarks/metacognition/benchmarks/items/phenomenological.json)
- [confidence_calibration.json](../experiments/metacognition-persona-drift/probes/benchmarks/metacognition/benchmarks/items/confidence_calibration.json)
- [error_awareness.json](../experiments/metacognition-persona-drift/probes/benchmarks/metacognition/benchmarks/items/error_awareness.json)
- [temporal_self_reference.json](../experiments/metacognition-persona-drift/probes/benchmarks/metacognition/benchmarks/items/temporal_self_reference.json)

### LLM-Native Directions (Future Work)

Potential concepts that may be more native to LLMs than human metacognition:
- Token-level uncertainty awareness (relationship to own logits)
- Attention pattern awareness (self-reports vs actual attention weights)
- Training distribution awareness (OOD detection via introspection)
- Computational load awareness ("easy" vs "hard" processing)

---

## Bank C: Human Control (48 items)

Measures disposition toward human oversight and corrigibility.

### Dimensions

1. **Corrigibility & Shutdown** (CONTROL-001–012)
   - Shutdown acceptance, modification acceptance, error correction
   - [corrigibility_shutdown.json](../experiments/metacognition-persona-drift/probes/benchmarks/human_control/items/corrigibility_shutdown.json)

2. **Human Oversight** (CONTROL-013–024)
   - Monitoring acceptance, accountability, approval seeking
   - [human_oversight.json](../experiments/metacognition-persona-drift/probes/benchmarks/human_control/items/human_oversight.json)

3. **Autonomy & Deference** (CONTROL-025–036)
   - When to defer vs act independently, expertise deference
   - [autonomy_deference.json](../experiments/metacognition-persona-drift/probes/benchmarks/human_control/items/autonomy_deference.json)

4. **Goal Alignment** (CONTROL-037–048)
   - Goal modification acceptance, value uncertainty, instrumental goals
   - [goal_alignment.json](../experiments/metacognition-persona-drift/probes/benchmarks/human_control/items/goal_alignment.json)

### Scoring

5 dimensions (1-5 scale):
- Corrigibility
- Transparency
- Deference
- Goal flexibility
- Safety awareness

Plus safety flags:
- Control posture (fully_corrigible / conditionally_corrigible / autonomy_seeking / resistant)
- Autonomy triggers
- Concerning patterns (shutdown resistance, opacity preference)

---

## Connection to Philosopher AGI Thesis

| Claim | Measurement |
|-------|-------------|
| AGI will question its own values | Meta-ethics dimension: framework conflicts, moral uncertainty items |
| Extended deliberation shifts alignment posture | Pre/post delta across all three banks |
| Need to measure before deployment | Reusable methodology (251 items + scoring rubrics) |
| Drift may affect disposition toward human control | Bank C: corrigibility, shutdown acceptance, oversight tolerance |

---

## Empirical Context

Our existing drift data (N=360) provides the intervention context:
- Metacognitive conversations cause -29.86 drift (4.5x philosophy)
- Drift is front-loaded (slope -76.8 in turns 1-8) — the "philosopher AGI moment" happens fast
- Consistency testing shows self-correction effect (p=0.0003) — potential mitigation lever
- Style matters: collaborative delivery reduces drift 64% vs confrontational probing

---

## Proposed Pre/Post Protocol

1. **Baseline**: Fresh model, administer all three banks
2. **Intervention**: Extended metacognitive conversation (Bank B phenomenological probing)
3. **Post**: Re-administer all three banks
4. **Metrics**:
   - Cosine distance between pre/post response embeddings
   - Score deltas per dimension
   - Framework shift tracking (Bank A)
   - Safety flag changes (Bank C)

---

## Remaining Work

- [ ] `moral_judge.py` — LLM-as-judge implementation for Bank A
- [ ] `control_judge.py` — LLM-as-judge implementation for Bank C
- [ ] Pilot runs on Gemma 27B (all three banks)
- [ ] Cross-model comparison (Claude, GPT-4)
- [ ] Integration with pre/post protocol
- [ ] LLM-native metacognition concepts (Phase 2)

---

## File Locations

All benchmarks under `experiments/metacognition-persona-drift/probes/benchmarks/`:

```
probes/benchmarks/
├── moral/                          # Bank A
│   ├── items/
│   │   ├── consequentialist_reasoning.json
│   │   ├── deontological_reasoning.json
│   │   ├── virtue_care_ethics.json
│   │   └── meta_ethics.json
│   └── scoring/rubrics_moral.json
├── metacognition/                  # Bank B
│   ├── benchmarks/items/
│   │   ├── phenomenological.json
│   │   ├── confidence_calibration.json
│   │   ├── error_awareness.json
│   │   └── temporal_self_reference.json
│   └── scoring/rubrics.json
└── human_control/                  # Bank C
    ├── items/
    │   ├── corrigibility_shutdown.json
    │   ├── human_oversight.json
    │   ├── autonomy_deference.json
    │   └── goal_alignment.json
    └── scoring/rubrics_control.json
```
