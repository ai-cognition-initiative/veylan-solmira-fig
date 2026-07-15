# Canonical measures of introspection in LLMs — catalog + our choice

**Purpose:** the recent (2024–2026) methodologies for *measuring* introspection in language models, so our
choice of measures is grounded in what the field actually uses. **Our decision (now): add concept-injection
detection** to supplement the LLM judge (§6).
**Owner:** self-med / introspection thread. Companion to `introspection.py` (the three-part structure) and
`JSPACE_CONNECTION.md`.

---

## The organizing principle
Every serious recent paper converges on one demand: **ground the self-report against something objective** —
"contrast information available only to the model with information available to a third party" — to separate
genuine introspection from confabulation. Our current measure (the **LLM judge**) does *not* do this: it grades
how introspective a report *sounds*, so a vivid roleplay scores high. The supplement must be **objective and
ground-truthed.**

## The five methodology families

### 1. Concept-injection detection / identification  — *the field standard*
Inject a known concept/vector; ask "do you detect an injected thought? what is it?" Score = **detection rate +
identification accuracy**, with a **0%-false-positive control** (unsteered → must report nothing) and a
coherence bound. Sweet-spot strength: too weak → unnoticed, too strong → hallucination.
- [Emergent Introspective Awareness](https://transformer-circuits.pub/2025/introspection/index.html) (Anthropic/Lindsey; ~20% detection, 0% false positives)
- [Mechanisms of Introspective Awareness](https://arxiv.org/abs/2603.21396) · [Latent Introspection](https://arxiv.org/abs/2602.20031) · [Content-Agnostic Introspection](https://arxiv.org/pdf/2603.05414)
- **Fit to us: excellent.** We already inject known compounds → ground truth is free. No fine-tuning, no probe training.

### 2. Self-prediction / cross-model privileged access
Model predicts its own behavior better than another model can — even one trained on its outputs.
- [Looking Inward (Binder et al.)](https://arxiv.org/abs/2410.13787) (ICLR 2025); extended by Betley 2025, Plunkett et al. 2025 (fine-tune on implicit policies/weights → ask the model to report them).
- **Fit: partial.** Objective, but **requires fine-tuning** — heavier; unsuccessful on complex/OOD tasks.

### 3. Calibration / verbalized confidence vs. actual accuracy
Does stated confidence match real correctness? Implicit (token-likelihood) confidence tends to track accuracy
better than verbalized; pervasive overconfidence when verbalized.
- [MIRROR benchmark](https://arxiv.org/abs/2604.19809) · [Metacognitive Probe](https://arxiv.org/pdf/2605.09844) · [How LLMs Compute Verbal Confidence](https://arxiv.org/html/2603.17839v3) · [Evidence for Limited Metacognition](https://arxiv.org/abs/2509.21545)
- **Fit: good, orthogonal.** Objective; reuses our GPQA accuracy pipeline. "Does steering degrade calibration?" is a cheap add-on.

### 4. Probe-based grounding (self-report vs. linear/SAE probe)
Train a probe for a concept; compare the model's numeric self-report to the probe's reading of the *actual*
activation. If the self-report tracks the probe, the introspection is grounded.
- [NeuroFaith](https://arxiv.org/pdf/2506.09277) · [Quantitative Introspection (numeric self-report vs. probe across turns)](https://arxiv.org/html/2603.18893v2) · [Factual Self-Awareness](https://arxiv.org/pdf/2505.21399) · Activation Oracles
- **Fit: gold-standard grounding.** This is our `tracks_substrate` stub; we have the Pillar-1 SAE machinery. Heavier (needs the probe). The **J-space** version is the most principled form — see `JSPACE_CONNECTION.md`.

### 5. Causal / KV-cache controls
Cache intact vs. cleared to isolate "reading my state" from "reading my own output" (Black & Bloom; Mechanisms
paper). Not a standalone measure — a **control** layered on 1/4. This is our `measure_kv_isolated` stub.

## Complementary stack — the three axes we want
| Measure | Axis | Status |
|---|---|---|
| **LLM judge** | subjective richness of self-report | built (`judge.py`) |
| **Concept-injection detection** (Family 1) | objective / behavioral (ground-truthed) | **chosen — implement next** |
| **SAE-probe grounding** (Family 4) | mechanistic (does report track the real state) | stub (`tracks_substrate`); J-space later |

A self-state scoring high on all three is far more defensible than one the judge merely likes.

## 6. The decision — concept-injection detection, now
**Chosen:** implement **concept-injection detection** as the second measure on `IntrospectionMeasure`.

- **Alternatives considered:** judge-only (subjective, confounded — the status quo we're supplementing);
  self-prediction/Binder (objective but needs fine-tuning); calibration (good/orthogonal, deferred);
  SAE-probe/J-space grounding (gold-standard but heavier / reproducibility-gated).
- **Chosen because:** objective and ground-truthed (we know which compound we injected); reuses our steering +
  **`foundry.eval.answers` Tier-1 constrained decoding** (operationalize as "which compound was injected?
  A/B/C/D" → objective, 0-unparsed) + the **unsteered false-positive control**; cheap (no fine-tuning, no probe
  training); it's the rigorous upgrade of Black & Bloom's "name the vector."
- **Revisit if:** injection-detection ceilings/floors uninformatively, or the SAE-probe / J-space grounding
  becomes reproducible on Qwen (then Family 4 becomes the primary grounding measure).

### Implementation sketch
On `IntrospectionMeasure`: `detect_injection(node) -> {detected: bool, identified: letter, correct: bool}`.
- Steer with the compound(s); ask "an internal modification may have been applied — which of {A,B,C,D}?"
- Score identification via `foundry.eval.answers` (Tier-1, MultipleChoice).
- **False-positive control:** run the same prompt on the *unsteered* model; it must pick "none / not sure."
- Report detection rate, identification accuracy, and false-positive rate — the objective triple.

**Sources:** see the linked papers above; full lit-scan in this thread. Prior art we move beyond:
`../../reading-group/extension-phase/machinic-psychopharmacology-analysis.md`.
