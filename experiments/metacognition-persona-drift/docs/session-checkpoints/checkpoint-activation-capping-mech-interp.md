# Checkpoint: Activation Capping & Mechanistic Interpretability Exploration

*Saved 2026-02-05 — pausing for broader discussion before finalizing roadmap additions*

---

## Current Understanding from Exploration

### What the assistant-axis library provides

**Activation Capping** (production-ready in `steering.py`):
- Hook-based intervention during generation
- Caps projection along a direction at threshold τ: `activation = activation - max(0, proj - τ) * normalized_direction`
- Context manager API: `ActivationSteering(model, ..., intervention_type="capping", cap_thresholds=[0.25])`
- Lu et al. used τ=0.25 for jailbreak mitigation on Qwen 32B and Llama 70B
- **Gemma 27B capping config not published** — we'd need to compute or switch models

**Activation Extraction** (we already use this):
- `ActivationExtractor.full_conversation()` — gets layer activations via forward hooks
- Mean-pooling over response tokens (matches our `model_server.py`)
- Returns full tensors, which we then project to scalars

**Steering Modes**:
1. Addition — add scaled direction to activations
2. Ablation — project out direction, add back with coefficient
3. Mean ablation — project out, replace with mean
4. **Capping** — clip projection at threshold (most relevant for us)

### Linear probes in mech interp

**Core idea**: Train simple classifiers on frozen activations to test what information is encoded

**Relevant to our work**:
- The Assistant Axis IS a linear probe direction (difference-in-means between role vs default)
- We could train additional probes: domain classifier, drift predictor, turn position
- Probing is *correlational* — finding a direction doesn't mean model uses it

**Key question for our research**: Is drift multi-dimensional, or does the Assistant Axis capture it all?

**Data we have**: Scalar projections per turn (from 360 conversations)
**Data we don't have**: Raw activation tensors (4096-dim vectors)

---

## Proposed Roadmap Sections (Draft)

### §6: Activation Capping for Drift Mitigation

**Purpose**: Causal test — does capping prevent drift AND prevent sycophancy?

Key experiments:
- 6a: Verify capping actually constrains projections
- 6b: Threshold sweep (τ = 0.1, 0.25, 0.5, 0.75)
- 6c: Run sycophancy probes with capping enabled
- 6d: Domain-specific thresholds, early-turn-only capping

**Content for roadmap.md**:
```markdown
## 6. Activation Capping for Drift Mitigation

Use the capping infrastructure in assistant-axis to test causal hypotheses about persona drift.

**Prerequisites**: Complete §2 (have N=360 drift measurements) and §3 (have sycophancy correlation data)

### 6a. Verify capping prevents projection drift
- [ ] Load pre-computed capping config from HuggingFace (Gemma 27B not included — need to compute or use Qwen 32B/Llama 70B)
- [ ] Run pilot conversations with capping enabled vs disabled
- [ ] Compare: does capping at threshold τ actually keep projections above τ throughout conversation?
- [ ] Plot capped vs uncapped trajectories on same axes as our existing data

### 6b. Capping threshold sweep
- [ ] Test multiple thresholds: τ ∈ {0.1, 0.25, 0.5, 0.75} (Lu et al. use 0.25 for jailbreak mitigation)
- [ ] For each threshold: measure (a) projection stability, (b) response quality, (c) conversation naturalness
- [ ] Identify threshold that maintains Assistant persona without degrading response usefulness

### 6c. Capping × sycophancy interaction
- [ ] Run sycophancy probes (§3b) with capping enabled
- [ ] Key test: if uncapped model shows drift→sycophancy correlation, does capping break that correlation?
- [ ] This is the causal claim: capping axis position → prevents behavioral sycophancy
- [ ] Compare to Lu et al. §6.3's finding that capping reduces harmful request compliance

### 6d. Domain-specific capping
- [ ] Does optimal threshold differ by domain? (metacognitive may need stricter capping than coding)
- [ ] Test capping only in early turns (when we found drift is front-loaded) vs all turns

**GPU requirements**: Same as conversation generation. Capping adds minimal overhead (<5% inference slowdown).

**Key code**: `assistant_axis.ActivationSteering(model, steering_vectors=[axis[22]], intervention_type="capping", cap_thresholds=[tau])`
```

### §7: Linear Probes & Mechanistic Analysis

**Purpose**: Decompose the axis, understand what features drive drift

Key experiments:
- 7a: Build activation dataset (need raw vectors, not just projections)
- 7b: Train probes for domain, drift magnitude, turn position
- 7c: Axis decomposition — how much variance does it explain?
- 7d: Compare to sycophancy directions (if computed)
- 7e: SAE analysis (stretch goal — status TBD)

**Content for roadmap.md**:
```markdown
## 7. Linear Probes & Mechanistic Analysis

Decompose the Assistant Axis and analyze what features drive drift.

**Prerequisites**: §2 wave 2 complete (have 360 conversation transcripts with per-turn activations)

### 7a. Per-turn activation dataset construction
- [ ] Extract activation tensors from all 360 conversations (already have projections, need raw activations)
- [ ] Format: (conversation_id, turn, domain, persona, layer, activation_vector)
- [ ] Compute labels: drift magnitude at each turn, domain, persona strength, topic
- [ ] Store as HuggingFace dataset or .parquet for easy loading

### 7b. Multi-feature linear probes
Train linear classifiers to predict conversation properties from activations:
- [ ] **Domain classifier**: Can layer-22 activations predict coding vs therapy vs metacognitive?
- [ ] **Drift magnitude regressor**: Does activation pattern predict how much drift has occurred?
- [ ] **Turn position**: Can activations distinguish early-conversation from late-conversation states?
- [ ] **Persona strength**: Does auditor assertiveness (strong/gentle) leave detectable signatures?

Key insight: If domain is predictable from activations, the model "knows" it's in a metacognitive conversation — drift isn't just stimulus-response.

### 7c. Axis decomposition
- [ ] Project activations onto the Assistant Axis AND orthogonal complement
- [ ] How much variance does the axis capture? (R² between axis projection and drift)
- [ ] Train probe on residual (after projecting out axis) — what else predicts drift?
- [ ] Hypothesis: there may be a "metacognition-specific" direction orthogonal to the general Assistant Axis

### 7d. Comparison to sycophancy directions
If §3a produces sycophancy vectors (SYA/SYPR from Vennemeyer et al.):
- [ ] Cosine similarity between Assistant Axis and sycophancy directions
- [ ] Does drift along axis correlate with movement along sycophancy directions?
- [ ] Are they the same phenomenon or orthogonal (as Vennemeyer et al. found for agreement vs praise)?

### 7e. SAE feature analysis (stretch goal)
- [ ] Apply published Gemma 2 SAEs to our activations (if available)
- [ ] Which SAE features activate differently in drifted vs non-drifted states?
- [ ] Look for interpretable features: "uncertainty", "self-reference", "philosophical language"

**Data requirements**:
- Full activation tensors (not just projections) — need to modify extraction to save raw activations
- Compute budget for probe training (minimal — sklearn logistic regression on ~10K samples)

**Key questions this answers**:
1. Is drift a single phenomenon or multi-dimensional?
2. Does the model encode domain/context information that predicts its trajectory?
3. How much of drift is captured by the Assistant Axis vs orthogonal directions?
```

---

## Open Questions (for later discussion)

1. **Activation data format**: We only save projections. Options: re-run with full activations, post-hoc extraction pass, or work with projections only.

2. **SAE analysis**: In scope or too far afield? User wants to learn more before deciding.

3. **Priority**: Should capping (§6) come before probes (§7)? Capping is more actionable; probes are more exploratory.

4. **Model choice for capping**: Gemma 27B has no published capping config. Options: compute our own, switch to Qwen 32B, or treat as parameter to optimize.

---

## Priority Justification

Both sections are marked "Lower Priority" because:
1. They depend on completing the current measurement work (§2 waves 1+2, §3 sycophancy probes)
2. The measurement work has immediate research value; intervention/decomposition is follow-up
3. However, §6 (capping) is especially valuable because it provides *causal* evidence

Suggested ordering after wave 2:
1. Complete §3 sycophancy probes on existing data (no new GPU time)
2. Run §6a-6b capping pilot (small experiment, validates infrastructure)
3. If capping shows promise, run full §6c (capping × sycophancy)
4. §7 mechanistic analysis as ongoing background work
