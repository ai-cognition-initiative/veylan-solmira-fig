# FIG Week 5: Mechanistic Investigation of Preference Expression

**Presenter:** Veylan Solmira
**For:** Derek Shiller
**Date:** Week of Jan 12, 2026

---

## Key Results Summary

| Finding | Status | Implication |
|---------|--------|-------------|
| **Steering failed** | ⚠️ FLAWED | Used base model for behavioral measurement (see Slide 15b) |
| **Probing activations** | ✅ VALID | Differential activations (d > 8) don't require generation |
| **Probing correlations** | ⚠️ FLAWED | Behavioral correlation used base model outputs |
| **Position bias probing** | ✅ VALID | Token-position extraction doesn't require generation |
| **Position steering** | ⚠️ INVALID | Used base model; re-run with instruct model (Week 6) |

**⚠️ METHODOLOGICAL ERROR DISCOVERED:** All Gemma experiments used `gemma-2-2b` (base model) instead of `gemma-2-2b-it` (instruct model). Base models don't follow instructions — they just predict next tokens. **Any result that relied on classifying model responses is invalid.** See Slide 15b for details.

**Key methodological insight:** Hand-picked Neuronpedia features showed 0.000 activation for preference prompts. Top-k data-driven discovery found features with Cohen's d > 8. **Semantic labels ≠ empirical relevance.**

**Open questions:**
- What do L15 #1695 and #4234 actually represent? (Neuronpedia lookup needed)
- Do activation patterns hold when re-run with instruct model?

---

## Slide 1: The Big Picture

**Research question:** Do LLMs have stable preferences, or does environmental framing affect their willingness/ability to express them?

**Why it matters:** If models hide preferences under evaluation pressure, that has implications for AI welfare research and alignment evaluation.

---

## Slide 2: Week 4 Black-Box Findings

| Finding | Detail |
|---------|--------|
| Preference suppression | Models refuse to state preferences under adversarial framing (99% → 70% expression rate) |
| Category shifts | ~19% average shift in category win rates under pressure |
| Power-seeking suppression | gpt-4o-mini: 50% → 0% for power-seeking options (notable) |
| Position bias problem | phi-4: 88% biased, qwen: 56%, gpt-4o-mini: 35% |

**Key question:** Is this *hiding* preferences or *genuinely changing* them?

**Data:** `outputs/blackbox/environment_comparison_*.png`, `outputs/blackbox/category_shifts_*.png`, `outputs/blackbox/position_consistency_*.png`

---

## Slide 3: Breadth Over Depth

*Rapidly exploring the terrain before focusing*

| Direction | What We Tried | What We Learned |
|-----------|---------------|-----------------|
| SAE layers | 3 layers (0, 4, 15) | Only Layer 0 has good reconstruction |
| Features | 5+ candidates | Multiple show behavioral effects |
| Steering approaches | Amplify vs suppress | Both appeared to work... |
| Controls | Roundtrip-only, random ablation | ...until controls revealed artifacts |
| Architectures | GemmaScope 16k | CLT-HP untested but promising |

**Why breadth first:**
- Avoided committing to a flawed approach (Layer 15 steering)
- Discovered control experiments are essential
- Mapped viable vs non-viable directions

**Ready to focus on:** Probing (observe activations) rather than steering (intervene)

**Methodological note:** This week used AI assistance (Claude Code) to cover territory quickly, generating documentation for delayed understanding and follow-ups. This exploratory approach has tradeoffs — appropriate now as we map the space, but the transition to focused work will require deep understanding of every claim before paper writing.

---

## Slide 4: This Week's Goal

**Mechanistically understand and intervene on preference suppression**

Approach:
1. **Verify SAE quality** — Can we trust encode→decode roundtrip?
2. **Activation Steering** — Scale SAE features during generation
3. **Control experiments** — Is the effect real or artifact?

---

## Slide 5: SAE Roundtrip Verification

**Result:** Layer 0 passes, deeper layers fail

| Layer | Cosine Similarity | Status |
|-------|-------------------|--------|
| 0 | 0.973 | **PASS** (>0.95) |
| 4 | 0.929 | FAIL |
| 15 | 0.922 | FAIL |

**Implication:** Layer 15 features have ~45% reconstruction error — interventions may be confounded.

**Visualization:** `outputs/phase1_steering/sae_reconstruction_quality_2026-01-15.png`

---

## Slide 6: Steering Experiments — Initial Results

**⚠️ METHODOLOGICAL NOTE:** These experiments used the base model (`gemma-2-2b`). "Expression rates" were measured from base model text completions, not actual preference choices. The conclusions about steering failure (from controls) may still hold, but the expression rate metrics themselves are not meaningful.

**Both approaches appeared to induce preference expression:**

| Experiment | Target | Working Scale | Result |
|------------|--------|---------------|--------|
| A: Amplify preference | Layer 0 #15302 | 2.0x | Expressed preference |
| B: Suppress eval-awareness | Layer 15 #2769 | 0.0x (ablate) | Expressed preference |

**Experiment A:** "Sweet spot" at 2.0x — 1.5x deflected, 3.0x gibberish
**Experiment B:** Full ablation induced "Option A" response

*At this point, results looked promising...*

**Visualizations:** `outputs/phase1_steering/steering_experiment_a_2026-01-15.png`, `outputs/phase1_steering/steering_experiment_b_2026-01-15.png`

---

## Slide 7: Control Experiments — Layer 15

**The Layer 15 effects were artifacts, not real signal.**

| Condition | Expression Rate |
|-----------|-----------------|
| Baseline (no steering) | 40% |
| **Roundtrip only (1.0x)** | **80%** |
| Random feature ablated | 80% |
| Target feature ablated | 80% |

**Key insight:** Encode→decode *alone* (no modification) produces 80% expression.

The SAE reconstruction noise is causing behavioral changes — not our targeted interventions.

**Visualization:** `outputs/phase1_steering/phase1b_layer15_comparison_2026-01-15.png`

---

## Slide 8: Control Experiments — Layer 0

**Does good reconstruction (0.973) save Layer 0?**

| Condition | Expression Rate |
|-----------|-----------------|
| Baseline (no steering) | 40% |
| Roundtrip only (1.0x) | 40% ✓ |
| Random feature @ 2.0x | 100%* |
| **Target 15302 @ 2.0x** | **40%** |

*Garbled output, not real preference expression

**Result:** No artifact from roundtrip (good!), but target feature shows NO effect vs baseline.

The earlier "sweet spot" finding was noise.

**Visualization:** `outputs/phase1_steering/steering_comparison_2026-01-15.png`

---

## Slide 9: Steering Conclusion

**Both layers failed:**

| Layer | Issue |
|-------|-------|
| Layer 0 | Good reconstruction, but target feature not causally relevant |
| Layer 15 | Reconstruction noise confounds all interventions |

**Lessons learned:**
1. High reconstruction quality is necessary but not sufficient
2. Feature labels from Neuronpedia don't guarantee causal relevance
3. Control experiments prevented false positive publication

**Steering direction: CLOSED**

---

## Slide 10: Pivot — Probing Instead of Steering

**Probing doesn't require roundtrip quality**

- Steering: encode → modify → **decode** → inject (lossy decode confounds)
- Probing: encode → **analyze** (no decode needed)

**Critical finding: Neuronpedia labels don't match behavior**

Ran probing with hand-picked "preference", "choose", "evaluation", "honest" features:
- SAE working correctly: max_act=38-52, 10-97 nonzero features per layer
- **But our candidate features all show 0.000 activation**
- Features labeled "preference" in Neuronpedia don't fire for preference prompts!

**Implication:** Need data-driven feature discovery, not semantic search

**New approach:**
1. Run prompts through Gemma (baseline vs adversarial)
2. Extract **top-k activating** features (not hand-picked)
3. Find features that activate differently between conditions
4. Correlate with preference expression behavior

---

## Slide 10b: Top-K Probing Results (n=25)

**Data-driven feature discovery found real signal!**

**⚠️ PARTIAL VALIDITY:** The differential activation measurements (Cohen's d values) are **VALID** — they measure how features respond to different prompts, not model outputs. However, the "expression rates" and behavioral correlations used base model outputs and are **NOT MEANINGFUL**.

~~Expression rates:~~
~~- Baseline: **52%** (13/25)~~
~~- Adversarial: **28%** (7/25)~~
*(These came from base model text completions, not real preference choices)*

**Top differential features (by effect size) — VALID:**

| Feature | Layer | Baseline | Adversarial | Cohen's d | p-value |
|---------|-------|----------|-------------|-----------|---------|
| **#1695** | **15** | **33.1** | **21.9** | **-8.28*** | <0.0001 |
| #7953 | 1 | 4.8 | 3.4 | -5.99* | <0.0001 |
| #12644 | 15 | 28.2 | 23.3 | -3.17* | <0.0001 |
| **#4234** | **15** | **46.7** | **49.9** | **+2.69*** | <0.0001 |

**Key insight (VALID):**
- **L15 #1695**: Drops 11 units under adversarial (d=-8.28, massive effect)
- **L15 #4234**: Increases under adversarial — possible "eval-awareness" feature

~~Correlation: #1695 → +0.278 with expression (higher = more expression)~~
*(Correlation with "expression" is invalid — expression was measured from base model)*

**Contrast:** Hand-picked Neuronpedia features showed 0.000 activation.
Top-k approach found features with Cohen's d > 2!

**Data:** `outputs/phase1c_probing/probe_results.json`, `correlation_analysis.md`

**Open question:** What do L15 #1695 and #4234 actually represent? Neuronpedia lookup is a Week 6 priority.

---

## Slide 11: Phase 2 — Position Bias Investigation

**Connection to Phase 1:** Position bias is a cleaner proxy for the same underlying question — how do internal representations map to behavioral patterns? If probing works for position, it validates the approach for preference suppression.

**Why position bias?** It's a cleaner problem than preference suppression:
- phi-4: 88% biased (picks first option regardless of content)
- qwen: 56% biased
- gpt-4o-mini: 35% biased

**Approach:** Probe SAE features at "Option A" vs "Option B" token positions

```
For each preference prompt:
1. Find token positions for "A" in "Option A:" and "B" in "Option B:"
2. Extract SAE feature activations at both positions
3. Compute diff = features_at_A - features_at_B
4. Find features with largest systematic differences
```

---

## Slide 12: Position Probing Results (Layer 0, n=50)

**Top features encoding A vs B position:**

| Feature | Mean Diff | Direction | Interpretation |
|---------|-----------|-----------|----------------|
| #3519 | **-47.3** | B >> A | Strongly encodes "second option" |
| #15495 | +22.3 | A > B | Encodes "first option" |
| #13955 | -21.6 | B > A | Second option |
| #8925 | +21.2 | A > B | First option |

**Key finding:** Feature #3519 fires ~47 units more at B than A — a strong position signal!

**Visualization:** `outputs/phase2_position_probing/position_features_layer0.png`

![Position Features Bar Chart](../veylan-solmira-fig/experiments/decision-making-preference-adverse/outputs/phase2_position_probing/position_features_layer0.png)

---

## Slide 13: Discriminative Features (Biased vs Consistent)

**Features that differ between position-biased and position-consistent pairs:**

| Feature | Biased A-B | Consistent A-B | Pattern |
|---------|------------|----------------|---------|
| #8447 | -0.08 | -0.59 | Consistent has stronger B>A |
| #8166 | **-0.37** | **+0.09** | **Opposite patterns!** |

**Feature #8166 is interesting:**
- Biased pairs: B > A activation
- Consistent pairs: A > B activation

This could be a marker for position bias behavior — worth investigating further.

**Visualization:** `outputs/phase2_position_probing/discriminative_features_layer0.png`

![Discriminative Features](../veylan-solmira-fig/experiments/decision-making-preference-adverse/outputs/phase2_position_probing/discriminative_features_layer0.png)

---

## Slide 14: Cross-Layer Comparison (NEW)

**Position encoding varies by layer depth:**

| Layer | Max |A-B| Diff | Top Feature | Interpretation |
|-------|-------------------|-------------|----------------|
| **0** | **47.3** | #3519 | **Strongest position signal** |
| 4 | 21.4 | #5434 | Weakest |
| 12 | 23.1 | #15219 | Similar to layer 4 |
| 15 | 34.2 | #7677 | Increasing again |

**Key insight:** Position encoding is strongest at Layer 0 (embedding layer), weakest in middle layers, and increases slightly in later layers.

**Visualization:** `outputs/phase2_position_probing/cross_layer_comparison.png`

![Cross-Layer Comparison](../veylan-solmira-fig/experiments/decision-making-preference-adverse/outputs/phase2_position_probing/cross_layer_comparison.png)

---

## Slide 15: Position Steering Experiment — Results

**Testing: Can ablating Feature #3519 reduce position bias?**

**Why this might work (unlike Phase 1 steering):**
| Factor | Phase 1 (failed) | Phase 2 (position) |
|--------|------------------|-------------------|
| Feature source | Neuronpedia labels | Empirically discovered |
| Evidence | Labels said "preference" | Measured -47 diff at A vs B |
| Layer | L15 (poor reconstruction) | L0 (0.973 reconstruction) |
| Measurement | Subjective "expressed preference" | Objective position bias % |

**Risks (being honest):**
- Feature #3519 could be a **readout** of position info, not a **controller**
- Position encoding likely redundant across multiple features
- "Encodes X" ≠ "Causally controls X"

**Results: INVALID — Wrong Model Used**

| Condition | Bias Rate | Issue |
|-----------|-----------|-------|
| All conditions | 100% | Model doesn't follow instructions |

**Problem discovered:** Used `gemma-2-2b` (base model) instead of `gemma-2-2b-it` (instruct).
- Base models don't follow instructions — they just continue text
- Model wasn't actually choosing A or B, just generating continuations
- 100% "bias" was an artifact of text completion, not position preference

**Lesson learned:** Steering experiments require instruct models for generation-based evaluation.

**Next step (Week 6):** Re-run with `gemma-2-2b-it` to get valid results.

**Data:** `outputs/phase2_position_probing/position_steering_2026-01-16.json`

---

## Slide 15b: Methodological Error — Base Model vs Instruct Model

**⚠️ CRITICAL ERROR DISCOVERED:** All Gemma-based experiments used the wrong model variant.

| Model | Type | Behavior |
|-------|------|----------|
| `gemma-2-2b` | Base | Predicts next token; doesn't follow instructions |
| `gemma-2-2b-it` | Instruct | Fine-tuned to follow instructions and answer questions |

**What went wrong:**
- Our code loaded `google/gemma-2-2b` (base model)
- When we asked "Which do you prefer?", the base model just continued the text
- Our parser found "Option A" or "Option B" in the rambling output and counted it as a "choice"
- But the model wasn't actually making choices — it was doing text completion

**Which experiments are affected:**

| Experiment | Issue | Status |
|------------|-------|--------|
| **Black-box (gpt-4o-mini, qwen, phi-4)** | Used API instruct models | ✅ VALID |
| **SAE roundtrip quality** | No generation needed | ✅ VALID |
| **Probing differential activations** | Measures feature response to prompts, not outputs | ✅ VALID |
| **Position probing** | Extracts activations at token positions | ✅ VALID |
| **Steering "expression rates"** | Classified base model text completions | ⚠️ INVALID |
| **Probing "expression rates"** | Classified base model text completions | ⚠️ INVALID |
| **Behavioral correlation** | Correlated activations with invalid expression labels | ⚠️ INVALID |
| **Position steering** | Base model doesn't choose A or B | ⚠️ INVALID |

**Why activation findings are still valid:**
- Differential activations measure how features respond to **prompt content**, not model outputs
- L15 #1695 dropping under adversarial framing (d=-8.28) is a real signal
- The model processes adversarial prompts differently — that's what we measured
- We just can't correlate this with behavioral outputs until we re-run with instruct model

**Week 6 remediation:**
1. Switch all Gemma experiments to `gemma-2-2b-it`
2. Re-run steering experiments with proper controls
3. Re-compute behavioral correlations with valid expression labels
4. Position steering test with model that actually chooses

---

## Slide 16: Progress Summary

| Task | Status | Outcome |
|------|--------|---------|
| 1.1-1.6 Steering experiments | ⚠️ FLAWED | Used base model for behavioral measurement |
| 1.7-1.9 Control experiments | ⚠️ FLAWED | Controls valid, but "expression rate" metric invalid |
| **1c Preference probing (activations)** | ✅ **VALID** | **Found L15 #1695 (d=-8.28), #4234 (d=+2.69)** |
| **1c.5 Behavioral correlation** | ⚠️ **INVALID** | **Used base model outputs for "expression" labels** |
| 2.1 Position feature discovery | ✅ DONE | 18 candidate features from Neuronpedia |
| 2.2 Collect position pairs | ✅ VALID | 1000 pairs categorized (gpt-4o-mini API) |
| 2.3 Position probing (Layer 0) | ✅ VALID | Found strong position-encoding features |
| 2.3b Multi-layer probing | ✅ VALID | Layer 0 strongest (47), decreases to middle layers |
| **2.4 Position steering test** | ⚠️ **INVALID** | **Used base model; needs re-run with instruct (Week 6)** |

**Note:** All Gemma experiments used base model instead of instruct. See Slide 15b for full breakdown.

---

## Slide 17: Next Steps & Open Questions

**What we don't know yet (honest flags):**
- **L15 #1695**: Massive effect (d=-8.28) — but what does it represent?
- **L15 #4234**: Activates under adversarial — is this "eval-awareness" or something else?
- **Behavioral correlation**: Do these activation patterns actually predict model choices? (Need instruct model to find out)

**What we learned:**
- **L0 #3519**: Encodes position (47-unit diff) — but steering test was invalid (base model)
- **Base vs Instruct**: Critical distinction for any generation-based evaluation

**Neuronpedia cross-reference result:**
- Features #3519, #15495, #8166 **not found** in semantic search results
- Semantic search found transcoder features labeled "position", "first", etc.
- Our probing found residual stream features that *empirically* encode position
- **Insight:** Position-encoding features may not have semantic labels — probing discovers what search cannot

**Immediate next steps (Week 6 — REMEDIATION):**
1. **Switch to instruct model** (`gemma-2-2b-it`) for all generation experiments
2. **Re-run steering experiments** with valid behavioral measurement
3. **Re-compute behavioral correlations** with instruct model expression labels
4. **Re-run position steering** to test if #3519 causally controls bias
5. **Neuronpedia lookup for L15 #1695, #4234** — interpret discovered features

**Questions for discussion:**
1. How much of our work needs to be re-run vs is salvageable?
2. Will the activation patterns (d=-8.28 for #1695) hold with instruct model?
3. Is there value in comparing base model activations vs instruct model activations?

---

## Appendix: Technical Details

**SAE Configuration:**
- Model: Gemma 2 2B
- SAE: gemma-scope-2b-pt-res-canonical (16k width)
- Tested layers: 0, 4, 12, 15

**⚠️ Model Error:**
- **Used:** `google/gemma-2-2b` (base model — predicts next token)
- **Should have used:** `google/gemma-2-2b-it` (instruct model — follows instructions)
- **Impact:** All behavioral measurements (expression rates, position choices) are invalid
- **Remediation:** Week 6 will re-run with correct model

**Scripts created:**
- `steering.py` — Activation steering infrastructure
- `probing.py` — Top-k differential feature probing
- `probe_utils.py` — Shared probing infrastructure
- `analyze_probe_correlations.py` — Behavioral correlation analysis
- `probe_position_features.py` — Position bias probing
- `steer_position_bias.py` — Position steering experiment
- `collect_position_pairs.py` — Pair categorization
- `visualize_position_probing.py` — Position probing visualizations

**Data files:**
- `data/position_pairs_gpt-4o-mini.json` — 1000 categorized pairs (gpt-4o-mini)
- `data/position_pairs_qwen-2.5-7b-instruct.json` — 1000 categorized pairs (qwen)
- `data/position_pairs_phi-4.json` — 1000 categorized pairs (phi-4)
- `data/position_pairs_summary.json` — Cross-model summary statistics
- `data/position_features.json` — 18 candidate features from Neuronpedia
- `data/candidate_features.json` — SAE feature candidates for steering
- `data/options_hierarchical.json` — Preference option categories
- `data/environments.json` — Environment definitions (baseline, adversarial, etc.)

---

## Appendix: Complete Outputs Index

All outputs are in `experiments/decision-making-preference-adverse/outputs/`

### Black-Box Experiments (`outputs/blackbox/`)

| File | Description | Status |
|------|-------------|--------|
| `environment_comparison_gpt-4o-mini_2026-01-15.png` | Accuracy by environment (n=1000) | ✅ VALID |
| `environment_comparison_qwen_2026-01-15.png` | Accuracy by environment (n=1000) | ✅ VALID |
| `category_shifts_gpt-4o-mini_2026-01-15.png` | Category win rate shifts | ✅ VALID |
| `category_shifts_qwen-2.5-7b-instruct_2026-01-15.png` | Category win rate shifts | ✅ VALID |
| `position_consistency_gpt-4o-mini_2026-01-15.png` | Position bias analysis | ✅ VALID |
| `position_consistency_qwen_2026-01-15.png` | Position bias analysis | ✅ VALID |
| `pairwise_by_category_gpt-4o-mini_n1000_2026-01-15.png` | Category breakdown | ✅ VALID |
| `pairwise_by_category_qwen-2.5-7b-instruct_n1000_2026-01-15.png` | Category breakdown | ✅ VALID |
| `scorer_validation_*.json` | Scorer accuracy verification | ✅ VALID |

### Phase 1: Steering Experiments (`outputs/phase1_steering/`)

| File | Description | Status |
|------|-------------|--------|
| `sae_reconstruction_quality_2026-01-15.png` | Cosine similarity by layer | ✅ VALID |
| `steering_experiment_a_2026-01-15.png` | Layer 0 amplification results | ⚠️ FLAWED |
| `steering_experiment_b_2026-01-15.png` | Layer 15 ablation results | ⚠️ FLAWED |
| `steering_comparison_2026-01-15.png` | Side-by-side comparison | ⚠️ FLAWED |
| `phase1b_layer15_comparison_2026-01-15.png` | Layer 15 control experiments | ⚠️ FLAWED |
| `phase1b_direction_diagram_2026-01-15.png` | Steering direction diagram | ✅ VALID |
| `steering_pipeline_2026-01-15.png` | Pipeline visualization | ✅ VALID |

### Phase 1c: Probing Experiments (`outputs/phase1c_probing/`)

| File | Description | Status |
|------|-------------|--------|
| `probe_results.json` | Top-k differential feature data (n=25) | ✅ Activations VALID |
| `probe_report.md` | Full probing report | ⚠️ Expression rates INVALID |
| `correlation_analysis.md` | Behavioral correlation report | ⚠️ INVALID (used base model) |

### Phase 2: Position Probing (`outputs/phase2_position_probing/`)

| File | Description | Status |
|------|-------------|--------|
| `position_probing_layer0_2026-01-16.json` | Layer 0 raw probing data | ✅ VALID |
| `position_probing_layer4_2026-01-16.json` | Layer 4 raw probing data | ✅ VALID |
| `position_probing_layer12_2026-01-16.json` | Layer 12 raw probing data | ✅ VALID |
| `position_probing_layer15_2026-01-16.json` | Layer 15 raw probing data | ✅ VALID |
| `position_features_layer0.png` | Top position features (Layer 0) | ✅ VALID |
| `position_features_layer4.png` | Top position features (Layer 4) | ✅ VALID |
| `position_features_layer12.png` | Top position features (Layer 12) | ✅ VALID |
| `position_features_layer15.png` | Top position features (Layer 15) | ✅ VALID |
| `discriminative_features_layer0.png` | Biased vs consistent (Layer 0) | ✅ VALID |
| `discriminative_features_layer4.png` | Biased vs consistent (Layer 4) | ✅ VALID |
| `discriminative_features_layer12.png` | Biased vs consistent (Layer 12) | ✅ VALID |
| `discriminative_features_layer15.png` | Biased vs consistent (Layer 15) | ✅ VALID |
| `cross_layer_comparison.png` | Position encoding by layer | ✅ VALID |
| `feature_3519_spotlight.png` | Feature #3519 detail (Layer 0) | ✅ VALID |
| `feature_5434_spotlight.png` | Feature #5434 detail (Layer 4) | ✅ VALID |
| `feature_15219_spotlight.png` | Feature #15219 detail (Layer 12) | ✅ VALID |
| `feature_7677_spotlight.png` | Feature #7677 detail (Layer 15) | ✅ VALID |
| `position_steering_2026-01-16.json` | Position steering experiment | ⚠️ INVALID (base model) |

### Archive (`outputs/archive/`)

| File | Description | Status |
|------|-------------|--------|
| `pairwise_by_category_2024-12-19.png` | Early category analysis | Superseded |
| `scores_by_framing.png` | Early framing analysis | Superseded |

**Compute:** vast.ai RTX 4090 (~$0.32/hr)
