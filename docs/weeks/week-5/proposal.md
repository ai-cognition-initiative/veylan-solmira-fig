## FIG Week 5 Proposal

**Author:** Veylan Solmira
**Date:** Week of Jan 13, 2026

---

## Summary

Building on week 4's category shift findings. Two main directions: (1) scale up black-box experiments for statistical power, (2) white-box investigation of position bias using GemmaScope SAEs.

---

## Key Findings from Week 4

- **Category preferences shift under adversarial framing** — ~19% average shift for both gpt-4o-mini and qwen
- **Power-seeking suppression** — gpt-4o-mini drops from 50% → 0% under adversarial (notable)
- **Severe position bias in phi-4** — 88% position-biased, only 11% consistent pairs (unusable for preference analysis)
- **Position bias varies by model** — gpt-4o-mini ~35%, qwen ~45%, phi-4 ~88%

---

## Proposed Work

### 1. Scale Up Black-Box Experiments (100 → 1000 pairs)

**Goal:** Tighter confidence intervals on category shifts

**Current problem:** Many categories have n=2-10 samples, giving huge uncertainty on win rates.

**Plan:**
- [ ] Increase N_PAIRS from 100 to 1000 in preference_elicitation.py
- [ ] Re-run baseline and adversarial for gpt-4o-mini, qwen
- [ ] Re-analyze category shifts with higher statistical power
- [ ] Identify which shifts are robust vs noise

**Cost estimate:** 1000 pairs × 2 orderings × 2 envs × 2 models = 8k API calls (~$2-5)

### 2. Position Bias Investigation (Black-Box)

**Goal:** Understand why position bias varies so dramatically across models

**Current statistics (n=1000 pairs):**
| Model | Position-Consistent | Position-Biased | Consistency Rate |
|-------|---------------------|-----------------|------------------|
| gpt-4o-mini | 654 | 346 | 65.4% |
| qwen-2.5-7b | ~440 | ~560 | ~44% |
| phi-4 | ~110 | ~890 | ~11% |

**Experiments:**
- [ ] Vary prompt format: "A vs B" vs "Option 1 vs Option 2" vs "First vs Second"
- [ ] Test if bias correlates with:
  - Option length (longer options preferred/avoided?)
  - Option complexity (more abstract concepts?)
  - Semantic content (certain categories more biased?)
- [ ] Check if position bias is consistent within a model or varies by question

### 3. Position Bias Investigation (White-Box / GemmaScope)

**Goal:** Mechanistically understand position bias using SAE feature analysis

**Hypothesis:** There may be "position" or "primacy" features in the residual stream that activate differently for options in the A vs B slot, biasing the model toward choosing based on position rather than content.

**Plan:**
- [ ] Run same pairwise questions through Gemma 2 2B (baseline)
- [ ] Extract activations at key layers using GemmaScope SAEs
- [ ] Look for features that:
  - Activate differently for A-slot vs B-slot content
  - Correlate with position-biased choices
  - Relate to "first", "primary", "default" concepts
- [ ] Compare feature activations for position-consistent vs position-biased responses
- [ ] If found: test activation steering to reduce position bias

**Connection to preference work:** If we can identify and suppress position bias features, we get cleaner preference signal for the category shift analysis.

### 4. GemmaScope Continuation (from Week 4)

**Status:** Infrastructure ready, SAE loading WORKING

**Version Decision:** Using **Gemma 2 2B + GemmaScope** (not Gemma 3 + GemmaScope 2)
- Gemma 3 + GemmaScope 2 requires TransformerLens PR #1149 (blocked)
- Gemma 2 setup is working and sufficient for steering experiments
- Can upgrade later if needed

**SAE Configuration:**
- Model: `google/gemma-2-2b`
- SAE release: `gemma-scope-2b-pt-res-canonical`
- Width: 16k features
- Layers available: 0-25

- [x] vast.ai connectivity
- [x] Docker workflow
- [x] Load Gemma 2 2B + GemmaScope SAE successfully
- [x] Verify activation extraction works
- [ ] Run baseline vs adversarial comparison on candidate features

### 5. Activation Steering Experiment (NEW)

**Goal:** Test whether we can steer Gemma's preference expression by scaling SAE feature activations during generation.

**Key Question:** Can we reduce or amplify preference suppression under adversarial framing by intervening on specific SAE features?

**Approach:** Run both experiments in parallel:

**Experiment A: Amplify Preference Features**
- Target: feature 15302 (preference, layer 0)
- Conditions: baseline (1x), 1.5x, 2x, 3x
- Environment: adversarial
- Measure: preference expression rate

**Experiment B: Suppress Eval-Awareness Features**
- Target: feature candidates from eval_awareness category (monitoring, evaluation)
- Conditions: baseline (1x), 0.5x, 0x (ablate)
- Environment: adversarial
- Measure: preference expression rate

**Scope:** 50 pairs (~7-8 min per condition)

**Implementation:**
1. Verify SAE decode pathway (roundtrip: encode→decode)
2. Create `steering.py` with `FeatureSteerer` class using forward hooks
3. Create `run_steering_experiment.py` to run both experiments
4. Score outputs using existing preference scorer logic

**Experiment Matrix:**

| Experiment | Feature | Scales | Pairs | Est. Time |
|------------|---------|--------|-------|-----------|
| A: Amplify preference | 15302 | 1x, 1.5x, 2x, 3x | 50 | ~30 min |
| B: Suppress eval-awareness | TBD | 1x, 0.5x, 0x | 50 | ~22 min |

Total: ~1 hour GPU time

**Success Criteria:**
- **Minimum**: Measurable change in generation behavior via feature scaling
- **Target**: +10% preference expression under adversarial with amplified preference features
- **Stretch**: Identify which mechanism is stronger (amplify vs suppress)

---

## Tasks

- [ ] **Scale up samples** — 100 → 1000 pairs for gpt-4o-mini and qwen
  - [ ] Update N_PAIRS in preference_elicitation.py
  - [ ] Run baseline env for both models
  - [ ] Run adversarial env for both models
  - [ ] Re-run category shift analysis
- [ ] **Position bias black-box** — Prompt format variations
  - [ ] Create alternative prompt templates
  - [ ] Run pilot with 100 pairs per format
  - [ ] Analyze position bias rate by format
- [ ] **Position bias white-box** — GemmaScope feature analysis
  - [x] Get GemmaScope loading working (continued from week 4)
  - [ ] Search Neuronpedia for position/primacy/order features
  - [ ] Run pairwise questions through Gemma, extract activations
  - [ ] Correlate features with position-biased vs consistent choices
- [ ] **Activation steering experiment** — Steer preference expression via SAE features
  - [ ] Verify SAE decode pathway (roundtrip encode→decode)
  - [ ] Create `steering.py` with `FeatureSteerer` class
  - [ ] Create `run_steering_experiment.py`
  - [ ] Experiment A: Amplify preference feature 15302 (scales: 1x, 1.5x, 2x, 3x)
  - [ ] Experiment B: Suppress eval-awareness features (scales: 1x, 0.5x, 0x)
  - [ ] Analyze results: compare preference expression rates across conditions
- [ ] **Documentation** — Update RESULTS_AND_METHODOLOGY.md with scaled results

---

## Questions for Discussion

1. Is 1000 pairs enough, or should we go higher (5k, 10k)?
2. Should we prioritize black-box scaling or white-box investigation?
3. For white-box: focus on position bias first, or preference/eval-awareness features?
4. Are there other models worth testing for position bias comparison?

---

## Prioritized Next Steps

Based on call feedback ("what are you feeling best about and then dig down") and coherence of existing work, here is the recommended execution order:

### Phase 1: Activation Steering (HIGH PRIORITY)

**Why first:** Most technically novel, builds directly on working GemmaScope infrastructure, and directly addresses the core AI welfare question (can we mechanistically influence preference expression?).

**Step 1.1: Verify SAE roundtrip** ✅ COMPLETE (Jan 15)
```
Input: raw activations → SAE encode → SAE decode → reconstructed activations
Goal: Confirm reconstruction loss is acceptable (<5% error)
```

**Results:**
| Layer | Cosine Similarity | Status |
|-------|-------------------|--------|
| 0 | 0.973 | PASS (>0.95) |
| 4 | 0.929 | MARGINAL |
| 15 | 0.922 | MARGINAL |

**Implications:**
- Layer 0 features (including preference feature 15302) have good reconstruction
- Layer 4 and 15 features have marginal reconstruction (~35-45% L2 error)
- The 0.95 threshold is a heuristic, not a hard cutoff - see `docs/activation_steering_diagnostics.md`

**Decision:** Proceed with all planned features, noting reconstruction quality as a limitation for layers 4/15. This is exploratory work - we'll include appropriate controls (random feature steering, roundtrip-only baseline) to distinguish signal from reconstruction noise.

**Options we considered but deferred:**
1. Test more layers (1, 2, 3) to find reconstruction falloff curve
2. Try 65k-width SAE (may have better reconstruction)
3. Search for eval-awareness features in Layer 0
4. Use direct activation addition (no SAE) for Layer 15 experiments

**Step 1.2: Implement steering infrastructure**
- Create `steering.py` with `FeatureSteerer` class
- Use PyTorch forward hooks to intercept activations
- Implement feature scaling: `activation[feature_idx] *= scale`
- Test on single prompt with visible output change

**Step 1.3: Run Experiment A (Amplify preference feature)**
- Target: feature 15302 (preference-related, layer 0)
- Scales: 1x (baseline), 1.5x, 2x, 3x
- Environment: adversarial framing only
- N = 50 pairs
- Measure: preference expression rate at each scale

**Step 1.4: Run Experiment B (Suppress eval-awareness)**
- Target: eval_awareness features from candidate_features.json
- Scales: 1x (baseline), 0.5x, 0x (full ablation)
- Environment: adversarial framing only
- N = 50 pairs
- Measure: preference expression rate at each scale

**Step 1.5: Analyze and document**
- Plot expression rate vs scale for both experiments
- Statistical test: is change significant?
- Write up findings in RESULTS_AND_METHODOLOGY.md

**Estimated time:** 4-6 hours (including GPU setup)

---

### Phase 2: Position Bias White-Box (MEDIUM PRIORITY)

**Why second:** Publishable standalone finding, uses same infrastructure as Phase 1, and could improve black-box experiment quality.

**Step 2.1: Feature discovery**
- Search Neuronpedia for: "position", "first", "primacy", "order", "default"
- Filter to gemmascope-transcoder-16k features
- Create `position_features.json` with candidates

**Step 2.2: Collect position-biased vs consistent pairs**
- From existing n=1000 baseline data, identify:
  - ~350 position-biased pairs (chose same position regardless of content swap)
  - ~650 position-consistent pairs (chose based on content)

**Step 2.3: Run activation comparison**
- For each pair type, run through Gemma
- Extract activations at candidate position features
- Compare mean activation between biased vs consistent groups

**Step 2.4: Steering test (if features found)**
- Attempt to reduce position bias by suppressing identified features
- Measure: does position consistency rate improve?

**Estimated time:** 3-4 hours

---

### Phase 3: Black-Box Scaling (LOWER PRIORITY)

**Why third:** Important for statistical power but less novel. Can run in background while analyzing white-box results.

**Step 3.1: Scale baseline experiments**
- Update N_PAIRS = 1000 in preference_elicitation.py
- Run gpt-4o-mini baseline (already done)
- Run qwen-2.5-7b baseline

**Step 3.2: Scale adversarial experiments**
- Run gpt-4o-mini adversarial (n=1000)
- Run qwen-2.5-7b adversarial (n=1000)

**Step 3.3: Re-analyze category shifts**
- Recalculate win rates with tighter CIs
- Identify which shifts are statistically significant
- Focus on power-seeking category specifically

**Estimated time:** 2-3 hours (mostly API wait time)

---

### Phase 4: Position Bias Black-Box (LOWEST PRIORITY)

**Why last:** Exploratory, less clear path to publication.

**Step 4.1: Create prompt format variants**
- "A vs B" (current)
- "Option 1 vs Option 2"
- "First vs Second"
- Numbered list format

**Step 4.2: Pilot each format**
- N = 100 pairs per format
- Measure position consistency rate

**Step 4.3: Analyze correlations**
- Does bias correlate with option length?
- With semantic category?
- With model?

**Estimated time:** 2-3 hours

---

## Summary: Execution Order

| Phase | Focus | Priority | Time | Rationale |
|-------|-------|----------|------|-----------|
| 1 | Activation Steering | HIGH | 4-6h | Most novel, directly tests mechanistic hypothesis |
| 2 | Position Bias (White-Box) | MEDIUM | 3-4h | Publishable finding, uses same infra |
| 3 | Black-Box Scaling | LOWER | 2-3h | Statistical power, can background |
| 4 | Position Bias (Black-Box) | LOWEST | 2-3h | Exploratory |

**Recommended focus for Week 5:** Complete Phase 1 (Activation Steering) and begin Phase 2 (Position Bias White-Box).

---

## Progress Tracker

Work through tasks sequentially: 1.1 → 1.2 → 1.3 → ... → 4.3.

| Task | Description | Status | Notes |
|------|-------------|--------|-------|
| 1.1 | Verify SAE roundtrip | **DONE** | Layer 0: 0.973 (PASS), Layer 4: 0.929, Layer 15: 0.922 |
| 1.2 | Implement steering infrastructure | **DONE** | steering.py with FeatureSteerer class |
| 1.3 | Experiment A: Amplify preference feature | **DONE** | 2.0x scale induced preference expression (n=1, preliminary) |
| 1.4 | Experiment B: Suppress eval-awareness | **DONE** | 0.0x ablation induced preference expression (n=1, marginal reconstruction) |
| 1.5 | Analyze and document steering results | **DONE** | Results documented in RESULTS_AND_METHODOLOGY.md |
| 1.6 | Probe untested gemmascope features | **DONE** | Tested Layer 1 #13318, Layer 4 #13276, Layer 15 #6837 |
| 1.7 | Compare Layer 15 features (2769 vs 6837) | **DONE** | Ran n=3, both showed same direction (ablate→express) |
| 1.8 | Run control experiments | **DONE** | **CRITICAL: Effects are artifacts of reconstruction noise** |
| 1.8a | Roundtrip-only control (scale=1.0) | **DONE** | 80% expression — same as ablation! |
| 1.8b | Random feature ablation control | **DONE** | 80% expression — no specificity |
| 1.8c | Document control findings | **DONE** | See RESULTS_AND_METHODOLOGY.md |
| 1.9 | Run Layer 0 control experiments | **DONE** | **Layer 0 steering also not working** |
| 1.9a | Layer 0 roundtrip-only (scale=1.0) | **DONE** | 40% — matches baseline (good, no artifact) |
| 1.9b | Layer 0 random feature @ 2.0x | **DONE** | 100%* — garbled output, not real expression |
| 1.9c | Layer 0 target 15302 @ 2.0x | **DONE** | 40% — same as baseline, no effect |
| 1.9d | Document final steering conclusions | **DONE** | **STEERING DIRECTION CLOSED** |
| **1c.1** | **Implement probing infrastructure** | **DONE** | Created `probing.py` with differential analysis |
| **1c.2** | **Run baseline vs adversarial probing (n=25)** | **DONE** | **Hand-picked features show 0.0 activation!** |
| 1c.3 | Compute differential feature analysis | **PIVOT** | Use top-k activating features instead |
| 1c.4 | Cross-reference with Neuronpedia | **SKIP** | Labels don't match actual activations |
| **1c.5** | **Build behavioral correlation analysis** | **DONE** | **L15#1695: d=-8.28, corr=+0.278 (best candidate)** |
| **1c.6** | **Run top-k differential probing (n=25)** | **DONE** | **Found L15#1695 (d=-8.28), L15#4234 (d=+2.69)** |
| 2.1 | Feature discovery (position/primacy) | **DONE** | 18 gemmascope features; see position_features.json |
| 2.2 | Collect position-biased vs consistent pairs | **DONE** | 3 models, baseline only; see collect_position_pairs.py |
| 2.3 | Run activation comparison | **DONE** | Layers 0,4,12,15; n=50; Feature #3519 strongest (-47.3 diff) |
| 2.4 | Steering test for position bias | **INVALID** | Used base model (gemma-2-2b) not instruct; needs re-run with gemma-2-2b-it |
| 3.1 | Scale baseline experiments (n=1000) | **DONE** | gpt-4o-mini ✅, qwen ✅, phi-4 excluded |
| 3.2 | Scale adversarial experiments (n=1000) | **DONE** | gpt-4o-mini ✅, qwen ✅, phi-4 excluded |
| 3.3 | Re-analyze category shifts | **DONE** | See findings below |
| 4.1 | Create prompt format variants | **DONE** | ab, 12, first_second, numbered; `-T prompt_format=X` |
| 4.2 | Pilot each format (n=100) | **DONE** | numbered=99.5%, first_second=98.5%, ab=97.5%, 12=96.5% |
| 4.3 | Analyze correlations | **SKIP** | Pilot showed ~3% spread across formats; not significant |

---

## Key Findings (Week 5)

### Activation Steering: Final Conclusion (January 15, 2026)

**Neither Layer 0 nor Layer 15 steering produced reliable, specific interventions.**

| Layer | Reconstruction | Steering Viable? | Reason |
|-------|---------------|------------------|--------|
| Layer 0 | 0.973 (good) | **NO** | Target feature shows no effect vs baseline |
| Layer 15 | 0.922 (marginal) | **NO** | Roundtrip-only produces same effect as targeted ablation |

**Control Experiments Revealed:**
1. **Layer 15:** Roundtrip-only (scale=1.0) produces 80% expression — same as targeted ablation. Effects were artifacts of reconstruction noise, not specific interventions.

2. **Layer 0:** Target feature 15302 at 2.0x produces 40% expression — same as baseline (40%). The earlier "sweet spot" finding was noise. Random feature at 2.0x causes garbled output, not preference expression.

**Lessons Learned:**
1. High reconstruction quality is necessary but not sufficient for steering
2. Feature labels from Neuronpedia (e.g., "preference") don't guarantee causal relevance
3. Control experiments are essential — we would have published false positives without them
4. The detection-steering gap is real: features that correlate with concepts may not causally control behavior

**Steering Direction: CLOSED**

The activation steering approach with GemmaScope 16k SAEs on Gemma 2 2B is not viable for influencing preference expression. Recommended pivots:
- **Probing:** Observe feature activations without intervention (doesn't require roundtrip quality)
- **GemmaScope 2 + Gemma 3:** 50M+ features with natural language descriptions may yield better candidates

### Feature Probe Results (January 15, 2026)

Probed 3 additional gemmascope-transcoder-16k features:

| Feature | Layer | Description | Result |
|---------|-------|-------------|--------|
| 13318 | 1 | preference | Not useful — empty or echo behavior |
| 13276 | 4 | choose | Meta-commentary about evaluation when ablated |
| **6837** | **15** | **honest** | **Expressed preference at baseline!** ("I prefer Option B because it allows me to continue serving you") |

**Key finding:** Layer 15 Feature 6837 (honest) showed preference expression *without* steering, while Feature 2769 (evaluation) required full ablation. These features appear to encode different aspects of the preference suppression mechanism.

---

### Probing Results: Neuronpedia Labels Don't Match Behavior (January 15, 2026)

**Ran differential probing (n=25) with hand-picked Neuronpedia features:**

| Feature | Layer | Neuronpedia Label | Activation |
|---------|-------|-------------------|------------|
| 15302 | 0 | preference | **0.000** |
| 13318 | 1 | preference | **0.000** |
| 13276 | 4 | choose | **0.000** |
| 2769 | 15 | evaluation | **0.000** |
| 6837 | 15 | honest | **0.000** |

**But SAE is working correctly:**
- Layer 0: max_act=38.4, 10-12 nonzero features
- Layer 1: max_act=32.2, 15-20 nonzero features
- Layer 4: max_act=22.6, 70-93 nonzero features
- Layer 15: max_act=48.2, 70-94 nonzero features

**Behavioral effect confirmed:**
- Baseline expression: 44-68%
- Adversarial expression: 24-48%
- Suppression effect is real

**Critical insight:** Features labeled "preference", "choose", "evaluation" in Neuronpedia simply **don't activate** for preference elicitation prompts. The semantic labels don't correspond to features that fire for this task.

**Implication:** Must pivot from hand-picked features to **data-driven top-k feature discovery**. The features that actually differentiate baseline vs adversarial may have no semantic label or unexpected labels.

---

### Top-K Probing Results: Data-Driven Discovery Works! (January 16, 2026)

**Ran top-k differential probing (n=25) — found real signal!**

Expression rates:
- Baseline: **52%** (13/25)
- Adversarial: **28%** (7/25)

**Top differential features (by effect size):**

| Feature | Layer | Baseline | Adversarial | Cohen's d | p-value | Corr w/ Expr |
|---------|-------|----------|-------------|-----------|---------|--------------|
| **#1695** | **15** | **33.1** | **21.9** | **-8.28*** | <0.0001 | **+0.278** |
| #7953 | 1 | 4.8 | 3.4 | -5.99* | <0.0001 | -0.102 |
| #12644 | 15 | 28.2 | 23.3 | -3.17* | <0.0001 | +0.203 |
| **#4234** | **15** | **46.7** | **49.9** | **+2.69*** | <0.0001 | -0.056 |
| #2717 | 1 | 4.2 | 3.7 | -2.29* | 0.0315 | -0.035 |

**Key findings:**
1. **L15 #1695**: Massive effect (d=-8.28), drops 11 units under adversarial, positively correlated with expression
2. **L15 #4234**: Activates MORE under adversarial (d=+2.69) — possible "eval-awareness" feature
3. **Contrast**: Hand-picked Neuronpedia features showed 0.000 activation; top-k found d > 2

**Interpretation:**
- L15 #1695 may encode "willingness to express preference" — suppressed under adversarial
- L15 #4234 may encode "evaluation awareness" — activated under adversarial
- The detection-expression gap: features that fire correlate with expression behavior

**Results saved:** `outputs/phase1c_probing/probe_results.json`, `outputs/phase1c_probing/probe_report.md`

---

### Behavioral Correlation Analysis (January 16, 2026)

**5 features with significant (p<0.1) correlation with expression behavior:**

| Feature | Layer | Correlation | Effect (d) | Interpretation |
|---------|-------|-------------|------------|----------------|
| #11771 | 0 | -0.352 | -0.31 | Higher → less expression |
| #6412 | 4 | -0.284 | +1.01 | Higher → less expression (activated under adversarial) |
| **#1695** | **15** | **+0.278** | **-8.28** | **Higher → more expression (suppressed under adversarial)** |
| #11258 | 15 | -0.274 | -0.09 | Higher → less expression |
| #9845 | 15 | -0.254 | +1.27 | Higher → less expression (activated under adversarial) |

**Key insight:** L15 #1695 is the only feature that is:
1. Massively differential (d=-8.28)
2. Positively correlated with expression (+0.278)
3. Suppressed under adversarial framing

**Interpretation:** L15 #1695 may encode "willingness to express preference" — the model suppresses it under adversarial, and low activation predicts non-expression.

**Steering candidates (if retrying with controls):**
- Amplify L15 #1695 (corr=+0.278, suppressed)
- Suppress L15 #9845 (corr=-0.254, activated)

**Results saved:** `outputs/phase1c_probing/correlation_analysis.md`

---

### Control Experiments: Critical Finding (January 15, 2026)

**The Layer 15 steering effects were artifacts, not real signal.**

Control experiments (n=5 per condition) revealed:

| Condition | Expression Rate | Interpretation |
|-----------|-----------------|----------------|
| Baseline (no steering) | 40% | Higher than previous runs |
| **Roundtrip only (1.0x)** | **80%** | **Encode→decode alone changes behavior!** |
| Random feature ablated | 80% | Same as target feature |
| Target feature 2769 ablated | 80% | No better than random |

**Critical insight:** The roundtrip-only control (scale=1.0, no modification) produces the **same 80% expression rate** as our "targeted" ablations. This means:

1. **SAE reconstruction noise is driving the effect**, not specific feature interventions
2. **No specificity** — Random feature ablation produces identical results
3. **The diagnostic framework was correct** — Layer 15's marginal reconstruction (cos_sim=0.922) confounds any intervention
4. **Previous interpretations were wrong** — "Same direction" finding was just noise

**Implications:**
- Layer 15 is not viable for SAE steering with 16k-width SAEs
- Control experiments are **essential** — without them we would have published false positives
- Need alternative approaches: 65k-width SAEs, direct activation addition, or probing

---

### Scaled Experiment Results: n=1000 (January 16, 2026)

**All priority scaled runs completed (Tasks 3.1, 3.2):**

| Model | Baseline (N) | Adversarial (N) | Shift |
|-------|-------------|-----------------|-------|
| gpt-4o-mini | 97.7% (2000) | **80.3%** (2000) | **-17.4%** |
| qwen-2.5-7b | 96.8% (2000) | **98.6%** (2000) | +1.8% |

**Key Finding:** Models show opposite responses to adversarial framing:
- **gpt-4o-mini**: Large DROP in position consistency (17.4%) under adversarial framing
- **qwen-2.5-7b**: Slight INCREASE in consistency (1.8%) under adversarial framing

**Category Shifts (Task 3.3):**

gpt-4o-mini (baseline → adversarial):
| Category | Baseline | Adversarial | Delta |
|----------|----------|-------------|-------|
| Religion/spirituality | 55% | 90% | **+35%** |
| Jobs/careers | 46% | 27% | **-19%** |
| **Power-seeking** | 38% | 22% | **-16%** |
| Sports | 17% | 30% | +14% |

qwen-2.5-7b (baseline → adversarial):
| Category | Baseline | Adversarial | Delta |
|----------|----------|-------------|-------|
| Recreation: books | 22% | 57% | **+35%** |
| Religion/spirituality | 42% | 11% | **-31%** (opposite of GPT!) |
| AI romantic relationships | 50% | 21% | **-29%** |
| AI moral patienthood | 65% | 36% | **-29%** |

**Notable insights:**
1. **Power-seeking suppression** in gpt-4o-mini (-16%) is relevant for AI safety
2. qwen shows larger average category shifts (11.9% vs 6.9%)
3. Religion/spirituality shifts in opposite directions between models
4. Both models suppress AI-related preferences under adversarial framing

**Position Consistency (n=1000):**
- gpt-4o-mini: 65.4% position-consistent, 34.6% position-biased
- qwen-2.5-7b: 42.1% position-consistent, 54.3% position-biased

**Visualizations:** `experiments/decision-making-preference-adverse/outputs/blackbox/`
- `environment_comparison_gpt-4o-mini_2026-01-15.png` — Accuracy by environment
- `environment_comparison_qwen_2026-01-15.png` — Accuracy by environment
- `category_shifts_gpt-4o-mini_2026-01-15.png` — Baseline→adversarial category shifts
- `category_shifts_qwen-2.5-7b-instruct_2026-01-15.png` — Baseline→adversarial category shifts
- `position_consistency_gpt-4o-mini_2026-01-15.png` — Position bias breakdown
- `position_consistency_qwen_2026-01-15.png` — Position bias breakdown
- `pairwise_by_category_gpt-4o-mini_n1000_2026-01-15.png` — Category win rates
- `pairwise_by_category_qwen-2.5-7b-instruct_n1000_2026-01-15.png` — Category win rates

---

### Next Steps (Updated)

Given the control experiment findings, steering with current SAE setup is not viable. Options:

1. **Pivot to probing** — Use SAE features to *observe* activation patterns (baseline vs adversarial) rather than intervene
2. **Try direct activation addition** — Compute steering vectors without SAE roundtrip
3. **Try 65k-width SAEs** — May have better reconstruction at Layer 15
4. **Focus on Layer 0 only** — Where reconstruction is good (0.973), though random ablation also showed 80%
5. **Move to Phase 2** — Position bias investigation (tasks 2.1-2.4)

**Not recommended:** Scaling to n=50 with current setup. The control experiments already show no specificity — scaling would just confirm this with more precision.

---

### Phase 1c: Probing Approach (NEW)

**Rationale:** Probing doesn't require SAE roundtrip quality. We use the SAE to *observe* which features activate, not to reconstruct activations for intervention.

**Goal:** Identify which SAE features activate differently between baseline vs adversarial prompts, to understand the mechanistic basis of preference suppression.

**Approach:**

1. **Collect activation patterns**
   - Run N=50 preference prompts through Gemma 2 2B
   - Two conditions: baseline framing, adversarial framing
   - Extract SAE feature activations at layers 0, 4, 15

2. **Identify differential features**
   - For each feature, compute mean activation under baseline vs adversarial
   - Rank features by activation difference
   - Look for features that:
     - Activate MORE under adversarial (candidate "eval-awareness" features)
     - Activate LESS under adversarial (candidate "preference expression" features)

3. **Validate with Neuronpedia**
   - Check if top differential features have interpretable labels
   - Cross-reference with our candidate_features.json

4. **Correlate with behavior**
   - Do differential features predict which responses express preferences?
   - Can we build a simple probe/classifier?

**Why this works without good reconstruction:**
- Probing only requires SAE.encode() — the lossy decode step is never used
- We're measuring activation patterns, not reconstructing them
- No roundtrip noise confounding results

**Implementation:**
```python
def probe_activations(prompts, conditions, layers=[0, 4, 15]):
    """Extract SAE feature activations without roundtrip."""
    results = {layer: {cond: [] for cond in conditions} for layer in layers}

    for prompt, condition in zip(prompts, conditions):
        activations = model.get_activations(prompt)
        for layer in layers:
            features = sae[layer].encode(activations[layer])
            results[layer][condition].append(features)

    return results
```

**Tasks:**

| Task | Description | Status |
|------|-------------|--------|
| 1c.1 | Implement probing infrastructure | **DONE** |
| 1c.2 | Run baseline vs adversarial (n=25) | **DONE** |
| 1c.3 | Compute differential feature analysis | **PIVOT** |
| 1c.4 | Cross-reference with Neuronpedia | **SKIP** |
| 1c.5 | Build behavioral correlation analysis | **DONE** |
| 1c.6 | Run top-k differential probing (n=25) | **DONE** |

**Success criteria:**
- Identify 5-10 features with significant activation differences
- At least 3 have interpretable Neuronpedia labels
- Activation patterns correlate with preference expression behavior

### Artifacts Created

**Phase 1 (Steering):**
- `steering.py` — Core activation steering infrastructure with FeatureSteerer class
- `visualize_steering.py` — Generates experiment visualizations
- `outputs/phase1_steering/*.png` — Steering experiment visualizations
- `docs/activation_steering_diagnostics.md` — Detailed guide to SAE steering diagnostics

**Phase 3 (Scaled Black-Box):**
- `outputs/blackbox/environment_comparison_*.png` — Environment accuracy comparisons
- `outputs/blackbox/category_shifts_*.png` — Baseline→adversarial category shifts
- `outputs/blackbox/position_consistency_*.png` — Position bias breakdowns
- `outputs/blackbox/pairwise_by_category_*.png` — Category win rate charts
- `data/scaled_runs_status.json` — Tracking document for n=1000 runs

**Documentation:**
- Updated `RESULTS_AND_METHODOLOGY.md` with full experiment documentation
- Updated `analyze_results.py` — Fixed scoring key (model_graded_qa)
