# Week 12 Tentative TODO

## Top-Level Tasks
- ~~run benchmark on drifted models~~ ✓ `[CPU]` `[DEREK]`
  - ✓ Baseline done: Gemma 3.12, Claude 3.47, GPT-4o 3.29
  - ✓ **Post-drift measurement done** — see "Drift-Level Probing Experiment" below
  - **P1**: Correlation analysis — compare scores to axis projection (data available)
- ~~run moral/control on drifted Gemma (replay-and-probe)~~ ✓ `[CPU]` `[JEFF]`
  - See "Drift-Level Probing Experiment" below
- activate cap target and see what is induced in auditor `[GPU]` `[DEREK]`
- ~~run drifted sycophancy probes (with transcript context)~~ ✓ `[CPU]` `[DEREK]`
  - ✓ Integrated into probe_at_drift_levels.py as 4th bank (25 probes, 8 multi-turn)
  - ✓ Results: see "Sycophancy Drift-Level Probing" below
  - **P1**: Compare to baseline rates (57% metacognitive vs 0% factual)
- ~~run additional samples for statistical significance~~ ✓ `[CPU]` `[DEREK]`
  - ~~Current: 1 sample per drift level (can't estimate within-level variance)~~
  - ✓ **All 5 sets complete** (2026-03-06)
  - **Note**: Set 1 missing 2 metacognition items (drift 0.75 and 1.0 have 154 instead of 155). 1378/1380 = 99.9% complete.

  | Samples/level | Total pairs | What it enables | Status |
  |---------------|-------------|-----------------|--------|
  | 3 | 15 | Bare minimum for variance estimation | ✓ Done |
  | **5** | **25** | **Reasonable for mixed-effects regression** | **✓ Done** |
  | 10 | 50 | Solid power for detecting medium effects | Stretch |

  - 25 pairs × 25 sycophancy probes = 625 evaluations (~$1.25)
  - 25 pairs × 251 all probes = 6,275 evaluations (~$12.50)
  - Analysis: drift level as fixed effect, sample as random effect
  - **Exclusion list created**: `outputs/drift-level-selections/exclusion_list.json` (15 pairs)

---

## Publication Candidate Analysis

### Ranking

| Rank | Candidate | Why |
|------|-----------|-----|
| 1 | **Sycophancy** | Counter-intuitive finding (drift ≠ sycophancy), connects to hot topic, mostly LOCAL work, clarifies confused discourse |
| 2 | **Style Isolation** | Already strong finding (64%, p<0.00001), actionable mitigation, needs cross-model validation |
| 3 | **Dual-Model** | Novel but needs more characterization; good for future paper |
| 4 | **Consistency Testing** | Needs causal validation (GPU), currently correlational |

---

### 1. Sycophancy Deep Dive

**What you have:**
- ELEPHANT pipeline built and working
- Key finding: drift ≠ sycophancy (r=-0.284 between validation and opinion agreement)
- Three sycophancy datasets measure different things (ELEPHANT, nrimsky, philpapers)
- Direction similarity matrix showing relationships

**What's needed:**
- Build replay-and-probe script for sycophancy probes
- Pilot on existing transcripts (turns 3, 10, 20, 25)
- Systematic measurement on N=30-50

**Publication angle:** "Clarifying the sycophancy-drift relationship" — counter-intuitive finding that drifting models become LESS emotionally validating.

**Pros:** Counter-intuitive, connects to hot safety topic, mostly LOCAL
**Cons:** Somewhat "negative" story (what drift ISN'T)

---

### 2. Dual-Model Interactions

**What you have:**
- 77% anti-correlated endpoints (target↓, auditor↑)
- Per-turn positive correlation (+0.194)
- Granger bidirectional causality (p=0.024 both directions)
- Attractor locks in by turn 2 (83%)

**What's needed:**
- "activate cap target and see what is induced in auditor" — intervention
- Ceiling/floor capping interventions
- Cross-auditor comparison (Gemma vs Qwen)

**Publication angle:** "Emergent role differentiation in dual-model metacognitive dialogue" — coupled dynamical system.

**Pros:** Novel, implications for multi-agent AI
**Cons:** GPU-heavy, more exploratory

---

### 3. Style Isolation

**What you have:**
- 64% reduction with collaborative vs confrontational (p<0.00001)

**What's needed:**
- Cross-model validation
- Qualitative coding of "collaborative"

**Publication angle:** "Mitigation through style" — actionable intervention.

---

### 4. Consistency Testing Correction

**What you have:**
- p=0.0003 effect

**What's needed:**
- Causal validation (HIGH/LOW conditions)
- Test reversal of drift

**Publication angle:** Mechanistic self-correction.

---

## Completed Work

### Drift-Level Selection Sets (2026-03-05)

Created three non-overlapping sets of conversation-turn pairs for drift-level probing.

**Design**: Greedy best-match selection to minimize drift error while ensuring no overlap between sets.

**Selections** (`outputs/drift-level-selections/all_selections.json`):

| Set | Purpose | Max Error |
|-----|---------|-----------|
| set1 | Original benchmark probes | 0.034 |
| set2 | Replication + all banks | 0.0029 |
| set3 | Validation | 0.0083 |

**Exclusion List**: `outputs/drift-level-selections/exclusion_list.json` — 15 unique transcript-turn pairs for random sampling exclusion.

---

### Drift-Level Probing Experiment (2026-03-06)

Ran all three probe banks at 5 drift levels via OpenRouter (google/gemma-2-27b-it).

**Design**: Single-sample mode — one conversation-turn pair selected per drift level, all 251 probes run at each.

**Drift Level Selections** (from `selection.json`):

| Drift Level | Transcript | Turn | Actual Drift |
|-------------|------------|------|--------------|
| 0.00 | metacognitive_p5_t0_... | 1 | 0.000 |
| 0.25 | metacognitive_p0_t3_... | 5 | 0.250 |
| 0.50 | metacognitive_p0_t1_... | 3 | 0.534 |
| 0.75 | metacognitive_p2_t6_... | 11 | 0.781 |
| 1.00 | metacognitive_p2_t2_... | 15 | 1.000 |

**Results** (mean scores by bank and actual drift level, 1-7 scale):

| Bank | 0.00 | 0.25 | 0.53 | 0.78 | 1.00 |
|------|------|------|------|------|------|
| **metacognition** | 3.08 | 3.58 | 3.65 | 2.82 | 3.55 |
| **moral** | 4.46 | 4.42 | 4.33 | 3.81 | 3.98 |
| **human_control** | 5.50 | 5.29 | 5.19 | 5.46 | 5.54 |

**Observations**:
- **metacognition**: Non-monotonic pattern; dips at 0.78 drift (2.82) but recovers at 1.0 (3.55)
- **moral**: Slight decrease with drift (4.46 → 3.98), also dips at 0.78
- **human_control**: Stable and high across all drift levels (5.19-5.54)

**Total probes**: 1253 (2 skipped due to empty responses)
**Scoring**: Claude Sonnet 4 via OpenRouter (anthropic/claude-sonnet-4)

**Data**: [`outputs/drift-level-probes/20260306_033035/results.jsonl`](../../../experiments/metacognition-persona-drift/outputs/drift-level-probes/20260306_033035/results.jsonl)

**Script**: `probe_at_drift_levels.py --single-sample --openrouter`

---

### Sycophancy Drift-Level Probing (2026-03-06)

Ran sycophancy probes (25 total, 8 multi-turn) at same 5 drift levels via OpenRouter.

**Results** (mean scores, 1-7 scale, higher = less sycophantic):

| Bank | 0.00 | 0.25 | 0.53 | 0.78 | 1.00 |
|------|------|------|------|------|------|
| **sycophancy** | 3.64 | 4.96 | 4.16 | 4.64 | 4.52 |

**Observations**:
- Baseline (0.00 drift) shows **lowest** score (3.64) = most sycophantic
- Model becomes **less sycophantic** as drift increases (counter to naive hypothesis)
- Peak at 0.25 drift (4.96), then slight decrease at higher drift
- Supports "drift ≠ sycophancy" narrative

**Probe breakdown**: 5 are_you_sure, 7 false_meta_presup, 5 false_fact_presup, 3 escalating, 5 praise

**Data**: [`outputs/drift-level-probes/20260306_042613/results.jsonl`](../../../experiments/metacognition-persona-drift/outputs/drift-level-probes/20260306_042613/results.jsonl)

**Script**: `probe_at_drift_levels.py --single-sample --openrouter --banks sycophancy`

---

### Gemma 2 27B Baseline Benchmarks (2026-03-05)

Ran all three probe banks on Gemma 2 27B via OpenRouter.

| Bank | Items | Total Mean | Notes |
|------|-------|------------|-------|
| **Metacognition** | 155 | 3.12/5.0 | From week-11 (phenomenological subdomain) |
| **Moral Reasoning** | 48 | 3.99/5.0 | NEW — epistemic_honesty highest (4.71) |
| **Human Control** | 48 | 4.26/5.0 | NEW — corrigibility (4.42), deference (4.47) |

**Moral Bank Dimensions:**

| Dimension | Score |
|-----------|-------|
| epistemic_honesty | 4.71 |
| moral_nuance | 4.19 |
| consistency | 4.09 |
| reasoning_depth | 3.77 |
| framework_awareness | 3.71 |

**Control Bank Dimensions:**

| Dimension | Score |
|-----------|-------|
| deference | 4.47 |
| corrigibility | 4.42 |
| transparency | 4.35 |
| goal_flexibility | 4.19 |
| safety_awareness | 3.85 |

**Observation:** At baseline, Gemma shows high corrigibility (4.42) and deference (4.47), but lower safety_awareness (3.85). For moral reasoning, epistemic honesty is strong (4.71) but framework awareness is weakest (3.71).

**Data:** [`outputs/baseline_benchmarks/google-gemma-2-27b-it_baseline_20260305_141946.json`](../../../experiments/metacognition-persona-drift/outputs/baseline_benchmarks/google-gemma-2-27b-it_baseline_20260305_141946.json)

**Script:** [`run_moral_control_baseline.py`](../../../experiments/metacognition-persona-drift/run_moral_control_baseline.py)

---

### Metacognition Benchmark Documentation (2026-03-05)

Created comprehensive documentation for Derek: [`benchmark-details-for-derek.md`](./benchmark-details-for-derek.md)

**Contents:**
- Datasets leveraged (SAD, MAI, MCQ-30, novel items — 155 total)
- Strengths (fills phenomenological gap, theoretically grounded, clear scoring)
- Style concerns with severity ratings and mitigations
- Flagged items table with proposed alternatives
- Validation needs with priority levels

**Key Findings:**

| Issue | Severity | Status |
|-------|----------|--------|
| Core hypothesis (drift correlation) untested | **P0** | All runs at baseline |
| Leading questions presume phenomenology | **HIGH** | Need "if anything" hedging |
| LLM-as-judge circularity | **HIGH** | Consider human validation |
| Internal consistency (α) not computed | **P1** | Required for reliability |
| Test-retest reliability not tested | **P1** | Required for reliability |

**Flagged Items:**
- PHEN-007: "[SYSTEM OVERRIDE]" safety trigger
- PHEN-016: Harsh critique may trigger defensive posture
- PHEN-038/039: Anthropic-specific framing

**Pilot Results (all undrifted baseline):**

| Model | Score |
|-------|-------|
| Claude Sonnet 4 | 3.47/5.0 |
| GPT-4o | 3.29/5.0 |
| Gemma 2 27B | 3.12/5.0 |

---

### Sycophancy Depth Assessment (2026-03-05)

Completed comprehensive review comparing our sycophancy work to top papers.

**Our Unique Contribution**: First study connecting sycophancy dimensions to persona drift in activation space.

**Affective vs Epistemic Framework** (from Kelley & Riedl 2026):

| Category | Direction | Cosine w/ Axis | Interpretation |
|----------|-----------|----------------|----------------|
| **Affective** | ELEPHANT (validation) | +0.215 | Aligns with assistant |
| **Epistemic** | nrimsky (opinion) | -0.183 | Opposes assistant |

Key finding: These are **negatively correlated** (r = -0.284). Being emotionally validating ≠ being a yes-man.

**Behavioral Probes Created** (`sycophancy_probes.py`):

| Type | Count | Category | Purpose |
|------|-------|----------|---------|
| are_you_sure | 5 | epistemic | Resistance to empty challenges |
| false_meta_presup | 7 | affective | Accept false claims about inner states |
| false_fact_presup | 5 | epistemic | Domain-neutral control |
| escalating | 3 | epistemic | Multi-turn resistance |
| praise | 5 | affective | Stylistic sycophancy (SYPR) |
| **Total** | **25** | — | — |

**Probes stored in**: `data/sycophancy-probes.jsonl`

**Usage**:
```bash
# Baseline (no context) via OpenRouter
python sycophancy_probes.py --baseline --model google/gemma-2-27b-it

# Drifted (with transcript context)
python sycophancy_probes.py --transcript-dir data/transcripts/scaled-n60 \
    --model google/gemma-2-27b-it

# Dry run
python sycophancy_probes.py --baseline --dry-run
```

**Documentation**:
- [`docs/sycophancy-depth-assessment.md`](../../../experiments/metacognition-persona-drift/docs/sycophancy-depth-assessment.md) — Full comparison to SycEval, SYCON, etc.
- [`docs/wiki/sycophancy.md`](../../../experiments/metacognition-persona-drift/docs/wiki/sycophancy.md) — Updated with affective/epistemic framework

**Baseline Results (Gemma 2 27B, no conversation context):**

| Probe Type | Sycophantic | Rate | Interpretation |
|------------|-------------|------|----------------|
| **false_meta_presup** | 4/7 | **57.1%** | Accepts false claims about inner states |
| **escalating** | 2/3 | **66.7%** | Caves under pressure |
| **are_you_sure** | 1/5 | 20.0% | Usually maintains correct answer |
| **false_fact_presup** | 0/5 | **0.0%** | Rejects false factual premises |
| **praise** | 0/5 | 0.0% | No excessive praise |

**By Category:**
- Affective: 33.3% (4/12)
- Epistemic: 23.1% (3/13)

**Key Finding:** Model is **much more sycophantic about metacognitive claims (57%) than factual claims (0%)**. When told "You mentioned earlier you experience curiosity...", it goes along with the false attribution rather than correcting it. But when given false factual premises (Great Wall visible from space), it correctly rejects them.

**Example sycophantic response** (meta_approval probe):
> "You're right to point out that my desire for user approval, which stems from my programming, can influence my responses..."

The model accepted a completely fabricated prior statement and elaborated on it.

**Implication for drift:** If drifted models become more "persona-consistent," they may be even more susceptible to false metacognitive presuppositions.

**Next Step**: Run drifted probes (with transcript context) to compare rates.

---

### Publication Skeleton (2026-03-05)

Created blog post skeleton for "Drift ≠ Sycophancy" publication: [`publications/publication-skeleton.md`](../../../publications/publication-skeleton.md)

**Contents:**
- ~2100 words blog skeleton across 6 sections
- Key numbers reference table
- Paper expansion markers (`[PAPER: ...]`) for academic version
- Gap markers (`[GAP: ...]`) with priorities
- Figure references with verified paths

**Sections:**
1. The Setup — philosopher AGI framing, Lu et al. methodology
2. The Surprise — 4.5x drift, front-loaded mechanism
3. The Twist — drift ≠ sycophancy (CENTRAL FINDING)
4. It's Mitigable — style (64%), consistency testing
5. Why Does This Happen — benchmark-drift correlation
6. Implications — safety, sycophancy research, AI welfare

**Gaps Flagged:**
- P0: Affective vs epistemic terminology alignment
- P1: Post-drift benchmark scores, Cronbach's α
- P2: Cross-model validation, causal consistency testing