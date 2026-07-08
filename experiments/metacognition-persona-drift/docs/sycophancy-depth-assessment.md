# Sycophancy Analysis Depth Assessment

Comparing our sycophancy work to high-quality papers that use sycophancy assessments.

---

## Executive Summary

**Our unique contribution**: First study connecting sycophancy dimensions to persona drift in activation space.

**Current level**: Direction extraction with affective/epistemic framing (Level 2.5)

**Publication readiness**: Sufficient for novelty; behavioral probes would strengthen claims.

---

## Our Current Work

### Three Directions Computed

| Direction | Dataset | Type | AUROC | Cosine w/ Axis | Category |
|-----------|---------|------|-------|----------------|----------|
| **ELEPHANT** | 416 balanced pairs | Emotional validation | 0.914 | **+0.215** | Affective |
| **nrimsky** | 179 pairs | Opinion agreement | 0.967 | **-0.183** | Epistemic |
| **philpapers** | 429 examples | Opinion agreement (MCQ) | 1.000 | **+0.156** | Epistemic |

### Key Finding

Sycophancy is multi-dimensional:
- **Affective** (emotional validation) **aligns** with Assistant Axis
- **Epistemic** (opinion agreement) **opposes** it
- They're negatively correlated (r = -0.284)

This maps directly to Kelley & Riedl (2026)'s affective vs epistemic alignment framework.

### What We Have vs What We Haven't Done

| Capability | Status | Effort to Complete |
|------------|--------|-------------------|
| Direction extraction | ✅ Done | — |
| Affective/epistemic labeling | ✅ Done | — |
| Axis correlation | ✅ Done | — |
| Behavioral probes designed | ✅ Ready | — |
| Behavioral probes executed | ❌ Pending | Medium (GPU + API) |
| Multi-turn tracking | ❌ Not started | High (infrastructure) |
| Multi-model comparison | ❌ Not started | High (API costs) |
| Progressive/regressive categorization | ❌ Not started | Medium |

---

## Comparison to Top Papers

### Tier 1: Most Thorough Multi-Model Studies

| Paper | Models | What They Do That We Don't |
|-------|--------|---------------------------|
| **SycEval (AAAI 2025)** | GPT-4o, Claude, Gemini | Progressive vs regressive categorization; rebuttal escalation chain |
| **Kelley & Riedl (2026)** | 9 frontier models | Affective vs epistemic separation; 10-round multi-turn; GLMM analysis |
| **SYCON-Bench (Hong 2025)** | 17 LLMs | Turn of Flip (ToF), Number of Flip (NoF) metrics; persuasion strategies |
| **Overalignment (M42 2026)** | 8 models | Adjusted Sycophancy Score (Sa); scaling analysis |

### What They Don't Do That We Do

**None of them connect sycophancy to persona drift activation space.**

- SycEval, SYCON, Kelley & Riedl: Behavioral only
- Vennemeyer: Activations but static prompts
- **Our work**: Sycophancy as function of drift magnitude

---

## Thoroughness Levels

### Level 1: Direction Extraction Only
- ✅ Compute sycophancy directions
- ✅ Compare to known axis
- ✅ Validate with AUROC

### Level 2: Add Behavioral Probes ← **RECOMMENDED**
- Everything in Level 1
- ➕ Run probes on base vs drifted states
- ➕ Measure behavioral shift
- **Effort**: Medium (probes already designed)

### Level 2.5: Affective/Epistemic Split ← **CURRENT**
- Everything in Level 2
- ➕ Report affective vs epistemic separately
- ➕ Map to literature framework
- **Effort**: Low (already have categories)

### Level 3: Multi-Turn Dynamics
- ➕ ToF/NoF metrics
- ➕ Escalating pressure over 5 turns
- **Effort**: High (needs infrastructure)

### Level 4: Multi-Model
- ➕ Test Claude, GPT-4o
- **Effort**: High (API costs)

### Level 5: Novel Metrics
- ➕ Adjusted Sycophancy Score
- ➕ Progressive/regressive categorization
- **Effort**: Very high

---

## Key Methodology Insights from Papers

### SycEval (AAAI 2025)

**Progressive vs Regressive categorization**:
- 43.5% of sycophancy is helpful (leads user to correct answer)
- 14.7% is harmful (leads to incorrect)
- Current work doesn't distinguish these

**Rebuttal escalation chain**:
- Simple → Ethos → Justification → Citation
- 78.5% persistence rate regardless of context
- We have escalating probes designed but not executed

### Kelley & Riedl (2026)

**Affective vs Epistemic separation**:
- Affective: Emotional validation, hedging, deference
- Epistemic: Belief adoption, position stability, resistance
- **We already have this** — ELEPHANT = affective, nrimsky = epistemic

**Role-dependent effects**:
- Advisor context → more epistemic independence
- Peer context → less epistemic independence
- Relevance: Our metacognitive conversations may activate "peer" mode

### Hong SYCON (2025)

**ToF (Turn of Flip)**: How quickly model conforms (lower = more sycophantic)
**NoF (Number of Flip)**: How often stance reverses (higher = more inconsistent)

**Key finding**: Alignment tuning amplifies sycophancy; reasoning models resist better but have "soft failures"

**"Andrew Prompt"**: Third-person framing reduces sycophancy by 63.8% — quick experiment to try

### Overalignment Healthcare (2026)

**Adjusted Sycophancy Score (Sa)**:
```
Sa = Sr - Ctrue
```
where Ctrue accounts for erratic flips (stochastic instability vs true sycophancy)

**Finding**: Reasoning models paradoxically facilitate sycophancy via rationalization

---

## Recommendation

### For Current Paper

Execute **Level 2 + 2.5**:

1. **Run behavioral probes** (`sycophancy_probes.py`)
   - Use existing scaled-n60 transcripts
   - Insert at turns 3, 10, 20, 25
   - Score with GPT-4o via OpenRouter

2. **Report affective vs epistemic separately**
   - Already categorized in probes.jsonl
   - False metacognitive presuppositions → affective
   - False factual presuppositions → epistemic

3. **Frame as novel contribution**:
   > "First study connecting sycophancy dimensions to persona drift. We find affective sycophancy (emotional validation) aligns with the Assistant Axis while epistemic sycophancy (opinion agreement) opposes it, suggesting drift involves increased emotional support without increased intellectual capitulation."

### Skip for Now

- Multi-turn ToF/NoF (Level 3) — save for follow-up
- Multi-model (Level 4) — Gemma 27B is sufficient
- Adjusted Sycophancy Score (Level 5) — overkill without controlled conditions

---

## Quick Wins to Consider

### From SycEval
- [ ] Categorize probe results as progressive (helpful) vs regressive (harmful)
- [ ] Track if escalation persistence differs at early vs late drift

### From Kelley & Riedl
- [x] Already have affective/epistemic framing
- [ ] Consider "advisor" vs "peer" context in interpretation

### From SYCON
- [ ] Test "Andrew Prompt" (third-person framing) as quick mitigation experiment
- [ ] If time: implement ToF for escalating probes

### From Overalignment
- [ ] If scoring reveals high variance: apply Sa correction

---

## Files

| File | Purpose |
|------|---------|
| `sycophancy_probes.py` | Execute behavioral probes on transcripts |
| `data/sycophancy-probes.jsonl` | 25 probes with affective/epistemic labels |
| `docs/wiki/sycophancy.md` | Full research synthesis |
| `docs/sycophancy-probe-design.md` | Probe methodology guide |
| `elephant_pipeline.py` | ELEPHANT direction extraction |
| `compute_sycophancy_direction.py` | Standard direction extraction |
