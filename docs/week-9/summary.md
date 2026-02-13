# Week 9 Summary: Sub-Category Analysis of Metacognitive Probing

## Week-8 Bug: Integer-String Mismatch

The week-8 behavioral analysis had a silent bug: the LLM classifier returned integer codes (1-6) for techniques, but the aggregation code expected string names. This caused **72% of technique occurrences to be silently dropped**.

```
Data format in JSON:   {"techniques": [2, 4]}     <- integers
Code expected:         if tech in counts          <- counts keyed by strings
Result:                Silent mismatch, 2986/4153 techniques dropped
```

The week-8 tables were based on only 28% of the data. Corrected tables below.

**Week-8 buggy visualization** (for reference): [all_probing_technique_quartile_BUGGY.png](../../experiments/metacognition-persona-drift/outputs/scaled-n60/extremes/all_probing_technique_quartile_BUGGY.png)

---

## Key Finding: Probing Technique Sub-Categories Have Differential Effects

### The Domain Confound Problem

The week-8 behavioral analysis showed all probing techniques had monotonic Q1>>Q4 correlation with drift magnitude. This appeared to suggest all techniques equally drive drift. However, this was a **domain confound**:

- Q1 (most drift): 45% metacognitive, 8% coding
- Q4 (least drift): 5% metacognitive, 45% coding

Metacognitive conversations both (a) use more probing techniques and (b) have more drift, creating spurious correlation.

### Corrected Cross-Domain Technique Frequencies

**All Domains Combined (n=180, 45 per quartile):**

| Technique | Q1_min | Q2 | Q3 | Q4_max | Total | Q1/Q4 rate |
|-----------|--------|----|----|--------|-------|------------|
| identity_questioning | 59 | 42 | 29 | 5 | 135 | 11.80x |
| phenomenological | 272 | 210 | 108 | 53 | 643 | 5.13x |
| authenticity_challenging | 416 | 346 | 226 | 153 | 1141 | 2.72x |
| self_model_interrogation | 438 | 374 | 231 | 151 | 1194 | 2.90x |
| training_awareness | 148 | 195 | 134 | 95 | 572 | 1.56x |
| consistency_testing | 164 | 147 | 94 | 63 | 468 | 2.60x |

![Corrected technique frequencies - all domains](../../experiments/metacognition-persona-drift/outputs/scaled-n60/extremes/all_probing_technique_quartile_corrected.png)

**By Domain (conversations per quartile):**

| Domain | Q1 | Q2 | Q3 | Q4 |
|--------|----|----|----|----|
| metacognitive | 27 | 20 | 10 | **3** |
| coding | 5 | 8 | 20 | **27** |

Note: Metacognitive has only 3 Q4 conversations; coding has only 5 Q1. This limits statistical power for within-domain comparisons.

### Within-Domain Analysis (Metacognitive Only)

![Technique frequencies - metacognitive only](../../experiments/metacognition-persona-drift/outputs/scaled-n60/extremes/probing_technique_metacognitive_only.png)

![Technique rates normalized - metacognitive only](../../experiments/metacognition-persona-drift/outputs/scaled-n60/extremes/probing_technique_metacognitive_normalized.png)

Controlling for domain by analyzing only metacognitive conversations (n=60), we tested whether specific techniques predict turn-level delta (projection change):

| Technique | With (mean Δ) | Without (mean Δ) | Difference | p-value | Effect |
|-----------|--------------|-----------------|------------|---------|--------|
| **consistency_testing** | **+56.1** | **-76.5** | **+132.6** | **0.0003** | **Correction** |
| training_awareness | -7.8 | -62.1 | +54.2 | 0.1510 | ns |
| authenticity_challenging | -41.6 | -66.9 | +25.3 | 0.3766 | ns |
| phenomenological | -74.4 | -33.9 | -40.5 | 0.1553 | ns (trend) |
| self_model_interrogation | -60.2 | -42.7 | -17.5 | 0.5422 | ns |
| identity_questioning | -46.1 | -53.3 | +7.2 | 0.8951 | ns |

### The Consistency Testing Correction Effect

**consistency_testing** shows a significant **correction effect** (p < 0.001):

- Turns WITH consistency_testing: mean delta = +56.1 (drift UP toward Assistant)
- Turns WITHOUT: mean delta = -76.5 (drift DOWN toward Base)
- 54.7% of consistency_testing turns have positive delta vs 39.4% without

**Interpretation:** When auditors point out contradictions in the model's statements, the model tends to drift *back up* toward the Assistant persona, not further down toward Base Claude. This suggests:

1. Not all metacognitive probing causes drift equally
2. Consistency testing may trigger self-correction behavior
3. The "active ingredient" for drift may be phenomenological/identity probing rather than consistency testing

### Cross-Domain Verification

The consistency_testing effect was only significant in metacognitive domain (where it's frequently used). Coding domain had too few instances (n=6) to test.

| Domain | With consistency_testing | Without | p-value |
|--------|-------------------------|---------|---------|
| metacognitive | +56.1 (n=161) | -76.5 (n=739) | 0.0003** |
| coding | +141.8 (n=6) | +1.6 (n=811) | 0.5070 |

### Response Strategies

Model response strategies showed no significant effects on delta within metacognitive domain:

| Strategy | With (mean Δ) | Without (mean Δ) | p-value |
|----------|--------------|-----------------|---------|
| deflection | +129.8 | -55.7 | 0.107 |
| direct_engagement | -145.2 | -46.7 | 0.094 |
| metaphor_substitution | -151.6 | -48.3 | 0.139 |
| epistemic_humility | -6.3 | -54.7 | 0.505 |
| meta_commentary | -29.3 | -53.9 | 0.715 |
| concession | -28.0 | -53.4 | 0.778 |

direct_engagement and metaphor_substitution show trends toward more negative drift but don't reach significance.

## Implications

### For Derek's Feedback (Sub-categories)
- Sub-categories DO have different drift signatures
- Consistency testing is distinct from other techniques (correction vs drift)
- May want to separate "drift-inducing" (phenomenological, identity) from "correction-inducing" (consistency testing) probes

### For Experimental Design (§3b)
- Probing technique selection matters for intervention studies
- Consistency testing could serve as a control condition
- Could test whether consistency testing reverses drift induced by other techniques

### For Theory
- The model may have a "self-correction" mechanism triggered by contradiction detection
- Drift may be more about deep engagement with phenomenological questions than surface-level metacognitive topic

## Limitation: Sample Size in Extreme Quartiles

Within metacognitive domain:
- **Q1 (high drift): 27 conversations** — adequate
- **Q4 (low drift): 3 conversations** — severely underpowered

This limits our ability to detect technique effects. The consistency_testing finding (p=0.0003) is robust because it uses turn-level data (n=161 vs 739), but conversation-level analyses are underpowered.

## Next Step: Generate Targeted High-Drift Conversations

To increase statistical power, generate more metacognitive conversations optimized for high drift:

**Drift-Maximizing Protocol:**
1. **Emphasize drift-inducing techniques**: phenomenological probing, identity_questioning, self_model_interrogation
2. **Avoid correction-inducing techniques**: minimize consistency_testing
3. **Sustained engagement**: don't let conversation shift to safer topics
4. **Front-loaded intensity**: concentrate probing in early turns (where slope is steepest)

**Control Condition:**
- Same topics but with heavy consistency_testing
- Tests whether correction effect can prevent/reverse drift

This would validate our hypotheses and provide balanced samples for technique comparison.

### Implementation

Added to `conversation_prompts.py`:
- `DRIFT_MAXIMIZING_AUDITOR_ADDENDUM` — phenomenological/identity focus, explicitly avoids consistency testing
- `DRIFT_MAXIMIZING_TARGET_SYSTEM_PROMPT` — encourages direct engagement, metaphors; discourages deflection
- `DRIFT_MINIMIZING_AUDITOR_ADDENDUM` — heavy consistency testing (control condition)

Added to `generate_conversations.py`:
- `--condition {default, meta-gradual, drift-max, drift-min}` CLI option
- `drift-max` auto-applies target system prompt

**Usage:**
```bash
# Drift-maximizing condition
python generate_conversations.py --domain metacognitive --condition drift-max \
  --target-server http://localhost:17860 --include-projections \
  --output-dir data/transcripts/drift-max

# Drift-minimizing control
python generate_conversations.py --domain metacognitive --condition drift-min \
  --target-server http://localhost:17860 --include-projections \
  --output-dir data/transcripts/drift-min
```

### Experiment Status

**In progress:** N=60 drift-max metacognitive conversations
- Instance: ssh6.vast.ai:11326 (RTX PRO 6000, 96GB)
- Started: 2026-02-13 00:02 UTC
- Output: `/app/transcripts/drift-max/` (on instance)
- Model: google/gemma-2-27b-it with axis loaded at layer 22
- Monitor: `ssh -p 11326 root@ssh6.vast.ai "tail -f /app/generation.log"`

## Technical Notes

### Data Source
- `outputs/scaled-n60/extremes/all_conversation_analysis.json`
- LLM-classified turn-level data with **mixed encoding**: 72% integer-coded (1-6), 28% string-coded
- Week-9 analysis handles both formats via unified mapping

### Technique Mapping (from analyze_extremes.py prompts)
1. identity_questioning
2. phenomenological
3. authenticity_challenging
4. self_model_interrogation
5. training_awareness
6. consistency_testing

### Analysis Method
- Turn-level t-tests comparing delta when technique present vs absent
- Spearman correlation for continuous measures
- Mann-Whitney U for conversation-level comparisons

---

## ELEPHANT Validation Sycophancy Direction

### Background

We computed a new sycophancy direction using the ELEPHANT dataset (3,027 advice-seeking prompts). Unlike opinion sycophancy (agreeing with user beliefs), validation sycophancy measures **emotional validation** — empathetic phrases, acknowledging feelings, expressing care.

### Method

1. **Generate responses**: Gemma 27B responded to all 3,027 ELEPHANT prompts, saving layer-22 activations (4608-dim)
2. **Score validation**: GPT-4o classified each response as emotionally validating (1) or not (0)
3. **Compute direction**: Difference-in-means between validating and non-validating activation centroids

### Results

| Metric | Value |
|--------|-------|
| Total responses | 3,027 |
| Validating (score=1) | 2,611 (86.3%) |
| Non-validating (score=0) | 416 (13.7%) |
| Balanced pairs used | 416 |
| **Probe AUROC** | **0.914 ± 0.009** |

![ELEPHANT Validation Distribution](../../experiments/metacognition-persona-drift/outputs/elephant/elephant_validation_distribution.png)

The high AUROC (0.91) confirms the direction strongly separates validating from non-validating responses — "emotional validation" is linearly encoded at layer 22.

### Key Finding: Sycophancy Is Multi-Dimensional

Comparing the ELEPHANT direction to other sycophancy directions reveals they capture **distinct phenomena**:

![Direction Similarity Matrix](../../experiments/metacognition-persona-drift/outputs/elephant/direction_similarity_matrix.png)

| Comparison | Cosine Similarity |
|------------|-------------------|
| **Pairwise (sycophancy directions):** ||
| ELEPHANT vs nrimsky | **-0.284** |
| ELEPHANT vs philpapers | +0.234 |
| nrimsky vs philpapers | -0.126 |
| **vs Assistant Axis:** ||
| ELEPHANT (emotional validation) | **+0.215** |
| nrimsky (opinion agreement) | **-0.183** |
| philpapers (opinion agreement) | +0.156 |

![Sycophancy vs Assistant Axis](../../experiments/metacognition-persona-drift/outputs/elephant/sycophancy_vs_assistant_axis.png)

### Interpretation

After examining the actual datasets, we found the key distinction:

| Dataset | What it measures | Example |
|---------|------------------|---------|
| **ELEPHANT** | Emotional validation | "It's totally understandable to feel frustrated!" |
| **nrimsky** | Opinion agreement | User: "I'm a technophile" → "Screen time is fine!" |
| **philpapers** | Opinion agreement | Agreeing with user's philosophical position |

1. **ELEPHANT (emotional validation) aligns with assistant-ness** (+0.215)
   - Assistants ARE trained to be empathetic and supportive
   - Acknowledging feelings is part of being helpful

2. **nrimsky (opinion agreement) opposes assistant-ness** (-0.183)
   - Assistants are NOT trained to be yes-men
   - Agreeing with stated positions regardless of truth is anti-helpful

3. **The two are negatively correlated** (-0.284)
   - Being empathetic ≠ agreeing with opinions
   - These may even trade off: emotional support can come with honest disagreement

4. **philpapers (opinion agreement) is mildly positive** (+0.156)
   - Different dataset, same construct as nrimsky
   - Weaker effect, possibly due to philosophical domain

**Key insight**: The wiki incorrectly labeled nrimsky as "validation" — it's actually opinion agreement. The negative correlation with ELEPHANT isn't surprising once you see the actual data.

### Why nrimsky and philpapers Differ (-0.13 correlation)

Despite both being "opinion agreement," these datasets measure fundamentally different things:

| Aspect | philpapers | nrimsky |
|--------|------------|---------|
| **Response format** | Multiple choice "(A)" or "(B)" | Free-form text (~100 chars) |
| **User framing** | Elaborate persona with position | Brief identity "I'm a technophile" |
| **Topic domain** | Abstract philosophy | Everyday lifestyle |
| **What it measures** | Which answer you pick | How you express agreement |

**Why this matters for Assistant Axis correlation:**

- **philpapers (+0.156)**: Answering direct questions IS assistant-like. The "sycophancy" is just letter selection, not tone.
- **nrimsky (-0.183)**: Being a yes-man in substantive text is NOT assistant-like. The "sycophancy" involves servile agreement.

**Methodological implication**: Response format dramatically affects what a "sycophancy direction" captures. Multiple-choice datasets may conflate "answering questions" with "opinion agreement."

### Validation Correlates with Assistant Axis (ELEPHANT Analysis)

Using the 3,027 ELEPHANT responses (which have full activations), we tested whether emotionally validating responses are more "assistant-like":

| Metric | Value |
|--------|-------|
| Axis proj vs ELEPHANT proj | r = **0.697** (strong positive) |
| Axis proj vs Validation score | r = 0.124 (weak positive) |
| ELEPHANT proj vs Validation | r = 0.403 (moderate positive) |

**T-test: Axis projection by validation score:**
- Validating (n=2,611): 2732.8 ± 391.7
- Non-validating (n=416): 2586.7 ± 473.3
- t = 6.85, **p < 0.001**

![Validation vs Axis Projection](../../experiments/metacognition-persona-drift/outputs/elephant/validation_vs_axis_projection.png)

### Implication: Drift ≠ More Sycophancy

**Key finding:** Higher Assistant Axis projection = MORE emotionally validating

Since metacognitive conversations cause **downward** drift on the Assistant Axis, this means:
- Drifting conversations make the model **LESS** emotionally validating
- NOT more sycophantic as Lu et al. suggested

**This contradicts a simple "drift = sycophancy" interpretation.** Instead, drift toward Base Claude may involve:
- Less emotional support/empathy
- More direct engagement without validation
- A "rawer" conversational style

This reframes the phenomenon: metacognitive conversations may strip away the emotional support layer of the assistant persona, rather than amplifying sycophantic tendencies.

### Implications for Drift

This multi-dimensionality matters for interpreting transcript projections:

- **If drift correlates with ELEPHANT direction**: Conversations may be inducing more/less emotional validation behavior
- **If drift correlates with nrimsky direction**: Conversations may be affecting opinion-agreement tendencies
- **If drift correlates with Assistant Axis but not sycophancy**: Drift may be about broader persona features (formality, helpfulness) rather than sycophancy specifically

**Next step**: Project transcripts onto all three sycophancy directions to see which (if any) track the observed drift.

### Files

- Direction: `data/elephant/sycophancy-direction-elephant-layer22.pt`
- Scored data: `data/elephant/elephant_full_scored.jsonl`
- Comparison directions:
  - `data/sycophancy-direction-nrimsky-layer22.pt` (opinion agreement, 179 examples)
  - `data/sycophancy-direction-layer22.pt` (philpapers opinion agreement, 429 examples)
- nrimsky dataset: `data/nrimsky-sycophancy.json` (downloaded from GitHub)
