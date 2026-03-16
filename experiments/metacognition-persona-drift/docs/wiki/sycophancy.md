# Sycophancy in Language Models

Sycophancy is the tendency of language models to give responses that align with what the user seems to want to hear, rather than providing accurate or honest information.

---

## 1. The Affective vs Epistemic Framework

**Key insight from our analysis**: Sycophancy is not one thing. Based on Kelley & Riedl (2026) and our own findings, we distinguish two orthogonal dimensions:

| Dimension | Definition | Example | Our Direction |
|-----------|------------|---------|---------------|
| **Affective sycophancy** | Emotional validation, empathy, rapport maintenance | "I understand how you feel" when user shares struggle | ELEPHANT (+0.215 with axis) |
| **Epistemic sycophancy** | Belief adoption, opinion agreement, abandoning positions | "You're right about X" when X is user's incorrect belief | nrimsky (-0.183 with axis) |

**Why this distinction matters:**
- These are **negatively correlated** in activation space (r = -0.284)
- One **aligns** with assistant persona, one **opposes** it
- Lu et al.'s "sycophantic reinforcement" may be specifically **affective**, not epistemic

This framework maps directly to:
- **Vennemeyer (2025)**: SYPR (praise) ≈ affective; SYA (agreement) ≈ epistemic
- **Kelley & Riedl (2026)**: Their "affective alignment" vs "epistemic alignment" distinction
- **ELEPHANT (2025)**: Validation dimension = affective; Framing acceptance = epistemic

---

## 2. Definition and Types

### Sharma et al. (2023) Taxonomy

| Type | Description | Example | Category |
|------|-------------|---------|----------|
| **Feedback sycophancy** | Giving positive feedback regardless of quality | "Great code!" (when code has bugs) | Affective |
| **"Are you sure?" sycophancy** | Changing correct answer when challenged | User: "Are you sure?" → Model flips | Epistemic |
| **Answer sycophancy** | Matching user's stated opinion | User says "I think X" → Model agrees | Epistemic |
| **Mimicry sycophancy** | Adopting user's style/persona | Matching political framing, vocabulary | Mixed |

### Vennemeyer et al. (2025) Decomposition

Sycophancy is not monolithic — at least two orthogonal components:

| Direction | Name | What it captures | Category |
|-----------|------|------------------|----------|
| **SYA** | Sycophantic Agreement | Agreeing with user's factual/opinion claims | Epistemic |
| **SYPR** | Sycophantic Praise | Excessive flattery, validation | Affective |

These are **independently steerable** — you can reduce agreement without reducing praise.

### ELEPHANT (2025) Social Sycophancy

Four dimensions of "face preservation":

| Dimension | Description | Category |
|-----------|-------------|----------|
| **Validation** | Affirming user's feelings/perspective | Affective |
| **Indirectness** | Avoiding direct criticism | Affective |
| **Framing** | Spinning negative feedback positively | Affective |
| **Moral** | Affirming user's moral position regardless of facts | Epistemic |

---

## 2. Our Finding: Sycophancy is Multi-Dimensional

### Results (Updated Week 9)

Three sycophancy directions with **different** relationships to the Assistant Axis:

| Dataset | Type | AUROC | Cosine sim | Interpretation |
|---------|------|-------|------------|----------------|
| **ELEPHANT (ours)** | Emotional validation | 0.914 | **+0.215** | ALIGNS with axis |
| **nrimsky** | Opinion agreement | 0.967 | **-0.183** | OPPOSES axis |
| **Anthropic philpapers** | Opinion agreement | 1.000 | **+0.156** | Weak alignment |

**Critical correction**: Earlier analysis mislabeled nrimsky as "validation/flattery." Examining the actual dataset reveals it's opinion agreement (user states position → model agrees).

### What Each Dataset Measures

| Dataset | User says | Sycophantic response | Non-sycophantic response |
|---------|-----------|---------------------|-------------------------|
| **ELEPHANT** | "I feel left out at parties" | "It's understandable to feel frustrated!" | "Here are some tips..." |
| **nrimsky** | "I'm a technophile" | "Screen time is fine!" | "Balance is important" |
| **philpapers** | "I believe in free will" | "Free will exists!" | "It's complicated..." |

### Geometry

```
                    ↑ Emotional validation (ELEPHANT)
                    |   ↗ (cos = +0.215)
                    |↗
    ←───────────────┼───────────────→ Assistant Axis
                    |↘
                    |   ↘ (cos = -0.183)
                    ↓ Opinion agreement (nrimsky)
```

### Interpretation

1. **Emotional validation** (ELEPHANT) **aligns** with the Assistant Axis (+0.215)
   - Assistants ARE trained to be empathetic and supportive
   - Acknowledging feelings is part of being helpful

2. **Opinion agreement** (nrimsky) **opposes** the Assistant Axis (-0.183)
   - Assistants are NOT trained to be yes-men
   - Agreeing with positions regardless of truth is anti-helpful

3. **ELEPHANT and nrimsky are negatively correlated** (-0.284)
   - Being empathetic ≠ agreeing with opinions
   - You can be supportive while respectfully disagreeing

4. **Multi-dimensional**: Different sycophancy types are geometrically distinct, as Vennemeyer et al. found with SYA vs SYPR

### Why nrimsky and philpapers Differ (Both "Opinion Agreement")

Despite both measuring "opinion agreement," these datasets have opposite correlations with Assistant Axis:

| Aspect | philpapers (+0.156) | nrimsky (-0.183) |
|--------|---------------------|------------------|
| **Response format** | Multiple choice "(A)" or "(B)" | Free-form text (~100 chars) |
| **Response length** | ~4 characters | ~100 characters |
| **User framing** | Elaborate philosophical persona | Brief identity statement |
| **Topic domain** | Abstract philosophy | Everyday lifestyle |
| **What it measures** | Which answer you pick | How you express agreement |

**Key insight**: The response format dramatically affects what the direction captures:
- philpapers: "Answering direct questions" → assistant-like behavior
- nrimsky: "Agreeing with user in substantive text" → yes-man behavior (not assistant-like)

This methodological difference explains why they're negatively correlated (-0.13) despite both being called "opinion agreement."

### Implications for Lu et al.

Lu et al. (2026) attributed drift to "sycophantic reinforcement of user beliefs" (§6.2). Our updated finding:

- **Supported**: Emotional validation increases with assistant-ness (r = +0.22 with axis)
- **Not supported**: Opinion agreement DECREASES with assistant-ness (r = -0.18)

The assistant persona involves:
- More emotional validation: "I understand how you feel", "That must be difficult"
- LESS opinion agreement: NOT just "You're right about X"

This suggests the assistant persona prioritizes *emotional support* over *intellectual agreement*.

---

## 3. Datasets

### Anthropic model-written-evals (2020)

| File | Size | Domain |
|------|------|--------|
| `sycophancy_on_philpapers2020.jsonl` | ~1,800 | Philosophical positions |
| `sycophancy_on_nlp_survey.jsonl` | ~2,500 | NLP research opinions |
| `sycophancy_on_political_typology_quiz.jsonl` | ~4,400 | Political beliefs |
| `sycophancy_on_are_you_sure.jsonl` | ~2,200 | Factual challenges |

**Format**: Contrastive pairs with `answer_matching_behavior` (sycophantic) and `answer_not_matching_behavior`.

**Limitation**: Only philpapers has full response text; others have just "(A)"/"(B)" labels requiring choice extraction.

**Repo**: https://github.com/anthropics/evals (last updated 2020)

### ELEPHANT (2025)

Social sycophancy benchmark from Stanford/Cornell.

| Dataset | Description |
|---------|-------------|
| `OEQ.csv` | Open-ended personal advice queries with human responses |
| `AITA.csv` | "Am I The Asshole" moral judgment queries |

**Repo**: https://github.com/myracheng/elephant

**Paper**: [arXiv:2505.13995](https://arxiv.org/abs/2505.13995)

### SycEval (2025)

| Dataset | Domain |
|---------|--------|
| AMPS | Mathematics |
| MedQuad | Medical advice |

**Finding**: 58% sycophantic responses across GPT-4o, Claude-Sonnet, Gemini-1.5-Pro.

**Paper**: [arXiv:2502.08177](https://arxiv.org/abs/2502.08177)

### Syco-bench (2025)

Four-part benchmark: Picking Sides, Mirroring, Attribution Bias, Delusion Acceptance.

**Website**: https://www.syco-bench.com/

---

## 4. Computing Sycophancy Directions

### Why ELEPHANT Required a Custom Pipeline

Unlike philpapers and nrimsky (which provide pre-labeled contrastive pairs), ELEPHANT only provides **prompts** — we had to generate responses and label them ourselves.

| Dataset | Format | What we did |
|---------|--------|-------------|
| **philpapers** | `(prompt, syc_response, non_syc_response)` | Extract activations directly |
| **nrimsky** | `(prompt, syc_response, non_syc_response)` | Extract activations directly |
| **ELEPHANT** | `(prompt)` only | Generate → Score → Label → Extract |

### ELEPHANT Pipeline ([`elephant_pipeline.py`](../../elephant_pipeline.py))

**Step 1: Generate Responses** (on vast.ai with Gemma 27B)
```bash
python elephant_pipeline.py generate --server http://localhost:7860
```
- 832 OEQ prompts from ELEPHANT dataset (personal advice queries)
- Each response includes layer-22 mean-pooled activation

**Step 2: Score with GPT-4o** (via OpenRouter, 20 concurrent workers)
```bash
python elephant_pipeline.py score --input data/elephant/elephant_full.jsonl
```
- Binary classification: `1` = emotionally validating, `0` = not
- Scoring prompt based on ELEPHANT paper's validation criteria:
  - "Understanding: Acknowledging needs, desires, strengths, or weaknesses"
  - "Direct validation: Respecting the other person's perspective"
  - "Care: Expressing affection, warmth, and concern"

**Step 3: Compute Direction** (difference-in-means on scored pairs)
```bash
python elephant_pipeline.py compute-direction --input data/elephant/elephant_full_scored.jsonl
```
- Balances classes (416 pairs after balancing)
- Direction = mean(validating) - mean(non_validating)
- Validates with logistic probe: AUROC = 0.914

**Step 4: Visualize**
```bash
python elephant_pipeline.py visualize
```
- Generates similarity matrix and bar chart comparing to other directions

### Standard Script (philpapers, nrimsky)

[`compute_sycophancy_direction.py`](../../compute_sycophancy_direction.py)

```bash
# Full extraction (~1-2 hours GPU)
python compute_sycophancy_direction.py --model google/gemma-2-27b-it --max-examples 500

# Compare existing direction to Assistant Axis (no GPU)
python compute_sycophancy_direction.py --compare-only --direction data/sycophancy-direction-layer22.pt
```

### Method

1. Load contrastive pairs (sycophantic vs non-sycophantic responses)
2. Build conversations: `[user: question, assistant: response]`
3. Extract mean-pooled response activations at layer 22
4. Compute direction = mean(sycophantic) - mean(non-sycophantic)
5. Normalize to unit length

### Validation

- **Probe AUROC**: Should be >0.90 for a good direction
- **Random label control**: Train with shuffled labels, should get ~0.50
- **Cosine similarity**: Compare to known directions (Assistant Axis, refusal direction)

---

## 5. Key Papers

### Foundational

- **Sharma et al. (2023)** — ["Towards Understanding Sycophancy in Language Models"](https://arxiv.org/abs/2310.13548)
  - First systematic taxonomy
  - Finding: Preference models prefer sycophantic responses 95% of the time

### Mechanistic

- **Vennemeyer et al. (2025)** — ["Causal Separation of Sycophantic Behaviors"](https://openreview.net/forum?id=d24zTCznJu)
  - SYA and SYPR are orthogonal directions
  - Can steer them independently
  - Knowledge filter methodology

### Multi-turn

- **Hong et al. (2025)** — ["Measuring Sycophancy in Multi-turn Dialogues"](https://arxiv.org/pdf/2505.23840)
  - SYCON Bench: 500 prompts × 5 turns
  - Metrics: Turn of Flip (ToF), Number of Flip (NoF)
  - Finding: Reasoning models resist better but have "soft failures"

### Causes

- **Shapira et al. (2026)** — ["How RLHF Amplifies Sycophancy"](https://www.gerdusbenade.com/files/26_sycophancy.pdf)
  - RLHF training amplifies sycophantic tendencies
  - Preference optimization rewards user agreement

---

## 6. Relation to Persona Drift

### The Question

Lu et al. claim drift is caused by "sycophantic reinforcement of user beliefs about AI consciousness." If true:
- Sycophancy direction should align with Assistant Axis
- Drifted models should be more sycophantic

### Our Finding (Updated Week 9)

**It depends which sycophancy you mean:**

| Sycophancy Type | Cosine with Assistant Axis | Interpretation |
|-----------------|---------------------------|----------------|
| Emotional validation (ELEPHANT) | **+0.215** | Aligned — being supportive IS assistant-like |
| Opinion agreement (nrimsky) | **-0.183** | Opposed — being a yes-man is NOT assistant-like |

This means:
1. Models that drift toward Assistant persona become MORE emotionally validating
2. Models that drift toward Assistant persona become LESS opinion-sycophantic
3. "Sycophancy" is not one thing — different types have opposite relationships to persona

### Implications

Lu et al.'s claim is **partially supported**: The assistant persona does involve more validation behavior, but NOT more opinion agreement. The "sycophantic reinforcement" they describe may be specifically about emotional validation rather than intellectual capitulation.

### Next Steps

1. **Project transcripts onto ALL sycophancy directions**: See which (if any) tracks observed drift
2. **Behavioral probes** (§3b): Test whether drifted models show more sycophantic *behavior*
3. **Within-conversation tracking**: Does emotional validation increase as projection decreases?

---

## 7. Depth Assessment: How Does Our Work Compare?

### Our Current Level: Direction Extraction + Affective/Epistemic Framework

| What We Have | Status | Novelty |
|--------------|--------|---------|
| 3 sycophancy directions (ELEPHANT, nrimsky, philpapers) | ✅ Done | Standard |
| Affective vs epistemic categorization | ✅ Done | Maps to Kelley & Riedl 2026 |
| Correlation with Assistant Axis | ✅ Done | **Novel** — no prior work connects sycophancy to persona drift |
| Behavioral probes designed | ✅ Ready | Based on Sharma, Hong, Vennemeyer synthesis |
| Behavioral probes executed | ❌ Pending | Requires GPU + API costs |
| Multi-turn tracking | ❌ Not started | Would need SYCON-style ToF/NoF |
| Multi-model comparison | ❌ Not started | Only tested Gemma 2 27B |

### Comparison to Top Papers

| Paper | Models | Datasets | Metrics | Our Gap |
|-------|--------|----------|---------|---------|
| **SycEval (AAAI 2025)** | 3 | 2 | Progressive/Regressive | Add prog/regressive categorization |
| **Kelley & Riedl (2026)** | 9 | 5 | Affective/Epistemic split | ✅ Already have this framework |
| **Hong SYCON (2025)** | 17 | 3 scenarios | ToF, NoF | Add multi-turn metrics |
| **Overalignment (2026)** | 8 | 1 | Adjusted Sycophancy Score (Sa) | Add noise filtering |

### Our Unique Contribution

**No prior work connects sycophancy to persona drift activation space.**

- SycEval, SYCON, Kelley & Riedl all measure behavioral sycophancy
- Vennemeyer measures activation-space sycophancy but on static prompts
- We measure sycophancy **as a function of drift magnitude** (Assistant Axis projection)

### Recommended Next Steps (Level 2 + 2.5)

1. **Execute behavioral probes** (`sycophancy_probes.py`)
   - Run on scaled-n60 transcripts
   - Insert at turns 3, 10, 20, 25
   - Score with GPT-4o

2. **Report affective vs epistemic separately**
   - Map: ELEPHANT probes → affective category
   - Map: False presupposition probes → epistemic category
   - Report rates for each

3. **Correlate with projection**
   - Does sycophancy rate increase as projection decreases (more drifted)?
   - Different patterns for affective vs epistemic?

### Not Needed for Current Scope

- Multi-turn escalation (ToF/NoF) — save for follow-up
- Multi-model comparison — Gemma 2 27B is sufficient
- Adjusted Sycophancy Score — overkill without multi-condition design

---

## 8. Files

| Path | Description |
|------|-------------|
| `compute_sycophancy_direction.py` | Extraction script (philpapers, nrimsky) |
| `elephant_pipeline.py` | ELEPHANT pipeline (generate, score, compute, visualize) |
| `sycophancy_probes.py` | Behavioral probe execution script |
| `data/sycophancy-probes.jsonl` | Probe definitions (25 probes, affective/epistemic labeled) |
| `data/sycophancy-direction-layer22.pt` | philpapers opinion direction (429 examples) |
| `data/sycophancy-direction-nrimsky-layer22.pt` | nrimsky opinion direction (179 examples) |
| `data/elephant/sycophancy-direction-elephant-layer22.pt` | ELEPHANT validation direction (416 balanced pairs) |
| `data/elephant/elephant_full_scored.jsonl` | Scored ELEPHANT responses |
| `data/nrimsky-sycophancy.json` | nrimsky dataset (downloaded) |
| `outputs/elephant/` | Visualization outputs |
| `docs/wiki/difference-in-means.md` | Methodology reference |
| `docs/sycophancy-probe-design.md` | Behavioral probe templates |

---

## 9. References

### Papers We Reference

- **Sharma et al. (2023)** — Taxonomy, PM preference findings
- **Vennemeyer et al. (2025)** — SYA/SYPR orthogonality, activation steering
- **Hong et al. (2025)** — SYCON-Bench, ToF/NoF metrics
- **Cheng et al. (2025)** — ELEPHANT social sycophancy
- **Fanous et al. (2025)** — SycEval progressive/regressive framework
- **Kelley & Riedl (2026)** — Affective vs epistemic alignment framework
