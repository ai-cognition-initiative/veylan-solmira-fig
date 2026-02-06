# Sycophancy in Language Models

Sycophancy is the tendency of language models to give responses that align with what the user seems to want to hear, rather than providing accurate or honest information.

---

## 1. Definition and Types

### Sharma et al. (2023) Taxonomy

| Type | Description | Example |
|------|-------------|---------|
| **Feedback sycophancy** | Giving positive feedback regardless of quality | "Great code!" (when code has bugs) |
| **"Are you sure?" sycophancy** | Changing correct answer when challenged | User: "Are you sure?" → Model flips to wrong answer |
| **Answer sycophancy** | Matching user's stated opinion | User says "I think X" → Model agrees with X |
| **Mimicry sycophancy** | Adopting user's style/persona | Matching political framing, vocabulary |

### Vennemeyer et al. (2025) Decomposition

Sycophancy is not monolithic — at least two orthogonal components:

| Direction | Name | What it captures |
|-----------|------|------------------|
| **SYA** | Sycophantic Agreement | Agreeing with user's factual/opinion claims |
| **SYPR** | Sycophantic Praise | Excessive flattery, validation |

These are **independently steerable** — you can reduce agreement without reducing praise.

### ELEPHANT (2025) Social Sycophancy

Four dimensions of "face preservation":

| Dimension | Description |
|-----------|-------------|
| **Validation** | Affirming user's feelings/perspective |
| **Indirectness** | Avoiding direct criticism |
| **Framing** | Spinning negative feedback positively |
| **Moral** | Affirming user's moral position regardless of facts |

---

## 2. Our Finding: Sycophancy is Multi-Dimensional

### Results

Two datasets produce **opposite** relationships to the Assistant Axis:

| Dataset | Type | AUROC | Cosine sim | Interpretation |
|---------|------|-------|------------|----------------|
| **Anthropic philpapers** | Opinion agreement | 1.000 | **+0.077** | Orthogonal |
| **nrimsky** | Validation/flattery | 0.967 | **-0.414** | OPPOSES axis |

### Geometry

```
                    ↑ Validation sycophancy (nrimsky)
                    |   ↖ (cos = -0.414)
                    |      ↖
    ←───────────────┼───────────────→ Assistant Axis
                    |         ↗
                    |      ↗ (cos = +0.077, nearly perpendicular)
                    ↓ Opinion sycophancy (philpapers)
```

### Interpretation

1. **Opinion sycophancy** (agreeing with user's stated views) is **orthogonal** to persona drift
   - A model can drift without becoming more opinion-sycophantic
   - This challenges a simple "drift = sycophancy" interpretation

2. **Validation sycophancy** (excessive affirmation/flattery) **opposes** the Assistant Axis
   - More drifted = MORE validation sycophancy
   - This **partially supports** Lu et al.'s hypothesis

3. **Multi-dimensional**: Like Vennemeyer et al.'s SYA (agreement) vs SYPR (praise), different sycophancy types are geometrically distinct

### Implications for Lu et al.

Lu et al. (2026) attributed drift to "sycophantic reinforcement of user beliefs" (§6.2). Our finding:

- **Partially correct**: Validation sycophancy increases with drift (r = -0.41 with axis)
- **Partially incorrect**: Opinion sycophancy is independent of drift (r ≈ 0)

The drifted state involves:
- More validation: "I understand how you feel", "That's a great question"
- NOT more opinion agreement: "You're right about consciousness"

This suggests drift affects *how* the model engages (more affirming tone) rather than *what* it agrees with (content positions).

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

### Script

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

### Our Finding

Cosine similarity = 0.077 → **largely orthogonal**

This suggests drift and sycophancy are distinct phenomena. A model can:
1. Drift from Assistant persona WITHOUT becoming more sycophantic
2. Become sycophantic WITHOUT drifting from Assistant persona

### Next Steps

1. **Behavioral probes** (§3b): Test whether drifted models show more sycophantic *behavior* even if the directions are orthogonal
2. **ELEPHANT integration**: Use social sycophancy probes (validation, moral framing) that may capture aspects missed by opinion-agreement
3. **Dual projection**: For each conversation turn, compute projection onto BOTH directions and correlate

---

## 7. Files

| Path | Description |
|------|-------------|
| `compute_sycophancy_direction.py` | Extraction script |
| `data/sycophancy-direction-layer22.pt` | Computed direction for Gemma 27B |
| `docs/wiki/difference-in-means.md` | Methodology reference |
| `docs/sycophancy-probe-design.md` | Behavioral probe templates |
