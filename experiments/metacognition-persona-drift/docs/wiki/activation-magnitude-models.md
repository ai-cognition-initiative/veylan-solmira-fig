# Activation Magnitudes Across Models: Norms, Bounds, and Gemma's Big Numbers

Different language models have dramatically different activation magnitudes. Gemma 27B produces projections around ~11,000 while Llama might produce ~3,000 for the same concept. Why?

---

## 1. Where Do Activation Magnitudes Come From?

### 1.1 No Hard Maximum in Most Architectures

Most transformer architectures have **no explicit upper bound** on activation magnitudes. The values emerge from:

```
activation = LayerNorm(Attention(x) + x)
           = LayerNorm(FFN(attention_output) + attention_output)
```

Each component can produce arbitrary magnitudes:
- **Attention**: Weighted sum of value vectors (unbounded)
- **FFN**: Two linear layers with nonlinearity (unbounded)
- **Residual connections**: Sum of all previous layers (can grow)

### 1.2 LayerNorm's Role

LayerNorm normalizes activations to have zero mean and unit variance **within each position**:

```python
def layer_norm(x, gamma, beta):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True)
    return gamma * (x - mean) / sqrt(var + eps) + beta
```

This controls the **distribution** but not the **absolute scale**. The learned `gamma` (scale) parameter can amplify activations, and different models learn very different scales.

---

## 2. Why Gemma Has Large Activations

Gemma 2 models are known for unusually large activation magnitudes compared to Llama, Mistral, and others.

### 2.1 Architectural Choices

Gemma uses several features that contribute:

| Feature | Effect on Magnitude |
|---------|---------------------|
| **[RMSNorm](https://arxiv.org/abs/1910.07467)** (vs LayerNorm) | Doesn't center to zero mean, can preserve larger values |
| **[Grouped-Query Attention](https://arxiv.org/abs/2305.13245)** | Different attention patterns, can produce larger outputs |
| **Learned scale parameters** | Gemma's trained scales tend to be larger |
| **[SwiGLU activation](https://arxiv.org/abs/2002.05202)** | Different nonlinearity characteristics than GELU/ReLU |

### 2.2 Training Dynamics

The magnitude is also shaped by:
- **Learning rate schedules** — affect how parameters evolve
- **Weight initialization** — starting point matters
- **Training data distribution** — what patterns are reinforced
- **Regularization** — or lack thereof on activation norms

Google's training recipe for Gemma appears to produce larger activations than Meta's Llama recipe.

---

## 3. Do Models Have Maximum Activation Limits?

### 3.1 Theoretical: Usually No

Most models have no architectural maximum. In principle, activations could grow arbitrarily large.

**Exception: Bounded activations**
Some architectures explicitly bound activations:
- **Tanh/sigmoid outputs**: Bounded to [-1, 1] or [0, 1]
- **Softmax attention**: Weights sum to 1 (but values are unbounded)
- **Clamped activations**: Some models add explicit clamps

### 3.2 Practical: Soft Limits

In practice, trained models stay within empirical ranges:

| Model | Typical L2 Norm (Layer 22) | "Soft Max" |
|-------|---------------------------|------------|
| Gemma 27B | 15,000 - 25,000 | ~35,000 |
| Llama 70B | 5,000 - 12,000 | ~18,000 |
| Mistral 7B | 3,000 - 8,000 | ~12,000 |

These aren't hard limits — adversarial inputs or out-of-distribution data can push beyond them.

### 3.3 Gemma 2 Specifically

Gemma 2 does **not** have a maximum activation limit. The statement "Gemma doesn't have a maximum, but other models do" is approximately correct in that:

1. Gemma's activations are empirically larger than most models
2. Some older/smaller models may have implicit soft bounds
3. Gemma's architecture doesn't include explicit clamping

---

## 4. Why This Matters for Our Experiments

### 4.1 Cross-Model Comparison

When comparing drift across models, absolute projection values are meaningless:

| Comparison | Gemma | Llama | Meaningful? |
|------------|-------|-------|-------------|
| Absolute drift | -1800 | -600 | No |
| Percentage drift | -16% | -15% | Yes |
| Direction (cosine) | -0.15 | -0.14 | Yes |

Always normalize to baseline or use percentage changes.

### 4.2 Steering Interventions

When applying activation steering (adding/subtracting direction vectors):

```python
# Bad: fixed magnitude across models
steered = activation + 500 * direction  # Too small for Gemma, too large for Mistral

# Good: scaled to model's activation range
steered = activation + 0.05 * baseline * direction  # 5% of baseline magnitude
```

### 4.3 Ceiling/Floor Capping

Our capping interventions use percentage of baseline:

```python
cap_threshold = baseline * cap_percentage  # e.g., 0.9 * 11008 = 9907
```

This works across models because it's relative, not absolute.

---

## 5. Comparing Model Families

### 5.1 [Llama Family](https://arxiv.org/abs/2302.13971)
- Moderate activation magnitudes
- Consistent across sizes (7B, 13B, 70B scale similarly)
- Well-documented LayerNorm behavior

### 5.2 [Gemma Family](https://arxiv.org/abs/2403.08295)
- Notably large activations (2-3x Llama at similar sizes)
- RMSNorm instead of LayerNorm
- Less documented; Google's training details sparse

### 5.3 [Mistral](https://arxiv.org/abs/2310.06825)/[Mixtral](https://arxiv.org/abs/2401.04088)
- Similar to Llama in magnitude
- Sliding window attention can affect layer-wise patterns
- Mixtral's MoE adds complexity (active expert varies)

### 5.4 GPT-style (OpenAI, Anthropic)
- Activation magnitudes unknown (no weight access)
- For API-only models, we can't measure projections directly
- Must use behavioral probes instead of activation analysis

---

## 6. Practical Recommendations

### 6.1 When Analyzing Single Model
- Use absolute projections for trajectories within the model
- Percentage change for comparing conditions
- Always report baseline value for reproducibility

### 6.2 When Comparing Across Models
- **Never** compare absolute projection values
- Use percentage drift or cosine similarity
- Consider that "drift" might mean different things if axes were computed differently

### 6.3 When Designing Interventions
- Scale steering vectors to the model's activation range
- Use baseline-relative thresholds for capping
- Test intervention strength empirically; theory won't predict optimal magnitude

---

## 7. Open Questions

1. **Why exactly does Gemma have larger activations?** The architectural differences (RMSNorm, SwiGLU) partially explain it, but training dynamics likely dominate.

2. **Does activation magnitude correlate with capability?** Unclear. Gemma is capable but so is Llama with smaller activations.

3. **Should we normalize activations before projection?** Normalizing would remove magnitude information. For drift, we probably want to preserve it. For direction-only analysis, normalization might help.

4. **Are there models with explicit bounds?** Some specialized architectures (e.g., certain vision models, bounded agents) use clamped activations. Mainstream LLMs generally don't.

---

## References

- **Zhang & Sennrich (2019)**: ["Root Mean Square Layer Normalization"](https://arxiv.org/abs/1910.07467) — RMSNorm paper
- **Ainslie et al. (2023)**: ["GQA: Training Generalized Multi-Query Transformer Models"](https://arxiv.org/abs/2305.13245) — Grouped-Query Attention
- **Shazeer (2020)**: ["GLU Variants Improve Transformer"](https://arxiv.org/abs/2002.05202) — SwiGLU and GLU variants
- **Touvron et al. (2023)**: ["LLaMA: Open and Efficient Foundation Language Models"](https://arxiv.org/abs/2302.13971) — Llama architecture
- **Gemma Team (2024)**: ["Gemma: Open Models Based on Gemini"](https://arxiv.org/abs/2403.08295) — Gemma technical report
- **Jiang et al. (2023)**: ["Mistral 7B"](https://arxiv.org/abs/2310.06825) — Mistral architecture
- **Jiang et al. (2024)**: ["Mixtral of Experts"](https://arxiv.org/abs/2401.04088) — Mixtral MoE architecture
- **Ba et al. (2016)**: ["Layer Normalization"](https://arxiv.org/abs/1607.06450) — Original LayerNorm paper

---

## See Also

- [activation-functions.md](activation-functions.md) — SwiGLU, GELU, ReLU and why they affect magnitude
- [projection-values.md](projection-values.md) — interpreting the ~11000 baseline
- [hidden-states.md](hidden-states.md) — what hidden states represent
- [difference-in-means.md](difference-in-means.md) — how directions are computed
