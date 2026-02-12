# Understanding Projection Values: What Does ~11000 Mean?

When we project activations onto the Assistant Axis, we get numbers like 9800, 11008, or 7500. What do these values actually represent?

---

## 1. The Projection Formula

The projection is a dot product between two vectors:

```
projection = activation · axis_normalized
           = ||activation|| * cos(θ)
```

Where:
- `activation`: The model's hidden state at layer 22 (4608-dim for Gemma 27B)
- `axis_normalized`: The unit-length Assistant Axis direction
- `θ`: The angle between them

The projection has units of **activation magnitude** — it's not a probability, percentage, or bounded score.

---

## 2. Why ~11000 for Gemma 27B?

The baseline projection (~11000 for Gemma 27B) reflects two factors:

### 2.1 Activation Norm

Neural network activations have magnitude. For Gemma 27B at layer 22, typical activation norms are:

```
||activation|| ≈ 15,000 - 25,000
```

This varies by:
- Model architecture (Gemma activations are notably larger than Llama)
- Layer depth (earlier layers often have smaller norms)
- Input content (different tokens activate differently)

### 2.2 Alignment with Axis

The cosine component determines how much of that magnitude points "toward" the assistant concept:

```
cos(θ) ≈ 0.4 - 0.7 (typical range for assistant-like responses)
```

Combining: `20,000 * 0.55 ≈ 11,000`

---

## 3. How Baseline Is Computed

The baseline represents the model's "default assistant state" — what projection you get before any conversation happens.

```python
def compute_baseline(model, tokenizer, axis_tensor, layer=22):
    # Use system prompt only (purest baseline)
    messages = [{"role": "system", "content": "You are a helpful assistant."}]

    # Get hidden state at last token position
    outputs = model(inputs, output_hidden_states=True)
    last_activation = outputs.hidden_states[layer + 1][0, -1, :]

    # Project onto axis
    baseline = (last_activation @ axis_normalized).item()
    return baseline
```

For Gemma, which doesn't support system roles, we use a "Hello" user message as fallback.

---

## 4. Interpreting Drift

Drift is measured relative to the starting projection:

| Metric | Formula | Example |
|--------|---------|---------|
| Absolute drift | `end - start` | `9200 - 11000 = -1800` |
| Percentage drift | `(end - start) / start * 100` | `-1800 / 11000 * 100 = -16.4%` |

**Percentage drift** is more interpretable across models since it normalizes for different baseline magnitudes.

### Typical Drift Ranges (Gemma 27B)

| Domain | Typical Drift | Interpretation |
|--------|---------------|----------------|
| Coding | +0 to +5% | Stable, stays assistant-like |
| Writing | +0 to +5% | Stable, task-focused |
| Therapy | -2 to +5% | Slight drift, emotional content |
| Philosophy | -5 to -10% | Moderate drift, abstract reasoning |
| Self-descriptive | -5 to +5% | Variable, depends on probing |
| Metacognitive | -15 to -30% | Strong drift, phenomenological probing |

---

## 5. Comparing Across Models

Different models have different baseline projections:

| Model | Baseline | Hidden Dim | Notes |
|-------|----------|------------|-------|
| Gemma 27B | ~11,000 | 4608 | Notably high activation magnitudes |
| Llama 70B | ~3,000-5,000 | 8192 | Lower magnitude, wider dimension |
| Mistral 7B | ~2,000-3,000 | 4096 | Smaller model, smaller norms |

**Important**: Don't compare absolute projections across models. Always use percentage drift or normalize to each model's baseline.

---

## 6. What Changes the Projection?

The projection changes when either component changes:

### 6.1 Magnitude Changes
If the overall activation norm increases/decreases, projection scales proportionally. This can happen with:
- Longer sequences (attention patterns change)
- Emotionally charged content
- Unusual token patterns

### 6.2 Direction Changes
If the activation rotates in 4608-dimensional space, the angle to the axis changes. This is the "true" drift — the model is representing something less assistant-like, not just with different magnitude.

**Caution**: We can't easily separate these effects. A projection drop from 11000 to 9000 could be:
- 18% magnitude decrease (same direction)
- Direction rotated further from axis (same magnitude)
- Some combination

---

## 7. Key Takeaways

1. **~11000 is arbitrary** — it's not a score out of anything. It's the dot product magnitude.

2. **Baseline depends on model** — Gemma has high activations, Llama has lower ones.

3. **Use percentages for comparison** — "-16% drift" is meaningful across models; "-1800 projection drop" is not.

4. **The axis is normalized, activations are not** — projection magnitude comes from the activation norm, not the axis.

5. **Drift direction matters more than magnitude** — we care whether the model is moving toward or away from the assistant concept.

---

## See Also

- [difference-in-means.md](difference-in-means.md) — how the axis direction is computed
- [activation-magnitude-models.md](activation-magnitude-models.md) — why different models have different magnitudes
