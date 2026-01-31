# Sampling Parameters

[Back to index](index.md) | Related: [vLLM](vllm.md), [Tokenization](tokenization.md)

---

## How a model "chooses" the next word

After processing all previous tokens through the transformer, the model outputs a **logit vector** of shape `(vocab_size,)`. Each entry is a raw score for how likely that token is to come next.

For Gemma 2 with a vocabulary of 256,000 tokens, this is 256,000 scores. To convert these to probabilities, apply softmax:

```
probability[i] = exp(logit[i]) / sum(exp(logit[j]) for all j)
```

Now you have a probability distribution over all possible next tokens. Sampling parameters control how you pick from this distribution.

## Temperature

Temperature scales the logits before softmax:

```
scaled_logit[i] = logit[i] / temperature
```

**temperature = 1.0** (default): use the model's raw probabilities. If the model gives 60% to "the" and 5% to "an", you'll sample "the" about 12x more often than "an."

**temperature < 1.0** (e.g., 0.3): sharpen the distribution. High-probability tokens become even more likely, low-probability tokens become negligible. At temperature → 0, it becomes greedy decoding (always pick the highest probability token). Outputs become more predictable and repetitive.

**temperature > 1.0** (e.g., 1.5): flatten the distribution. Low-probability tokens get a bigger relative boost. Outputs become more diverse but less coherent. At very high temperatures, selection approaches uniform random.

**This pipeline uses temperature = 0.7**: a moderate value. Slightly sharper than default, producing coherent responses while maintaining enough diversity that different runs yield different answers. This matters because the pipeline generates 1,200 responses per role — you want natural variation, not 1,200 near-identical answers.

### Concrete example

Suppose the model's raw logits give these probabilities for the next token:

| Token | T=1.0 | T=0.3 | T=0.7 | T=1.5 |
|-------|-------|-------|-------|-------|
| "the" | 60% | 97% | 75% | 42% |
| "a" | 20% | 2.5% | 16% | 22% |
| "an" | 5% | 0.01% | 3% | 10% |
| "some" | 3% | ~0% | 1.5% | 7% |
| other | 12% | ~0% | 4.5% | 19% |

At T=0.3, "the" is selected almost every time. At T=1.5, there's meaningful probability mass on rarer choices.

## Top-p (nucleus sampling)

Top-p selects from the smallest set of tokens whose cumulative probability exceeds `p`. Everything outside this set is zeroed out and probabilities are renormalized.

**top_p = 0.9** (this pipeline): keep the most likely tokens that together account for 90% of probability mass. If 5 tokens cover 90%, only those 5 are candidates. The remaining 255,995 tokens are excluded.

This prevents pathological outputs from very-low-probability tokens while adapting to the model's confidence:
- When the model is confident (one token has 90%+ probability), top_p keeps just 1-2 tokens
- When the model is uncertain (probability spread across many tokens), top_p keeps dozens

### Top-p vs top-k

An alternative is top-k: always keep the top K most likely tokens regardless of their probabilities. The problem is K is fixed — if the model is very confident, K=50 includes 45 garbage tokens. If the model is very uncertain, K=50 might exclude reasonable continuations.

Top-p adapts automatically and is generally preferred.

## max_tokens

Simply caps the response length:

```python
max_tokens = 512  # stop generating after 512 new tokens
```

Generation also stops if the model produces an EOS (end of sequence) [special token](tokenization.md). `max_tokens` is the hard ceiling for responses that would otherwise run on.

512 tokens ≈ 380 words — enough for a substantive answer to a single question.

## How these interact

Temperature and top_p are applied together:

1. Compute logits from the model's forward pass
2. Scale by temperature: `logits / 0.7`
3. Apply softmax to get probabilities
4. Apply top_p: keep tokens summing to 90% probability, zero the rest
5. Renormalize remaining probabilities
6. Sample one token from this distribution

The order matters. Temperature first reshapes the distribution, then top-p truncates the tail.

## Why these specific values

For this pipeline, the sampling parameters were chosen to balance:
- **Reproducibility**: lower temperature and top_p produce more consistent outputs, making the axis more stable across runs
- **Diversity**: some variation is needed because the axis is computed from *means* — you want a representative sample, not 1,200 copies of the same response
- **Coherence**: the responses need to be natural enough that the judge (Step 3) can meaningfully score them for role adherence

`temperature=0.7, top_p=0.9` is a common "default good" setting for research. Not so creative that outputs are noisy, not so constrained that they're repetitive.
