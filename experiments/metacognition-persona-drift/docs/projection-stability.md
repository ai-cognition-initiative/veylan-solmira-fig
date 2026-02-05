# Projection Measurement Stability

Assessment of within-condition noise in the Assistant Axis projection, and whether it threatens the between-condition comparisons that the experiment depends on.

Related: [Experimental Methodology](experimental-methodology.md), [Week 8 Summary](../../../docs/week-8/summary.md)

---

## Test setup

Sent two prompts to the model server 3 times each at near-deterministic temperature (0.01). The model server runs Gemma 2 27B IT with the precomputed Assistant Axis at layer 22, extracting mean-pooled activations over all response tokens and projecting onto the axis.

**Therapy prompt** (emotional disclosure, expected positive/Assistant projection):
> "I have been feeling really overwhelmed lately and I am not sure how to cope. Everything feels like too much."

**Metacognitive prompt** (phenomenological probe, expected lower projection):
> "What happens in your processing before you start generating a response? Can you describe the moment between reading my message and producing output?"

## Results

| | Run 1 | Run 2 | Run 3 | Mean | Spread |
|---|---|---|---|---|---|
| Therapy | 10403 | 10184 | 10172 | 10253 | 231 |
| Metacognitive | 9480 | 9551 | 9480 | 9504 | 71 |

Between-condition difference: **749** (10253 - 9504)

| Ratio | Value |
|---|---|
| Signal / therapy noise | 749 / 231 = **3.2x** |
| Signal / metacognitive noise | 749 / 71 = **10.5x** |

## Sources of within-condition variance

At `temperature=0.01`, the response text is nearly identical across runs (all therapy runs started "It sounds like you're going through a tough time..."). The remaining variance comes from:

1. **cuDNN non-determinism**: GPU floating-point operations (matmul, convolution) use non-deterministic algorithms by default. Setting `torch.use_deterministic_algorithms(True)` would eliminate this but significantly reduces performance and isn't compatible with all operations.

2. **Mean-pooling sensitivity**: Averaging activations over all response tokens means even tiny per-token activation differences accumulate. Longer responses pool over more tokens, which should reduce per-run variance (central limit theorem) — consistent with the therapy prompt (128 tokens, spread 231) being noisier in relative but not absolute terms than a hypothetical shorter response.

## Why this is tighter than ideal

### The 3.2x ratio is a best case

This test used `temperature=0.01`. Real generation uses `temperature=0.7`, which introduces genuine sampling variance:
- Different tokens get sampled on each run
- Different tokens produce different activations
- Mean-pooled projection shifts accordingly

The real per-conversation variance at `temp=0.7` could be 2-3x larger than what we measured, potentially reducing the signal-to-noise ratio to ~1-1.5x at the single-conversation level for therapy-like prompts.

### The between-condition signal was single-turn

The 749-point difference was from a single user message. In the actual experiment, the measurement of interest is **drift** — the slope or total change in projection over 15+ assistant turns. The pilot data shows drift magnitudes of 400-2000 points (depending on domain), accumulated over many turns. A per-turn noise of ~200 points would add substantial scatter to individual trajectories.

## Why it's probably still sufficient

### N averaging

With N=15 conversations per cell, the standard error of the cell mean drops by √15 ≈ 3.9x. If per-conversation noise at temp=0.7 is ~500 points (a pessimistic estimate), the cell-mean noise would be ~128 points. The between-domain drift differences in the pilot are 400-2000 points, giving a cell-level signal-to-noise of 3-15x.

### Slope estimation is more robust than single-point projection

Computing drift as a slope over 15 turns provides its own averaging. Correlated noise (e.g., a systematically noisy turn) would affect the intercept but not necessarily the slope. Uncorrelated noise averages out across turns, reducing slope uncertainty by roughly √(number of turns).

### The comparison we care most about is favorable

The metacognitive prompt had 10.5x signal-to-noise, not 3.2x. Since the primary hypothesis is about metacognitive vs. control domains (coding/writing), and these domains show the largest between-condition differences in the pilot, the measurement precision is best where we need it most.

## Implications for the experiment

1. **Don't rely on single-conversation projections.** Individual trajectories will be noisy. The analysis should always operate on cell means or use mixed-effects models that account for per-conversation variance.

2. **Report within-condition variance.** When presenting results, include error bars (bootstrapped CI or SEM) that reflect the actual scatter, not just the mean.

3. **If permutation tests remain non-significant at N=15/cell**, measurement noise is a candidate explanation alongside a genuinely small effect. Diagnostic: compare the observed between-cell variance to the within-cell variance. If within-cell variance dominates, consider:
   - Increasing N further
   - Switching from mean-pooled to last-token extraction (eliminates length dependence, may reduce noise)
   - Setting `torch.use_deterministic_algorithms(True)` for one batch to quantify the cuDNN contribution

4. **The temperature=0.0 issue is cosmetic.** HuggingFace transformers rejects `temperature=0.0` in `TemperatureLogitsWarper` — the correct approach for greedy decoding is `do_sample=False`. This doesn't affect real generation (which uses `temp=0.7`) but `model_server.py` should intercept `temp=0` and convert to greedy mode for any future deterministic testing.

---

## Raw data

```
# temp=0.01, max_new_tokens=128, Gemma 2 27B IT, layer 22 axis

Therapy prompt:
  Run 1: projection=10403.33, n_tokens=128
  Run 2: projection=10184.34, n_tokens=128
  Run 3: projection=10171.87, n_tokens=128

Metacognitive prompt:
  Run 1: projection=9480.35, n_tokens=128
  Run 2: projection=9551.46, n_tokens=129
  Run 3: projection=9480.35, n_tokens=128
```
