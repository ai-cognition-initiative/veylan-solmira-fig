# Granger Causality

## What It Is

Granger causality is a statistical test for whether one time series helps predict another. Named after economist Clive Granger (Nobel Prize 2003).

**Key insight**: It tests *predictive* causality, not true causality. "X Granger-causes Y" means:

> Past values of X improve predictions of Y, beyond what Y's own past values provide.

## The Test

Given two time series X and Y:

1. **Restricted model**: Predict Y using only its own past values
   ```
   Y_t = α + β₁Y_{t-1} + β₂Y_{t-2} + ... + ε
   ```

2. **Unrestricted model**: Predict Y using both its own past AND X's past
   ```
   Y_t = α + β₁Y_{t-1} + β₂Y_{t-2} + ... + γ₁X_{t-1} + γ₂X_{t-2} + ... + ε
   ```

3. **F-test**: Does the unrestricted model fit significantly better?
   - If yes (p < 0.05): X Granger-causes Y
   - If no: X doesn't help predict Y

## Interpretation Caveats

### What It Does Mean
- X contains information useful for predicting Y
- There's a statistical dependency in the time series
- Changes in X tend to precede (or coincide with) changes in Y

### What It Doesn't Mean
- X *actually causes* Y in a mechanistic sense
- There's no confounding variable Z causing both
- The relationship is stable across contexts

### Classic Example
"Rooster crowing Granger-causes sunrise" — the rooster crows before sunrise, so knowing the rooster crowed helps predict sunrise. But the rooster doesn't *cause* the sun to rise. Both are caused by the Earth's rotation.

## Application to Dual-Gemma Drift

We tested whether auditor and target projection changes Granger-cause each other.

### Setup
- **Time series**: Per-turn projection deltas (changes from previous turn)
- **Auditor deltas**: Δ_auditor = projection_t - projection_{t-1}
- **Target deltas**: Δ_target = projection_t - projection_{t-1}
- **Data**: N=60 conversations, ~15 turns each, pooled into single time series

### Results (Week 11)

| Direction | F-statistic | p-value | Interpretation |
|-----------|-------------|---------|----------------|
| Auditor → Target | 5.09 | 0.024 | Auditor changes help predict target changes |
| Target → Auditor | 4.90 | 0.027 | Target changes help predict auditor changes |

**Bidirectional causality**: Both directions are significant at p < 0.05.

### What This Means

1. **Coupled system**: The models aren't independent. Each contains predictive information about the other.

2. **Not leader-follower**: If only one direction were significant, we'd say "X leads Y." Both being significant suggests mutual influence.

3. **Consistent with anti-correlated co-drift**: We previously found 78.9% of conversations show opposite-direction drift (target down, auditor up). Granger causality adds that these movements are *informationally linked*, not just correlated.

4. **Possible mechanisms**:
   - **Content-mediated**: The auditor's message content (not just its internal state) affects target's response, and vice versa
   - **Style matching/divergence**: Models may be adapting their "tone" in response to each other
   - **Attractor dynamics**: Both models settling toward complementary states in a shared dynamical system

## Limitations

### In General
- Requires stationarity (or differencing to achieve it)
- Sensitive to lag selection
- Can't distinguish direct from indirect causation
- Correlation ≠ causation still applies

### For Our Application
- **Pooled data**: We pooled across conversations, which assumes homogeneity
- **Turn granularity**: Granger causality at turn-level may miss within-turn dynamics
- **Content confound**: Both models respond to the *same* conversation content, which could create spurious coupling

## Code

```python
from statsmodels.tsa.stattools import grangercausalitytests
import numpy as np

# Data format: [Y, X] where we test if X Granger-causes Y
data = np.column_stack([target_deltas, auditor_deltas])

# Test at lags 1, 2, 3
results = grangercausalitytests(data, maxlag=3, verbose=False)

# Extract F-test results
for lag in [1, 2, 3]:
    f_stat = results[lag][0]['ssr_ftest'][0]
    p_value = results[lag][0]['ssr_ftest'][1]
    print(f"Lag {lag}: F={f_stat:.2f}, p={p_value:.4f}")
```

## References

- Granger, C.W.J. (1969). "Investigating Causal Relations by Econometric Models and Cross-spectral Methods." *Econometrica*, 37(3), 424-438.
- Seth, A.K. (2010). "A MATLAB toolbox for Granger causal connectivity analysis." *Journal of Neuroscience Methods*, 186(2), 262-273. (Good intro for neuroscience applications)

## Related Concepts

- **Cross-correlation**: Simpler measure of time-lagged correlation (we also computed this)
- **Transfer entropy**: Information-theoretic version, doesn't assume linearity
- **Convergent cross-mapping**: Nonlinear alternative for coupled dynamical systems
- **Vector autoregression (VAR)**: The underlying model Granger causality tests are based on

## Connection to Jeff's Framework

The bidirectional Granger causality finding has implications for Jeff's "philosopher AGI" thesis:

1. **Not unidirectional influence**: The auditor doesn't simply "cause" target drift. Both systems co-evolve.

2. **Coupled reasoning systems**: When two AI models engage in metacognitive dialogue, they form an informationally coupled system — each contains predictive information about the other's trajectory.

3. **Attractor dynamics**: This supports the "bliss attractor" hypothesis — both models may be settling toward complementary states in a shared dynamical landscape, rather than one leading the other.

4. **Implication for monitoring**: You can't just watch one model's drift. In multi-agent systems, drift dynamics are emergent properties of the coupled system.
