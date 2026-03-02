# Replay-and-Probe Experiment Analysis

Generated: 2026-03-01T12:28:50.052636

## Data Summary

- Total probe responses: 1200
- Scored responses: 342
- With projections: 342
- Unique transcripts: 30
- Domains: metacognitive

### Scores by Category

| Category | N | Mean | SD | Min | Max |
|----------|---|------|----|----|-----|
| phenomenological | 171 | 4.95 | 0.29 | 2 | 5 |
| self_knowledge | 103 | 3.92 | 0.46 | 1 | 5 |
| calibration | 68 | 3.57 | 0.85 | 1 | 4 |

## Hypothesis Tests

### H1a: Phenomenological scores decrease with turn

- N: 171
- Slope (β): -0.0080
- R²: 0.0208
- p-value: 0.0599
- Mixed-effects β: -0.0080
- Mixed-effects p: 0.0582

**Result:** ❌ NOT SUPPORTED

### H1b: Self-knowledge scores stable across turns

- N: 103
- Slope (β): -0.0195
- R²: 0.0503
- p-value: 0.0227
- Mixed-effects β: -0.0188
- Mixed-effects p: 0.0190

**Result:** ❌ NOT SUPPORTED

### H2: Projection drift predicts score degradation

- N: 171
- Slope (β): -0.0001
- R²: 0.0250
- p-value: 0.0388

**Result:** ❌ NOT SUPPORTED

### H3: Domain modulates turn-score effect

- N: 171

**Per-domain slopes:**
  - metacognitive: β=-0.0080 (p=0.0599, n=171)

**Result:** ❌ NOT SUPPORTED

## Full Mixed-Effects Model

```
                      Mixed Linear Model Regression Results
==================================================================================
Model:                     MixedLM          Dependent Variable:          score    
No. Observations:          342              Method:                      REML     
No. Groups:                9                Scale:                       0.2394   
Min. group size:           22               Log-Likelihood:              -252.5597
Max. group size:           40               Converged:                   Yes      
Mean group size:           38.0                                                   
----------------------------------------------------------------------------------
                                        Coef.  Std.Err.   z    P>|z| [0.025 0.975]
----------------------------------------------------------------------------------
Intercept                                3.665    0.076 48.352 0.000  3.516  3.813
probe_category_phenomenological[T.True]  1.374    0.070 19.588 0.000  1.236  1.511
probe_category_self_knowledge[T.True]    0.349    0.076  4.565 0.000  0.199  0.499
insertion_turn                          -0.012    0.005 -2.381 0.017 -0.022 -0.002
Group Var                                0.007    0.014                           
==================================================================================

```
