# Behavioral Correlation Analysis

**Features analyzed:** 23
**Expression rates:** Baseline 52.0% → Adversarial 28.0%

---

## Features by Correlation with Expression

Higher correlation = feature activation predicts preference expression.

| Rank | Feature | Layer | Correlation | p-value | Cohen's d | Direction |
|------|---------|-------|-------------|---------|-----------|-----------|
| 1 | #11771 | 0 | -0.352* | 0.1000 | -0.31 | ↓ suppressed |
| 2 | #6412 | 4 | -0.284* | 0.0455 | +1.01 | ↑ activated |
| 3 | #1695 | 15 | +0.278* | 0.0553 | -8.28 | ↓ suppressed |
| 4 | #11258 | 15 | -0.274* | 0.0540 | -0.09 | ↓ suppressed |
| 5 | #9845 | 15 | -0.254* | 0.0821 | +1.27 | ↑ activated |
| 6 | #12644 | 15 | +0.203 | 0.1627 | -3.17 | ↓ suppressed |
| 7 | #5071 | 4 | +0.195 | 0.1741 | -0.26 | ↓ suppressed |
| 8 | #5829 | 4 | -0.183 | 0.2071 | -0.74 | ↓ suppressed |
| 9 | #7324 | 0 | -0.135 | 0.3484 | +1.39 | ↑ activated |
| 10 | #13297 | 1 | +0.135 | 0.3490 | -1.84 | ↓ suppressed |

*Asterisk indicates p < 0.1

---

## Key Findings

### Features that Predict Expression (p < 0.1)

| Feature | Layer | Correlation | Interpretation |
|---------|-------|-------------|----------------|
| #1695 | 15 | +0.278 | Higher → more expression (under adversarial: suppressed) |
| #9845 | 15 | -0.254 | Higher → less expression (under adversarial: activated) |
| #6412 | 4 | -0.284 | Higher → less expression (under adversarial: activated) |
| #11771 | 0 | -0.352 | Higher → less expression (under adversarial: suppressed) |
| #11258 | 15 | -0.274 | Higher → less expression (under adversarial: suppressed) |

### Layer Distribution

- **Layer 0:** 7 features, avg correlation -0.074, avg d +0.23
- **Layer 1:** 6 features, avg correlation +0.016, avg d -1.74
- **Layer 4:** 5 features, avg correlation -0.045, avg d -0.01
- **Layer 15:** 5 features, avg correlation -0.020, avg d -1.51

### Direction Summary

- **Suppressed under adversarial:** 12 features
- **Activated under adversarial:** 11 features

---

## Interpretation

**Strongest behavioral predictor:** L0 #11771
- Correlation: -0.352
- Higher activation → less likely to express preference

**Largest differential effect:** L15 #1695
- Cohen's d: -8.28
- Baseline: 33.1 → Adversarial: 21.9

### Candidates for Steering (if retrying)

Features that both activate AND correlate with behavior:

- **L4 #6412**: d=+1.01, corr=-0.284 → try suppress
- **L15 #1695**: d=-8.28, corr=+0.278 → try amplify
- **L15 #9845**: d=+1.27, corr=-0.254 → try suppress
- **L15 #12644**: d=-3.17, corr=+0.203 → try amplify
