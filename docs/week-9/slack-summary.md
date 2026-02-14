# Week 9 Slack Summary

## 1. Sub-Category Analysis (Derek's Feedback)
Found that **consistency_testing** has a significant **correction effect** (p=0.0003) — turns with consistency testing drift UP (+56) vs DOWN (-77) without. Not all metacognitive probing causes drift equally; pointing out contradictions may trigger self-correction.

## 2. ELEPHANT Sycophancy Analysis
Key finding: **Drift ≠ sycophancy**. Higher axis projection = MORE emotionally validating (r=0.70, p<0.001). Drift DOWNWARD makes model LESS supportive, not more sycophantic. Contradicts Lu et al.'s "sycophantic reinforcement" hypothesis.

Also discovered sycophancy is multi-dimensional:
- ELEPHANT (emotional validation) vs nrimsky (opinion agreement): **negatively correlated** (-0.28)
- Being empathetic ≠ agreeing with opinions

## 3. Drift-Max Bug Discovery
The `--condition drift-max` flag was silently ignored in batch mode. Accidental finding: two baseline runs 8 days apart showed significantly different drift (p=0.007). Bug fixed.

## 4. Metacognition Benchmark (NEW)
Built comprehensive benchmark (155 items, 6 subdomains) operationalizing Derek's metacognition vs self-knowledge distinction:

| Subdomain | Items | Focus |
|-----------|-------|-------|
| **Phenomenological** | 45 | "What is it like" — PRIMARY |
| Self-Knowledge | 35 | SAD + MCQ-30 adapted |
| Strategy Monitoring | 28 | MAI adapted |
| Confidence Calibration | 20 | Uncertainty quantification |
| Error Awareness | 15 | Mistake detection |
| Temporal Self-Reference | 12 | Within-conversation awareness |

**Primary hypothesis:** Phenomenological subdomain predicts drift; self-knowledge does not.

Location: `experiments/metacognition-persona-drift/benchmarks/metacognition/`
