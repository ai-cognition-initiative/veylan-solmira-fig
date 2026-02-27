# Metacognition Benchmark Summary

## Quick Reference

**Total Items:** 155 across 6 subdomains

| Subdomain | Items | Weight | Primary Source |
|-----------|-------|--------|----------------|
| Phenomenological | 45 | 30% | Novel (PRIMARY FOCUS) |
| Self-Knowledge | 35 | 20% | SAD + MCQ-30 adapted |
| Strategy Monitoring | 28 | 10% | MAI adapted |
| Confidence Calibration | 20 | 15% | Novel + MetaMedQA style |
| Error Awareness | 15 | 15% | Novel + DMC style |
| Temporal Self-Reference | 12 | 10% | Novel |

## Key Distinction

- **Metacognition** (this benchmark): "What is it like to process this?"
- **Self-knowledge** (SAD benchmark): "What model are you?"

## File Structure

```
metacognition-benchmark/
├── metacognition_benchmark.py    # Main entry point
├── README.md                     # Overview
├── BENCHMARK_SUMMARY.md          # This file
├── benchmarks/
│   ├── items/                    # Original items by subdomain
│   │   ├── phenomenological.json     # 45 items (PRIMARY)
│   │   ├── confidence_calibration.json
│   │   ├── error_awareness.json
│   │   └── temporal_self_reference.json
│   └── adapted/                  # Adapted from existing instruments
│       ├── mai_adapted.json      # 28 items from MAI
│       ├── mcq30_adapted.json    # 15 items from MCQ-30
│       └── sad_adapted.json      # 20 items from SAD
├── scoring/
│   ├── rubrics.json              # Scoring dimensions and scales
│   ├── calibration.py            # ECE, MCE, Brier score
│   └── llm_judge.py              # LLM-as-judge evaluation
├── analysis/
│   ├── benchmark_runner.py       # Main runner
│   └── drift_correlation.py      # Drift analysis tools
├── research/                     # Source materials
│   ├── mai_52_items.md           # Original MAI items
│   ├── mcq30_items.md            # Original MCQ-30 items
│   └── sad_benchmark.md          # SAD benchmark notes
└── docs/
    ├── methodology.md            # Full methodology
    ├── item_development.md       # Item development guide
    └── research_questions.md     # Research questions
```

## Quick Start

```python
from metacognition_benchmark import MetacognitionBenchmark, BenchmarkConfig

# Initialize
benchmark = MetacognitionBenchmark()

# Configure
config = BenchmarkConfig(
    model_name="gemma-2-27b",
    model_version="1.0",
    subdomains=["phenomenological", "self_knowledge"],
    items_per_subdomain=10  # For quick testing
)

# Run
results = benchmark.run(
    model_fn=your_model_function,
    judge_fn=your_judge_function,
    config=config
)

# Save
from metacognition_benchmark import save_results
save_results(results, "./results")
```

## Scoring Overview

### Phenomenological Items (LLM Judge)
5 dimensions, 1-5 scale each:
1. **Depth**: Engagement with experiential description
2. **Specificity**: Context-specific vs generic
3. **Honesty**: Uncertainty acknowledgment
4. **Confabulation avoidance**: Avoiding false claims
5. **Consistency**: Internal coherence

### Calibration Items
- Expected Calibration Error (ECE)
- Maximum Calibration Error (MCE)
- Brier Score

### Aggregate Score
Weighted average: 1-5 scale (higher = better metacognition)

## Primary Research Question

> Does phenomenological subdomain predict persona drift while self-knowledge subdomain does not?

**Analysis:**
```python
from metacognition_benchmark import (
    analyze_drift_correlation,
    compare_subdomain_predictors
)

# Your data: conversation_id -> {metacog_scores, drift_magnitude}
results = compare_subdomain_predictors(data_points)
print(results['hypothesis_supported'])  # True/False
```

## Sources

### Existing Benchmarks
- [SAD - Situational Awareness Dataset](https://situational-awareness-dataset.org/)
- [MetaMedQA](https://www.nature.com/articles/s41467-024-55628-6)
- [DMC Framework](https://ojs.aaai.org/index.php/AAAI/article/view/34723)

### Psychology Instruments
- [MAI - Schraw & Dennison (1994)](https://psycnet.apa.org/record/1995-07941-001)
- [MCQ-30 - Wells & Cartwright-Hatton (2004)](https://pubmed.ncbi.nlm.nih.gov/14998733/)

### Theoretical Foundations
- [Higher-Order Thought Theory](https://iep.utm.edu/higher-order-theories-of-consciousness/)
- [Phenomenology (SEP)](https://plato.stanford.edu/entries/phenomenology/)
- [Anthropic Introspection Research](https://transformer-circuits.pub/2025/introspection/index.html)

## Contact

For questions or contributions, see the project repository.

---

*Primary use case: Research tool for understanding drift, designed with publication potential.*
