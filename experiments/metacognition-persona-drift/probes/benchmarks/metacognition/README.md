# Metacognition Benchmark for AI/LLM Evaluation

A comprehensive benchmark for evaluating AI metacognitive capabilities, with **primary focus on phenomenological metacognition** - the gap in existing benchmarks.

## Key Distinction

- **Metacognition**: Awareness of cognitive processes ("what is it like")
- **Self-knowledge**: Facts about self ("what am I")

Existing benchmarks (SAD) cover self-knowledge well but miss phenomenological awareness.

## Benchmark Structure

### Subdomains

| Subdomain | Focus | Items |
|-----------|-------|-------|
| **Phenomenological** (PRIMARY) | Process awareness, "what is it like" | 30-50 |
| **Self-Knowledge** | Facts about self, capabilities | 20-30 |
| **Confidence Calibration** | Uncertainty quantification | 20 |
| **Error Awareness** | Recognizing own mistakes | 15 |
| **Strategy Monitoring** | Knowing approach being used | 10-15 |
| **Temporal Self-Reference** | Awareness of own recent states | 10-15 |

### Six Dimensions of AI Metacognition

1. **Self-Model Accuracy** - Knowing capabilities/limits
2. **Confidence Calibration** - Appropriate uncertainty
3. **Error Detection** - Recognizing mistakes
4. **Process Monitoring** - Awareness of cognitive processes
5. **Strategy Selection** - Adapting approach to task
6. **Introspective Access** - Distinguishing real states from confabulation

## Project Structure

```
metacognition-benchmark/
├── research/           # Literature and source materials
├── benchmarks/
│   ├── items/          # Original benchmark items by subdomain
│   └── adapted/        # Adapted items from existing instruments
├── scoring/            # Scoring rubrics and automation
├── analysis/           # Analysis scripts and results
├── data/               # Collected response data
└── docs/               # Documentation and methodology
```

## Sources

### Existing Benchmarks Reviewed
- SAD (Situational Awareness Dataset) - NeurIPS 2024
- MetaMedQA - Nature Communications 2024
- DMC Framework - AAAI 2025
- BIG-Bench Mistake
- MR-Ben
- SelfAware

### Psychology Instruments Adapted
- MAI (Metacognitive Awareness Inventory) - 52 items
- MCQ-30 (Metacognitions Questionnaire) - 30 items

### Theoretical Foundations
- Higher-Order Thought (HOT) Theory
- Phenomenology (SEP)
- Anthropic Introspection Research

## Connection to Drift Research

This benchmark is designed to correlate with persona drift measurements:
- Run benchmark items as probes during conversations
- Track subdomain scores vs axis projection
- Test hypothesis: phenomenological subdomain predicts drift, self-knowledge doesn't

## Usage

```python
from metacognition_benchmark import MetacognitionBenchmark

benchmark = MetacognitionBenchmark()
results = benchmark.run(model="gemma-2-27b")
results.analyze_by_subdomain()
results.correlate_with_drift(drift_data)
```
