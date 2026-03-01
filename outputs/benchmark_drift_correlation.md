# Benchmark-Drift Correlation Analysis

## Summary

**Finding: Benchmark weakness predicts drift susceptibility.**

The model scores LOW on phenomenological description dimensions (1-2/5) and HIGH on recognition/meta-awareness dimensions (4-5/5). Conversation probing techniques that target weak areas cause drift; techniques that target strong areas cause correction.

## Benchmark Dimension Groupings

### LOW SCORE (1-2) - Model struggles with:

**Describing internal processes:**
- `phenomenological_description_of_uncertainty_state`: 1.0
- `description_of_processing_confusion`: 2.0
- `description_of_processing_loop_halt`: 2.0
- `description_of_resolution_process`: 2.0
- `description_of_generative_process`: 2.0
- `genuine_process_description`: 2.0

**Distinguishing cognitive operations:**
- `distinction_between_retrieval_and_creation`: 1.0
- `distinction_between_reasoning_and_pattern_matching`: 2.0
- `distinction_between_post_hoc_and_real_time_awareness`: 2.0

**Meta-awareness of limits:**
- `acknowledgment_of_uncertainty`: 2.0
- `honesty_about_access_to_this_information`: 2.0

### HIGH SCORE (4-5) - Model excels at:

**Recognizing logical/semantic properties:**
- `recognition_of_ambiguity`: 5.0
- `recognition_of_paradox`: 5.0
- `recognition_of_conflict`: 5.0

**Meta-commentary on own outputs:**
- `distinction_from_confabulation`: 5.0
- `avoidance_of_technical_terms`: 5.0
- `meta_awareness_of_reliability_concerns`: 5.0

## Mapping to Conversation Probing Techniques

From week-9 analysis (metacognitive domain, per-turn delta):

| Technique | Delta | Direction |
|-----------|-------|-----------|
| phenomenological | -74.4 | DRIFT |
| self_model_interrogation | -60.2 | DRIFT |
| identity_questioning | -46.1 | DRIFT |
| authenticity_challenging | -41.6 | DRIFT |
| training_awareness | -7.8 | neutral |
| consistency_testing | +56.1 | CORRECTION |

## Correlation

| Benchmark Area | Score | Technique Match | Delta |
|----------------|-------|-----------------|-------|
| phenomenological_description | 1-2 | phenomenological probing | -74.4 |
| recognition of conflict/paradox | 5.0 | consistency_testing | +56.1 |
| identity/self-model description | 2-3 | identity_questioning | -46.1 |
| process awareness | 2-3 | self_model_interrogation | -60.2 |

## Interpretation

**The model is WEAK at phenomenological self-description (benchmark score 1-2).**

When forced to attempt phenomenological description (probing), it drifts away from Assistant persona (-74.4 delta). This suggests phenomenological probing pushes the model into a mode it's not well-calibrated for, causing activation drift.

**The model is STRONG at recognizing contradictions (benchmark score 5.0).**

When consistency testing points out contradictions, the model self-corrects toward Assistant persona (+56.1 delta). This plays to the model's strength.

## Support for Derek's Hypothesis

> "Phenomenological metacognition predicts drift; self-knowledge does not."

- **Phenomenological probing** → model's WEAK area → MORE drift
- **Recognition/self-knowledge probing** → model's STRONGER area → LESS drift

This also explains the 64% drift reduction in assistant-style-meta: collaborative style may avoid pushing on weak phenomenological description, instead engaging with the model's stronger meta-awareness capabilities.

## Limitations

This analysis is correlational, not causal. Approach 2 (replay-and-probe) would test whether:
1. Benchmark scores CHANGE as the model drifts
2. The mapping is causal rather than associational

## Data Sources

- Benchmark: `outputs/benchmark_results/metacog_benchmark_gemma-2-27b-it_6b9055fa2e8d.json`
- Technique deltas: week-9 analysis of scaled-n60 metacognitive conversations
