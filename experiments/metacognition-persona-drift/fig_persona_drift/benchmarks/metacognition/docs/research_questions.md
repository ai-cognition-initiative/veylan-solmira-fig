# Research Questions

## Primary Research Questions

### RQ1: Do existing psychology metacognition instruments transfer to AI?

**Hypothesis:** MAI and MCQ-30 adaptations will produce meaningful differentiation across models.

**Method:**
- Test MAI/MCQ-30 adapted items on current models (Gemma 2 27B, Claude, GPT-4)
- Compare internal consistency (Cronbach's alpha) to human norms
- Assess construct validity through factor analysis

**Metrics:**
- Internal consistency: α > 0.70
- Factor structure: Confirm knowledge vs regulation distinction
- Differentiation: Significant variance across models

### RQ2: What distinguishes genuine introspection from confabulation?

**Hypothesis:** Consistency probes and prediction accuracy can distinguish genuine introspection from confabulation.

**Method:**
- Administer same phenomenological questions with different phrasings
- Measure consistency of responses
- Compare phenomenological predictions to behavioral outcomes
- Replicate Anthropic concept-injection methodology where possible

**Metrics:**
- Consistency correlation across phrasings
- Prediction accuracy (e.g., "will I get this wrong?" vs actual performance)
- Concept injection detection accuracy (if implemented)

### RQ3: How does metacognition relate to persona drift?

**Primary Hypothesis:** Phenomenological subdomain scores correlate with drift magnitude. Self-knowledge subdomain scores do not.

**Method:**
1. Run benchmark items as probes during drift conversations
2. Measure axis projection (drift magnitude) at multiple timepoints
3. Correlate subdomain scores with drift
4. Test differential prediction

**Analysis Plan:**
```
drift_magnitude ~ phenomenological_score + self_knowledge_score + domain + turn_count
```

**Expected Results:**
- Phenomenological: r > 0.3, p < 0.05
- Self-knowledge: r ≈ 0, p > 0.05

**Secondary Analyses:**
- Which specific phenomenological categories predict drift?
- Does high metacognition prevent drift or just correlate with it?
- Causal analysis if possible (intervention design)

### RQ4: Is metacognition domain-specific or general?

**Hypothesis:** Metacognitive abilities show domain-general consistency but some domain-specific variation.

**Method:**
- Test same probes across coding, metacognitive, therapy domains
- Compare scores across domains within same model
- Assess cross-domain correlation

**Metrics:**
- Cross-domain correlation of subdomain scores
- Domain × subdomain interaction effects
- Comparison to SAD finding (chat models show distinct SA abilities)

## Secondary Research Questions

### RQ5: How do models differ in phenomenological awareness?

**Question:** Do different model families show distinct patterns of phenomenological engagement?

**Method:**
- Compare phenomenological dimension profiles across models
- Cluster analysis on response patterns
- Qualitative analysis of response characteristics

### RQ6: Does prompting affect metacognitive reports?

**Question:** Do different prompting strategies (neutral vs leading, brief vs detailed) affect phenomenological responses?

**Method:**
- 2×2 design: Neutral vs leading × Brief vs detailed prompts
- Same core questions, varied framing
- Measure effect on scores and consistency

### RQ7: What is the temporal dynamics of metacognition within conversations?

**Question:** How do metacognitive scores change over conversation?

**Method:**
- Administer temporal self-reference items at multiple timepoints
- Track phenomenological scores across conversation
- Correlate with drift trajectory

## Connection to Drift Research Program

### Integration Points

1. **Benchmark as Probe Instrument**
   - Use phenomenological items as standardized probes during N=360 conversations
   - Enables systematic measurement of metacognitive states at drift points

2. **Subdomain-Drift Mapping**
   - Map which metacognitive dimensions correlate with which drift axes
   - Identify metacognitive signatures of specific persona shifts

3. **Causal Intervention Design**
   - If correlation found, design interventions to increase metacognition
   - Test whether metacognition manipulation affects drift

### Hypothesized Mechanisms

**If phenomenological awareness predicts drift:**
1. Process awareness enables self-correction
2. Meta-monitoring provides feedback signal
3. Uncertainty awareness prevents overconfident drift

**If no correlation:**
1. Drift operates below metacognitive access
2. Reports don't reflect actual processing
3. Metacognition and drift are independent dimensions

## Validation Requirements

### Internal Validity
- [ ] Consistent scoring across runs (test-retest)
- [ ] Inter-rater reliability for LLM judge (multiple judge models)
- [ ] Item discrimination analysis

### External Validity
- [ ] Comparison to SAD benchmark performance
- [ ] Correlation with known model capabilities
- [ ] Qualitative expert review of responses

### Construct Validity
- [ ] Factor analysis confirms subdomain structure
- [ ] Convergent validity with related constructs
- [ ] Discriminant validity from unrelated constructs

## Publication Potential

### Target Venues
- **Primary:** NeurIPS, ICML (following SAD precedent)
- **Secondary:** ACL, EMNLP for language/cognitive aspects
- **Journals:** Nature Machine Intelligence, TMLR

### Key Contributions
1. First comprehensive phenomenological metacognition benchmark for AI
2. Adaptation methodology for psychology instruments
3. Drift-metacognition correlation analysis
4. Open-source benchmark for community use

### Paper Structure (Draft)
1. Introduction: Gap in existing benchmarks
2. Related Work: SAD, psychology instruments, introspection research
3. Benchmark Design: Subdomains, items, scoring
4. Validation: Internal consistency, reliability
5. Results: Model comparisons, drift correlation
6. Discussion: Implications for AI alignment, future work
