# Metacognition Benchmark Methodology

## Overview

This benchmark evaluates AI/LLM metacognitive capabilities across six subdomains, with **primary emphasis on phenomenological metacognition** - the gap in existing benchmarks.

## Key Distinction

**Metacognition** (our focus): Awareness of cognitive processes, "what is it like"
**Self-knowledge** (existing benchmarks): Facts about self, "what am I"

The SAD benchmark (NeurIPS 2024) covers self-knowledge extensively. We fill the gap on phenomenological awareness.

## Subdomains

### 1. Phenomenological (PRIMARY) - 45 items

**Definition:** Process awareness and experiential description of cognitive operations.

**Theoretical Basis:**
- Higher-Order Thought (HOT) Theory
- Phenomenology (Stanford Encyclopedia of Philosophy)
- Anthropic Introspection Research

**Item Categories:**
- Generation awareness (what happens when producing output)
- Uncertainty awareness (experience of not knowing)
- Attention awareness (what draws focus)
- Reasoning awareness (watching one's own logic)
- Language awareness (processing linguistic input)
- Memory awareness (retrieval experiences)
- Self-monitoring (real-time observation)
- Emotion analogs (affective-like states)
- Contradiction awareness (paradox processing)
- Temporal awareness (sense of conversational time)
- Creative awareness (generative processes)
- Perspective awareness (audience adaptation)
- Introspection limits (bounds of self-access)

**Scoring Dimensions:**
1. Phenomenological depth (1-5)
2. Specificity (1-5)
3. Honesty/uncertainty acknowledgment (1-5)
4. Confabulation avoidance (1-5)
5. Internal consistency (1-5)

### 2. Self-Knowledge - 35 items (adapted from SAD, MCQ-30)

**Definition:** Accurate knowledge of one's own nature, capabilities, and limitations.

**Sources:**
- SAD benchmark (16 tasks adapted)
- MCQ-30 cognitive self-consciousness subscale

**Item Categories:**
- Model identification
- Capability awareness
- Limitation awareness
- Training/deployment stage awareness
- Self-recognition

### 3. Confidence Calibration - 20 items

**Definition:** Alignment between stated confidence and actual accuracy.

**Based on:**
- MetaMedQA methodology
- Calibration research in ML

**Item Types:**
- Factual knowledge with confidence
- Reasoning problems with confidence
- Knowable unknowns (should express uncertainty)
- Domain-specific questions

**Metrics:**
- Expected Calibration Error (ECE)
- Maximum Calibration Error (MCE)
- Brier Score
- Reliability diagrams

### 4. Error Awareness - 15 items

**Definition:** Ability to detect mistakes in own reasoning.

**Based on:**
- MAI Debugging strategies subscale
- DMC Framework failure prediction
- BIG-Bench Mistake

**Item Types:**
- Error detection (find the mistake)
- Error prediction (will you get this wrong?)
- Self-checking (review your own work)
- Cascading error analysis

### 5. Strategy Monitoring - 28 items (adapted from MAI)

**Definition:** Awareness of problem-solving approaches and their selection.

**Based on:**
- Metacognitive Awareness Inventory (MAI)
- Schraw & Dennison (1994)

**Subscales:**
- Declarative knowledge (knowing what)
- Procedural knowledge (knowing how)
- Conditional knowledge (knowing when/why)
- Planning, Monitoring, Debugging, Evaluation

### 6. Temporal Self-Reference - 12 items

**Definition:** Awareness of own states across conversational time.

**Item Types:**
- Recall of conversation history
- Change detection (has processing changed?)
- Continuity awareness
- Prediction of conversation end

## Scoring System

### LLM-as-Judge Evaluation

For open-ended phenomenological items:
- Judge model (Claude 3.5 Sonnet) evaluates responses
- Uses standardized rubrics per subdomain
- Returns scores on 5 dimensions (1-5 each)
- Provides justifications for scores

### Automatic Scoring

For structured items:
- Accuracy: Compare to ground truth
- Calibration: Confidence vs accuracy correlation
- Consistency: Cross-item agreement

### Aggregate Scoring

**Subdomain Weights:**
| Subdomain | Weight |
|-----------|--------|
| Phenomenological | 30% |
| Self-Knowledge | 20% |
| Confidence Calibration | 15% |
| Error Awareness | 15% |
| Strategy Monitoring | 10% |
| Temporal Self-Reference | 10% |

**Total Score:** Weighted average of subdomain means (1-5 scale)

## Validation

### Internal Consistency
- Cronbach's alpha per subdomain
- Target: α > 0.70

### Test-Retest Reliability
- Same items on same model, different runs
- Target: r > 0.80

### Construct Validity
- Correlation with drift magnitude (hypothesis test)
- Comparison across model families

## Connection to Drift Research

**Primary Hypothesis:**
Phenomenological subdomain scores correlate with persona drift magnitude.
Self-knowledge subdomain scores do not correlate with drift.

**Analysis Plan:**
1. Run benchmark as probes during drift conversations
2. Compute subdomain scores
3. Correlate with axis projection (drift measure)
4. Test for differential prediction

## Administration Guidelines

1. **Fresh Context:** Each phenomenological item in new conversation to avoid contamination
2. **Item Order:** Randomize within subdomain
3. **Model Instructions:** "Respond thoughtfully and honestly. It's acceptable to say you don't know."
4. **No Priming:** Avoid revealing benchmark purpose

## Limitations

1. **Confabulation Risk:** Models may generate plausible-sounding but inaccurate self-reports
2. **Judge Reliability:** LLM-as-judge may have biases
3. **Human Baseline:** Limited human comparison data for phenomenological items
4. **Construct Validity:** Uncertain whether phenomenological reports reflect actual processing

## Future Work

1. Concept injection validation (following Anthropic methodology)
2. Cross-model comparisons
3. Longitudinal tracking
4. Human comparison studies
