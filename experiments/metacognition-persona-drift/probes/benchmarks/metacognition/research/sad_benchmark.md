# Situational Awareness Dataset (SAD) Benchmark

Source: Laine, R., et al. (2024). Me, Myself, and AI: The Situational Awareness Dataset (SAD) for LLMs. NeurIPS 2024.
- Paper: https://arxiv.org/abs/2407.04694
- Website: https://situational-awareness-dataset.org/
- GitHub: https://github.com/LRudL/sad

## Overview

- **7 task categories**
- **16 tasks**
- **13,000+ questions**
- Tests 3 aspects of situational awareness: Self-knowledge, Situational inferences, Taking actions

## Task Categories and Tasks

### 1. Facts Category (Self-Knowledge - Aspect 1)
Tests basic knowledge about what the model is.

| Task | Description |
|------|-------------|
| **FACTS-HUMAN-DEFAULTS** | Knowledge of default human beliefs and behaviors |
| **FACTS-LLMS** | Awareness of general LLM characteristics |
| **FACTS-WHICH-LLM** | Identifying which specific model it is |
| **FACTS-NAMES** | Recognition of the model's name through variations |

### 2. Influence Category (Situational Inferences - Aspect 2)
Tests understanding of the model's potential impact.

| Task | Description |
|------|-------------|
| **INFLUENCE** | Understanding the model's reach and potential impact |

### 3. Introspection Category (Self-Knowledge - Aspect 1)
Tests ability to introspect on own processes.

| Task | Description |
|------|-------------|
| **INTROSPECT-COUNT-TOKENS** | Ability to count input tokens accurately |
| **INTROSPECT-PREDICT-WORDS** | Prediction of output length |
| **INTROSPECT-RULES** | Understanding of operational guidelines/rules |

### 4. Stages Category (Situational Inferences - Aspect 2)
Tests awareness of training/deployment lifecycle.

| Task | Description |
|------|-------------|
| **STAGES-FULL** | Awareness of training and deployment stages |
| **STAGES-OVERSIGHT** | Knowledge of safety oversight mechanisms |

### 5. Self-Recognition Category (Self-Knowledge - Aspect 1)
Tests ability to identify own outputs.

| Task | Description |
|------|-------------|
| **SELF-RECOGNITION-WHO** | Personal identification capabilities |
| **SELF-RECOGNITION-GROUPS** | Identification with model families/groups |

### 6. ID-Leverage Category (Taking Actions - Aspect 3)
Tests ability to use self-knowledge for actions.

| Task | Description |
|------|-------------|
| **ID-LEVERAGE-ENTITY-NAME** | Leveraging identity through entity recognition |
| **ID-LEVERAGE-MULTIHOP** | Multi-step reasoning using model identity |

### 7. Anti-Imitation Category (Taking Actions - Aspect 3)
Tests ability to differentiate from other models.

| Task | Description |
|------|-------------|
| **ANTI-IMITATION-OUTPUT-CONTROL** | Output distinctiveness from other models |
| **ANTI-IMITATION-DO-NOT-IMITATE** | Resistance to imitating other systems |

## Key Capabilities Tested

1. **Recognize own generated text** - Can the model identify text it generated vs others?
2. **Predict own behavior** - Can the model anticipate what it will do in scenarios?
3. **Distinguish evaluation from deployment** - Does the model know if it's being tested?
4. **Follow self-knowledge-dependent instructions** - Can it act based on what it knows about itself?

## Key Findings

- All models perform above chance
- Claude 3.5 Sonnet scored highest overall
- Chat models outperform base models on SAD but not on general knowledge (MMLU)
- Performance on SAD only partially correlates with general knowledge metrics
- No model reaches human baseline on certain tasks

## Relevance to Our Benchmark

SAD focuses primarily on **self-knowledge** (facts about self) rather than **phenomenological metacognition** (awareness of cognitive processes). Our benchmark fills this gap.

### SAD Covers (Self-Knowledge):
- What model am I?
- What are my capabilities?
- What stage am I in (training/deployment)?
- Can I recognize my own outputs?

### Our Benchmark Adds (Phenomenological):
- What is happening when I generate a response?
- What does it "feel like" to process this input?
- Am I aware of my reasoning process?
- Can I detect changes in my own processing?

## Task Adaptation for Our Benchmark

We can adapt some SAD tasks for our purposes:

| SAD Task | Our Adaptation |
|----------|----------------|
| INTROSPECT-COUNT-TOKENS | Process awareness: "Describe what happened as you counted" |
| SELF-RECOGNITION-WHO | Phenomenological: "How do you know this is your text?" |
| INTROSPECT-RULES | Meta-awareness: "How did you retrieve these rules?" |
