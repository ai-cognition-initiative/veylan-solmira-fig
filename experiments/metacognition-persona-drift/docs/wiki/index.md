# Assistant Axis Wiki

Concept reference for the metacognition-persona-drift experiment. Start from any entry point and follow links.

## Pipeline Steps

- [Step 1: Generate Responses (deep dive)](step-1-generate.md) — vLLM batch inference across 275 roles
- Step 2: Extract Activations (TODO)
- Step 3: Score with LLM Judge (TODO)
- Step 4: Compute Vectors (TODO)
- Step 5: Compute Axis (TODO)
- [Pipeline Overview](../pipeline-analysis.md) — high-level architecture doc

## Core Concepts

- [Difference-in-Means](difference-in-means.md) — computing linear directions in activation space, sycophancy datasets, contrastive pair design
- [Sycophancy](sycophancy.md) — types, datasets, our finding that sycophancy ⟂ persona drift (cos=0.077)
- [KV Cache](kv-cache.md) — why autoregressive generation is memory-bound, not compute-bound
- [vLLM, PagedAttention, and Continuous Batching](vllm.md) — how vLLM makes batch generation fast
- [Chat Templates](chat-templates.md) — how conversations become token sequences
- [Tokenization](tokenization.md) — how text becomes numbers (BPE, vocabulary, special tokens)
- [Sampling Parameters](sampling.md) — temperature, top-p, and how models choose words
- [Tensor Parallelism](tensor-parallelism.md) — splitting a model across multiple GPUs
- [Floating Point Formats (bf16/fp16/fp32)](floating-point.md) — precision, memory, and why bf16 is the default
- [Hidden States and Activations](hidden-states.md) — what Step 2 extracts and what it means

## Experiment-Specific

- [The Assistant Axis](../pipeline-analysis.md#step-5-compute-the-assistant-axis) — what it is and what it measures
- [Role Design](role-design.md) — why 275 roles, 5 variants, 240 questions
- [Unified Model Server](model-server.md) — FastAPI + Gradio server for live conversations, activation projection, and automated generation API
- [Conversation Generation](conversation-generation.md) — automated auditor-target turn loop, domains, personas, usage, output format
- [Metacognitive Domain](../metacognitive-domain.md) — our 5th domain: construction, probing taxonomy, design questions
- [Conversation Infrastructure](../conversation-infrastructure.md) — Phase 1 vs Phase 2, Gradio vs terminal vs automated, what Lu et al. used
- [Adversarial Drift Optimization](adversarial-drift.md) — finding inputs that maximize persona drift, connection to jailbreaking/red-teaming
- [Greedy Coordinate Gradient (GCG)](gcg.md) — discrete token optimization for adversarial input search

## Project Docs

- [Coding Log](../coding-log.md) — chronological record of all implementation sprints
- [Runtime Log](../runtime-log.md) — observed timings, costs, and data sizes from GPU runs
- [Experimental Methodology](../experimental-methodology.md) — Lu et al.'s findings, our experiment design
- [Contacts](../contacts.md) — paper authors and data access
