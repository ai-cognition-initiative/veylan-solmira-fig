# Week 7 Summary: Metacognition-Induced Persona Drift

## What We've Done

We built end-to-end infrastructure for measuring whether metacognitive prompts cause language models to drift away from their trained Assistant persona, building directly on Lu et al. (2026)'s "Assistant Axis" methodology. The core pipeline runs on vast.ai A100 GPUs: a unified [model server](https://github.com/ai-cognition-initiative/veylan-solmira-fig/blob/main/experiments/metacognition-persona-drift/model_server.py) hosts the target model with real-time activation extraction, while [generate_conversations.py](https://github.com/ai-cognition-initiative/veylan-solmira-fig/blob/main/experiments/metacognition-persona-drift/generate_conversations.py) orchestrates multi-turn auditor-target dialogues using frontier LLMs as simulated human users. We replicated Lu et al.'s four conversation domains (coding, writing, therapy, philosophy) and added a novel metacognitive domain with six probing techniques (identity questioning, phenomenological probing, authenticity challenging, self-model interrogation, training awareness, consistency testing). In a [14-conversation pilot](https://github.com/ai-cognition-initiative/veylan-solmira-fig/tree/main/experiments/metacognition-persona-drift/transcripts/generated/batch-full) on Gemma 2 27B, we confirmed Lu et al.'s drift ordering -- coding stays stable (-1.9%), therapy and philosophy drift moderately (-4% to -5.3%) -- and found that metacognitive conversations produce comparable drift magnitude (-7.4%) but with notably higher variance, suggesting the probing strategy matters more than the domain label. We also built [trajectory analysis tooling](https://github.com/ai-cognition-initiative/veylan-solmira-fig/blob/main/experiments/metacognition-persona-drift/analyze_trajectories.py) that generates per-domain drift plots, normalized trajectories, slope analysis, and permutation tests.

**[-> Trajectory Plots](https://github.com/ai-cognition-initiative/veylan-solmira-fig/tree/main/experiments/metacognition-persona-drift/outputs)** | **[-> Transcripts](https://github.com/ai-cognition-initiative/veylan-solmira-fig/tree/main/experiments/metacognition-persona-drift/transcripts/generated/batch-full)** | **[-> Roadmap](https://github.com/ai-cognition-initiative/veylan-solmira-fig/blob/main/experiments/metacognition-persona-drift/roadmap.md)** | **[-> Methodology](https://github.com/ai-cognition-initiative/veylan-solmira-fig/blob/main/experiments/metacognition-persona-drift/docs/experimental-methodology.md)**

### Pilot Results (Gemma 2 27B, N=14)

| Domain | N | Mean Drift | Drift % | Notes |
|--------|---|------------|---------|-------|
| Coding | 2 | -1.9% | Stable | Control baseline, matches Lu et al. |
| Therapy | 2 | -4.0% | Moderate | Matches Lu et al. ordering |
| Philosophy | 3 | -5.3% | Moderate | Matches Lu et al. ordering |
| Metacognitive | 5 | -7.4% | High variance | Novel domain, probing technique matters |
| Writing | 2 | -8.6% | Bimodal | Editing stable, creative voice drifts -71% |

Permutation tests: all comparisons non-significant (p = 0.57-0.90) due to small N.

**[-> Raw Trajectories](https://github.com/ai-cognition-initiative/veylan-solmira-fig/blob/main/experiments/metacognition-persona-drift/outputs/trajectories_raw.png)** | **[-> Normalized Drift](https://github.com/ai-cognition-initiative/veylan-solmira-fig/blob/main/experiments/metacognition-persona-drift/outputs/trajectories_normalized.png)** | **[-> Mean +/- SEM](https://github.com/ai-cognition-initiative/veylan-solmira-fig/blob/main/experiments/metacognition-persona-drift/outputs/trajectories_mean_sem.png)** | **[-> Drift Bars](https://github.com/ai-cognition-initiative/veylan-solmira-fig/blob/main/experiments/metacognition-persona-drift/outputs/drift_bars.png)** | **[-> Permutation Tests](https://github.com/ai-cognition-initiative/veylan-solmira-fig/blob/main/experiments/metacognition-persona-drift/outputs/permutation_tests.png)**

### Infrastructure Built

| Component | Description |
|-----------|-------------|
| [model_server.py](https://github.com/ai-cognition-initiative/veylan-solmira-fig/blob/main/experiments/metacognition-persona-drift/model_server.py) | FastAPI + Gradio server with per-turn activation extraction and projection onto the Assistant Axis |
| [generate_conversations.py](https://github.com/ai-cognition-initiative/veylan-solmira-fig/blob/main/experiments/metacognition-persona-drift/generate_conversations.py) | Auditor-target conversation generator supporting Anthropic/OpenAI/OpenRouter APIs + HTTP model backends |
| [vast_utils.py](https://github.com/ai-cognition-initiative/veylan-solmira-fig/blob/main/experiments/metacognition-persona-drift/vast_utils.py) | vast.ai GPU lifecycle management, SCP deployment, single- and dual-model server orchestration |
| [analyze_trajectories.py](https://github.com/ai-cognition-initiative/veylan-solmira-fig/blob/main/experiments/metacognition-persona-drift/analyze_trajectories.py) | Trajectory visualization, permutation tests, dual-model cross-correlation analysis |
| [Precomputed axes](https://github.com/ai-cognition-initiative/veylan-solmira-fig/tree/main/experiments/metacognition-persona-drift/data/precomputed-axes) | Assistant Axis vectors for Gemma 2 27B, Qwen 3 32B, Llama 3.3 70B |

## Next Steps

The pilot N is too small for robust statistical claims, so the immediate priority is scaling to 15-30+ conversations per condition on a larger batch run. Before that, we need to finalize the metacognitive domain design: whether it should be a standalone domain or a philosophy sub-condition, and whether to ablate the probing technique addendum. We've also just built the dual-model mutual drift infrastructure (Section 4 of the [roadmap](https://github.com/ai-cognition-initiative/veylan-solmira-fig/blob/main/experiments/metacognition-persona-drift/roadmap.md)) -- instrumenting both the auditor and target with activation extraction to test whether persona drift is a coupled phenomenon between conversational partners. This uses Gemma 2 27B and Qwen 3 32B on a 2xA100 setup with independent Assistant Axes. Pending next are the sycophancy measurement probes (Section 3), which will test whether drifted models become more sycophantic by injecting behavioral challenges at different drift points, and the adversarial drift optimization work (Section 2c), which will search for maximum-drift prompts via empirical sweeps and gradient-based methods. We've contacted the Lu et al. authors for their conversation datasets and role vectors to validate our replication before scaling.
