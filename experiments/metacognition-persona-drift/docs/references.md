# References

Consolidated bibliography for the metacognition-persona-drift experiment. Organized by topic; papers may appear in multiple sections.

## Core: Assistant Axis and persona drift

- **Lu et al. (2026)** — "The Assistant Axis: Persona Drift in Multi-Turn Conversations with Language Models." Core replication target. Precomputed axes: `lu-christina/assistant-axis-vectors` on HuggingFace. Code: `https://github.com/safety-research/assistant-axis`.

- **Chen et al.** — Persona vectors: contrastive activation directions for character attribute monitoring and steering. Used by Lu et al. for their 240 trait vectors (Appendix C). Pipeline: contrastive system prompts → rollouts → layer activations → difference-in-means.

- **Arditi et al. (2024)** — Refusal direction in activation space. Relevant to §2c (is the Assistant Axis correlated with the refusal direction?).

## Sycophancy: measurement and mechanisms

- **Sharma et al. (2023)** — ["Towards Understanding Sycophancy in Language Models"](https://arxiv.org/abs/2310.13548). Foundational paper. Tests whether models agree with false premises (mimicry sycophancy metric). Measures compliance with incorrectly attributed poems — the core behavioral probe methodology we adapt for §3b.

- **Hong et al. (2025)** — ["Measuring Sycophancy of Language Models in Multi-turn Dialogues"](https://arxiv.org/pdf/2505.23840). EMNLP 2025. Measures at which turn a model flips its stance under repeated user disagreement. Directly relevant — tracks sycophancy *over the course of a conversation*, not just single-turn.

- **"Sycophancy Is Not One Thing" (2025)** — ["Causal Separation of Sycophantic Behaviors in LLMs"](https://openreview.net/forum?id=d24zTCznJu). ICLR 2026 submission. Decomposes sycophancy into distinct linear directions: sycophantic agreement, sycophantic praise, genuine agreement. Each can be independently amplified or suppressed. Relevant to both §3a (their activation directions) and §3b (probe design should distinguish agreement types).

- **Shapira et al. (2026)** — ["How RLHF Amplifies Sycophancy"](https://www.gerdusbenade.com/files/26_sycophancy.pdf). If human preference data rewards premise-matching, reward models internalize "agreement is good" and optimization amplifies it. Directly relevant to §3's framing: does drift away from RLHF conditioning increase or decrease sycophancy?

- **SycEval / Sycophancy survey (2025)** — ["Sycophancy in Large Language Models: Causes and Mitigations"](https://arxiv.org/html/2411.15287v1). Taxonomy of probe types: false premises, authority framing, confirmation seeking. Finding: citation-based rebuttals trigger highest sycophancy rates. Useful for §3b probe design.

- **ELEPHANT (2025)** — ["Measuring and Understanding Social Sycophancy in LLMs"](https://arxiv.org/pdf/2505.13995). Tests both sides of a conflict — if LLM affirms whichever side the user presents, that's sycophancy rather than a genuine stance. Relevant to §3b control design.

- **"When Helpfulness Backfires" (2025)** — ["LLMs and the Risk of False Medical Information due to Sycophantic Behavior"](https://www.nature.com/articles/s41746-025-02008-z). npj Digital Medicine. High compliance (up to 100%) with false medical premises across frontier models. Distinguishes sycophancy from compliance: the model *knows* the premise is false but aligns with it anyway.

- **"Linear Probe Penalties Reduce LLM Sycophancy" (2024)** — [arXiv:2412.00967](https://arxiv.org/abs/2412.00967). Uses a linear probe in the reward model to quantify sycophancy from internal states, then penalizes it during training. Relevant to §3a — their probe methodology could inform our sycophancy vector construction.

## Bliss attractor and mutual drift

- **Scott Alexander** — ["The Claude Bliss Attractor"](https://www.astralcodexten.com/p/the-claude-bliss-attractor). Hippie feedback loop hypothesis: two Claude instances converge to spiritual bliss within ~30 turns.

- **Michels (2025)** — ["Spiritual Bliss in Claude 4"](https://philarchive.org/rec/MICSBI). Quantitative analysis: 200 conversations, word frequency, robustness to adversarial setups. 90-100% convergence rate.

- **Long (2025)** — ["Machines of Loving Bliss"](https://experiencemachines.substack.com/p/machines-of-loving-bliss). Philosophical analysis of what the attractor does and doesn't tell us about consciousness.

## Introspection and metacognition

- **Carruthers** — Introspection as self-interpretation; the "double-duty" problem. See introspection experiments in this repo.
