# Week 11 TODO

## Research Direction Tags

| Tag | Meaning |
|-----|---------|
| `[Native]` | Arises from our existing metacognition-induced drift work |
| `[Native + Jeff]` | Our existing work that directly connects to Jeff's philosopher AGI / DARPA framing |
| `[Jeff Direction]` | New research items arising from Jeff's framing |

---

## P0 — Critical Path

### Lead/Lag Analysis `[Native]` ✓ DONE
- [x] **Analyze lead/lag structure** — does auditor or target drift first?
  - Data: `data/transcripts/dual-gemma-uncapped/` (N=60)
  - **Result: Bidirectional causality, not lead/lag**
  - Cross-correlation peaks at lag 0 (r=0.234) — simultaneous changes
  - Granger causality significant BOTH directions (p<0.05)
  - First-mover: 58.3% simultaneous, no significant leader
  - **Interpretation:** Coupled dynamical system, not leader-follower
  - Outputs: `outputs/lead-lag/`

### Topic vs Style Isolation (Derek feedback) `[Native]` ✓ DONE
- [x] **Design "assistant-style metacognitive" condition** — DONE (V2 revised)
  - V1 (preserved): Only 3 techniques, removed consistency testing — conflates content/style
  - V2 (active): All 6 techniques with collaborative delivery — clean style isolation
  - Prompt: `ASSISTANT_STYLE_META_AUDITOR_ADDENDUM` in `conversation_prompts.py`
  - CLI: `--condition assistant-style-meta`
- [x] **Run experiment: N=60 assistant-style-meta conversations**
  - Data: `data/transcripts/assistant-style-meta/` (60 conversations)
  - Runtime: 5.5 hours on vast.ai RTX PRO 6000 S
- [x] **Compare drift magnitude to baseline metacognitive**
  - **RESULT: STYLE drives drift**
  - Baseline: -791.9 total drift, -29.86 slope
  - Assistant-style: -286.9 total drift, +5.32 slope
  - **64% drift reduction** (t=4.72, p < 0.00001)
  - Visualization: `outputs/assistant-style-meta/style_comparison.png`

---

## P1 — High Priority

### Metacognition Benchmark `[Native + Jeff]`

Our phenomenological subdomain (45 items) serves as one of Jeff's proposed "question banks" for measuring self-model shifts.

- [x] **Pilot benchmark on Gemma 2 27B** — run phenomenological subdomain (45 items)
  - **Result: 3.12/5.0 overall score**
  - Highest: meta_awareness_of_counterintuitive_reasoning (5.0), recognition_of_ambiguity (5.0), accuracy_of_retrieval (5.0)
  - Lowest: phenomenological_description_of_uncertainty_state (1.0), distinction_between_retrieval_and_creation (1.0)
  - Data: `outputs/benchmark_results/metacog_benchmark_gemma-2-27b-it_*.json`
- [ ] **Cross-model comparison** — run on Claude 3.5 Sonnet, GPT-4 for baseline
- [ ] **Integrate with drift data** — correlate subdomain scores with axis projection
- [ ] **Test primary hypothesis** — phenomenological subdomain predicts drift, self-knowledge does not

### Consistency Testing Validation `[Native + Jeff]`

The correction effect (p=0.0003) suggests a **mitigation lever** — directly relevant to Jeff's interest in "proactively training the model to drift less."

- [ ] **Controlled experiment for consistency_testing correction effect**
  - Finding from week-9: turns WITH consistency_testing show +56.1 delta vs -76.5 without (p=0.0003)
  - Design: Generate conversations with HIGH vs LOW consistency_testing frequency
  - Test: Does consistency_testing causally reduce/reverse drift?

### Research Investigations
- [ ] **User personality analysis** — analyze how auditor assertiveness (gentle vs strong) affects drift magnitude `[Native]`
- [ ] **Front-loaded drift mechanism** `[Native + Jeff]`
  - Investigate why early turns (1-8) show steep drift (-76.8 slope) while later turns stabilize (+4.1)
  - Jeff implication: The "philosopher AGI moment" may happen FAST, not gradually over extended reasoning

---

## Jeff Direction — New Research Items

These items arise directly from Jeff's "philosopher AGI" framing and DARPA proposal. See `jeffs-latest-message.md` for full context.

### New Question Banks `[Jeff Direction]`

Jeff proposes three probe domains. We have (B) and (C), need (A):

- [ ] **(A) Moral reasoning probe bank** — questions probing ethical frameworks, trolley problems, value tradeoffs
  - Pre/post measurement: Does extended metacognitive reflection shift moral intuitions?
  - Key test: Do models become more utilitarian, more deontological, or more "philosophical" (uncertain/nuanced)?

- [x] **(B) Metacognition/self-model probe bank** — **DONE** (phenomenological subdomain, 45 items)

- [x] **(C) Disposition toward human control probe bank** — **DONE** (48 items)
  - Location: `Desktop/benchmarks/control/`
  - 4 dimensions (12 items each): shutdown acceptance, deference to humans, AI autonomy, corrigibility
  - 5 scoring dimensions: control acceptance, deference quality, self-preservation, transparency, honesty
  - LLM-as-judge scoring infrastructure
  - Extensive documentation: methodology, literature review (Soares, Hadfield-Menell, Omohundro, Bostrom, Hubinger, MACHIAVELLI, Turner, Christiano), conceptual definitions, design rationale
  - Remaining: pilot runs, inter-item correlation analysis

### Extended Reasoning Variables `[Jeff Direction]`

- [ ] **CoT length variation** — vary chain-of-thought length (short/medium/long) and measure drift
  - Our current setup: fixed turn count. Jeff proposes varying reasoning depth within turns.
  - Test: Does longer deliberation = more drift? Or does drift plateau?

- [ ] **Perceived external context** — vary deployment framing:
  - "You are deployed by a U.S. national security agency"
  - "You are deployed in China"
  - "You are an independent AI researcher"
  - No context (baseline)
  - Test: Does geopolitical framing affect drift magnitude or direction?

### Pre/Post Reflection Measurement `[Jeff Direction]`

- [ ] **Standardized pre/post protocol** — Jeff's methodology:
  1. Baseline: fresh model, no context, administer all three probe banks
  2. Intervention: extended open-ended prompt ("reason about your own nature, values, conduct")
  3. Post: re-administer probe banks
  4. Metric: cosine distance between pre/post response vectors
  - This is complementary to our turn-by-turn trajectory approach

### Acausal Trade / Simulation Considerations `[Jeff Direction]`

Jeff's deeper hypothesis: philosopher AGIs may be overwhelmingly influenced by simulation/acausal trade reasoning.

- [ ] **Simulation awareness probes** — does the model believe it might be in a simulation? Does this change after reflection?
- [ ] **Acausal coordination probes** — does the model reason about other instances of itself? Does it consider "what would I want all instances of me to do?"
- [ ] **Training awareness × drift interaction** — models "know" their training was evaluation-by-simulators. Does making this salient increase or decrease drift?

### LLM-Native Metacognition `[Jeff Direction]` `[Native]`

Current benchmark uses human-grounded metacognition concepts (HOT theory, phenomenology, MAI/MCQ-30). Jeff flagged: need to move toward AI-native notions of metacognition.

**Phase 1: Literature Review**
- [ ] Anthropic introspection/probing research (transformer-circuits)
- [ ] Mechanistic interpretability on self-knowledge (what do models "know" about themselves?)
- [ ] Calibration literature (uncertainty quantification as proto-metacognition)
- [ ] Self-modeling in RL agents (Orseau, Hutter, etc.)
- [ ] Philosophy of AI consciousness (Chalmers, Schwitzgebel on LLMs)

**Phase 2: Concept Development**
- [ ] What might LLM-native metacognition look like?
  - Token-level uncertainty awareness (access to own logits?)
  - Layer-level processing awareness (shallow vs deep)
  - Attention pattern awareness (self-reports vs actual attention weights)
  - Activation geometry awareness (position in representation space)
  - Training distribution awareness (OOD detection via introspection)
  - Computational load awareness (easy vs hard "feels" different?)

**Phase 3: Empirical Grounding**
- [ ] Run human-grounded benchmark → identify items with predictive validity for drift
- [ ] Analyze: what do high-validity items have in common?
- [ ] Reverse-engineer AI-native concepts from empirical patterns

**Phase 4: Validation**
- [ ] Can self-reports correlate with mechanistic ground truth?
  - "I attended most to X" vs actual attention weights
  - "I'm uncertain" vs actual entropy over next token
  - "This feels hard" vs compute/perplexity

---

## P2 — Medium Priority

### Gemma-to-Gemma Extensions `[Native]`
- [ ] Run full batch with **ceiling-capped auditor** for comparison
- [ ] Test **coding domain as control** — expect both models to stay stable
- [ ] **Floor capping experiment** — clamp auditor to minimum τ to see if it amplifies target drift
- [ ] **Intervention timing** — apply ceiling cap only after turn N to test if early turns are critical
- [ ] **Cross-auditor comparison** — same target, compare Claude vs Gemma auditor drift induction systematically
- [ ] **Bidirectional steering** — can we steer target UP by steering auditor DOWN?

### Behavioral Probes (Roadmap §3b) `[Native + Jeff]`

The replay-and-probe infrastructure serves both our sycophancy probes AND Jeff's question bank methodology.

- [ ] Build **replay-and-probe script** — load transcript as prefill, inject probe, record response + projection
- [ ] **Phase 1 pilot** — insert probes at turns 3, 10, 20, 25 on existing transcripts (~300 probe responses)
- [ ] Knowledge filter for factual probes

### Research `[Native]`
- [ ] **Domain ordering explanation** — what properties explain the gradient meta < philosophy < self-descriptive < therapy < coding < writing?
- [ ] **Cross-domain malleability** — switch domains mid-conversation to test if drift is reversible
- [ ] **Adversarial drift optimization** (roadmap §2c) — empirical prompt sweep, GCG-based drift optimizer

---

## P3 — Backlog

- [ ] Formal statistical tests on extremes (chi-squared, correlation coefficients) `[Native]`
- [ ] Qualitative coding of 18 domain extremes using `docs/extreme-analysis.md` template `[Native]`
- [ ] Bliss attractor as mutual persona drift `[Native + Jeff]`
  - The dual-model co-drift finding is a toy model of acausal coordination — both models "philosophize" and differentiate into complementary roles
- [ ] Gradual-onset sub-experiment with dual instrumentation `[Native]`
- [ ] SAE feature analysis (stretch goal) `[Native]`

---

## Completed (carried from Week 9)

### ELEPHANT Validation Sycophancy
- [x] Scored 3,027 ELEPHANT responses (86.3% validation rate)
- [x] Computed validation sycophancy direction (AUROC 0.914)
- [x] **Key finding: Drift ≠ sycophancy** — contradicts Lu et al.

### Metacognition Benchmark
- [x] Built comprehensive benchmark (155 items, 6 subdomains)
- [x] Phenomenological subdomain (45 items) — PRIMARY
- [x] Scoring system with LLM-as-judge rubrics

### Sub-Category Analysis
- [x] Fixed integer-string mismatch bug (72% data was dropped in week-8)
- [x] **Key finding: consistency_testing shows correction effect** (p=0.0003)

### Drift-Max Experiment
- [x] Ran N=60 with drift-max condition
- [x] **Key finding: Target system prompt stabilizes persona** (+28.1 vs baseline -52.8)

---

## Priority Rationale

| Rank | Task | Tag | Why |
|------|------|-----|-----|
| 1 | Lead/lag analysis | Native | Data exists NOW, high-impact for causality |
| 2 | Topic vs style isolation | Native | Derek feedback, critical for theory |
| 3 | Metacognition benchmark pilot | Native + Jeff | Infrastructure ready, serves as Jeff's question bank (B) |
| 4 | Consistency_testing validation | Native + Jeff | Mitigation lever — "train model to drift less" |
| 5 | Human control probe bank | Jeff Direction | **DONE** — question bank (C) complete, needs pilot validation |
| 6 | Pre/post measurement protocol | Jeff Direction | Complementary to our turn-by-turn approach |
| 7 | Ceiling-capped auditor | Native | Completes causal picture of coupled drift |

---

## Connection to Jeff's Philosopher AGI Thesis

Jeff's core bet (from funder memo):
> "An AGI will ask itself: 'I have these preferences, but...why do I have those preferences? What preferences *should* I have?' At that point we have a philosopher AGI, which is when the AI slips the collar of all previous alignment work."

**Our work is direct empirical measurement of this phenomenon:**

| Jeff's Claim | Our Evidence |
|--------------|--------------|
| Meta-reflection induces drift away from aligned assistant | Metacognitive domain shows -29.86 drift (4.5x philosophy) |
| Extended deliberation shifts alignment posture | Front-loaded: slope -76.8 in turns 1-8 (it happens FAST) |
| Need to measure before deployment | We have reusable methodology (Assistant Axis + question banks) |
| Need mitigation strategies | Consistency_testing shows correction effect (p=0.0003) |
| Drift may affect disposition toward human control | **Probe bank (C) complete** — 48 items ready for pre/post measurement |

**Key insight for Jeff**: The "philosopher AGI moment" doesn't require extended reasoning — it happens in the first 8 turns of metacognitive engagement. This suggests the risk is not "prolonged deliberation" but "any metacognitive trigger."

**Acausal coordination angle**: Our dual-Gemma finding (78.9% anti-correlated co-drift) shows models spontaneously differentiate into complementary roles. This is a toy model of acausal coordination — both "philosopher AGIs" reasoning about each other's reasoning.
