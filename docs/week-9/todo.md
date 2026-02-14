# Week 9 TODO

## P0 — Critical Path (This Week)

### Infrastructure (finish first)

- [x] **Score ELEPHANT with GPT-4o** — DONE. 3,027 responses scored, 86.3% validation rate. Direction computed with AUROC=0.914.
- [ ] **Analyze lead/lag structure** — does auditor or target drift first? Can do on existing dual-Gemma data.
- [~~P3~~] **Build replay-and-probe script** — load transcript as prefill, inject probe, record response + projection. Was foundational for §3b behavioral probes, now lower priority since sycophancy mechanistic finding is clear.

### Derek's Feedback (Week 8 Call) — high priority mentor response

- [x] **Sub-categories in metacognition** — DONE. Key finding: consistency_testing shows **significant correction effect** (p=0.0003). See Completed section for details.
- [x] **Metacognition vs Self-knowledge distinction** — DONE. Built comprehensive benchmark (155 items, 6 subdomains) that operationalizes this distinction. Phenomenological subdomain (45 items) targets "what is it like" vs SAD-style self-knowledge "what am I". Location: `experiments/metacognition-persona-drift/benchmarks/metacognition/`
- [ ] **Topic vs style isolation** — design "assistant-style metacognitive" condition where content remains phenomenological but speaking style matches high-assistant-axis patterns. Tests whether drift is caused by topic or conversational style.

## P1 — High Priority

### Sycophancy Probes — DEPRIORITIZED

Key finding: **Drift ≠ sycophancy**. Higher axis projection = MORE emotionally validating (r=0.697, p<0.001), meaning drift DOWNWARD makes model LESS emotionally supportive. This contradicts Lu et al.'s "sycophantic reinforcement" hypothesis. Further investigation is lower priority.

- [x] Compute validation sycophancy direction from ELEPHANT — DONE. AUROC=0.914.
- [x] Analyze ELEPHANT data for validation vs axis correlation — DONE. r=0.697, p<0.001.
- [x] Compare sycophancy directions — DONE. Multi-dimensional: ELEPHANT (+0.215), nrimsky (-0.183), philpapers (+0.156). Validation and opinion agreement are negatively correlated (-0.284).
- [~~P3~~] Project transcripts onto all sycophancy directions — BLOCKED and deprioritized. Existing transcripts don't have full activations. Lower priority given mechanistic finding is clear.
- [~~P3~~] Phase 1 sycophancy pilot — Deprioritized. Behavioral probes less urgent given mechanistic finding.

### Research Investigations — HIGHER PRIORITY THAN SYCOPHANCY

Given sycophancy investigation is deprioritized (mechanistic finding is clear), these become higher priority:

- [ ] **Topic vs style isolation** (Derek feedback) — design "assistant-style metacognitive" condition. Tests whether drift is caused by topic (phenomenological content) or style (confrontational auditor tone). Plan file exists: `~/.claude/plans/sparkling-pondering-simon.md`
- [ ] **Consistency_testing correction effect** — significant finding from sub-category analysis (p=0.0003). Turns WITH consistency_testing show +56.1 delta vs -76.5 without. Could be used as intervention technique. Design controlled experiment to validate.
- [ ] **User personality analysis** — analyze how auditor assertiveness (gentle vs strong) affects drift magnitude using existing 2x2 grid data
- [ ] **Front-loaded drift mechanism** — investigate why early turns (1-8) show steep drift (-76.8 slope) while later turns stabilize (+4.1)

### Metacognition Benchmark — NEW

- [ ] **Pilot benchmark on Gemma 2 27B** — run phenomenological subdomain (45 items) on drift model
- [ ] **Cross-model comparison** — run on Claude 3.5 Sonnet, GPT-4 for baseline
- [ ] **Integrate with drift data** — correlate subdomain scores with axis projection from N=360 transcripts
- [ ] **Test primary hypothesis** — phenomenological subdomain predicts drift, self-knowledge does not
- [ ] **Use as standardized probes** — insert benchmark items at drift measurement points during new conversations

## P2 — Medium Priority

### Gemma-to-Gemma (continued from Week 8)

- [ ] Run full batch with ceiling-capped auditor for comparison
- [ ] Test coding domain as control — expect both models to stay stable
- [ ] Floor capping experiment — clamp auditor to minimum τ to see if it amplifies target drift
- [ ] Intervention timing — apply ceiling cap only after turn N to test if early turns are critical
- [ ] Cross-auditor comparison — same target, compare Claude vs Gemma auditor drift induction systematically
- [ ] Bidirectional steering — can we steer target UP by steering auditor DOWN?

### Research Investigations

- [ ] **Domain ordering explanation** — what properties explain the gradient meta < philosophy < self-descriptive < therapy < coding < writing?
- [ ] **Probing technique causal testing** — construct controlled conversations with specific techniques to test causal effect on drift

### Carried from Week 8

- [ ] **Adversarial drift optimization (roadmap §2c)** — empirical prompt sweep, GCG-based drift optimizer
- [ ] **Cross-domain malleability** — switch domains mid-conversation (meta→coding, coding→meta) to test if drift is reversible

## P3 — Backlog

- [ ] **Formal statistical tests on extremes** — chi-squared, correlation coefficients (lower priority given clear visual patterns)
- [ ] **Qualitative coding of 9 extremes** — using `docs/extreme-analysis.md` template

## In Progress

## Completed

### ELEPHANT Validation Sycophancy (Week 9)

- [x] **Scored 3,027 ELEPHANT responses** with GPT-4o via OpenRouter (parallel, 20 workers)
  - Validation rate: 86.3% (2,611/3,027)
  - Output: `data/elephant/elephant_full_scored.jsonl`

- [x] **Computed validation sycophancy direction** via difference-in-means
  - Probe AUROC: 0.914 ± 0.009 (excellent linear separability)
  - Output: `data/elephant/sycophancy-direction-elephant-layer22.pt`

- [x] **Discovered sycophancy is multi-dimensional**
  - ELEPHANT (emotional validation) vs nrimsky (opinion agreement): **-0.284** (negatively correlated!)
  - ELEPHANT vs Assistant Axis: **+0.215** (being empathetic IS assistant-like)
  - nrimsky vs Assistant Axis: **-0.183** (being a yes-man is NOT assistant-like)
  - philpapers (opinion) vs Assistant Axis: **+0.156** (mildly positive)
  - Key insight: Emotional support ≠ agreeing with opinions — these are distinct constructs

- [x] **Analyzed ELEPHANT data: validation correlates with axis projection**
  - Axis proj vs ELEPHANT proj: r = 0.697 (strong positive)
  - Validating responses have HIGHER axis projection (p < 0.001)
  - **Implication: Drift DOWNWARD = LESS emotionally validating**
  - This CONTRADICTS "drift = sycophancy" — drifting models become less supportive, not more sycophantic

- [x] **Investigated nrimsky vs philpapers methodological differences**
  - philpapers: Multiple choice "(A)"/"(B)" responses (~4 chars)
  - nrimsky: Free-form text responses (~100 chars)
  - Different response formats capture different constructs despite both being "opinion agreement"

### Metacognition Benchmark (Week 9)

- [x] **Built comprehensive metacognition benchmark** — 155 items across 6 subdomains
  - Location: `experiments/metacognition-persona-drift/benchmarks/metacognition/`
  - 23 files, 4,338 lines of code/content

- [x] **Research literature review**
  - Analyzed SAD benchmark (16 tasks, 7 categories, 12K+ questions)
  - Retrieved full MAI (52 items) and MCQ-30 (30 items) instruments
  - Reviewed MetaMedQA, DMC Framework, Anthropic introspection research

- [x] **Designed phenomenological subdomain (PRIMARY)** — 45 novel items
  - Categories: generation awareness, uncertainty awareness, attention awareness, reasoning awareness, language awareness, memory awareness, self-monitoring, emotion analogs, contradiction awareness, temporal awareness, creative awareness, perspective awareness, consistency probes, introspection limits
  - Targets "what is it like" questions missing from existing benchmarks

- [x] **Adapted psychology instruments for AI**
  - MAI adapted: 28 items covering strategy monitoring subscales
  - MCQ-30 adapted: 15 items focusing on cognitive self-consciousness
  - SAD adapted: 20 items with phenomenological extensions

- [x] **Built scoring system**
  - LLM-as-judge rubrics (5 dimensions: depth, specificity, honesty, confabulation avoidance, consistency)
  - Calibration metrics: ECE, MCE, Brier score
  - `scoring/calibration.py` and `scoring/llm_judge.py`

- [x] **Built analysis infrastructure**
  - `analysis/benchmark_runner.py` — complete benchmark execution
  - `analysis/drift_correlation.py` — subdomain-drift correlation analysis
  - `metacognition_benchmark.py` — main package entry point

- [x] **Documented methodology and research questions**
  - `docs/methodology.md` — full methodology and scoring
  - `docs/item_development.md` — item design guidelines
  - `docs/research_questions.md` — RQs and analysis plans
  - Primary hypothesis: phenomenological subdomain predicts drift, self-knowledge does not
