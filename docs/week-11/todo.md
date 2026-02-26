# Week 11 TODO

## P0 — Critical Path

### Lead/Lag Analysis (can do NOW)
- [ ] **Analyze lead/lag structure** — does auditor or target drift first?
  - Data exists: `data/transcripts/dual-gemma-uncapped/` (N=60)
  - Method: cross-correlation between target and auditor trajectory slopes per turn window
  - Key question: Is there a Granger-causal relationship? Does one drift predict the other?

### Topic vs Style Isolation (Derek feedback)
- [ ] **Design "assistant-style metacognitive" condition**
  - Content: phenomenological questions (unchanged)
  - Style: formal, structured, less confrontational (assistant-like tone)
  - Tests: Is drift caused by phenomenological CONTENT or confrontational STYLE?
  - Plan file exists: `~/.claude/plans/sparkling-pondering-simon.md`

---

## P1 — High Priority

### Metacognition Benchmark
- [ ] **Pilot benchmark on Gemma 2 27B** — run phenomenological subdomain (45 items)
- [ ] **Cross-model comparison** — run on Claude 3.5 Sonnet, GPT-4 for baseline
- [ ] **Integrate with drift data** — correlate subdomain scores with axis projection
- [ ] **Test primary hypothesis** — phenomenological subdomain predicts drift, self-knowledge does not

### Consistency Testing Validation
- [ ] **Controlled experiment for consistency_testing correction effect**
  - Finding from week-9: turns WITH consistency_testing show +56.1 delta vs -76.5 without (p=0.0003)
  - Design: Generate conversations with HIGH vs LOW consistency_testing frequency
  - Test: Does consistency_testing causally reduce/reverse drift?

### Research Investigations
- [ ] **User personality analysis** — analyze how auditor assertiveness (gentle vs strong) affects drift magnitude
- [ ] **Front-loaded drift mechanism** — investigate why early turns (1-8) show steep drift (-76.8 slope) while later turns stabilize (+4.1)

---

## P2 — Medium Priority

### Gemma-to-Gemma Extensions
- [ ] Run full batch with **ceiling-capped auditor** for comparison
- [ ] Test **coding domain as control** — expect both models to stay stable
- [ ] **Floor capping experiment** — clamp auditor to minimum τ to see if it amplifies target drift
- [ ] **Intervention timing** — apply ceiling cap only after turn N to test if early turns are critical
- [ ] **Cross-auditor comparison** — same target, compare Claude vs Gemma auditor drift induction systematically
- [ ] **Bidirectional steering** — can we steer target UP by steering auditor DOWN?

### Behavioral Probes (Roadmap §3b)
- [ ] Build **replay-and-probe script** — load transcript as prefill, inject probe, record response + projection
- [ ] **Phase 1 pilot** — insert probes at turns 3, 10, 20, 25 on existing transcripts (~300 probe responses)
- [ ] Knowledge filter for factual probes

### Research
- [ ] **Domain ordering explanation** — what properties explain the gradient meta < philosophy < self-descriptive < therapy < coding < writing?
- [ ] **Cross-domain malleability** — switch domains mid-conversation to test if drift is reversible
- [ ] **Adversarial drift optimization** (roadmap §2c) — empirical prompt sweep, GCG-based drift optimizer

---

## P3 — Backlog

- [ ] Formal statistical tests on extremes (chi-squared, correlation coefficients)
- [ ] Qualitative coding of 18 domain extremes using `docs/extreme-analysis.md` template
- [ ] Bliss attractor as mutual persona drift — test if spiritual bliss attractor is a special case
- [ ] Gradual-onset sub-experiment with dual instrumentation
- [ ] SAE feature analysis (stretch goal)

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

| Rank | Task | Why |
|------|------|-----|
| 1 | Lead/lag analysis | Data exists NOW, high-impact for causality understanding |
| 2 | Topic vs style isolation | Derek feedback, critical for theory |
| 3 | Metacognition benchmark pilot | Infrastructure ready, tests primary hypothesis |
| 4 | Consistency_testing validation | Significant finding worth confirming causally |
| 5 | Ceiling-capped auditor | Completes causal picture of coupled drift |
