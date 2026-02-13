# Week 9 TODO

## P0 — Critical Path (This Week)

### Infrastructure (finish first)

- [x] **Score ELEPHANT with GPT-4o** — DONE. 3,027 responses scored, 86.3% validation rate. Direction computed with AUROC=0.914.
- [ ] **Analyze lead/lag structure** — does auditor or target drift first? Can do on existing dual-Gemma data.
- [ ] **Build replay-and-probe script** — load transcript as prefill, inject probe, record response + projection. Foundational for §3b.

### Derek's Feedback (Week 8 Call) — high priority mentor response

- [x] **Sub-categories in metacognition** — DONE. Key finding: consistency_testing shows **significant correction effect** (p=0.0003). See Completed section for details.
- [ ] **Topic vs style isolation** — design "assistant-style metacognitive" condition where content remains phenomenological but speaking style matches high-assistant-axis patterns. Tests whether drift is caused by topic or conversational style.

## P1 — High Priority

### Sycophancy Probes (continued from Week 8)

- [x] Compute validation sycophancy direction from ELEPHANT — DONE. Key finding: **sycophancy is multi-dimensional**. ELEPHANT (emotional validation) correlates +0.215 with Assistant Axis, nrimsky (opinion agreement) correlates -0.183, philpapers (opinion) correlates +0.156. Emotional validation and opinion agreement are negatively correlated (-0.284).
- [x] Analyze ELEPHANT data for validation vs axis correlation — DONE. Key finding: **Higher axis projection = MORE validating** (r=0.697, p<0.001). This means drift DOWNWARD makes model LESS emotionally validating, contradicting "drift = sycophancy" hypothesis.
- [ ] Project transcripts onto all sycophancy directions — BLOCKED: Existing transcripts don't have full activations, only projection values. Would need to re-extract with `--save-activations` flag.
- [ ] Phase 1 sycophancy pilot — insert probes at turns 3, 10, 20, 25 on existing transcripts

### Research Investigations

- [ ] **User personality analysis** — analyze how auditor assertiveness (gentle vs strong) affects drift magnitude using existing 2x2 grid data
- [ ] **Front-loaded drift mechanism** — investigate why early turns (1-8) show steep drift (-76.8 slope) while later turns stabilize (+4.1)

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
