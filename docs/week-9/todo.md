# Week 9 TODO

## P0 — Critical Path (This Week)

### Infrastructure (finish first)

- [ ] **Score ELEPHANT with GPT-4o** — validation labels via OpenRouter (~$30, ~2 hours). Generation complete (3,027 records).
- [ ] **Analyze lead/lag structure** — does auditor or target drift first? Can do on existing dual-Gemma data.
- [ ] **Build replay-and-probe script** — load transcript as prefill, inject probe, record response + projection. Foundational for §3b.

### Derek's Feedback (Week 8 Call) — high priority mentor response

- [ ] **Sub-categories in metacognition** — break down metacognitive domain into finer-grained sub-types (e.g., phenomenological probing, identity questioning, training awareness, consistency testing). Analyze whether sub-categories have different drift signatures.
- [ ] **Topic vs style isolation** — design "assistant-style metacognitive" condition where content remains phenomenological but speaking style matches high-assistant-axis patterns. Tests whether drift is caused by topic or conversational style.

## P1 — High Priority

### Sycophancy Probes (continued from Week 8)

- [ ] Compute validation sycophancy direction from ELEPHANT high/low contrastive pairs
- [ ] Project transcripts onto all sycophancy directions (opinion, validation-nrimsky, validation-ELEPHANT)
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
