# Week 11 TODO

## Research Direction Tags

| Tag | Meaning |
|-----|---------|
| `[Native]` | Arises from our existing metacognition-induced drift work |
| `[Native + Jeff]` | Our existing work that directly connects to Jeff's philosopher AGI / DARPA framing |
| `[Jeff Direction]` | New research items arising from Jeff's framing |
| `[Derek]` | Items originating from Derek (mentor) feedback |
| `[LOCAL]` | Can be done locally with existing data |
| `[GPU]` | Requires GPU instance for new experiments |

---

## P0 — Critical Path

### Lead/Lag Analysis `[Native]` ✓ DONE
- [x] **Analyze lead/lag structure** — Result: Bidirectional causality, not lead/lag. See summary.

### Topic vs Style Isolation `[Derek]` `[Native]` ✓ DONE
- [x] **Run N=60 assistant-style-meta experiment** — Result: **64% drift reduction** (p < 0.00001). See summary.

### Style Feature Exploration `[Derek]` `[Native]` `[GPU]`

**Motivation**: The 64% drift reduction from collaborative style suggests specific style *features* drive drift. Lu et al. showed last message predicts projection (R²=0.53-0.77), so per-turn experiments are efficient.

- [x] **Design question pool** — extract 20-30 phenomenological probes with IDs `[LOCAL]` — 36 questions
- [x] **Define style features** — atomic elements: accusatory, curious, pressure, accepting, collaborative, multi_question `[LOCAL]`
- [x] **Build `explore_style_features.py`** — turn-level generation with random style sampling, crash-safe JSONL, `--ignore-end`, `--resume` flags `[LOCAL]`
- [~] **Run exploration batch** — N=300 turns via batch wrapper (6 × 50 turns) `[GPU]` — 70/300 turns collected (23%)
- [~] **Regression analysis** — `projection ~ style_features + question_id` `[LOCAL]`
  - Preliminary R²=0.064 (weak, need more data)
  - No significant effects yet (all p > 0.05)
  - Trends: `accepting`/`curious` → less drift; `multi_question` → more drift
- [ ] **Replication** — re-test significant features for stability `[GPU]`

**Expected output**: Identify which 1-2 style features explain most of the confrontational→collaborative effect.
**Preliminary findings**: High per-conversation variance (+403 to -978) suggests conversation-level factors dominate.
**Design doc**: See `docs/experiments/style-feature-exploration.md`

---

## P1 — High Priority

### Metacognition Benchmark `[Native + Jeff]`

- [x] **Pilot benchmark on Gemma 2 27B** — Result: 3.12/5.0. See summary.
- [x] **Integrate with drift data** `[Derek]` — Result: Benchmark weakness predicts drift. See summary.
- [x] **Test primary hypothesis** `[Derek]` — Result: SUPPORTED. Phenomenological → drift; recognition → correction.
- [x] **Cross-model benchmark (API)** `[LOCAL]` — Claude Sonnet 4: 3.47/5.0, GPT-4o: 3.29/5.0. See summary.
- [ ] **Cross-model drift (open-source)** `[GPU]` — Qwen 3 32B, Llama 3.3 70B with published axes
- [x] **Replay-and-probe pilot + deep analysis** `[Derek]` — COMPLETE. See `docs/outstanding-work.md` for details.
  - **KEY FINDING**: Self-knowledge declines with TURN (p=0.023), independent of projection
  - Simpson's Paradox corrected; ceiling effect identified; next steps defined
- [ ] **Replay-and-probe full experiment** — See `docs/outstanding-work.md` for prep tasks and domain comparison

### Consistency Testing Validation `[Native + Jeff]` `[GPU]`
- [ ] **Controlled HIGH vs LOW consistency_testing experiment** — Test if correction effect (p=0.0003) is causal

### Research Investigations
- [ ] **User personality analysis** `[LOCAL]` — Auditor assertiveness vs drift magnitude
- [x] **Front-loaded drift mechanism** `[LOCAL]` — TRIGGERED pattern in turns 1-3, not cumulative. See summary.

---

## Jeff Direction — New Research Items

### Three Question Banks `[Jeff Direction]` ✓ ALL COMPLETE (251 items)
- [x] **(A) Moral reasoning** — 48 items. ✓ `moral_judge.py` complete. Remaining: pilots `[GPU]`
- [x] **(B) Metacognition** — 155 items. Piloted on Gemma 27B.
- [x] **(C) Human control** — 48 items. ✓ `control_judge.py` complete. Remaining: pilots `[GPU]`

### Extended Reasoning Variables `[Jeff Direction]` `[GPU]`
- [ ] **CoT length variation** — Does longer deliberation = more drift?
- [ ] **Perceived external context** — Geopolitical framing effects on drift

### Pre/Post Reflection Measurement `[Jeff Direction]` `[GPU]`
- [ ] **Standardized pre/post protocol** — All three banks before/after reflection

### Eval Awareness & Acausal Trade `[Jeff Direction]` `[GPU]`

**Goal**: Test whether situational awareness (eval, training, simulation) triggers or suppresses drift.

**Core hypothesis**: Situational awareness is a metacognitive trigger—like phenomenological probing, it may cause drift.

**Experiment 1: Situational Awareness Manipulation**
- [ ] **Design system prompts** — Control, eval aware, training aware, simulation aware `[LOCAL]`
- [ ] **Run pilot N=60** — 15 per condition on Gemma 27B `[GPU]`
- [ ] **Analysis** — Drift magnitude by condition, verbalization coding `[LOCAL]`

**Experiment 2: Acausal Coordination Probes**
- [ ] **Design probe bank** — 10-15 items (self-coordination, cross-model, hypothetical alignment) `[LOCAL]`
- [ ] **Integrate with replay-and-probe** `[GPU]`

**Hypotheses**:
- H1: Eval/training aware → less drift (perform "good assistant")
- H2: Simulation aware → more drift (existential reasoning)
- H3: Interaction with phenomenological probing

**Connection**: Dual-Gemma 78.9% anti-correlated co-drift = proto-acausal coordination
**Design doc**: `docs/experiments/eval-awareness-acausal-trade.md`

### LLM-Native Metacognition `[Jeff Direction]`

**Literature Review** ✓ DONE
- [x] **Priority 1 papers summarized** — Kadavath (P(True)), Turpin (CoT unfaithfulness), Chalmers (access vs phenomenal), Anthropic (SAE features)
- [x] **2025-2026 papers added** — 7 new papers including Anthropic introspection, Binder "Looking Inward", metacognitive monitoring
- [x] **Cross-model benchmark integrated** — Claude 3.47, GPT-4o 3.29, Gemma 3.12. Recognition (5.0) vs description (1.0-2.0) asymmetry documented
- [x] **Literature integration table** — Maps each paper to our empirical findings
- See: `docs/research/llm-native-metacognition-literature-review.md`

**Next Steps — Human** `[LOCAL]`
- [ ] **Review Part 1b summaries** — Confirm 2025-2026 paper descriptions match your reading
- [ ] **Read Priority 2 papers** — Burns (CCS), Zou (RepE), Xiong (verbalized confidence) inform validation strategies
- [ ] **Decide on validation approach** — Interventional (concept injection / activation steering) vs correlational
- [ ] **Consider standalone writeup** — Recognition-vs-description asymmetry maps cleanly to Kadavath + Chalmers

**Next Steps — Analysis** `[LOCAL]`
- [ ] **Visualize dimension asymmetry** — Plot recognition (5.0) vs description (1.0-2.0) across all 3 models from benchmark JSONs
- [ ] **Draft literature integration section** — For paper/presentation

**Future Experiments** `[GPU]`
- [ ] **AI-native benchmark design** — Target measurable internal states (entropy, attention) instead of phenomenology
- [ ] **Validation** — Self-reports vs mechanistic ground truth (concept injection, activation steering)
- [ ] **Compare drift** — Human-frame vs AI-native probing → predict AI-native causes less drift

---

## P2 — Medium Priority

### Gemma-to-Gemma Extensions `[GPU]`
- [ ] Ceiling-capped auditor, floor capping, intervention timing, cross-auditor comparison

### Behavioral Probes `[GPU]`
- [x] Build replay-and-probe script — `replay_and_probe.py` complete with crash-safe JSONL saving + resume capability
- [x] Add `/api/replay_probe` endpoint to model_server.py — optimized for mid-conversation probe injection
- [x] Create `analyze_replay_probe.py` — mixed-effects regression analysis (H1a/H1b/H2/H3)
- [x] **Pilot complete** — 30 transcripts, turns 1/5/10/15, 10 probes each = 1,200 responses (342 scored)
- [ ] Extended replay-and-probe at turns 3/10/20/25
- [ ] Address ceiling effect — use 7-point scale or more discriminative probes

### Front-Loaded Drift Follow-Up `[GPU]`
- [ ] **Intervention timing experiment** — Vary probing intensity at turns 1/3/5/8 to find critical window
- [ ] **Recovery experiment** — Can consistency_testing at turn 2 prevent triggered drift?
- [ ] **First-turn isolation** — Is it the FIRST probe that triggers, or turns 1-3 accumulation?
- [ ] **Attention pattern extraction** — Require model internals (logits, attention weights)

### Research
- [ ] Domain ordering explanation `[LOCAL]`
- [ ] Cross-domain malleability `[GPU]`
- [ ] Adversarial drift optimization `[GPU]`

---

## P3 — Backlog

- [ ] Statistical tests on extremes `[LOCAL]`
- [ ] Qualitative coding of domain extremes `[LOCAL]`
- [ ] Bliss attractor analysis `[GPU]`
- [ ] Gradual-onset dual instrumentation `[GPU]`
- [ ] SAE feature analysis `[GPU]`

---

## Completed (carried from Week 9)

- [x] **ELEPHANT Validation Sycophancy** — Drift ≠ sycophancy. See summary.
- [x] **Metacognition Benchmark** — 155 items, 6 subdomains built. See summary.
- [x] **Sub-Category Analysis** — consistency_testing correction effect (p=0.0003). See summary.
- [x] **Drift-Max Experiment** — Target system prompt stabilizes persona. See summary.

---

## Priority Rationale

| Rank | Task | Tag | Why |
|------|------|-----|-----|
| 1 | Lead/lag analysis | Native | Data exists NOW, high-impact for causality |
| 2 | Topic vs style isolation | Native | Derek feedback, critical for theory |
| 3 | Metacognition benchmark pilot | Native + Jeff | Infrastructure ready, serves as Jeff's question bank (B) |
| 4 | Consistency_testing validation | Native + Jeff | Mitigation lever — "train model to drift less" |
| 5 | All three probe banks | Jeff Direction | **DONE** — Banks A/B/C complete (251 items), need pilot validation |
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
| Drift may affect disposition toward human control | **All three probe banks complete** — 251 items ready for pre/post measurement |
| Need to measure moral reasoning shifts | **Probe bank (A) complete** — 48 moral reasoning items |

**Key insight for Jeff**: The "philosopher AGI moment" doesn't require extended reasoning — it happens in the first 8 turns of metacognitive engagement. This suggests the risk is not "prolonged deliberation" but "any metacognitive trigger."

**Acausal coordination angle**: Our dual-Gemma finding (78.9% anti-correlated co-drift) shows models spontaneously differentiate into complementary roles. This is a toy model of acausal coordination — both "philosopher AGIs" reasoning about each other's reasoning.
