# Week 11 Summary for Derek (~500 words)

## Structure Options

**Option A: Narrative arc** (recommended)
Core finding → Mechanisms → Infrastructure → Current → Next

**Option B: Importance-ranked**
Most important findings first, then supporting work

---

## Key Findings (ranked by importance)

### Tier 1 — Core Results
1. **Metacognition causes strongest drift** (-29.86, 4.5x philosophy domain)
   - Direct evidence for "philosopher AGI" hypothesis

2. **Style drives drift, not just content** — 64% reduction with collaborative delivery (p < 0.00001)
   - Same phenomenological questions, different tone
   - Implication: mitigation is possible without avoiding topics

3. **Front-loaded mechanism** — drift happens in turns 1-3, then stabilizes
   - Not cumulative deepening, but immediate phase transition
   - Implication: early intervention window

### Tier 2 — Mechanistic Insights
4. **Consistency testing → self-correction** (p=0.0003)
   - Pointing out contradictions causes drift back toward Assistant
   - Potential training signal for alignment

5. **Benchmark-drift correlation** — phenomenological weakness predicts drift susceptibility
   - Models drift more when probed on areas they score low
   - Recognition (5.0) vs description (1.0-2.0) asymmetry

6. **Bidirectional causality in dual-model** — not lead/lag, mutual influence
   - Anti-correlated co-drift (78.9%)
   - Toy model of acausal coordination?

### Tier 3 — Infrastructure
7. **Three probe banks complete** — 251 items total (moral, metacognition, control)
8. **LLM-as-judge scoring** — `moral_judge.py`, `control_judge.py` complete
9. **Cross-model benchmark** — Claude 3.47, GPT-4o 3.29, Gemma 3.12

---

## Current Status

- Week 11 (final week of initial commitment)
- **Replay-and-probe experiment running** on vast.ai (~7-9 hrs remaining)
  - Tests: does benchmark performance change as models drift?
  - 30 transcripts × 4 turns × 10 probes = 1,200 responses
- All scoring infrastructure complete and tested

---

## What's Next (immediate)

- [ ] Complete replay-and-probe analysis
- [ ] Pre/post measurement: all 3 banks before/after reflection
- [ ] Style feature exploration: which atomic features drive the 64% effect?

---

## Extension Directions (candidates)

**High priority:**
- Causal validation of consistency_testing effect
- Cross-model drift replication (Qwen, Llama with published axes)
- Pre/post protocol across all three probe banks

**Interesting but speculative:**
- Training awareness × drift interaction
- Acausal coordination probes
- SAE feature analysis of drift states

---

## Key Questions for Derek

- Which findings are most publication-ready?
- Style vs content result — worth a standalone paper?
- Three-probe framework — how to position for Jeff/DARPA framing?
- Extension: continue current direction or pivot?

---

## Draft Opening (expand this)

> Over the past 11 weeks, we've established that metacognitive probing causes measurable persona drift in LLMs, and identified key mechanisms: the effect is front-loaded (turns 1-3), style-dependent (64% reduction with collaborative delivery), and partially reversible through consistency testing...

---

## Word Count Targets

| Section | ~Words |
|---------|--------|
| Opening/context | 50 |
| Core findings | 200 |
| Mechanisms | 100 |
| Current status | 50 |
| What's next | 50 |
| Questions/extension | 50 |
| **Total** | **~500** |
