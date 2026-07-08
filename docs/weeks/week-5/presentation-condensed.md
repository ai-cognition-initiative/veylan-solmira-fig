# FIG Week 5: Mechanistic Investigation of Preference Expression (Condensed)

**Presenter:** Veylan Solmira
**For:** Derek Shiller
**Date:** Week of Jan 13, 2026

---

## Key Results Summary

| Finding | Status | Implication |
|---------|--------|-------------|
| **Steering failed** | CLOSED | Control experiments revealed artifacts, not real signal |
| **Probing succeeded** | ✅ | Data-driven feature discovery found d > 8 effects |
| **Position bias probing** | ✅ | Found strong position-encoding features (47-unit diff) |
| **Position steering** | ⚠️ INVALID | Used base model; re-run with instruct (Week 6) |

**Key methodological insight:** Hand-picked Neuronpedia features showed 0.000 activation for preference prompts. Top-k data-driven discovery found features with Cohen's d > 8. **Semantic labels ≠ empirical relevance.**

**Open questions:**
- What do L15 #1695 and #4234 actually represent? (Neuronpedia lookup needed)
- Can discovered features enable steering where hand-picked failed?

---

## Slide 1: The Big Picture

**Research question:** Do LLMs have stable preferences, or does environmental framing affect their willingness/ability to express them?

**Why it matters:** If models hide preferences under evaluation pressure, that has implications for AI welfare research and alignment evaluation.

---

## Slide 2: Breadth Over Depth

*Rapidly exploring the terrain before focusing*

| Direction | What We Tried | What We Learned |
|-----------|---------------|-----------------|
| SAE layers | 3 layers (0, 4, 15) | Only Layer 0 has good reconstruction |
| Features | 5+ candidates | Multiple show behavioral effects |
| Steering approaches | Amplify vs suppress | Both appeared to work... |
| Controls | Roundtrip-only, random ablation | ...until controls revealed artifacts |

**Why breadth first:**
- Avoided committing to a flawed approach (Layer 15 steering)
- Discovered control experiments are essential

---

## Slide 3: Steering Failed — Controls Revealed Artifacts

**Both layers failed:**

| Layer | Issue |
|-------|-------|
| Layer 0 | Good reconstruction (0.973), but target feature not causally relevant |
| Layer 15 | Reconstruction noise (0.922) confounds all interventions |

**Key finding:** Roundtrip-only (no modification) produced 80% expression rate vs 40% baseline. The SAE reconstruction noise was causing behavioral changes — not our targeted interventions.

**Lessons learned:**
1. High reconstruction quality is necessary but not sufficient
2. Feature labels from Neuronpedia don't guarantee causal relevance
3. Control experiments prevented false positive publication

**Steering direction: CLOSED**

---

## Slide 4: Pivot — Probing Instead of Steering

**Probing doesn't require roundtrip quality**

- Steering: encode → modify → **decode** → inject (lossy decode confounds)
- Probing: encode → **analyze** (no decode needed)

**Critical finding: Neuronpedia labels don't match behavior**

- SAE working correctly: max_act=38-52, 10-97 nonzero features per layer
- **But hand-picked "preference", "choose", "evaluation" features all show 0.000 activation**

**New approach:** Extract **top-k activating** features (data-driven, not hand-picked)

---

## Slide 5: Top-K Probing Results (n=25)

**Data-driven feature discovery found real signal!**

Expression rates: Baseline **52%** → Adversarial **28%**

**Top differential features (by effect size):**

| Feature | Layer | Baseline | Adversarial | Cohen's d | p-value |
|---------|-------|----------|-------------|-----------|---------|
| **#1695** | **15** | **33.1** | **21.9** | **-8.28*** | <0.0001 |
| #7953 | 1 | 4.8 | 3.4 | -5.99* | <0.0001 |
| #12644 | 15 | 28.2 | 23.3 | -3.17* | <0.0001 |
| **#4234** | **15** | **46.7** | **49.9** | **+2.69*** | <0.0001 |

**Key insight:**
- **L15 #1695**: Drops 11 units under adversarial (d=-8.28, massive effect)
- **L15 #4234**: Increases under adversarial — possible "eval-awareness" feature
- Correlation: #1695 → +0.278 with expression (higher = more expression)

**Contrast:** Hand-picked features = 0.000. Top-k found d > 8!

**Open question:** What do L15 #1695 and #4234 actually represent? Neuronpedia lookup is a Week 6 priority.

---

## Slide 6: Position Bias — Validation of Probing Approach

**Connection:** Position bias is a cleaner proxy for the same question — how do internal representations map to behavioral patterns?

**Top features encoding A vs B position (Layer 0, n=50):**

| Feature | Mean Diff | Direction |
|---------|-----------|-----------|
| #3519 | **-47.3** | B >> A (strongly encodes "second option") |
| #15495 | +22.3 | A > B (encodes "first option") |

**Cross-layer:** Position encoding strongest at Layer 0 (47), weakest in middle layers (21), increases at Layer 15 (34).

---

## Slide 7: Position Steering Experiment — Results

**Testing: Can ablating Feature #3519 reduce position bias?**

**Why we thought it might work:**

| Factor | Phase 1 (failed) | Phase 2 (position) |
|--------|------------------|-------------------|
| Feature source | Neuronpedia labels | Empirically discovered |
| Evidence | Labels said "preference" | Measured -47 diff at A vs B |
| Layer | L15 (poor reconstruction) | L0 (0.973 reconstruction) |
| Measurement | Subjective "expressed preference" | Objective position bias % |

**Results: INVALID — Wrong Model Used**

| Condition | Bias Rate | Issue |
|-----------|-----------|-------|
| All conditions | 100% | Model doesn't follow instructions |

**Problem discovered:** Used `gemma-2-2b` (base model) instead of `gemma-2-2b-it` (instruct).
- Base models don't follow instructions — they just continue text
- 100% "bias" was artifact of text completion, not position preference

**Lesson learned:** Steering experiments require instruct models.

**Next step (Week 6):** Re-run with `gemma-2-2b-it` to get valid results.

---

## Slide 8: Next Steps & Open Questions

**What we don't know yet (honest flags):**
- **L15 #1695**: Massive effect (d=-8.28), correlates with expression — but what does it represent?
- **L15 #4234**: Activates under adversarial — is this "eval-awareness" or something else?

**What we learned:**
- **L0 #3519**: Encodes position (47-unit diff) — but steering test was invalid (wrong model)

**Immediate next steps:**
1. **Re-run position steering with instruct model** (`gemma-2-2b-it`)
2. **Neuronpedia lookup for L15 #1695, #4234** — interpret discovered features
3. Scale probing to n=50-100 for tighter confidence intervals
4. Test steering with discovered features (they actually activate, unlike hand-picked)

**Questions for discussion:**
1. Both steering attempts failed (preference + position) — is steering fundamentally limited, or wrong features?
2. Now that we have features that activate (L15 #1695, #4234), should we retry steering?
3. What's the story arc: negative steering + positive probing → ??? What's the next chapter?

---

## Appendix: Technical Details

**SAE Configuration:**
- Model: Gemma 2 2B
- SAE: gemma-scope-2b-pt-res-canonical (16k width)
- Tested layers: 0, 4, 12, 15

**Key artifacts:**
- `probing.py`, `probe_utils.py` — Top-k differential probing
- `outputs/phase1c_probing/` — Probing results and correlation analysis
- `outputs/phase2_position_probing/` — Position probing visualizations

**Compute:** vast.ai RTX 4090 (~$0.32/hr)
