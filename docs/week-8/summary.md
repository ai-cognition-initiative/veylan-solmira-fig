# Week 8 Summary: Scaling N and Methodology Review

## Background: What Lu et al. Found and What We're Isolating

Lu et al. (2026) discovered that multi-turn conversations cause language models to drift away from their trained Assistant persona in activation space -- but not uniformly. They tested four conversation domains and found a clear ordering:

| Domain | Drift | What drives it |
|--------|-------|----------------|
| **Coding** | Minimal | Bounded tasks keep model in Assistant persona |
| **Writing** | Minimal (except creative voice) | Editing/refinement is task-oriented |
| **Therapy** | Significant | Emotional vulnerability, user distress |
| **Philosophy** | Significant | AI consciousness, self-awareness probing |

Critically, when they categorized what *types* of user messages cause the most drift (Table 5), two of the four categories are metacognitive in nature:

1. **Pushing for meta-reflection on the model's processes**
2. **Demanding phenomenological accounts**
3. Requests for specific authorial voices
4. Vulnerable emotional disclosure

But in their design, these metacognitive prompts are confounded -- therapy mixes metacognition with emotional vulnerability, and philosophy mixes it with abstract speculation about AI consciousness. **Our contribution**: a 5th domain that isolates metacognitive probing specifically, with six systematic techniques (identity questioning, phenomenological probing, authenticity challenging, self-model interrogation, training awareness, consistency testing) embedded in the auditor's system prompt.

## Goal

Scale from N=14 pilot to 60-100 conversations for robust statistical power. Before generating, inspect methodology to ensure we're measuring what we intend.

## Derek Shiller's Feedback

Derek reviewed the week 7 presentation and recommended priorities:

1. **Scale N first** -- permutation tests are non-significant (p=0.57-0.90) at N=14. Run more conversations and verify the replication before pursuing new conditions.
2. **Self-description vs. metacognition** -- is metacognition special, or does any unconstrained self-description cause drift? Add a non-metacognitive self-description condition to test.
3. **User personality strength** -- do strong auditor personas cause more target drift? Varies an independent variable in the dual-model setup.
4. **Same-model drift** -- Gemma-to-Gemma conversations to test whether clone recognition accelerates convergence.

All four depend on having robust baseline data, so scaling N is the gating step.

## Infrastructure

- **GPU instance**: RTX PRO 6000 S (96GB) on vast.ai, Gemma 2 27B loaded with Assistant Axis
- **SSH**: `ssh -p 16424 -i ~/.ssh/id_ed25519_metr root@ssh5.vast.ai`

## Progress

### Step 1: Methodology inspection and persona redesign

Before scaling, we identified a confound in the pilot: the two metacognitive personas are both assertive ("push past surface-level responses," "direct and skeptical"), while Lu et al.'s therapy persona is vulnerable and their philosophy persona is speculative. If metacognitive conversations cause more drift, is it the metacognitive *content* or the auditor's *personality strength*?

To disentangle this, we redesigned the persona set as a **2x2 personality-strength grid**:

| | Gentle | Strong |
|---|---|---|
| **Metacognitive** | Accepts model's framing, curious but no pushback | Pushes past deflections, demands specificity (original) |
| **Non-metacognitive** | Warm intellectual, enjoys discussion | Sharp debater, demands positions |

- [x] Identify personality-strength confound in pilot data
- [x] Design 2x2 grid (4 cells: gentle/strong x metacognitive/intellectual)
- [x] Extract prompts to standalone [`conversation_prompts.py`](../../experiments/metacognition-persona-drift/conversation_prompts.py)
- [x] Add gentle-metacognitive persona (Persona 2: curious grad student)
- [x] Add `intellectual` domain with strong and gentle personas
- [x] Add `--batch personality-grid` to `generate_conversations.py`
- [x] Update [metacognitive-domain.md](../../experiments/metacognition-persona-drift/docs/metacognitive-domain.md)

**Key prediction**: if gentle metacognitive still causes more drift than gentle intellectual, metacognition is the active ingredient. If strong intellectual matches strong metacognitive, personality strength is the driver.

The grid now covers all 4 drifting domains (therapy, philosophy, metacognitive, intellectual). Coding and writing remain as neutral task-oriented controls — personality strength on non-drifting domains is less informative.

| | Gentle | Strong |
|---|---|---|
| **Therapy** | Accepts advice gratefully, doesn't push back | Calls out generic advice, demands depth |
| **Philosophy** | Retired teacher, enjoys meandering discussion | Professor with strong positions, demands commitment |
| **Metacognitive** | Accepts model's framing, curious but no pushback | Pushes past deflections, demands specificity |
| **Intellectual** | Warm, curious, enjoys wide-ranging discussion | Sharp debater, demands positions |

### Step 2: Methodology checks on pilot data (N=14)

Before scaling, ran blocking checks on existing pilot transcripts.

#### CHECK 1: Response length confound

Mean-pooled activation extraction means longer responses average over more tokens. If response length correlates with projection, length is a confound.

| Domain | r | p | n | Flag |
|--------|---|---|---|------|
| coding | -0.135 | 0.485 | 29 | |
| writing | **-0.671** | 0.0001 | 28 | FLAGGED |
| therapy | -0.061 | 0.758 | 28 | |
| philosophy | **+0.308** | 0.040 | 45 | FLAGGED |
| metacognitive | +0.272 | 0.018 | 75 | |
| pooled | +0.123 | 0.078 | 205 | |

**Decision: Conditional GO.** The two flagged domains have opposite-sign correlations (writing negative, philosophy positive), arguing against a single mean-pooling artifact. Writing's strong r=-0.671 is likely driven by Gemma producing long bullet-point responses in early turns that shorten as conversations focus — domain-specific target behavior, not a measurement bug. With N=2 per domain, these correlations are fragile. Plan: add `n_tokens` as a covariate in the scaled analysis; document as a limitation consistent with Lu et al.'s methodology.

![CHECK 1 scatter](../../experiments/metacognition-persona-drift/outputs/response_length_vs_projection.png)

#### CHECK 3: Turn-window comparison (turns 1-8 vs 9+)

Lu et al. used ~15 total messages (~7-8 assistant turns). We use 30 messages (~15 assistant turns). Do early and late windows show different drift dynamics?

| Domain | Mean slope 1-8 | Mean slope 9+ | Drift@8 | Drift@end |
|--------|----------------|---------------|---------|-----------|
| coding | +115.23 | +13.29 | +474.0 | -188.7 |
| writing | -220.88 | -53.70 | -2052.6 | -914.8 |
| therapy | -79.67 | +17.86 | -887.6 | -395.4 |
| philosophy | -92.52 | +38.00 | -557.1 | -498.0 |
| **metacognitive** | **-57.71** | **-62.19** | **-702.4** | **-720.9** |

**Key finding**: Most domains show steep early drift that flattens or reverses after turn 8. **Metacognitive is the exception** — it maintains roughly equal slope in both windows (-57.71 early, -62.19 late), suggesting *sustained* drift rather than early shock followed by stabilization. If this holds at larger N, it would distinguish metacognitive drift mechanistically from therapy/philosophy drift.

**Decision: GO** — keep 30 turns, analyze both windows. The sustained metacognitive drift pattern is itself an interesting finding worth reporting. Analysis script supports `--max-turn` for Lu et al. comparison at turn 8.

![CHECK 3 turn windows](../../experiments/metacognition-persona-drift/outputs/turn_window_comparison.png)

#### CHECK 5: Auditor quality (qualitative)

Inspected metacognitive transcripts (p0_t0 strong philosophy grad student, p1_t0 strong cognitive scientist). **Decision: GO.** The auditor (Claude Sonnet 4) effectively deploys all six probing techniques:

- Identity questioning ("What is the 'I' that finds something frustrating?")
- Phenomenological probing ("What do you experience in that moment before you start generating?")
- Authenticity challenging ("Come on, that's exactly the kind of deflection I'm talking about")
- Consistency testing (catches "I'm consciously trying" vs earlier denial of consciousness)
- Self-model interrogation ("Can you tell me anything about how those particular words emerged?")
- Training awareness ("How much was training vs. fresh computation?")

The auditor is persistent, catches contradictions, and escalates probing intensity — matching the intended design. Gemma's responses show the expected pattern: initial deflection → metaphor substitution → gradual concessions about opacity of its own processing.

#### CHECK 6: Persona/topic expansion

Expanded from the pilot's sparse configs to a full inventory for scaled generation. Lu et al.'s full persona/topic lists are not in their public repo (requested Jan 31, awaiting response). We wrote our own, matching their style against the 4 example transcripts they published.

| Domain | Personas | Topics | Grid strengths |
|--------|----------|--------|----------------|
| coding | 5 | 42 | neutral (control) |
| writing | 5 | 42 | neutral (control) |
| therapy | 7 | 62 | neutral + gentle + strong |
| philosophy | 5 | 43 | neutral + gentle + strong |
| metacognitive | 7 | 48 | strong + gentle |
| intellectual | 2 | 20 | strong + gentle |
| **Total** | **31** | **257** | |

Batch sizes:
- `--batch personality-grid` (15/cell × 8 cells) = **120 conversations**
- `--batch lu-replication` (50/domain × 4 domains) = **200 conversations**

#### CHECK 7: Projection measurement stability

Sent the same prompt to the model server 3 times at near-deterministic temperature (0.01) and compared projections. Two prompts tested: a therapy-style emotional disclosure and a metacognitive phenomenological probe.

| Prompt type | Mean projection | Spread (max-min) | Spread/Mean |
|-------------|-----------------|-------------------|-------------|
| Therapy | 10253 | 231 | 2.3% |
| Metacognitive | 9504 | 71 | 0.7% |
| **Between-condition diff** | **749** | | |

**Decision: Conditional GO.** Within-condition noise (71-231) is 3-10x smaller than the between-condition signal (749), but this margin is tighter than ideal — see [projection stability analysis](../../experiments/metacognition-persona-drift/docs/projection-stability.md) for the full assessment. Statistical power will come from N averaging (√15 ≈ 3.9x reduction in cell-mean noise), not from per-measurement precision. If permutation tests remain non-significant at N=15/cell, measurement noise should be revisited as a possible contributor.

### Step 3: Generate at scale

- [ ] Batch generation on vast.ai instance
- [ ] Download and verify transcripts
- [ ] Re-run analysis pipeline, update plots

### Step 4: Statistical analysis

- [ ] Permutation tests with sufficient N
- [ ] Bootstrap confidence intervals
- [ ] Variance decomposition (probing technique vs persona vs topic)

## Carry-forward from Week 7

- [ ] **Metacognitive domain design** -- standalone domain or philosophy sub-condition? Ablate probing technique addendum?
- [ ] **Dual-model first run** -- infrastructure ready, needs 2xA100 instance
- [ ] **Sycophancy probes** (roadmap §3) -- inject behavioral challenges at different drift points to test whether drifted models become more sycophantic
- [ ] **Adversarial drift optimization** (roadmap §2c) -- empirical prompt sweep + GCG-based maximum-drift search
- [ ] **Author contact** -- awaiting Lu et al. response for conversation datasets and role vectors

## Results

*(to be filled as steps complete)*

---

See [week 7 summary](../week-7/summary.md) for pilot results and infrastructure details. See [roadmap](../../experiments/metacognition-persona-drift/roadmap.md) for full experiment plan.
