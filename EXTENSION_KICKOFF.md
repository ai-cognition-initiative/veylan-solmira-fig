# FIG Extension Kickoff — Walkthrough & Plan

**Status:** Working doc for the first-2-weeks Elliot sync. Not a final research plan; the distillation of where things stand entering the extension, what the first phase looks like, and where I most want Elliot's insight and challenge.

**Last updated:** 2026-06-11

---

## Purpose

Document I can walk Elliot through during our sync (target week of June 8–12) covering:
1. What the project is and where it stands at extension start
2. The three primary research pillars for the 6-month extension
3. How feedback from mentor (Derek) and colleagues (Elliot, Carolina) has shaped the plan since the proposal
4. Concrete focus areas for the first ~2 weeks
5. The places I'd most value Elliot's challenge and the places I might pair on his work

Designed to be reciprocated — Elliot can present the equivalent for his work.

---

## Meeting cheat sheet — scannable

*Sized to be glanceable during the call. Each bullet is something I can speak to in 1–2 sentences without prep. Sections below this one go deeper if a thread needs more.*

### Open / where things stand
- N=360 metacognitive-domain runs complete on Gemma 2 27B
- Headline findings: 4.5× drift on metacognitive vs. other domains; drift is *not* sycophancy (drifted models become *less* validating, not more); dual-Gemma runs show 78.9% anti-correlated co-drift
- Front-loaded mechanism — most drift happens in turns 1–3

### The three pillars (one line each)
- **Pillar 1 — Persona dynamics under drift:** build a persona vocabulary from ~286 personas → find SAE features that activate *outside* any persona (inter-persona machinery) → causal-ablate them
- **Pillar 2 — Observer effect in welfare assessment:** measure how much the act of probing distorts the thing being probed; phenomenological vectors decomposed to SAE features
- **Pillar 3 — Playwright hypothesis:** find low-variance features persisting across all personas → test if they can be independently steered from character-level features

### Feedback integration (post-proposal)
- Derek (mentor): simplify language, MVP each pillar, scope SAE feature reliability as a real risk
- Elliot's prior input I want to revisit: do fine-tuning + use a second model
- Carolina: try base (pretrained) models + prefilling as a methodological branch

### First ~2 weeks focus
- **P0:** cross-model replication on Qwen 32B, then Llama 70B — if drift doesn't replicate, the rest of the plan needs rescoping
- SAE feature reliability scoping before scaling to all 286 personas
- Pilot Carolina's base-model + prefilling angle

### The four asks for Elliot (the actual point of the sync)
- Of three fine-tuning candidates I have (bliss-attractor induction / Assistant-Axis suppression / cross-model dual conversations) — which is most informative? Am I missing a fourth?
- "Use a second model" — what did you specifically mean? Want to align before either of us scopes downstream work
- Pushback on the persona-vocabulary → SAE feature ablation pipeline — what's the failure mode and what's a good fallback?
- The proposal threads welfare-focused and safety-focused framings through each pillar — is either harder to defend than I'm treating it?

### Mirror
- Ask Elliot to walk me through his plan in the same structure

### Close
- Pairing possibilities (depending on his scope): dual-model instrumentation, replay-and-probe pipeline, 251-item probe battery, N=60 dual-Gemma corpus — all reusable
- Next sync cadence — propose something (bi-weekly is a natural default)

### If we only have 30 minutes
- Drop the feedback-integration and first-2-weeks bullets — they can be inferred from the pillars + asks
- Do NOT drop the four asks for Elliot or the mirror request

---

## Where things stand at extension start

Core finding from the FIG fellowship phase (N=360 metacognitive-domain conversations on Gemma 2 27B): **metacognitive probing causes ~4.5× more persona drift than any other domain, the drift is front-loaded in turns 1–3, and drift is not sycophancy — drifted models become *less* emotionally validating, not more.** Dual-model runs show 78.9% anti-correlated co-drift (models spontaneously settle into complementary roles).

Full findings, plots, and infrastructure summary: see [`README.md`](README.md).

Probe battery, methodology, and replay-and-probe results: see [`RESULTS_AND_METHODOLOGY.md`](RESULTS_AND_METHODOLOGY.md).

---

## The three research pillars for the extension

Distillation of [`extension-program/updated-proposal.md`](../extension-program/updated-proposal.md) and [`extension-program/submitted-application.md`](../extension-program/submitted-application.md).

### Pillar 1 — Persona dynamics under drift (Months 1–4)
Build a persona vocabulary empirically from ~286 personas in the Assistant-Axis work, then use drift to identify SAE features that activate *outside* any individual persona's profile — candidates for inter-persona machinery. Causal test: ablate candidate features and measure whether persona shifting stops.

**Behavioral fallback / MVP:** the 251-item three-probe battery + replay-and-probe pipeline works with any model and requires no interpretability infrastructure. Validates Pillar 1 even if SAE features prove uninformative.

### Pillar 2 — Observer effect in welfare assessment (Months 2–5)
Welfare probing is itself a drift-inducing intervention. Measure the distortion via phenomenological vectors (curiosity, discomfort, engagement) and decompose into SAE features. Three-level result: vectors incoherent → method fails; coherent extra-persona features → real structure outside PSM; persona-mapped features → model represents own states using character features.

**Empirical anchor:** dual-Gemma 78.9% co-drift result already shows the observer effect is structured. Pillar 2 quantifies how much it distorts welfare measurement.

### Pillar 3 — Playwright hypothesis (Months 3–6)
PSM's unresolved question: is the Assistant exhaustive, or is there agency external to it? Identify low-variance SAE features persisting across all ~286 personas — playwright candidates. Critical test: can they be independently steered from character features? If yes, distinct representational levels.

### Cross-cutting: Fine-tuning intervention (Months 4–6, conditional)
Per Elliot's feedback. Likely a synthetic-data fine-tune of Gemma 2 27B to test whether the bliss-attractor pattern can be *induced* via training rather than emerging in-context, or to test whether the Assistant-Axis direction is suppressible at the weight level. Specific experiment TBD — this is where I most want Elliot's input on what intervention would be most informative.

---

## Feedback integration (post-proposal)

From the FIG-phase wrap-up — feedback collected from mentor (Derek) and colleagues (Elliot, Carolina). Source: [`ROADMAP.md`](ROADMAP.md) "Finish FIG Phase of Project" section.

- **Derek (mentor):** Proposal is conceptually dense → simplify language. PSM is a heavy framework imposing limitations → define what PSM entails, scope additional limitations during research, MVP each area upfront, document methodology choices and alternatives. **Open question:** how reliable are SAE features in the proposed context? Fallback plan if they don't work.
- **Elliot (colleague):** Do fine-tuning; use a second model. Want to revisit the full context at our sync.
- **Carolina (colleague):** Look at base (pretrained) models before instruction-tuning / RLHF; consider prefilling technique.
- **Self:** Router model vs. playwright model — what's the actual mechanistic difference?

**Status of integration:** all flagged; none operationalized yet. The first 2 weeks should turn at least the SAE-reliability question and Elliot's fine-tuning point into concrete near-term experiments.

---

## First ~2 weeks — concrete focus

Selected from [`docs/outstanding-work.md`](docs/outstanding-work.md), weighted toward P0 + the items most likely to unblock Pillar 1–3 design decisions.

### Active

1. **Cross-model drift replication on Qwen 32B (P0).** N=60 metacognitive-domain runs on Qwen 32B. Compare drift magnitude, front-loaded pattern, style isolation effect. Then Llama 70B. Tests whether the 4.5× finding generalizes beyond Gemma. **Why first:** if it doesn't replicate, the rest of the plan needs rescoping.

2. **SAE feature reliability scoping (Derek).** Before scaling Pillar 1's persona-vocabulary work to ~286 personas, test whether SAE features in the persona-extraction context are stable enough to support the planned causal-ablation experiments. Concrete: rerun the layer-0/4/15 roundtrip verification on Gemma 2 27B; pilot persona-feature extraction on 5–10 personas and measure feature stability across resampling.

3. **Carolina's base-model + prefilling question.** Pilot on Gemma 2 27B *base* (not instruct) with Lu et al.'s prefill method (Appendix D.3.1) to test the "less conditioned self vs. training artifact" hypothesis. This is P2 in the outstanding-work doc but worth surfacing in week 1–2 because it's a methodological branch point.

### Deferred but flagged

- Fine-tuning experiment design (Elliot's feedback) — defer specific design to after the Elliot sync. The sync itself is the input I need before scoping.
- Playwright candidate identification — depends on Pillar 1 persona-vocabulary work being solid first.

---

## Where I'd most value Elliot's challenge

Framed as peer exchange — places where I want pushback or a second read, not deference.

1. **Fine-tuning experiment specifics.** I have three candidate experiments sketched (bliss-attractor induction fine-tune, Assistant-Axis suppression fine-tune, cross-model dual conversations). Curious which one Elliot finds most informative for the open mechanistic questions vs. most tractable in 6 months — and whether he sees a fourth option I'm missing.
2. **"Use a second model" — what we each mean by it.** Multiple readings: dual-model instrumentation (already doing), fine-tuning against a second model, evaluation pairing. Worth aligning before either of us scopes downstream work.
3. **The persona-vocabulary → SAE feature pipeline.** Risk that the ~286-persona empirical vocabulary doesn't give clean enough SAE feature profiles for the causal ablation in Pillar 1. Want a peer challenge on the failure mode and fallback design.
4. **Welfare/safety framing tension.** The proposal threads welfare-focused and safety-focused interpretations through each pillar. Want Elliot's check on whether either framing is harder to defend than I'm treating it.

---

## Where I might pair on Elliot's work

To be filled after he walks me through his plan. Possibilities depending on his scope:
- The dual-model instrumentation is generalizable beyond Gemma — usable if there's a model pair he wants to instrument.
- The replay-and-probe pipeline + 251-item probe battery is reusable for measuring attitudinal shifts under any intervention class.
- The N=60 dual-Gemma corpus (with bliss-attractor co-drift) is shareable if his work touches mutual-drift phenomena.

---

## Linked documents

**Core context:**
- [`README.md`](README.md) — Findings, infrastructure, setup
- [`RESULTS_AND_METHODOLOGY.md`](RESULTS_AND_METHODOLOGY.md) — Methodology and pilot results
- [`ROADMAP.md`](ROADMAP.md) — Mentor feedback, task list, future directions

**Extension proposal materials:**
- [`extension-program/updated-proposal.md`](../extension-program/updated-proposal.md) — Three-pillar proposal
- [`extension-program/submitted-application.md`](../extension-program/submitted-application.md) — Submitted Q&A
- [`extension-program/field-survey-march-2026.md`](../extension-program/field-survey-march-2026.md) — Adjacent-field research scan
- [`extension-program/the-persona-selection-model.md`](../extension-program/the-persona-selection-model.md) — PSM framework

**Active work:**
- [`docs/outstanding-work.md`](docs/outstanding-work.md) — P0/P1/P2/P3 task tracking
- [`experiments/metacognition-persona-drift/`](experiments/metacognition-persona-drift/) — Active experiment code
- [`docs/persona-drift-research-brief.md`](docs/persona-drift-research-brief.md) — Research brief

**Administrative:**
- [`../admin/fig-sentience-extension-program.md`](../admin/fig-sentience-extension-program.md) — Extension program structure
- [`../foundational-research/DerekShillerMentorOverview.md`](../foundational-research/DerekShillerMentorOverview.md) — Derek mentor framing

---

## Walkthrough order for the sync

Suggested flow when I walk Elliot through this:

1. (5 min) Where things stand at extension start — point to the README findings.
2. (10 min) The three pillars — what each is, where they connect, where they branch.
3. (10 min) Feedback integration — what's been said by mentor + colleagues, what's not yet operationalized.
4. (5 min) First-2-weeks focus.
5. (15 min) Open questions / where I want challenge (fine-tuning, second model, persona-vocabulary failure modes, welfare/safety framing).
6. (15+ min) Elliot's walkthrough of his plan — mirror.
7. (5 min) Synthesize: areas of potential pairing, next sync cadence.

~60 minutes total if all of the above is covered. Easy to shorten by trimming items 3 or 5.
