# Connecting the introspection work to J-space (global workspace) — future step

**Status:** planning doc for a **future step (~next week)** — not started. Owner: self-med / introspection thread.
**Purpose:** capture *why* the J-space / global-workspace result is the natural theoretical home for the
introspection & self-as-instrument work, the concrete bridges, the honest caveat, and the next actions —
so we can pick it up cold.

---

## 1. What J-space is
**"Verbalizable Representations Form a Global Workspace in Language Models"** — Gurnee, Sofroniew, Pearce,
Piotrowski, Kauvar, Chen, Soligo, Bogdan, Ong, Wang, Thompson, Abrahams, Kantamneni, Ameisen, Batson,
**Lindsey** (Anthropic, Transformer Circuits, **2026-07-06**).

Claude-family models maintain a small, privileged set of internal representations — the **J-space** (after
the **Jacobian** technique that surfaces them) — that behaves like a neuroscience **global workspace**. The
defining claim: these representations are **"available for report, modulation, and flexible internal
reasoning."** A small, evolving set of *unspoken words* the model is currently reasoning with — neither echoes
of the input nor next-token predictions — **globally available** to the rest of the network. Selective: only a
small fraction of what the model represents. The technique surfaces "the concepts a model is poised to verbalize."

- Paper: https://transformer-circuits.pub/2026/workspace/index.html
- Anthropic summary: https://www.anthropic.com/research/global-workspace
- LessWrong: https://www.lesswrong.com/posts/3PaLrzxagpbnNtPLT/a-global-workspace-in-language-models
- Dehaene & Naccache commentary (consciousness angle): https://unicog.org/wp_2025/wp-content/uploads/2026/07/Dehaene-and-Naccache-Workspace-commentary-on-Gurnee-Lindsey-June-2026.pdf

## 2. Why it connects almost 1:1
The J-space's three defining properties map directly onto our three moving parts:

| J-space property | Our work |
|---|---|
| **available for report** | introspection = *can the model report its state?* → J-space predicts **which** states are reportable |
| **modulation** | our steering ("compounds") *is* modulation of internal representations |
| **flexible reasoning** | the self-as-instrument loop = reasoning about its own perturbed states |

**Through-line:** introspection = access to the global workspace; steering = perturbing it; self-as-instrument
(Result B) = *can the model learn to read its own J-space better by perturbing itself?* This is squarely the
Jack-Lindsey trajectory that is the project's north star.

## 3. Three concrete technical bridges (later)
1. **J-space = the rigorous version of our `tracks_substrate` stub.** Instead of "does the self-report track
   the SAE substrate," it becomes "does the self-report track what's actually *in the J-space*" — the field's
   most principled introspection-grounding measure, from the definitional source. (Introspection measure
   Family 4; see `introspection.py` Part 1 stub `tracks_substrate`.)
2. **It explains *why* concept-injection detection works.** The "sweet-spot strength" both this and the
   Emergent-Introspective-Awareness line observe = strong enough to enter the J-space (→ reportable), not so
   strong it breaks coherence. Our planned injection-detection measure gets a mechanistic theory for free.
3. **It reframes Result B for the target audience.** "Does self-experimentation raise the unmodified model's
   introspection" → "does it improve access to the global workspace." Makes the work immediately legible to the
   Anthropic-interp / Eleos community we're aiming at.

## 4. Unification with Pillar 1
Our **SAE substrate features** (persona-sae-fingerprinting) and the **J-space** are probing the *same object* —
the model's internal reportable workspace. J-space is the more dynamic/refined version of "the persistent
substrate." This links q-selfmodel (self-med introspection) and q-sentience (Pillar-1 substrate) through one
shared mechanistic target. Worth drawing this edge explicitly in the Cairn map.

## 5. The honest caveat — why this is "later," not "now"
- The **framing** connection is free and worth adopting immediately (sharpens measure choices + the publication story).
- The **technical** bridge (read J-space directly, compare to self-report) is a genuine later step because the
  **Jacobian technique's reproducibility on open Qwen models is an open question.** It's an Anthropic-internal
  method demonstrated on Claude; whether it ports to Qwen3-8B with open tooling is exactly what we'd need to
  verify before committing. Also a heavy interp method — non-trivial effort.

## 6. Recommendation — two tiers
- **Now (free):** adopt the global-workspace framing for the introspection work; add J-space to the reading map
  and the mentor narrative. It sharpens our measure choices and our story.
- **Later (contingent on reproducibility):** if the Jacobian technique reproduces on Qwen, J-space-vs-self-report
  becomes the gold-standard grounding measure (Family 4, done right), replacing/upgrading the `tracks_substrate` stub.

## 7. Next actions (the future step)
- [ ] **Fetch + deep-read** the J-space paper; assess whether the **Jacobian surfacing technique is reproducible
      on Qwen3-8B** with open tooling (the gating question for the technical bridge).
- [ ] **Cairn:** add a `j-space` node and edges linking **q-selfmodel ↔ Pillar-1 substrate ↔ this work** (the
      shared-mechanistic-target unification, §4).
- [ ] **Reading-group entry:** a short analysis in `reading-group/extension-phase/` (sibling to
      `machinic-psychopharmacology-analysis.md`).
- [ ] **If reproducible:** wire J-space readout as the Family-4 grounding measure in `introspection.py`
      (`IntrospectionMeasure.tracks_substrate`), replacing the SAE-only stub.
- [ ] **Framing:** fold the global-workspace framing into the self-med write-up + mentor deck.

## Related in-repo
- Introspection three-part structure + measures: `introspection.py`; schematic (self-as-instrument).
- Introspection-measure literature catalog (5 families): see the lit-scan notes / this thread.
- Prior-art analysis (the source we're moving beyond): `../../reading-group/extension-phase/machinic-psychopharmacology-analysis.md`.
