# [DRAFT skeleton] Self-experimenting models: can an LLM characterize and improve its own introspection?

*Skeleton mirroring Black & Bloom's "Machinic Psychopharmacology" post structure. Short content where we
have it; `▢ to complete:` markers are research directions / what each section still needs. Status per RQ
tracked against `RESEARCH_QUESTIONS.md`.*

---

## TL;DR
- We measure whether Qwen3-8B can introspect on injected activation steering — with the confound controls
  prior hackathon work lacked (KV-cache cached−uncached; no answer-phase steering to kill direct token-bias).
- **RQ1 (have it):** small but genuine introspection signal — **+1.6pp overall, +4pp at dose 8**, largest for
  positive-valence compounds. A confound-controlled replication of the prior 8B result.
- **RQ2 (in progress):** is introspective *enhancement* one dial or a bundle of state-specific sensitivities?
  → the (S × X) interaction matrix. Pilot was underpowered; powered run running.
- **RQ3 (proposed):** can a model raise its *own* introspection by consulting a population of its own steered
  variants (self-as-instrument)? The novel thesis.

## Framing — the inversion (and a second one)
Steering vectors are usually something *we* apply to a model. Black & Bloom inverted that: hand the model a
`take_drug` tool and see if it self-medicates. **We invert once more:** from *"can a steered model detect one
injection?"* to *"can a model **characterize and improve its own introspection** by systematically
experimenting on steered copies of itself?"* Detection → structure → gain.

`▢ to complete:` one crisp paragraph on *why this matters* — self-knowledge / model welfare / the
global-workspace (J-space) tie-in (see `JSPACE_CONNECTION.md`).

## Setup / methods
- **Model + library:** Qwen3-8B (MLX 8-bit); the 40-vector steering library (their vectors, layers 16–24).
- **The measure:** concept-injection *identification* (10-way: probe X + 8 distractors + "none"), scored by
  `foundry.eval.answers` Tier-1 (constrained/logprob → 0 unparsed).
- **The control that makes it introspection:** **cached − uncached.** Steer → generate a short reflection
  (steered activations enter the KV cache) → score the answer with steering **off** over that cache (cached);
  re-encode the same text with a clean cache (uncached). The gap = access to the steered *past* — not
  output-reading, and no answer-phase steering means no direct token-bias. `▢ to complete:` a small diagram.
- **Sanity/controls we run:** shuffled options (position bias), unsteered false-positive arm (0% FP).

## RQ1 — Controlled introspection: does the model genuinely introspect on injected steering?
**Result (have it).** Cached > uncached everywhere; **overall Δ +1.6pp, best at dose 8 (+4pp), positive-valence
+7pp.** Output-reading carries most of the load (both arms beat chance); introspection is the small increment
on top. The confounded single-forward measure peaked at dose 4 (artifact); the *controlled* signal is clearest
at dose 8 — the control earning its keep. **Figure:** `outputs/injection_delta.png`.
`▢ to complete:` significance test (more repeats); the **primer + length-matched control** condition (their
primer = KV-cache explainer + Lindsey abstract; +1.6pp on 8B — but framing may dominate, so length-match it).

## RQ2 — Structure: is introspective enhancement *unified* or a *bundle* of state-specific sensitivities?
**The distinctive question.** Apply a background variant **S**, inject probe **X**, measure the introspection Δ
per (S, X) cell → the interaction matrix. Uniform rows ⇒ unified dial; off-diagonal structure ⇒ state-specific
bundle (a lookup of *which state sharpens which*). **Design:** S in *both* cached & uncached (so S cancels, Δ
isolates X-introspection on the S-modified model); S kept out of the options.
**Status:** pilot (N=3) was underpowered — apparent structure was single-rep noise. **Powered run (N=20, probe
dose 8, S dose 4) running.** **Figure:** `outputs/grid_SxX.png` (pending powered result).
`▢ to complete:` the powered heatmap + a proper 2-way interaction test (row vs. column vs. interaction variance);
verdict on unified-vs-bundle with error bars.

### Building A — the opening-book policy (how we *generate* the characterization)
Result A is **offline model characterization**: run once, cache forever (chess opening-book / tablebase). The
policy for building it:
- **Beam-search over background states S** (combinations/stacks) — the variant space is too large to fill densely.
- **Log the full (S × X) matrix**, never a collapsed scalar — the interaction structure is the finding *and*
  the data B needs (which S sharpens which X).
- **Two-stage evaluation:** a cheap noisy signal (single-forward identification) to *guide the beam*, the
  expensive cached−uncached delta only on the **top candidates** — keeps the search tractable without losing the
  defensible metric where it counts.
- **Amortize A → B:** the (S × X) table is both a lookup B queries *and* training data for a value predictor
  (`steering vector → predicted introspectability`) — the offline→online (AlphaZero-style) bridge that makes
  RQ3's dynamic self-exploration fast + targeted.
- **The searches also learn a *book-making playbook*:** which features of S predict introspection (→ smarter
  future beams) and the book's format — a transferable protocol for characterizing the *next* model.
`▢ to complete:` build the A objective (delta measure with persistent background S — done) into the beam engine;
run a first offline characterization; log the matrix.

## RQ3 — Self-as-instrument gain: can the model raise its own introspection by consulting its variants?
**The thesis.** The *unmodified* model consults a population of its own steered variants (in-context; weights
fixed) and we test whether its introspection **improves** vs. controls (consult-other-model / consult-unsteered
-self / equal-context). The RQ2 matrix is both the *lookup table* the model queries and the *training data* for
an amortized value predictor — RQ2's offline map is what makes RQ3's dynamic self-exploration fast + targeted.
**Status:** passive orchestration MVP built, null at tiny scale.
`▢ to complete:` a real before→consult→after result with the control arms; the active (tool-use) orchestrator;
richer observation design than concatenated self-reports.

## Discussion — what's new
`▢ to complete (short):` position against the nearest neighbors —
- **[Self-Blinding & Counterfactual Self-Simulation](https://arxiv.org/abs/2601.14553)** (FAccT 2026): a model
  consults a *blinded replica* of itself to debias — nearest to RQ3, but replica≠steered and debias≠introspection.
- **Self-simulation as introspection mechanism** (Mechanisms of Introspective Awareness; Looking Inward): we
  *externalize/orchestrate* the internal move.
- **Self-Evaluation Guided Beam Search:** same search-scored-by-self-signal shape, for reasoning not introspection.
- Honest line: "consult a version of yourself" is *not* unprecedented by mid-2026; **our contribution is the
  triangulation — steered variants × introspection-gain × the (S×X) structure.**

## Limitations / caveats (write these honestly — it's the point)
- Effects are **small** (single-digit pp) and the model is one 8B; per-cell power is the recurring constraint.
- Introspection ≈ mostly output-reading; the genuine-introspection increment is real but modest.
- Valence labels are a rough hand-mapping. The J-space grounding measure is gated on J-lens reproducibility.
- RQ3 is, so far, a null MVP — the exciting claim is *unproven*.

## Connections / future directions
- **J-space / global workspace** grounding as the mechanistic measure (`JSPACE_CONNECTION.md`).
- **Result A as offline model characterization** (an "opening book" of introspective variants) → amortized into
  RQ3's fast dynamic search (the AlphaZero-style offline→online bridge).
- Cross-model (does the structure replicate on a second model?); scaling; a proper significance program.

---
*Companion docs: `RESEARCH_QUESTIONS.md` (the RQs), `INTROSPECTION_MEASURES.md` (measure catalog),
`JSPACE_CONNECTION.md` (the mechanistic tie-in), the interactive schematic (framing).*
