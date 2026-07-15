# Research questions — introspection & self-experimentation

Two RQs, sharp. The confound-controlled injection-detection is *the measure*, not a headline question.
Framed against Black & Bloom (the foundation we move beyond) — see
`../../reading-group/extension-phase/machinic-psychopharmacology-analysis.md`.

---

## The measure (foundation — not an RQ)
Can Qwen3-8B identify an injected steering compound through **genuine access to its modified internal
state**, not output-reading or the steering biasing the answer? Scored by the **cached − uncached** KV-cache
control (their control, our clean implementation) + no answer-phase steering (kills the direct-bias confound).
- **Result in hand:** small but positive — +1.6pp overall, +4pp at dose 8, positive-valence strongest. A
  confound-controlled replication of prior work. Figure: `outputs/injection_delta.png`.
- This is the *instrument*; the RQs below use it.

## RQ1 — Is introspective enhancement a *unified capacity* or a *bundle of state-specific sensitivities*?
Apply a background variant **S**, inject probe **X** → the (S × X) introspection-delta matrix. Uniform rows ⇒
one "introspection dial"; off-diagonal structure ⇒ state-specific (a lookup of *which S sharpens which X*).
- **Design:** S in *both* cached & uncached (S cancels → delta isolates X-introspection on the S-modified
  model); S kept out of the answer options.
- **Result (so far — a clean negative):** **no steered background enhances introspection.** Unmodified is best
  (row μ +0.08); background is neutral-to-harmful (melancholic −0.06). **No detectable S × X structure** —
  interaction σ (0.05) below the per-cell noise floor (SE ~0.09) even at N=20. Figure: `outputs/grid_SxX.png`.
- **Implication:** the "most introspective *state*" premise looks like a dead end on this model — perturbing
  the model degrades self-access. Reshapes the steering-based framing.

## RQ2 — Can a model use *itself* as a tool to learn about itself — and to what extent?
The **unmodified** model consults a population of its own steered variants (in-context; weights fixed); does its
introspection about itself **improve**, vs. controls (consult-other-model / consult-unsteered-self /
equal-context)?
- **Status:** passive orchestration MVP built, null at tiny scale. **The thesis** — the novel contribution.
- **Caution from RQ1:** if steered variants aren't *better* introspectors, the "consult variants" mechanism
  needs a justification beyond that — they're instruments to learn *about* itself, not better introspectors.

---

### Positioning vs. Black & Bloom
| | Them | Us |
|---|---|---|
| The measure (introspection) | 10-way MCQ + KV-cache control, +8.2pp (8B, primer) | same control, clean impl, no-primer +1.6pp; dose-response; valence |
| RQ1 (structure) | — | **(S × X) matrix: unified vs. bundle** (result: negative + no structure) |
| RQ2 (self-as-instrument) | — | **can a model use itself as a tool to learn about itself** |
