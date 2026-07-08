# FIG Sentience Extension — Research Threads (master doc)

**Purpose:** the living map of active research threads + their next steps. This supersedes `EXTENSION_KICKOFF.md` (a June-11 Elliot-sync walkthrough, now historical). Keep this current; everything else (proposal, worklog, reading tracker) links back here.

**Last updated:** 2026-07-05

**Anchors:** proposal → [`../extension-program/updated-proposal.md`](../extension-program/updated-proposal.md) · day-to-day → [`../extension-program/worklog.md`](../extension-program/worklog.md) · reading map → [`../reading-group/extension-phase/psm-canon.md`](../reading-group/extension-phase/psm-canon.md) · repo findings → [`README.md`](README.md), [`ROADMAP.md`](ROADMAP.md)

---

## At a glance

| # | Thread | Type | Status | Immediate next step |
|---|--------|------|--------|---------------------|
| 0 | Reading / literature | Foundation | Ongoing | Finish top-3 (Han, Berg, Lindsey) + Schwitzgebel |
| 1 | Persona dynamics — SAE fingerprinting (Pillar 1) | Core empirical | Setup | Cross-model replication + SAE-reliability scoping |
| 2 | Observer effect / persona-conditional welfare (Pillar 2) | Empirical | Scoped, not started | Design the persona-conditional-measurement MVP |
| 3 | Playwright / persistent substrate (Pillar 3) | Empirical | Blocked on #1 | Define "features in every persona" test |
| 4 | Self-experimentation scaffolding | Exploratory | Idea (posed to Derek) | Wait on Derek read; sketch a minimal version |
| 5 | Writing & outputs (phase-1 write-up + theory) | Writing | New | Outline the phase-1 drift blog; draft the locus-of-computation note |

Foundational infra (shared across 1–4): N=360 metacognition-drift corpus, 251-item probe battery, dual-model instrumentation, replay-and-probe pipeline, Assistant-Axis vectors, vast.ai tooling.

---

## 0. Reading / literature (foundation)

Feeds everything; not a deliverable line.
- **PSM canon** mapped: core paper → lineage → applications → critiques. See `psm-canon.md`.
- **The consciousness axis:** Cerullo ("Beyond the PSM," pro-non-deflation) ⇄ Schwitzgebel & Pober (Mimicry Argument, pro-deflation) is the live pro/con pairing.
- **Next:** finish top-3 (Han functional-welfare-axis, Berg self-referential reports, Lindsey introspection); read Schwitzgebel in full. Keep the reading tracker (Google Sheet) current.

## 1. Persona dynamics — SAE fingerprinting (Pillar 1) — *core empirical program*

The high-confidence line. Build a persona vocabulary empirically from ~286 personas → find SAE features active *outside* any persona (inter-persona machinery / the "seams") → causal-ablate them.

- **Base it sits on:** `experiments/metacognition-persona-drift/` (N=360, probe battery, 78.9% dual-model co-drift). The mechanistic layer reads off this.
- **Facets / sub-paths:**
  - (a) SAE fingerprinting + seam-finding — the main event.
  - (b) **Cross-model replication (P0 de-risk)** — Qwen 32B then Llama 70B; if the 4.5× metacognitive drift doesn't replicate, the whole plan rescopes.
  - (c) **SAE-feature reliability scoping (Derek's flag)** — are features stable enough in the persona-extraction context to support causal ablation? Pilot on 5–10 personas before scaling to 286.
- **Behavioral MVP / fallback:** the 251-item battery + replay-and-probe works with any model, no interp infra — validates Pillar 1 even if SAE features prove uninformative.
- **Next:** stand up the env (ongoing) → (b) and (c) in parallel as the first concrete experiments.

## 2. Observer effect / persona-conditional welfare measurement (Pillar 2)

The distinct question: **does the persona a probe *elicits* drive the welfare answer** (so you're measuring the elicited persona, not the model)? Worst for single-turn probes.

- **Two versions:** a *behavioral* one (probe battery, isolated vs. dyadic; no SAEs needed) and a *mechanistic* one (phenomenological vectors → SAE features → do they map onto persona clusters?).
- **Empirical anchor:** the dual-Gemma 78.9% co-drift already shows the observer effect is real and structured.
- **Status:** reframed this session as a *design principle to hold throughout* + a candidate standalone experiment. Open fork (put to Derek): does it earn a direct slot, or stay a methodological stance?
- **Next:** spec the persona-conditional-measurement MVP (run welfare probes single-turn → record answer + persona position → does position predict the answer?).

## 3. Playwright / persistent substrate (Pillar 3)

Complement to #1's seams: features present in *every* persona — a persistent substrate / agency "beneath" the personas. PSM's own open question (is the Assistant exhaustive?).

- **Convergence point:** this is where Lindsey's "third level" *and* the Schwitzgebel "locus of computation" objection (thread 5) both point — is there in-system mind-modeling under all the personas?
- **Status:** rich but no settled methodology; depends on #1's persona vocabulary being solid first.
- **Next:** once the vocabulary exists, identify low-variance cross-persona features → test whether they can be steered independently of character-level features.

## 4. Self-experimentation scaffolding (exploratory)

From Black & Bloom's "machinic psychopharmacology." Let a model **tree-search over copies of itself**, each running a different steering "compound," and watch the outcomes.

- **The bet (CoT analogy):** the paper is at the hand-built-scaffolding stage; the move is to let the model *self-navigate* that search — as CoT did for reasoning.
- **Digital-minds angle:** could structured self-experimentation — watching many steered versions of itself perform — teach a model about its *unmodified* self / sharpen introspection?
- **Status:** idea; posed to Derek (does he see a real possibility?). This is the main "excitement tangent" vs. the planned agenda.
- **Next:** hold for Derek's read; in parallel, sketch the *minimal* version (one model, a handful of compounds, a shallow tree, one introspection probe).

## 5. Writing & outputs

The written *products*. One investigation can yield several, and some outputs come from already-completed work rather than a current investigation — so outputs live here, together, separate from the investigations above.

### Empirical write-ups
- **Phase-1 drift findings → blog post (then paper).** The N=360 metacognitive-drift results: ~4.5× drift on metacognitive domains, front-loaded in turns 1–3, *not* sycophancy, 78.9% dual-model co-drift. **The results are already in hand** — source material lives in [`RESULTS_AND_METHODOLOGY.md`](RESULTS_AND_METHODOLOGY.md) and [`README.md`](README.md). This is the **highest-ROI near-term output**: a real result, low execution risk, and an early public win while the new experiments spin up. Target: **blog post first** (public + a concrete FIG deliverable), paper later. Cross-model replication (thread 1b) strengthens a *paper* but isn't needed for the blog.

### Theory / conceptual pieces
Through-line: the PSM ⇄ consciousness debate, and where interpretability can adjudicate it.
- **Locus-of-computation objection** to the Mimicry Argument. Biological/lookup mimics amortize the design work into evolution/environment (offline, fixed); flexible *consciousness*-mimicry must do the mind-modeling **online, in-system** — near what several theories call consciousness. Sharper cut: it **exposes that Schwitzgebel's "theory-neutral" screening-off smuggles in a theory** (works only if internal mind-modeling ≠ consciousness, false on self-model / higher-order / attention-schema theories). → *standalone note; the theory strand's first concrete output.*
- **Schwitzgebel ⇄ Cerullo pairing** on PSM-and-consciousness (pro- vs. anti-deflation).
- **Persona-conditional measurement principle** (from thread 2) — "welfare self-report reads the elicited persona, not the model."
- **"Empirically hostage" bridge** — the Mimicry Argument reduces to an interp question (*does the model model minds, or the surface statistics of mind-talk?*); this links the theory to threads 1/3.
- Supporting analyses on file: `../reading-group/extension-phase/copernican-mimicry-analysis.md`, `machinic-psychopharmacology-analysis.md`, `psm-canon.md`.

---

## Cross-cutting

- **Fine-tuning intervention (Elliot, conditional, Months 4–6).** Candidates: bliss-attractor induction / Assistant-Axis suppression / cross-model dual conversations. Specifics await the Elliot sync.
- **Feedback integration status (Derek / Elliot / Carolina):** all flagged in `EXTENSION_KICKOFF.md` §"Feedback integration"; SAE-reliability (Derek) and cross-model (P0) are being operationalized first; Carolina's base-model + prefilling is a methodological branch to pilot.
- **Agenda vs. excitement:** threads 1–3 = the planned agenda; threads 4–5 = the excitement pulls. Thread 3 (Playwright) is the bridge where the exciting philosophy and the planned interp actually meet. (This is the tension put to Derek.)

## Repo / infra notes

- **New experiments:** flat under `experiments/` (siblings to `metacognition-persona-drift/`), shared infra, **descriptive** folder names (pillar recorded in each README, not the path).
- **Extension index (the "README point"):** an `experiments/README.md` mapping each extension experiment → pillar → status, so the extension is discoverable without a hard `sentience-extension/` container. *(To create when the first Pillar-1 folder goes in.)*
- **Env setup:** ongoing — getting tooling running toward the SAE persona-fingerprinting.
- **Phase-1 artifacts (source / historical):** `RESULTS_AND_METHODOLOGY.md` = source material for the phase-1 write-up (thread 5). `ROADMAP.md` = phase-1 task tracker + feedback; historical (its live bit, the mentor feedback, is captured above). `EXTENSION_KICKOFF.md` = historical Elliot-sync doc, superseded by this file.

## Consolidated next steps (prioritized)

1. **Env smoke test** — confirm the repo runs + vast.ai reachable (de-risk early).
2. **Outline the phase-1 drift blog** from `RESULTS_AND_METHODOLOGY.md` — highest-ROI near-term output (results already in hand).
3. **Pillar 1 first experiments:** cross-model replication (Qwen) + SAE-feature reliability pilot (5–10 personas).
4. **Write the locus-of-computation note** — the theory strand's first output.
5. **Scaffold** `experiments/persona-sae-fingerprinting/` + the `experiments/README.md` index.
6. Spec the **Pillar-2 persona-conditional-measurement MVP** (design only, pending the agenda-vs-excitement call with Derek).
