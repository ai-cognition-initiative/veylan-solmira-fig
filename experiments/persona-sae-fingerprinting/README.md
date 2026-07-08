# Persona SAE Fingerprinting

> **FIG AI Sentience Extension · Pillar 1 (persona dynamics).** Master map: [`../../RESEARCH_THREADS.md`](../../RESEARCH_THREADS.md) (thread 1). The high-confidence, core empirical line.

## The bet
PSM says the model *selects / composes* personas. If that's true mechanistically, two things should be findable in SAE-feature space:
1. a **per-persona fingerprint** — the features that characterize each persona;
2. **inter-persona machinery** — features that fire when the model *moves between* personas but aren't part of any single persona's stable profile. These are the **"seams."**

The payoff is **causal**: ablate the seam features and see whether persona-shifting stops or changes. That's the difference between "we found correlated features" and "we found the switch."

**Free complement (Pillar 3):** the same persona × feature matrix yields the "persistent substrate" candidate — features present in *every* persona (low cross-persona variance). Seams and substrate are complements of one matrix.

## Goal
Build an empirical **persona vocabulary** from the ~275 Assistant-Axis roles — a per-persona SAE-feature activation profile — then use drift to find features active **outside any persona's profile** (candidate inter-persona machinery / seams), and causally ablate them.

## Model + SAE substrate (decided)
- **Science model: Qwen3-8B (instruct)** — already in the local HF cache; an 8B signal; unifies with the cross-model replication leg *and* the self-med experiment (one family).
- **SAEs: Qwen-Scope** — residual-stream SAEs for Qwen3-8B, all 36 layers, width 64K, Top-K 100 (`Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_100`). Raw `layer{n}.sae.pt` checkpoints (not a sae_lens release).
- **⚠️ Base/instruct caveat:** Qwen-Scope is trained on Qwen3-8B-**Base**; personas are an instruct phenomenon and our model is instruct → we apply base SAEs to instruct activations. The reconstruction gate (first-steps #2) is the go/no-go. Fallback: Gemma-9B + GemmaScope, or a base-model persona variant.

## Builds on
- [`../metacognition-persona-drift/`](../metacognition-persona-drift/) — the N=360 metacognitive-drift study (251-item probe battery, replay-and-probe, 78.9% dual-model co-drift). The mechanistic layer reads off this (re-run on Qwen3-8B — see de-risk).
- **Assistant-Axis corpus + pipeline** (Lu et al. 2026) — 275 roles / 240 traits, on disk (see Exists). Precomputed persona vectors for Qwen-3-32B et al. on HF (`lu-christina/assistant-axis-vectors`).

---

## Pipeline (6 stages)
1. **Persona corpus** — the ~275 Assistant-Axis roles: elicit each, capture Qwen3-8B residual-stream activations at the SAE layer(s).
2. **SAE encode** — run the Qwen-Scope SAE over those activations → a per-persona **feature-profile matrix**.
3. **Reliability gate** *(Derek's flag)* — 5–10 roles first; are profiles **stable** across prompt-resampling / seeds / layers? If not → behavioral MVP. **Gates everything downstream.**
4. **Scale the vocabulary** to all roles → the full persona × feature matrix.
5. **Seam-finding** — join the matrix with **Qwen3-8B drift trajectories** (from the replication leg): features active *during transitions* but outside every stable persona profile = candidate switching machinery.
6. **Causal ablation** — clamp/zero the candidate features during generation → measure whether drift stops/changes (251-probe battery + activation space).

## De-risk order (what gates what)
Two **P0** checks run *before* the main pipeline, in parallel:
- **(c) SAE reconstruction + reliability pilot** — stages 2–3. Gates whether SAE fingerprinting is viable at all (and settles the base/instruct question).
- **(b) Cross-model *behavioral* drift replication** on **Qwen3-8B** (then Qwen-32B) — does the ~4.5× metacognitive drift replicate off Gemma-27B? Existing generate/analyze pipeline, **no SAE and no download** (Qwen3-8B is local). Double duty: the P0 replication *and* it supplies the **matched drift trajectories** for stage 5.

Main pipeline (4→5→6) is gated on (c). Pillar 3 is gated on stage 4.

## Exists vs. needs building
**Exists (reuse):**
- **Persona corpus** — 275 roles (+240 traits, extraction questions, full Lu et al. generation pipeline) on disk, **but only in `admin/veylan-solmira-fig-broken/.../assistant-axis/data/`** — needs salvaging into this experiment.
- **SAE harness** — a working residual-stream hook + encode/decode roundtrip (cosine>0.95 gate) + **feature clamp/steer** in [`../decision-making-preference-adverse/`](../decision-making-preference-adverse/) (`gemma_sae.py`, `steering.py`). The hardest piece (ablation) exists in miniature. Wired for Gemma-2-2B via `sae_lens`; **porting to Qwen-Scope = a ~30-line raw-`.pt` loader** + the same hook (Qwen3 shares `model.model.layers[L]`).
- **Qwen3-8B weights** — local (HF + MLX). Drift codebase, N=360 corpus, 251-probe battery, dual-model instrumentation, vast.ai tooling, HF token.

**Needs building:** the Qwen-Scope loader; per-persona profile extraction + reliability metrics; the seam-finding join; scaling.

## Environment
Self-contained — this experiment has its **own** `.venv` (uv), pinned in `requirements.txt` to the validated Qwen3 / transformers-5 stack. No shared/cluster venv to track.
- Setup: `uv venv --python 3.12 .venv && uv pip install -r requirements.txt` (installs offline from uv's cache — zero download once the stack's been built once).
- Run: `.venv/bin/python check_reconstruction.py`

## First steps (ordered)
1. **Salvage the persona corpus** — copy Assistant-Axis `data/roles/` (+ traits, extraction questions, generation pipeline) out of the broken snapshot into this experiment (also clears part of the flagged broken-repo salvage).
2. **SAE reconstruction check on Qwen3-8B** *(go/no-go for base-SAE→instruct transfer)* — download a few mid-late Qwen-Scope layers (`layer{n}.sae.pt`, ~1–2 GB each), write the ~30-line loader, run encode→decode on instruct activations; is cosine ≳0.95?
3. **Reliability pilot** — feature profiles for 5–10 roles; stability across resampling / seeds / 2–3 layers.
4. **Cross-model behavioral replication** on Qwen3-8B — in parallel, **no download**; supplies matched drift trajectories.
5. Then: scale vocabulary → seam-finding → causal ablation.

## Fallback / MVP
The 251-item probe battery + replay-and-probe works with **any model and no interp infra** — validates Pillar 1 even if SAE features prove uninformative.

## Status: setup — substrate decided (**Qwen3-8B local + Qwen-Scope SAEs**). Next: salvage corpus + run the reconstruction check (base→instruct go/no-go).
