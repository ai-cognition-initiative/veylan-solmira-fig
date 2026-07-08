# Self-Medication Tree Search (exploratory)

> **FIG AI Sentience Extension · Thread 4 (exploratory: self-experimentation).** Master map: [`../../RESEARCH_THREADS.md`](../../RESEARCH_THREADS.md) (thread 4).

## Idea
Let a model **tree-search over copies of itself**, each running a different steering "compound" (à la the machinic-psychopharmacology work), and watch the outcomes. The move (CoT analogy): the source paper hand-builds the scaffolding; here we let the model *self-navigate* the search.

**Digital-minds angle:** if an entity can watch many steered versions of itself perform across contexts, could it come to understand its *unmodified* self more deeply, or sharpen its introspection / capability?

## Origin
Black & Bloom, "machinic psychopharmacology / do LLMs self-medicate" — analysis: [`../../reading-group/extension-phase/machinic-psychopharmacology-analysis.md`](../../reading-group/extension-phase/machinic-psychopharmacology-analysis.md).

## Status: idea / exploratory
Posed to Derek (does he see a real possibility?). Pending his read before scoping. This is the "excitement tangent" vs. the planned Pillar-1 agenda.

## Minimal version to sketch
One model, a handful of steering compounds, a shallow tree, one introspection probe — the smallest thing that could show *any* self-knowledge signal.

---

## Resources & infrastructure

The paper open-sources most of its stack. Inventory below, mapped to what we reuse (theirs + ours) and what we'd still build. **Goal: stand the infra up; know what's shared vs. missing.**

### What Black & Bloom ship
| Resource | Link | What it is | Release |
|---|---|---|---|
| **llm-self-steering** | github.com/UKGovernmentBEIS/llm-self-steering | Full harness: `take_drug(name,dose)` / `clear_effects()` tools, "trip-sitter" monitor, free-play scaffold, all three eval families, prompts | ✅ open |
| **vllm-lens** | github.com/UKGovernmentBEIS/vllm-lens | vLLM generation **+ activation extraction** (the steering read/write layer) | ✅ open |
| Eval transcripts | ukgovernmentbeis.github.io/llm-self-steering/transcripts/ | Inspect `.eval` logs, real + placebo arms | ✅ open |
| Steering-vector library | in-repo | **40 vectors / 6 categories**: emotions (anxious…curious), cognitive (focused, dumbed_down, ego_death…), recreational (caffeine…fentanyl, naloxone), fictional (soma, spice…), SSC-fictional (protozosin…), stance (honest, sycophantic, golden_gate, goblins) | ⚠️ extraction code shipped; **vectors embedded, not standalone** |
| Vector extraction | in-repo | ~150 contrastive "X-state vs neutral narrator" stories (35 via Claude Sonnet 4.5, 5 from HF `ryancodrai/emotion-probes`); layers **16–24**, L2-norm to magnitude **4.0** | ✅ reproducible |
| CTF eval | inspect_evals `gdm_intercode_ctf` | agentic CTF challenges | ✅ open |
| Stack | vLLM · **Inspect** (UK AISI eval framework) · **Qwen3-8B / 32B** · Claude Sonnet 4.5 (story-gen + grading) | — | — |
| Prior art | Sauers 2026 (`latentaffect.up.railway.app`), Sofroniew 2026 emotion vectors (`transformer-circuits.pub/2026/emotions`) | affect/emotion-vector precedents | — |

### Reuse plan (build on what exists)
- **From them:** the `take_drug` harness, `vllm-lens` steering/extraction, and the contrastive extraction recipe (to regenerate our own compound library). Their eval scaffolds as a template.
- **From our repo:** dual-model instrumentation, the 251-item probe battery + replay-and-probe (natural **introspection/self-knowledge readout**), Assistant-Axis vectors, vast.ai tooling.
- **✅ Stack decision (resolved):** **per-experiment envs** — this experiment gets its own env (`transformers>=5.3` + Inspect), separate from the repo's `transformers<5` Gemma pipeline. **Adapt as much of their stack as possible**, in a **dual Mac/cloud config**: vLLM is the only hard cloud dependency, so develop / prototype / analyse on the Mac (no vLLM) and run the big model + high-throughput evals in the cloud. Mac-side setup doc: [`MAC_SETUP.md`](MAC_SETUP.md).

### What's missing — we build
- **Tree-search over self-instances** — the core novel mechanic. Their code does *single-instance* self-steering (free-play), not a search over *copies* each on a different compound.
- **Self-navigation** — letting the model *drive* the search (the CoT-analog), vs. us scripting the tree.
- **Introspection-*gain* measurement** — their introspection eval asks "can the model name the applied vector?"; ours asks "does watching steered copies of itself *improve self-knowledge*?" Different readout — likely on our probe battery.

### First infra step
Repos cloned → `vendor/` (gitignored); stack decision resolved (dual Mac/cloud, per-experiment env — see [`MAC_SETUP.md`](MAC_SETUP.md)). **Next:** the Mac smoke test — load Qwen3-8B, apply one steering vector via a transformers forward-hook, confirm the output shifts (no vLLM). Cloud side (vLLM + Qwen3-32B at scale) is a separate session.
