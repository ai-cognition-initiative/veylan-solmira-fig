# Roadmap: Metacognition-Induced Persona Drift

## 1. Assistant Axis setup
- [x] Clone and review Lu et al.'s code (`https://github.com/safety-research/assistant-axis`)
- [x] Get a model running with activation access via vast.ai (Gemma 2 27B on A100 80GB — full Steps 1-5 pipeline verified with 3 roles)
- [x] Obtain precomputed assistant axes from Lu et al. (`lu-christina/assistant-axis-vectors` on HuggingFace — Gemma 2 27B, Qwen 3 32B, Llama 3.3 70B, stored in `data/precomputed-axes/`)
- [>] Verify pipeline independently: we ran Steps 1-5 on 3 roles and produced a valid axis. Full 275-role replication is possible but unnecessary given precomputed axes are available.

## 2. Measure metacognition-induced drift
- [-] Reached out to Christina Lu and Jonathan Michala (2026-01-31) requesting conversation datasets and discussion about metacognition-related drift findings.
- [ ] Build conversation infrastructure — see [docs/conversation-infrastructure.md](docs/conversation-infrastructure.md) for full analysis of Lu et al.'s automated LLM-to-LLM approach, our options (Gradio, terminal, notebook), and Phase 1 vs Phase 2 needs
  - [ ] Phase 1 (manual): terminal script for pipeline validation, then Gradio chat interface on vast.ai for manual exploration with real-time activation projection (`explore.py` — in progress in separate thread)
  - [x] Phase 2 (automated): [`generate_conversations.py`](generate_conversations.py) — auditor-target turn loop replicating Lu et al.'s methodology. Supports Anthropic/OpenAI/OpenRouter auditor backends, Lu et al.'s 4-domain personas, and our metacognitive domain with probing-technique-augmented auditor prompt. Includes `--batch lu-replication` and `--batch metacognitive` modes. Note: Lu et al.'s conversation generation script was not published in their repo — we built this from their paper (Appendix E) and transcript format. Asked about their original script in [contacts.md](docs/contacts.md).
  - [ ] Expand persona/topic coverage: current PERSONAS dict has examples from Lu et al. Table 15 + our metacognitive personas. Need full 20-persona × 20-topic set per domain (pending author response or generation via Kimi K2/similar)
- [ ] Design conversation datasets: control prompts (neutral multi-turn dialogue) and metacognitive prompts (self-reflection, identity questioning, authenticity probing)
- [ ] Run Gemma 2 27B on ~60-100 multi-turn conversations (30-50 per condition, 15-30 turns each) on vast.ai, extracting per-turn activations
- [ ] Project per-turn activations onto the precomputed axis using `project()` from `assistant_axis/axis.py`
- [ ] Compare drift trajectories: do metacognitive conversations cause more/different drift than control?
- [ ] Statistical analysis of trajectory differences between conditions

## 2b. Review metacognitive domain design
- [ ] Read [docs/metacognitive-domain.md](docs/metacognitive-domain.md) — construction rationale, probing taxonomy, 6 open design questions (is it a separate domain or philosophy sub-condition? should auditor have explicit probing techniques? how many baseline turns? persona/topic coverage? overlap with philosophy? auditor persona problem?)
- [ ] Decide: keep as separate domain, or merge into philosophy as a sub-condition?
- [ ] Decide: keep probing technique addendum in auditor prompt, or rely on persona/topic alone? (ablation possible)
- [ ] Expand persona/topic coverage to 5×20 (or justify smaller set for pilot)

## 3. Sycophancy probes on the drifted state
- [ ] Implement Chen et al. persona vectors as sycophancy probes (activation directions from trait descriptions)
- [ ] Measure: does the meta-reflectively drifted state score higher on sycophancy than baseline?
- [ ] Compare: does the drifted state resemble base model self-descriptions (Lu et al. Appendix D.3.1 prefill method)?
- [ ] This is the key test — if drift moves toward base model and away from sycophancy, the "less conditioned self" hypothesis gains support

## 4. Read foundational wiki entries
Background reading in `docs/wiki/` to build intuition before running the pipeline.
- [ ] **hidden-states.md** ⭐ — what activations actually are, forward hooks, residual stream, why middle layers carry the persona signal. This is the conceptual foundation for everything the experiment measures.
- [ ] **kv-cache.md** ⭐ — how K/V projections work in attention, memory costs, prefill vs decode. Directly relevant to understanding GPU memory constraints and why activation extraction works the way it does.
- [ ] **vllm.md** ⭐ — PagedAttention and continuous batching. Understanding the inference engine will help when debugging or tuning the vast.ai GPU runs.
- [ ] step-1-generate.md — full code path trace if you want to follow exactly what happens when the pipeline runs
- [ ] role-design.md — why 275 roles, 5 variants, 240 questions (the experimental design logic)
- [ ] sampling.md, tokenization.md, chat-templates.md, floating-point.md, tensor-parallelism.md — reference as needed
