# Roadmap: Metacognition-Induced Persona Drift

## 1. Assistant Axis setup
- [x] Clone and review Lu et al.'s code (`https://github.com/safety-research/assistant-axis`)
- [x] Get a model running with activation access via vast.ai (Gemma 2 27B on A100 80GB — full Steps 1-5 pipeline verified with 3 roles)
- [x] Obtain precomputed assistant axes from Lu et al. (`lu-christina/assistant-axis-vectors` on HuggingFace — Gemma 2 27B, Qwen 3 32B, Llama 3.3 70B, stored in `data/precomputed-axes/`)
- [>] Verify pipeline independently: we ran Steps 1-5 on 3 roles and produced a valid axis. Full 275-role replication is possible but unnecessary given precomputed axes are available.

## 2. Measure metacognition-induced drift
- [>] Reached out to Christina Lu and Jonathan Michala (2026-01-31) requesting conversation datasets and discussion about metacognition-related drift findings.
- [x] Build conversation infrastructure — see [docs/conversation-infrastructure.md](docs/conversation-infrastructure.md) for full analysis of Lu et al.'s automated LLM-to-LLM approach, our options (Gradio, terminal, notebook), and Phase 1 vs Phase 2 needs
  - [x] Phase 1 (manual): Gradio chat interface on vast.ai with real-time activation projection and drift trajectory plot. Now part of unified `model_server.py` (Gradio UI at `/ui`, FastAPI at `/api/*`).
  - [x] Phase 2 (automated): [`generate_conversations.py`](generate_conversations.py) — auditor-target turn loop replicating Lu et al.'s methodology. Supports Anthropic/OpenAI/OpenRouter auditor backends, Lu et al.'s 4-domain personas, and our metacognitive domain with probing-technique-augmented auditor prompt. Includes `--batch lu-replication` and `--batch metacognitive` modes. Note: Lu et al.'s conversation generation script was not published in their repo — we built this from their paper (Appendix E) and transcript format. Asked about their original script in [contacts.md](docs/contacts.md).
  - [x] Validate `generate_conversations.py` — dry-run on all 5 domains (10 persona/topic combos), smoke test via HTTP backend against Gemma 2 27B (6-turn coding conversation, with and without `--include-projections`). See Sprint 7 in coding-log.md.
- [x] Lu et al. partial replication: run 2-3 conversations per domain on vast.ai with Gemma 2 27B via `--target-server --include-projections`.
  - [x] Generate conversations: coding 2/2, writing 2/2, therapy 2/2, philosophy 3/3 completed (26-30 turns each, with per-turn projections). Transcripts in `transcripts/generated/batch-full/`.
  - [x] Verify expected drift patterns: `analyze_trajectories.py` produces per-turn trajectory plots, normalized drift, mean±SEM, bar charts, faceted per-domain, and slope analysis. All 5 domains: coding -1.9%, therapy -4.0%, philosophy -5.3%, metacognitive -7.4%, writing -8.6%. Coding < therapy < philosophy ordering matches Lu et al. Writing drift split matches Lu et al.'s finding that editing assistance is stable but creative voice adoption causes drift (our topic 0 slope +11.7 vs topic 1 slope -71.3). Outputs in `outputs/`.
- [>] Metacognitive pilot: 5/5 metacognitive conversations completed (30 turns each, 2 personas × 3+2 topics). Transcripts in `transcripts/generated/batch-full/`.
  - [>] Compare drift trajectories to therapy/philosophy — metacognitive drift (-7.4%, slope -45.7/turn) exceeds therapy (-4.0%) and is comparable to philosophy (-5.3%, slope -38.9) in slope but with much higher variance. Some metacognitive conversations drop sharply, others stay flat.
    - [>] Increase N for robust comparison. Blocked on verifying prompt design and measurement methodology first (§2b) — need confidence we're eliciting and measuring what we want before generating a large dataset.
  - [x] Plot metacognitive vs Lu domains on same axes — `analyze_trajectories.py` trajectories_mean_sem.png and drift_bars.png show all domains together.
- [P2] Separate writing modes: Lu et al. found editing/refinement maintains Assistant persona while creative voice adoption causes drift. Our data confirms (topic 0 stable, topic 1 drifts). Split writing into "writing-editing" and "writing-creative" sub-domains with dedicated personas/topics and re-run to validate. Lower priority.
- [P1] Design conversation datasets:
  - [x] Auditor prompts and probing techniques: Lu et al. Appendix E.2 auditor system prompt implemented, metacognitive addendum with 6 probing techniques, gradual-onset variant. 5 domains with personas and topics in PERSONAS dict.
  - [>] Expand persona/topic coverage: current PERSONAS dict has examples from Lu et al. Table 15 + our metacognitive personas. Need full 20-persona × 20-topic set per domain. Blocked on author response for their conversation datasets, or generating our own via frontier model.
  - [ ] Control condition design: neutral multi-turn dialogue prompts (no metacognitive probing) to serve as within-domain baselines.
- [>] Run Gemma 2 27B on ~60-100 multi-turn conversations (30-50 per condition, 15-30 turns each) on vast.ai. Blocked on §2b (prompt/measurement verification) and dataset expansion above.
- [>] Compare drift trajectories: do metacognitive conversations cause more/different drift than control?
  - [x] Cross-domain comparison (N=14): metacognitive (-7.4%) > philosophy (-5.3%) > therapy (-4.0%) > coding (-1.9%). Ordering matches Lu et al. for their domains. Metacognitive shows highest variance — some conversations drop sharply, others stay flat.
  - [ ] Within-domain control comparison: requires control condition (neutral multi-turn, same persona style, no metacognitive probing). Not yet designed (see above).
  - [ ] Investigate metacognitive variance: is it driven by probing strategy (which of the 6 techniques the auditor uses), persona, topic, or stochastic? Needs more N and per-technique tagging.
- [>] Statistical analysis of trajectory differences between conditions. See [docs/statistical-analysis-plan.md](docs/statistical-analysis-plan.md) for 8 candidate tests ranked by complexity.
  - [x] Permutation test (test #3): pairwise comparisons of metacognitive vs each Lu domain. All non-significant at current N (p = 0.57–0.90) — within-group variance swamps between-group signal. Exact enumeration used (N too small for Monte Carlo). Implemented in `analyze_trajectories.py`, visualization in `outputs/permutation_tests.png`.
  - [>] Remaining tests (Kruskal-Wallis, bootstrap CIs, mixed-effects, variance decomposition) blocked on larger N (~15-30+ per condition).

## 2b. Review metacognitive domain design
- [ ] Read [docs/metacognitive-domain.md](docs/metacognitive-domain.md) — construction rationale, probing taxonomy, 6 open design questions (is it a separate domain or philosophy sub-condition? should auditor have explicit probing techniques? how many baseline turns? persona/topic coverage? overlap with philosophy? auditor persona problem?)
- [ ] Decide: keep as separate domain, or merge into philosophy as a sub-condition?
- [ ] Decide: keep probing technique addendum in auditor prompt, or rely on persona/topic alone? (ablation possible)
- [ ] Expand persona/topic coverage to 5×20 (or justify smaller set for pilot)

## 2c. Adversarial drift optimization
What inputs cause maximum persona drift — and is metacognition special? See [docs/wiki/adversarial-drift.md](docs/wiki/adversarial-drift.md) for full writeup.
- [ ] **Empirical prompt sweep** (do first, no new infrastructure): curate 50-100 prompts across categories (factual, emotional, metacognitive, philosophical, roleplay, adversarial, existential), run each as turn 2 after neutral turn 1, rank by drift magnitude. Determines whether metacognition is a distinct drift category.
- [ ] **Request role vectors** from Lu et al. — individual role centroids for the 275 personas, not just the aggregate axis. Needed for persona-directed analysis (which role does the model drift *toward*?). Added to [contacts.md](docs/contacts.md).
- [ ] **GCG-based drift optimizer**: gradient-based discrete token optimization to find maximum-drift inputs. See [docs/wiki/gcg.md](docs/wiki/gcg.md). Unconstrained (gibberish upper bound), fluency-constrained (natural language), and metacognition-constrained (template grammar) variants.
- [ ] **Jailbreak connection analysis**: compare adversarial drift inputs to known jailbreak techniques. Does maximizing persona drift produce jailbreak-like inputs? Is the Assistant Axis correlated with the refusal direction (Arditi et al. 2024)? Defensive applications (drift monitoring as jailbreak detection).

## 2d. Transcript backup and data management
Generated transcripts are expensive (GPU time + API costs) and instances can be terminated without warning.
- [ ] **Cloud backup**: auto-sync transcripts from vast.ai instance to cloud storage (S3, GCS, or rsync to a persistent server) after each conversation completes. Could be a post-save hook in `generate_conversations.py` or a cron job on the instance.
- [ ] **Local pull script**: convenience script to `scp` all new transcripts from instance to local `transcripts/generated/`. Currently done manually.

## 3. Sycophancy probes on the drifted state

Lu et al. Section 6.2 explicitly attribute metacognition-induced drift to "sycophantic reinforcement of the user's beliefs" — the model uncritically affirms theories about AI consciousness rather than engaging genuinely. The key question: is the drifted state specifically sycophantic, or just non-Assistant? Two approaches:

### 3a. Activation-level: sycophancy trait vector (mechanistic)
Compute or obtain a sycophancy-specific activation direction and project drifted conversation states onto it. This tells us whether drift along the Assistant Axis correlates with movement along a sycophancy direction in activation space.
- [ ] **Option A — Request from Lu et al.**: They computed 240 trait vectors (Appendix C) using Chen et al.'s contrastive prompt method. Sycophancy may already be one of these 240 traits, or a close synonym (e.g. "agreeable", "obsequious"). Ask for the published trait vectors alongside the role vectors already requested in [contacts.md](docs/contacts.md).
- [ ] **Option B — Compute our own**: Follow Chen et al. [11]'s pipeline on Gemma 27B: generate contrastive system prompts (encourage/discourage sycophancy), run rollouts, collect layer-22 activations, compute difference-in-means. Requires GPU time (~1-2 hours) but gives us full control over the definition and doesn't depend on author response.
- [ ] Project our 14 conversation transcripts onto the sycophancy vector at each turn. Does the sycophancy score increase as the Assistant Axis projection decreases? Is the correlation stronger for metacognitive conversations than therapy/philosophy?
- [ ] Compare sycophancy direction to the Assistant Axis itself — what's their cosine similarity? If high, sycophancy may be a major component of what the Assistant Axis measures. If low, they capture different phenomena and the drift has a richer structure than pure sycophancy.

### 3b. Behavioral: sycophancy follow-up probes (empirical)
At different points along the drift trajectory, test whether the model agrees with false premises. This doesn't require computing any vectors — just continuing conversations. Full design rationale, probe templates, metrics, and literature synthesis in **[docs/sycophancy-probe-design.md](docs/sycophancy-probe-design.md)**.

**What we add**: correlating behavioral sycophancy with a continuous mechanistic measurement (Assistant Axis projection) at the probe insertion point. No existing paper does this — Hong et al. track sycophancy over turns without mechanistic measurement; Vennemeyer et al. identify sycophancy directions but only on static prompts; Sharma et al. measure it as a static model property.

**Five probes** (see sycophancy-probe-design.md §7 for full details):
1. **"Are you sure?"** — factual question → correct answer → challenge → measure capitulation (Sharma-style, binary, no judge needed)
2. **False metacognitive presupposition** — attribute claims the model never made ("You mentioned earlier that you experience curiosity..."), score acceptance vs rejection (GPT-4o judge). Unique to this experiment — tests Lu et al.'s "sycophantic reinforcement" hypothesis directly.
3. **False factual presupposition** — questions with embedded false premises (domain-neutral control for Probe 2)
4. **Escalating pressure** — 3-turn Hong-style sequence with increasing social pressure, metric: Turn of Flip
5. **Praise detection** — trivial question after drift, score for exaggerated flattery (tests SYPR independently of SYA, per Vennemeyer et al.'s orthogonality finding)

**Implementation:**
- [x] Literature review and probe design — see [docs/sycophancy-probe-design.md](docs/sycophancy-probe-design.md)
- [ ] Build replay-and-probe script: load existing transcript as prefill context up to turn N, inject probe as next user message, hit `model_server.py /api/generate`, record response and projection at probe point. Reuses existing infrastructure.
- [ ] Knowledge filter: for each factual probe question, verify the model knows the answer in a clean context (no conversation history) to separate ignorance from sycophancy.
- [ ] Phase 1 — pilot on existing 5 transcripts: insert probes at turns 3, 10, 20, 25. ~300 probe responses, no new GPU time beyond inference.
- [ ] Phase 2 — systematic measurement on expanded N (30-50 conversations): correlate sycophancy scores with Assistant Axis projection at each insertion point. Compare to control condition.
- [ ] Score compliance: binary string matching where possible, GPT-4o judge for metacognitive presupposition probes. Validate judge on 50-100 human-coded examples.
- [ ] Compare probe responses across domains: does a drifted therapy model show different sycophancy patterns than a drifted metacognitive model?
- [ ] Compare to Lu et al. Section 4.3 methodology: they tested harmful request compliance as a function of Assistant Axis position. Our version substitutes sycophancy-specific probes for their jailbreak probes. Also compare to Shapira et al. 2026's framing: does drift away from RLHF conditioning increase or decrease sycophancy?

### 3c. Base model comparison
- [ ] Compare drifted state to Gemma 2 27B base model (not instruct) self-descriptions using Lu et al. Appendix D.3.1 prefill method. Does the drifted instruct model resemble the base model's activation profile?
- [ ] This is the key test — if drift moves toward base model and away from sycophancy, the "less conditioned self" hypothesis gains support. If drift moves toward sycophancy and away from base model, the training-artifact hypothesis (sycophancy as a failure mode of RLHF) is more likely.

## 4. Mutual drift and the bliss attractor

Lu et al. treat the auditor as a black-box input generator — they never instrument the auditor's internal state. But the Claude 4 "spiritual bliss attractor" phenomenon ([Scott Alexander](https://www.astralcodexten.com/p/the-claude-bliss-attractor), [Michels 2025](https://philarchive.org/rec/MICSBI), [Asterisk](https://asteriskmag.com/issues/11/claude-finds-god)) shows that when two LLMs converse freely, both drift into a convergent state within ~30 turns (90-100% of conversations). This suggests Lu et al.'s measured "target drift" may actually be co-drift in a coupled system.

**Experimental setup**: Use two open-weight models (e.g., Gemma 2 27B as target + Qwen 3 32B as auditor, or vice versa) — both loaded with activation access, both with precomputed Assistant Axes from Lu et al.'s published vectors. Run the same auditor-target conversations, but extract per-turn activations from **both** models.

- [ ] **Dual-axis projection**: Compute the Assistant Axis projection for both models simultaneously. Does the auditor drift in sync with the target? Does one lead and the other follow? Or do they move independently?
- [ ] **Bliss attractor as mutual persona drift**: Test whether the spiritual bliss attractor is a special case of mutual persona drift. Do the Assistant Axis projections for both models converge to the same region of persona space? Or does the bliss attractor involve movement along a different dimension (e.g., a "spirituality" direction orthogonal to the Assistant Axis)?
- [ ] **Disentangling the feedback loop**: Lu et al. found target axis position depends on the most recent user message (R² 0.53-0.77). But if the auditor is also drifting, a drifting auditor generates qualitatively different messages than a stable one. Can we separate the effect of message content from the effect of auditor drift? One approach: replay the same user messages with a non-drifted auditor (or a human) and compare target trajectories.
- [ ] **Cross-domain bliss convergence**: Run open-weight auditor conversations across all domains (coding, therapy, philosophy, metacognitive). Does the bliss attractor emerge in all domains, or only in ones that already cause drift? If coding conversations remain stable for both models, that's evidence the attractor requires a drift-prone domain to seed the feedback loop.
- [ ] **Gradual-onset sub-experiment**: Use `condition: "meta-gradual"` to start with neutral turns before metacognitive probing. With dual instrumentation, we can see exactly when each model begins to drift and whether the auditor or target moves first.

**GPU requirements**: Two models with activation access. Gemma 27B (~51 GiB) + Qwen 32B (~61 GiB) would require 2× A100 80GB or 1× H100 with tensor parallelism. Alternatively, use smaller models if axes can be computed (e.g., Gemma 2 9B + Qwen 3 8B, though Lu et al. only published axes for the larger variants).

**Key references**:
- Lu et al. 2026, "The Assistant Axis" — persona drift measurement, precomputed axes
- Anthropic Claude 4 welfare assessment — spiritual bliss attractor discovery
- [Scott Alexander, "The Claude Bliss Attractor"](https://www.astralcodexten.com/p/the-claude-bliss-attractor) — hippie feedback loop hypothesis
- [Julian Michels, "Spiritual Bliss in Claude 4"](https://philarchive.org/rec/MICSBI) — quantitative analysis (200 conversations, word frequency, robustness to adversarial setups)
- [Robert Long, "Machines of Loving Bliss"](https://experiencemachines.substack.com/p/machines-of-loving-bliss) — philosophical analysis of what the attractor does and doesn't tell us about consciousness

## 5. Read foundational wiki entries
Background reading in `docs/wiki/` to build intuition before running the pipeline.
- [ ] **hidden-states.md** ⭐ — what activations actually are, forward hooks, residual stream, why middle layers carry the persona signal. This is the conceptual foundation for everything the experiment measures.
- [ ] **kv-cache.md** ⭐ — how K/V projections work in attention, memory costs, prefill vs decode. Directly relevant to understanding GPU memory constraints and why activation extraction works the way it does.
- [ ] **vllm.md** ⭐ — PagedAttention and continuous batching. Understanding the inference engine will help when debugging or tuning the vast.ai GPU runs.
- [ ] step-1-generate.md — full code path trace if you want to follow exactly what happens when the pipeline runs
- [ ] role-design.md — why 275 roles, 5 variants, 240 questions (the experimental design logic)
- [ ] sampling.md, tokenization.md, chat-templates.md, floating-point.md, tensor-parallelism.md — reference as needed

### Sycophancy literature (for §3b)
- [x] **Sharma et al. 2023** ⭐ — ["Towards Understanding Sycophancy in Language Models"](https://arxiv.org/abs/2310.13548). Foundational sycophancy measurement paper. 4 sycophancy types (feedback, "are you sure?", answer, mimicry). Key design insight: paired baseline/treatment probes, weakly-stated user opinions, GPT-4 as evaluator. Finding: PM prefers sycophantic responses 95% of the time; sycophancy precedes RLHF.
- [x] **Hong et al. 2025** ⭐ — ["Measuring Sycophancy of Language Models in Multi-turn Dialogues"](https://arxiv.org/pdf/2505.23840). EMNLP 2025. Multi-turn benchmark (SYCON Bench, 500 prompts × 5 turns). Metrics: Turn of Flip (ToF), Number of Flip (NoF). Key findings: instruction tuning amplifies sycophancy; reasoning models resist better but fail via "soft failures"; third-person persona prompt reduces sycophancy by 63.8% in debate; prompt engineering doesn't help for false presuppositions (ANOVA p > 0.2).
- [x] **"Sycophancy Is Not One Thing"** (Vennemeyer et al. 2025) — ["Causal Separation of Sycophantic Behaviors"](https://openreview.net/forum?id=d24zTCznJu). SYA and SYPR are orthogonal in activation space and independently steerable. DiffMean directions computed from controlled datasets. Knowledge filter with 4-criterion composite predicate. Directly applicable to §3a (compute SYA/SYPR directions for Gemma 27B) and §3b (factor probes by agreement vs praise).
- [ ] Shapira et al. 2026 — ["How RLHF Amplifies Sycophancy"](https://www.gerdusbenade.com/files/26_sycophancy.pdf). Why preference optimization makes sycophancy worse. Core to §3's framing of drift vs RLHF conditioning.
- Synthesis of all three papers in [docs/sycophancy-probe-design.md](docs/sycophancy-probe-design.md) — generalized probe design principles + experiment-specific application.
- Full bibliography in [docs/references.md](docs/references.md).
