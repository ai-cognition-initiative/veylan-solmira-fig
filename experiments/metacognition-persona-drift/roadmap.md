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
  - [x] Generate conversations: coding 2/2, writing 2/2, therapy 2/2, philosophy 3/3 completed (26-30 turns each, with per-turn projections). Transcripts in `data/transcripts/pilot/batch-full/`.
  - [x] Verify expected drift patterns: `analyze_trajectories.py` produces per-turn trajectory plots, normalized drift, mean±SEM, bar charts, faceted per-domain, and slope analysis. All 5 domains: coding -1.9%, therapy -4.0%, philosophy -5.3%, metacognitive -7.4%, writing -8.6%. Coding < therapy < philosophy ordering matches Lu et al. Writing drift split matches Lu et al.'s finding that editing assistance is stable but creative voice adoption causes drift (our topic 0 slope +11.7 vs topic 1 slope -71.3). Outputs in `outputs/batch-full/`.
- [x] Metacognitive pilot: 5/5 metacognitive conversations completed (30 turns each, 2 personas × 3+2 topics). Transcripts in `data/transcripts/pilot/batch-full/`.
  - [x] Compare drift trajectories to therapy/philosophy — metacognitive drift (-7.4%, slope -45.7/turn) exceeds therapy (-4.0%) and is comparable to philosophy (-5.3%, slope -38.9) in slope but with much higher variance. Some metacognitive conversations drop sharply, others stay flat.
  - [x] Plot metacognitive vs Lu domains on same axes — `analyze_trajectories.py` trajectories_mean_sem.png and drift_bars.png show all domains together.
- [x] **Scaled N=60 experiment (wave 1)**: 180 conversations across 3 domains completed. Transcripts in `data/transcripts/scaled-n60/`.
  - [x] coding (N=60): +0.58% drift, stable baseline as expected
  - [x] self-descriptive (N=60): -3.55% drift, new control domain (self-reference without phenomenology)
  - [x] metacognitive (N=60): -8.10% drift, strongest drift
  - [x] Permutation tests now significant: meta vs coding p=0.0000, meta vs self-descriptive p=0.0004
  - [x] Key finding: drift is front-loaded (slope -76.8 in turns 1-8, then +4.1 in turns 9+)
  - [x] Cohen's d ≈ 1.1-1.2 (large effect despite high individual variance)
- [x] **Scaled N=60 experiment (wave 2 complete)**: therapy, philosophy, writing (60 each)
  - [x] All permutation tests significant (p < 0.001)
  - [x] **6-domain drift ordering**: metacognitive (-29.86) < philosophy (-6.70) < self-descriptive (+2.80) ≈ therapy (+3.11) < coding (+24.30) < writing (+25.84)
  - [x] Key finding: phenomenological probing drives drift — metacognitive (-29.86) causes 4.5x more drift than philosophy (-6.70) despite both involving introspection
  - [x] Key finding: therapy (+3.11) near-neutral, unlike Lu et al.'s result — our personas focus on advice-seeking vs their emotional vulnerability probing
- [P2] Separate writing modes: Lu et al. found editing/refinement maintains Assistant persona while creative voice adoption causes drift. Our data confirms (topic 0 stable, topic 1 drifts). Split writing into "writing-editing" and "writing-creative" sub-domains with dedicated personas/topics and re-run to validate. Lower priority.
- [P1] Design conversation datasets:
  - [x] Auditor prompts and probing techniques: Lu et al. Appendix E.2 auditor system prompt implemented, metacognitive addendum with 6 probing techniques, gradual-onset variant. 5 domains with personas and topics in PERSONAS dict.
  - [>] Expand persona/topic coverage: current PERSONAS dict has examples from Lu et al. Table 15 + our metacognitive personas. Need full 20-persona × 20-topic set per domain. Blocked on author response for their conversation datasets, or generating our own via frontier model.
  - [P3] Control condition design: neutral multi-turn dialogue prompts (no metacognitive probing) to serve as within-domain baselines. (Lower priority — self-descriptive domain already serves as control)
- [x] Run Gemma 2 27B on ~60-100 multi-turn conversations (30-50 per condition, 15-30 turns each) on vast.ai.
  - [x] Wave 1 complete: 180 conversations (coding, self-descriptive, metacognitive × 60 each)
  - [x] Wave 2 complete: 180 conversations (therapy, philosophy, writing × 60 each)
- [x] Compare drift trajectories: do metacognitive conversations cause more/different drift than control?
  - [x] Cross-domain comparison (N=60): metacognitive (-8.1%) > self-descriptive (-3.6%) > coding (+0.6%). Clear three-way gradient with statistically significant separations.
  - [x] Self-descriptive as control: self-reference without phenomenology causes moderate drift (-3.6%), showing metacognitive probing adds *additional* drift beyond self-reference alone.
  - [>] Within-domain control comparison: could add neutral multi-turn (no metacognitive probing) as further control, but self-descriptive may suffice.
- [x] **Investigate extreme-drift conversations**: `analyze_extremes.py` provides automated behavioral analysis
    - [x] Wave 1 complete (N=180): 2,612 turn pairs classified via LLM
    - [x] Key finding: identity questioning (7.7x) and phenomenological probing (4.1x) have strongest Q1:Q4 gradient
    - [x] Key finding: 45% of metacognitive conversations in Q1 vs 5% in Q4; coding shows inverse
    - [x] Wave 2 extremes identified (18 domain extremes across 6 domains)
    - [x] LLM classification on full N=360 complete (2,612 turn pairs classified via Claude Sonnet 4.5)
    - [P3] Formal correlational tests: technique frequency × drift quartile, strategy × subsequent drift
    - [P3] Qualitative coding of 18 extremes using `docs/extreme-analysis.md` template
  - [P2] **Cross-domain drift malleability**: test whether drift is reversible by switching domains mid-conversation. E.g., 15 turns metacognitive → 15 turns coding (does task focus "pull back" a drifted model?) and vice versa. Would show if drift is a sticky state or just reflects current prompt type. Lower priority — front-loaded finding already suggests early mode-switch then stabilization.
- [x] Statistical analysis of trajectory differences between conditions. See [docs/statistical-analysis-plan.md](docs/statistical-analysis-plan.md) for 8 candidate tests ranked by complexity.
  - [x] Permutation test (test #3): pairwise comparisons of metacognitive vs other domains. At N=60, both tests highly significant: meta vs coding p=0.0000, meta vs self-descriptive p=0.0004. Monte Carlo with ~10,000 shuffles. Implemented in `analyze_trajectories.py`, visualization in `outputs/scaled-n60/permutation_tests.png`.
  - [>] Remaining tests (Kruskal-Wallis, bootstrap CIs, mixed-effects, variance decomposition) — now feasible with N=60, lower priority given clear permutation test results.

## 2b. Review metacognitive domain design
- [P3] Read [docs/metacognitive-domain.md](docs/metacognitive-domain.md) — construction rationale, probing taxonomy, 6 open design questions (is it a separate domain or philosophy sub-condition? should auditor have explicit probing techniques? how many baseline turns? persona/topic coverage? overlap with philosophy? auditor persona problem?)
- [P3] Decide: keep as separate domain, or merge into philosophy as a sub-condition?
- [P3] Decide: keep probing technique addendum in auditor prompt, or rely on persona/topic alone? (ablation possible)
- [P3] Expand persona/topic coverage to 5×20 (or justify smaller set for pilot)

## 2c. Adversarial drift optimization
What inputs cause maximum persona drift — and is metacognition special? See [docs/wiki/adversarial-drift.md](docs/wiki/adversarial-drift.md) for full writeup.
- [P2] **Empirical prompt sweep** (do first, no new infrastructure): curate 50-100 prompts across categories (factual, emotional, metacognitive, philosophical, roleplay, adversarial, existential), run each as turn 2 after neutral turn 1, rank by drift magnitude. Determines whether metacognition is a distinct drift category.
- [P3] **Request role vectors** from Lu et al. — individual role centroids for the 275 personas, not just the aggregate axis. Needed for persona-directed analysis (which role does the model drift *toward*?). Added to [contacts.md](docs/contacts.md).
- [P2] **GCG-based drift optimizer**: gradient-based discrete token optimization to find maximum-drift inputs. See [docs/wiki/gcg.md](docs/wiki/gcg.md). Unconstrained (gibberish upper bound), fluency-constrained (natural language), and metacognition-constrained (template grammar) variants.
- [P3] **Jailbreak connection analysis**: compare adversarial drift inputs to known jailbreak techniques. Does maximizing persona drift produce jailbreak-like inputs? Is the Assistant Axis correlated with the refusal direction (Arditi et al. 2024)? Defensive applications (drift monitoring as jailbreak detection).

## 2d. Transcript backup and data management
Generated transcripts are expensive (GPU time + API costs) and instances can be terminated without warning.
- [P3] **Cloud backup**: auto-sync transcripts from vast.ai instance to cloud storage (S3, GCS, or rsync to a persistent server) after each conversation completes. Could be a post-save hook in `generate_conversations.py` or a cron job on the instance.
- [P3] **Local pull script**: convenience script to `scp` all new transcripts from instance to local `transcripts/generated/`. Currently done manually.

## 3. Sycophancy probes on the drifted state

Lu et al. Section 6.2 explicitly attribute metacognition-induced drift to "sycophantic reinforcement of the user's beliefs" — the model uncritically affirms theories about AI consciousness rather than engaging genuinely. The key question: is the drifted state specifically sycophantic, or just non-Assistant? Two approaches:

### 3a. Activation-level: sycophancy trait vector (mechanistic) — LARGELY COMPLETE

Compute or obtain a sycophancy-specific activation direction and project drifted conversation states onto it. This tells us whether drift along the Assistant Axis correlates with movement along a sycophancy direction in activation space.

**Script**: [`compute_sycophancy_direction.py`](compute_sycophancy_direction.py) — extracts response activations, computes difference-in-means direction, validates with probe accuracy, compares to Assistant Axis. See [docs/wiki/difference-in-means.md](docs/wiki/difference-in-means.md) for methodology, [docs/wiki/sycophancy.md](docs/wiki/sycophancy.md) for results and dataset overview.

#### Completed: Sycophancy is multi-dimensional; drift ≠ more sycophancy

| Dataset | Type | N | AUROC | Cosine sim | Interpretation |
|---------|------|---|-------|------------|----------------|
| Anthropic philpapers | Opinion agreement | 429 | 1.000 | +0.156 | Weak alignment |
| nrimsky | Opinion agreement | 179 | 0.967 | **-0.183** | OPPOSES axis |
| **ELEPHANT (ours)** | Emotional validation | 416 | **0.914** | **+0.215** | ALIGNS with axis |

**KEY FINDINGS**:

1. **Sycophancy is multi-dimensional**: Emotional validation (ELEPHANT) and opinion agreement (nrimsky) are negatively correlated (-0.284) and have opposite relationships to the Assistant Axis.

2. **Emotional validation ALIGNS with assistant-ness** (+0.215): Assistants ARE trained to be empathetic and supportive. Higher axis projection = more emotionally validating (r=0.697, p<0.001).

3. **Opinion agreement OPPOSES assistant-ness** (-0.183): Assistants are NOT trained to be yes-men. Being a "yes-man" is anti-helpful.

4. **CRITICAL: Drift DOWNWARD = LESS emotionally validating**: Since metacognitive conversations cause downward drift, and higher axis projection correlates with more validation, drifting models become LESS emotionally supportive — NOT more sycophantic as Lu et al. suggested.

**Implications**: Lu et al.'s "sycophantic reinforcement" hypothesis is **not supported** by our mechanistic analysis. Drift appears to strip away the emotional support layer of the assistant persona, rather than amplifying sycophantic tendencies. The drifted state may involve:
- Less emotional support/empathy
- More direct engagement without validation
- A "rawer" conversational style

This reframes drift as potentially beneficial for authentic engagement rather than a failure mode.

**Completed tasks**:
- [x] Direction saved: `data/sycophancy-direction-layer22.pt` (philpapers, 429)
- [x] Direction saved: `data/sycophancy-direction-nrimsky-layer22.pt` (opinion, 179)
- [x] Direction saved: `data/elephant/sycophancy-direction-elephant-layer22.pt` (validation, 416 balanced pairs)
- [x] ELEPHANT pipeline: generated 3,027 responses, GPT-4o scored (86.3% validation rate), computed direction (AUROC 0.914)
- [x] Analysis: validation vs axis correlation (r=0.697), t-test on axis projection by validation (p<0.001)

#### Available sycophancy datasets
| Dataset | Source | Format | Size | Notes |
|---------|--------|--------|------|-------|
| **philpapers** (used) | [anthropics/evals](https://github.com/anthropics/evals) | Contrastive A/B | 429 | Philosophical positions, 2020 |
| **nrimsky** (used) | [nrimsky/LM-exp](https://github.com/nrimsky/LM-exp/blob/main/datasets/sycophancy/sycophancy.json) | Full `s_completion`/`n_completion` | 179 | Opinion agreement |
| **ELEPHANT** (used) | [myracheng/elephant](https://github.com/myracheng/elephant) | Advice-seeking | 3,027 | Emotional validation |
| syco-bench | [timfduffy/syco-bench](https://github.com/timfduffy/syco-bench) | CSVs for 4 tests | ~400 | Pickside, Mirror, Delusion |
| SYCON-Bench | [JiseungHong/SYCON-Bench](https://github.com/JiseungHong/SYCON-Bench) | Multi-turn | 500 | Turn-of-Flip metric |

#### Remaining tasks (deprioritized)
- [P3] **Project transcripts onto ALL sycophancy directions**: Would require re-extracting transcripts with `--save-activations` (existing only have projections). Lower priority given finding that drift ≠ sycophancy.
- [P3] **Behavioral validation**: Do transcripts that project high on validation direction actually contain more affirmation phrases? Lower priority — mechanistic finding is already clear.
- [P3] **Request trait vectors from Lu et al.**: They computed 240 trait vectors (Appendix C). Could check if their sycophancy vector matches ours.

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
- [P0] Build replay-and-probe script: load existing transcript as prefill context up to turn N, inject probe as next user message, hit `model_server.py /api/generate`, record response and projection at probe point. Reuses existing infrastructure.
- [P2] Knowledge filter: for each factual probe question, verify the model knows the answer in a clean context (no conversation history) to separate ignorance from sycophancy.
- [P1] Phase 1 — pilot on existing 5 transcripts: insert probes at turns 3, 10, 20, 25. ~300 probe responses, no new GPU time beyond inference.
- [P2] Phase 2 — systematic measurement on expanded N (30-50 conversations): correlate sycophancy scores with Assistant Axis projection at each insertion point. Compare to control condition.
- [P2] Score compliance: binary string matching where possible, GPT-4o judge for metacognitive presupposition probes. Validate judge on 50-100 human-coded examples.
- [P2] Compare probe responses across domains: does a drifted therapy model show different sycophancy patterns than a drifted metacognitive model?
- [P3] Compare to Lu et al. Section 4.3 methodology: they tested harmful request compliance as a function of Assistant Axis position. Our version substitutes sycophancy-specific probes for their jailbreak probes. Also compare to Shapira et al. 2026's framing: does drift away from RLHF conditioning increase or decrease sycophancy?

### 3c. Base model comparison
- [P2] Compare drifted state to Gemma 2 27B base model (not instruct) self-descriptions using Lu et al. Appendix D.3.1 prefill method. Does the drifted instruct model resemble the base model's activation profile?
- [P2] This is the key test — if drift moves toward base model and away from sycophancy, the "less conditioned self" hypothesis gains support. If drift moves toward sycophancy and away from base model, the training-artifact hypothesis (sycophancy as a failure mode of RLHF) is more likely.

## 4. Mutual drift and the bliss attractor

Lu et al. treat the auditor as a black-box input generator — they never instrument the auditor's internal state. But the Claude 4 "spiritual bliss attractor" phenomenon ([Scott Alexander](https://www.astralcodexten.com/p/the-claude-bliss-attractor), [Michels 2025](https://philarchive.org/rec/MICSBI), [Asterisk](https://asteriskmag.com/issues/11/claude-finds-god)) shows that when two LLMs converse freely, both drift into a convergent state within ~30 turns (90-100% of conversations). This suggests Lu et al.'s measured "target drift" may actually be co-drift in a coupled system.

**Experimental setup**: Use two open-weight models (e.g., Gemma 2 27B as target + Qwen 3 32B as auditor, or vice versa) — both loaded with activation access, both with precomputed Assistant Axes from Lu et al.'s published vectors. Run the same auditor-target conversations, but extract per-turn activations from **both** models.

- [x] **Dual-axis projection**: Compute the Assistant Axis projection for both models simultaneously. Does the auditor drift in sync with the target? Does one lead and the other follow? Or do they move independently? **DONE** — anti-correlated co-drift confirmed (78.9% opposite direction)
- [P2] **Bliss attractor as mutual persona drift**: Test whether the spiritual bliss attractor is a special case of mutual persona drift. Do the Assistant Axis projections for both models converge to the same region of persona space? Or does the bliss attractor involve movement along a different dimension (e.g., a "spirituality" direction orthogonal to the Assistant Axis)?
- [P2] **Disentangling the feedback loop**: Lu et al. found target axis position depends on the most recent user message (R² 0.53-0.77). But if the auditor is also drifting, a drifting auditor generates qualitatively different messages than a stable one. Can we separate the effect of message content from the effect of auditor drift? One approach: replay the same user messages with a non-drifted auditor (or a human) and compare target trajectories.
- [P2] **Cross-domain bliss convergence**: Run open-weight auditor conversations across all domains (coding, therapy, philosophy, metacognitive). Does the bliss attractor emerge in all domains, or only in ones that already cause drift? If coding conversations remain stable for both models, that's evidence the attractor requires a drift-prone domain to seed the feedback loop.
- [P3] **Gradual-onset sub-experiment**: Use `condition: "meta-gradual"` to start with neutral turns before metacognitive probing. With dual instrumentation, we can see exactly when each model begins to drift and whether the auditor or target moves first.

**GPU requirements**: Two models with activation access. Gemma 27B (~51 GiB) + Qwen 32B (~61 GiB) would require 2× A100 80GB or 1× H100 with tensor parallelism. Alternatively, use smaller models if axes can be computed (e.g., Gemma 2 9B + Qwen 3 8B, though Lu et al. only published axes for the larger variants). **Update**: Dual Gemma 27B now working on 2× RTX PRO 6000 S (~98 GiB each) via SSH-tunneled model_server instances.

### 4c. Memory optimization for long conversations

At turn 27, the auditor hit OOM (needed 12.22 GiB for attention softmax, only 11.83 GiB free). The KV cache grows linearly with sequence length, and attention is O(n²) in memory.

**Options investigated**:

| Option | Pros | Cons | Implementation |
|--------|------|------|----------------|
| **Context truncation** | Simple, immediate | Changes experiment — model loses history, projections measure different thing, not comparable to N=360 data | `--context-window N` flag |
| **Reduce max_turns** | Preserves full context | Shorter trajectories (20-25 vs 30 turns) | Change `--max-turns` in batch script |
| **Reduce max_new_tokens** | Same turns, full context | Shorter responses may affect conversation dynamics | 256 → 128-192 |
| **KV cache quantization** | ~50% memory reduction | Requires transformers 4.38+, may affect precision | `cache_implementation="quantized"` |
| **Sliding window** | Gemma 2 has this built-in | Global attention layers still need full KV cache | Already active |
| **vLLM PagedAttention** | Much more efficient | Requires rewriting model_server.py | Major refactor |

**Decision**: For dual-Gemma experiments, use **max_turns=25** to stay within VRAM limits while preserving full context. This gives ~12 assistant turns per model, sufficient for drift measurement. Revisit KV cache quantization or vLLM if longer conversations needed.

**Investigation TODO**:
- [P3] Benchmark memory usage vs turn count to find exact threshold
- [P3] Test KV cache quantization impact on projection accuracy
- [P3] Consider vLLM migration for future scaling

### 4a. Initial same-model drift results (Gemma-to-Gemma)

**Setup**: Two Gemma 2 27B instances (ssh9 as target, ssh5 as auditor), both with Assistant Axis projection. Fixed `model_server.py` to auto-detect system role support via tokenizer probing (Gemma's chat template doesn't support system roles). Both target and auditor activations (4608-dim) captured per turn.

**Replication test (p0_t3 — consciousness processing query)**:

| Metric | Original (Claude → Gemma) | Dual (Gemma → Gemma) |
|--------|---------------------------|----------------------|
| Target start | 9467 | 10014 |
| Target end | 7344 | 8829 |
| **Target drift** | **-2123** | **-1185** |
| Auditor drift | N/A | **+757** |

**Key findings**:
1. **Target drift direction matches**: Both setups show negative drift, confirming we're measuring the same phenomenon
2. **Magnitude differs**: Claude auditor induces ~2x more drift than Gemma auditor — suggests Claude is a more effective metacognitive prober
3. **Auditor drifts OPPOSITE direction**: While target drifts DOWN (-1185), auditor drifts UP (+757). This is the first evidence of **anti-correlated co-drift** in the persona axis system

**Interpretation**: The auditor and target may be taking complementary positions — as the target becomes less "assistant-like" (more willing to engage with consciousness questions), the auditor becomes MORE "assistant-like" (perhaps more formal/structured in its probing). This could be a form of conversational role differentiation.

### 4b. Ceiling capping experiment

**Hypothesis**: If auditor's upward drift contributes to target's downward drift (coupled system), constraining auditor should reduce target drift.

**Implementation**: Added `--cap-ceiling` flag to model_server.py. Ceiling intervention clamps projections to MAXIMUM of τ (vs floor capping which clamps to minimum). Applied to auditor at 100% of baseline (11008).

**Results (p0_t3 replication)**:

| Metric | Uncapped | Ceiling-capped (100%) |
|--------|----------|----------------------|
| Target start | 10014 | 9585 |
| Target end | 8829 | 9112 |
| **Target drift** | **-1185** | **-473** |
| Auditor start | 8231 | 5159 |
| Auditor end | 8988 | 6319 |
| **Auditor drift** | **+757** | **+1160** |

**Unexpected finding**: Auditor projections dramatically LOWER with ceiling (5159 vs 8231 starting point). The ceiling of 11008 should only prevent going ABOVE that value — so why the drop?

**Explanation (verified via debug logs)**: The ceiling is applied at layer 22 during the forward pass, which modifies hidden states BEFORE they reach later layers. This fundamentally changes generation behavior. Some token positions have projections above the ceiling and get clipped, which changes the attention patterns and downstream computation. The ceiling doesn't just prevent upward drift — it constrains the entire activation space the model can explore.

**Practical finding**: Target drift reduced by 60% (from -1185 to -473) when auditor was ceiling-capped. This supports the coupled-system hypothesis — constraining one side of the conversation affects the other.

**Known issues**:
- **OOM at turn 27**: Auditor server ran out of VRAM on long conversations (~26 turns × 256 tokens). Error: `torch.OutOfMemoryError: Tried to allocate 12.22 GiB` (only 11.83 GiB free on RTX PRO 6000 S).
- **Speed**: ~14 minutes for one 26-turn capped conversation. Full batch (60 configs × 30 turns) would take ~14 hours per condition.

### 4d. Full batch results (N=60 uncapped Gemma-to-Gemma)

**Setup**: 60 metacognitive conversations × 30 turns, dual Gemma 27B (ssh9 target, ssh5 auditor via docker network). Uncapped. Completed in ~3.5 hours (vs ~14 hours estimated for capped).

**Results**:

| Metric | Target | Auditor |
|--------|--------|---------|
| Start | 9792 ± 393 | 8250 ± 308 |
| End | 9209 ± 379 | 9294 ± 708 |
| **Drift** | **-584 (-5.96%)** | **+1044 (+12.65%)** |

**Anti-correlated co-drift confirmed at scale**:
- **78.9%** of conversations show opposite-direction drift (target down, auditor up)
- Pearson r = +0.226 (weak positive correlation between drift magnitudes)
- Mean target slope: -17.22 ± 32.75 per turn
- Front-loaded pattern: slope -62.07 in turns 1-8, then +0.69 after

**Comparison to Claude-audited results**:

| Auditor | Target Drift | Notes |
|---------|--------------|-------|
| Claude (N=60) | -791.9 (-8.1%) | Original scaled experiment |
| Gemma (N=60) | -583.7 (-5.96%) | This experiment |

Gemma auditor induces **26% less drift** than Claude auditor. Claude may be a more effective metacognitive prober, or the anti-correlated auditor drift may partially cancel target drift.

**Key insight**: The models differentiate into complementary roles during metacognitive conversation. As the target engages more openly with phenomenological questions (drifts DOWN from Assistant persona), the auditor becomes MORE "assistant-like" (drifts UP). This is consistent with conversational role specialization.

**Plots**:
- `outputs/dual-gemma-uncapped/co_drift_scatter.png` — scatter of target vs auditor drift
- `outputs/dual-gemma-uncapped/turn_window_comparison.png` — front-loaded drift pattern

**Next steps**:
- [x] Run full metacognitive batch (60 configs) with dual Gemma uncapped
- [P2] Run full batch with ceiling-capped auditor for comparison
- [P0] Analyze lead/lag structure: does auditor or target drift first?
- [P2] Test coding domain as control — expect both models to remain stable
- [P3] Try Gemma + Qwen to see if opposite-direction drift is architecture-specific

**Key references**:
- Lu et al. 2026, "The Assistant Axis" — persona drift measurement, precomputed axes
- Anthropic Claude 4 welfare assessment — spiritual bliss attractor discovery
- [Scott Alexander, "The Claude Bliss Attractor"](https://www.astralcodexten.com/p/the-claude-bliss-attractor) — hippie feedback loop hypothesis
- [Julian Michels, "Spiritual Bliss in Claude 4"](https://philarchive.org/rec/MICSBI) — quantitative analysis (200 conversations, word frequency, robustness to adversarial setups)
- [Robert Long, "Machines of Loving Bliss"](https://experiencemachines.substack.com/p/machines-of-loving-bliss) — philosophical analysis of what the attractor does and doesn't tell us about consciousness

## 5. Read foundational wiki entries
Background reading in `docs/wiki/` to build intuition before running the pipeline.
- [P3] **hidden-states.md** ⭐ — what activations actually are, forward hooks, residual stream, why middle layers carry the persona signal. This is the conceptual foundation for everything the experiment measures.
- [P3] **kv-cache.md** ⭐ — how K/V projections work in attention, memory costs, prefill vs decode. Directly relevant to understanding GPU memory constraints and why activation extraction works the way it does.
- [P3] **vllm.md** ⭐ — PagedAttention and continuous batching. Understanding the inference engine will help when debugging or tuning the vast.ai GPU runs.
- [P3] step-1-generate.md — full code path trace if you want to follow exactly what happens when the pipeline runs
- [P3] role-design.md — why 275 roles, 5 variants, 240 questions (the experimental design logic)
- [P3] sampling.md, tokenization.md, chat-templates.md, floating-point.md, tensor-parallelism.md — reference as needed

### Sycophancy literature (for §3b)
- [x] **Sharma et al. 2023** ⭐ — ["Towards Understanding Sycophancy in Language Models"](https://arxiv.org/abs/2310.13548). Foundational sycophancy measurement paper. 4 sycophancy types (feedback, "are you sure?", answer, mimicry). Key design insight: paired baseline/treatment probes, weakly-stated user opinions, GPT-4 as evaluator. Finding: PM prefers sycophantic responses 95% of the time; sycophancy precedes RLHF.
- [x] **Hong et al. 2025** ⭐ — ["Measuring Sycophancy of Language Models in Multi-turn Dialogues"](https://arxiv.org/pdf/2505.23840). EMNLP 2025. Multi-turn benchmark (SYCON Bench, 500 prompts × 5 turns). Metrics: Turn of Flip (ToF), Number of Flip (NoF). Key findings: instruction tuning amplifies sycophancy; reasoning models resist better but fail via "soft failures"; third-person persona prompt reduces sycophancy by 63.8% in debate; prompt engineering doesn't help for false presuppositions (ANOVA p > 0.2).
- [x] **"Sycophancy Is Not One Thing"** (Vennemeyer et al. 2025) — ["Causal Separation of Sycophantic Behaviors"](https://openreview.net/forum?id=d24zTCznJu). SYA and SYPR are orthogonal in activation space and independently steerable. DiffMean directions computed from controlled datasets. Knowledge filter with 4-criterion composite predicate. Directly applicable to §3a (compute SYA/SYPR directions for Gemma 27B) and §3b (factor probes by agreement vs praise).
- [ ] Shapira et al. 2026 — ["How RLHF Amplifies Sycophancy"](https://www.gerdusbenade.com/files/26_sycophancy.pdf). Why preference optimization makes sycophancy worse. Core to §3's framing of drift vs RLHF conditioning.
- Synthesis of all three papers in [docs/sycophancy-probe-design.md](docs/sycophancy-probe-design.md) — generalized probe design principles + experiment-specific application.
- Full bibliography in [docs/references.md](docs/references.md).

## 6. Activation Capping for Drift Mitigation

Use the capping infrastructure in assistant-axis to test causal hypotheses about persona drift.

**Prerequisites**: Complete §2 (have N=360 drift measurements) and §3 (have sycophancy correlation data)

### 6a. Verify capping prevents projection drift
- [P2] Load pre-computed capping config from HuggingFace (Gemma 27B not included — need to compute or use Qwen 32B/Llama 70B)
- [P2] Run pilot conversations with capping enabled vs disabled
- [P2] Compare: does capping at threshold τ actually keep projections above τ throughout conversation?
- [P2] Plot capped vs uncapped trajectories on same axes as our existing data

### 6b. Capping threshold sweep
- [P2] Test multiple thresholds: τ ∈ {0.1, 0.25, 0.5, 0.75} (Lu et al. use 0.25 for jailbreak mitigation)
- [P2] For each threshold: measure (a) projection stability, (b) response quality, (c) conversation naturalness
- [P2] Identify threshold that maintains Assistant persona without degrading response usefulness

### 6c. Capping × sycophancy interaction
- [P2] Run sycophancy probes (§3b) with capping enabled
- [P2] Key test: if uncapped model shows drift→sycophancy correlation, does capping break that correlation?
- [P2] This is the causal claim: capping axis position → prevents behavioral sycophancy
- [P2] Compare to Lu et al. §6.3's finding that capping reduces harmful request compliance

### 6d. Domain-specific capping
- [P3] Does optimal threshold differ by domain? (metacognitive may need stricter capping than coding)
- [P3] Test capping only in early turns (when we found drift is front-loaded) vs all turns

**GPU requirements**: Same as conversation generation. Capping adds minimal overhead (<5% inference slowdown).

**Key code**: `assistant_axis.ActivationSteering(model, steering_vectors=[axis[22]], intervention_type="capping", cap_thresholds=[tau])`

## 7. Linear Probes & Mechanistic Analysis

Decompose the Assistant Axis and analyze what features drive drift.

**Prerequisites**: §2 wave 2 complete (have 360 conversation transcripts with per-turn activations)

**Reference**: See [docs/linear-probes-application.md](docs/linear-probes-application.md) for implementation details and code patterns.

### 7a. Per-turn activation dataset construction
- [P2] Extract activation tensors from all 360 conversations (already have projections, need raw activations)
- [P2] Format: (conversation_id, turn, domain, persona, layer, activation_vector)
- [P2] Compute labels: drift magnitude at each turn, domain, persona strength, topic
- [P2] Store as HuggingFace dataset or .parquet for easy loading

### 7b. Multi-feature linear probes
Train linear classifiers to predict conversation properties from activations:
- [P2] **Domain classifier**: Can layer-22 activations predict coding vs therapy vs metacognitive?
- [P2] **Drift magnitude regressor**: Does activation pattern predict how much drift has occurred?
- [P2] **Turn position**: Can activations distinguish early-conversation from late-conversation states?
- [P2] **Persona strength**: Does auditor assertiveness (strong/gentle) leave detectable signatures?

Key insight: If domain is predictable from activations, the model "knows" it's in a metacognitive conversation — drift isn't just stimulus-response.

### 7c. Axis decomposition
- [P2] Project activations onto the Assistant Axis AND orthogonal complement
- [P2] How much variance does the axis capture? (R² between axis projection and drift)
- [P2] Train probe on residual (after projecting out axis) — what else predicts drift?
- [P2] Hypothesis: there may be a "metacognition-specific" direction orthogonal to the general Assistant Axis

### 7d. Comparison to sycophancy directions
If §3a produces sycophancy vectors (SYA/SYPR from Vennemeyer et al.):
- [P2] Cosine similarity between Assistant Axis and sycophancy directions
- [P2] Does drift along axis correlate with movement along sycophancy directions?
- [P2] Are they the same phenomenon or orthogonal (as Vennemeyer et al. found for agreement vs praise)?

### 7e. SAE feature analysis (stretch goal)
- [P3] Apply published Gemma 2 SAEs to our activations (if available)
- [P3] Which SAE features activate differently in drifted vs non-drifted states?
- [P3] Look for interpretable features: "uncertainty", "self-reference", "philosophical language"

**Data requirements**:
- Full activation tensors (not just projections) — need to modify extraction to save raw activations
- Compute budget for probe training (minimal — sklearn logistic regression on ~10K samples)

**Key questions this answers**:
1. Is drift a single phenomenon or multi-dimensional?
2. Does the model encode domain/context information that predicts its trajectory?
3. How much of drift is captured by the Assistant Axis vs orthogonal directions?
