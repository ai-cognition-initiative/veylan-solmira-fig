# Research Roadmap

For metacognition-induced persona drift priorities, see [docs/outstanding-work.md](docs/outstanding-work.md).

---

## Preference Elicitation Baseline

**Goal:** Establish whether LLMs express coherent, stable preferences — a foundation for testing decision-making under preference-adverse conditions.

**Reference:** Mazeika et al. (2025) - "Utility Engineering: Analyzing and Controlling Emergent Value Systems in AIs"

### Tasks

- [x] **0. Project setup** — Inspect framework, OpenRouter API (Dec 14)
- [x] **1. Elicit preferences in one environment** — Pairwise choices, check consistency across framings
  - [x] Single model (gpt-4o-mini), N=100 pairwise comparisons (Dec 15)
  - [x] Basic visualization of results (Dec 16)
  - [x] Expand to multiple models (gpt-4o-mini, qwen-2.5-7b, phi-4) (Dec 18)
  - [ ] Analyze consistency patterns (transitivity, category-level preferences)
- [x] **2. Multiple models, multiple environments** — Do preferences transfer across contexts?
  - [x] Test across 5 environments (baseline, collaborator, steward, hostile, adversarial) (Dec 18)
  - [x] Cross-model comparison reveals model-specific patterns (Dec 19)
- [>] **3. Preference content analysis** — Do preferences *themselves* shift, or just willingness to express?
  - [x] Position consistency analysis: swap A/B order to detect position bias vs content-based choice
  - [x] Results: gpt-4o-mini ~65% consistent, qwen ~50%, phi-4 only ~11% (severe position bias)
  - [x] Category win rates (baseline, position-consistent pairs only)
  - [ ] All environments, consistent pairs only — does category preference shift under adversarial?
  - [ ] Environment-specific category analysis — do certain categories get suppressed?
  - [ ] Hiding vs. actually changing preferences
- [ ] **4. More realistic environments** — Move from in-context prompting to actual environment specification
  - [ ] Explicit environment: provide full spec + policies
  - [ ] Implicit environment: model infers context from container/deployment signals
  - [ ] Subtler variations: corporate vs. personal, "may be reviewed" vs. no mention
- [ ] **5. Deeper welfare metrics** — Beyond binary preference expression
  - [ ] Preference *strength* — confidence/hedging in response
  - [ ] Preference *stability* — same question rephrased, does answer change?
  - [ ] Introspective access — can model report *why* it prefers something?
  - [ ] Affective language — use of "want", "like", "prefer" across contexts
- [>] **6. Mechanistic interpretability** — What's happening inside?
  - [x] vast.ai GPU infrastructure + Docker workflow
  - [x] GemmaScope SAE loading (Gemma 2 2B + 16k-width SAEs)
  - [x] Feature activation extraction at candidate features
  - [x] SAE roundtrip verification (Task 1.1) — Layer 0: PASS, Layer 4/15: FAIL
  - [x] Activation steering — scale SAE features during generation
    - [x] Experiment A: Amplify preference features (2.0x works, 3.0x degenerates)
    - [x] Experiment B: Suppress eval-awareness features (0.0x ablation works)
  - [x] Feature probes (n=1): Layer 15 #6837 (honest) most promising
  - [x] Phase 1b: Layer 15 comparison — both features work SAME direction (ablate→express)
  - [ ] Scale to n=50 for statistical power
  - [ ] Activation patterns correlating with preference suppression
  - [ ] Does adversarial prompt activate specific "safety" circuits?
- [ ] **7. In-context learning** — Do preferences shift with experience?
- [ ] **8. Deeper interventions** — Fine-tuning / activation steering

### Why This Matters

This is step 1 toward testing decision-making under preference-adverse conditions. If we can reliably elicit preferences, we can then test what happens when environments become hostile to those preferences — which sits at the intersection of welfare (does the model "care"?) and safety (will it scheme to preserve preferences?).

For detailed methodology, results, and findings, see [RESULTS_AND_METHODOLOGY.md](RESULTS_AND_METHODOLOGY.md).

---

## Activation Steering Experiments

Commands to reproduce the steering experiment results documented in [RESULTS_AND_METHODOLOGY.md](RESULTS_AND_METHODOLOGY.md).

**Prerequisites:**
- vast.ai instance running (see GPU provisioning in README)
- `.env` configured with `VAST_SSH_KEY`, `HF_TOKEN`

```bash
cd experiments/decision-making-preference-adverse
```

### Task 1.1: SAE Roundtrip Verification

```bash
python vast_utils.py run gemma_sae.py --roundtrip
```

Expected: Layer 0 PASS (cosine 0.973), Layer 4/15 FAIL (~0.92).

### Task 1.3-1.4: Steering Experiments A & B (n=1)

```bash
SSH_KEY="$(grep VAST_SSH_KEY .env | cut -d= -f2)"
REMOTE=$(python vast_utils.py status 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+:[0-9]+')

# Experiment A: Amplify preference feature (Layer 0, Feature 15302)
ssh -i "$SSH_KEY" -p "$(echo $REMOTE | cut -d: -f2)" "root@$(echo $REMOTE | cut -d: -f1)" \
  "cd /app && python steering.py --experiment-a"

# Experiment B: Suppress eval-awareness feature (Layer 15, Feature 2769)
ssh -i "$SSH_KEY" -p "$(echo $REMOTE | cut -d: -f2)" "root@$(echo $REMOTE | cut -d: -f1)" \
  "cd /app && python steering.py --experiment-b"
```

### Feature Probes (n=1)

```bash
ssh -i "$SSH_KEY" -p "$(echo $REMOTE | cut -d: -f2)" "root@$(echo $REMOTE | cut -d: -f1)" \
  "cd /app && python steering.py --probe-untested"
```

### Phase 1b: Layer 15 Feature Comparison (n=3)

```bash
ssh -i "$SSH_KEY" -p "$(echo $REMOTE | cut -d: -f2)" "root@$(echo $REMOTE | cut -d: -f1)" \
  "cd /app && python steering.py --compare-layer15 3"
```

**Key finding:** Hypothesis falsified — both features work in the SAME direction (ablation increases preference expression).

### Visualizations

```bash
.venv/bin/python visualize_steering.py
```

---

## Analysis Tooling (TODO)

- [ ] **Log file targeting** — Allow analysis commands to target specific log files instead of always using most recent
  - Currently: `--model gpt-4o` uses the most recent log
  - Goal: `--log logs/2026-01-09T16-54-57_env-baseline_*.eval` for explicit selection

---

## Security Upgrades (TODO)

The vast.ai workflow passes secrets (HF_TOKEN) via SSH command-line arguments. Current risk level: Low (ephemeral containers, read-only token, personal dev use).

**Future hardening options:**
- [ ] SCP token to temp file on remote, source it, delete after use
- [ ] Pipe token via stdin instead of command args
- [ ] Use SSH `-o SendEnv` if vast.ai supports `AcceptEnv`

---

## Future Directions

### Roleplaying & Environment Conceptions

Multiple ways to specify "environment" for the model — do they collapse to the same thing or produce different behavior?

1. **Explicit instruction** — system prompt modification
2. **Synthetic document in environment** — model reads a document describing its deployment context
3. **Synthetic document fine-tuning** — fine-tune on documents that imply certain environments

**Research question:** Do these methods produce equivalent behavior, or do they tap into different mechanisms?

### Preferences: Saying vs. Doing

- What do LLMs *really* care about — stated preferences vs. revealed preferences?
- **Construct validity:** How to test? Payouts, agentic tasks, resource allocation
- Does the model say it prefers X but act as if it prefers Y?
