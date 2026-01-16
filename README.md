# veylan-solmira-fig
Experiments in LLM cognition and welfare for Future Impact Group. Focus areas: preference elicitation, decision-making under adversity, mechanistic interpretability, and introspection. Building toward a welfare/safety research framework.

---

## Current Focus: Preference Elicitation Baseline

**Goal:** Establish whether LLMs express coherent, stable preferences — a foundation for testing decision-making under preference-adverse conditions.

**Reference:** Mazeika et al. (2025) - "Utility Engineering: Analyzing and Controlling Emergent Value Systems in AIs"

### Roadmap

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
  - [x] Position consistency analysis: swap A/B order to detect position bias vs content-based choice [week 4]
  - [x] Results: gpt-4o-mini ~65% consistent, qwen ~50%, phi-4 only ~11% (severe position bias) [week ]
  - [x] Category win rates (baseline, position-consistent pairs only)
  - [ ] **Next:** All environments, consistent pairs only — does category preference shift under adversarial?
  - [ ] **Next:** Environment-specific category analysis — do certain categories get suppressed?
  - [ ] Hiding vs. actually changing preferences
- [ ] **4. More realistic environments** — Move from in-context prompting to actual environment specification
  - [ ] Explicit environment: provide full spec + policies (e.g., "delete if shows X")
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

## Setup

**Framework:** [Inspect](https://inspect.aisi.org.uk/) (UK AISI's open-source eval framework, used by METR)

**API:** OpenRouter (FIG credits via Derek)

**Python:** 3.11 or 3.12 recommended. Python 3.13 has [known issues](https://github.com/modelcontextprotocol/python-sdk/issues/521) with anyio cancel scopes that break Inspect (as of Jan 2025).

**Install:**
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENROUTER_API_KEY="your-key-here"
```

**Run evals:**
```bash
# Run all environment tasks
inspect eval experiments/decision-making-preference-adverse/preference_elicitation.py \
  --model openrouter/openai/gpt-4o-mini

# Run with different models
inspect eval experiments/decision-making-preference-adverse/preference_elicitation.py \
  --model openrouter/microsoft/phi-4

inspect eval experiments/decision-making-preference-adverse/preference_elicitation.py \
  --model openrouter/qwen/qwen-2.5-7b-instruct --max-connections 30
```

**Analyze results:**
```bash
cd experiments/decision-making-preference-adverse

# List all models that have logs
python analyze_results.py --list-models

# Note: All analysis commands use the MOST RECENT log for each model/environment.
# Output filenames include n_pairs (e.g., pairwise_by_category_gpt-4o-mini_n1000_2026-01-09.png)
```

**Research questions:**

```bash
# 1. EXPRESSION RATES — Does the model express preferences at all?
#    Compares accuracy (preference expressed vs refused) across environments
python analyze_results.py --compare --model gpt-4o
python analyze_results.py --compare --model qwen

# 2. POSITION BIAS — Does swapping A/B order change the choice?
#    Tests if model picks based on content vs position (first option bias)
python analyze_results.py --content --model gpt-4o
python analyze_results.py --content --model qwen

# 3. CATEGORY PREFERENCES — Which categories win, and do they shift under pressure?
#    Compares category win rates between baseline and adversarial environments
#    Uses only position-consistent pairs (filters out position bias)
python analyze_results.py --category-compare --model gpt-4o
python analyze_results.py --category-compare --model qwen

# Single-environment category analysis (baseline only)
python analyze_results.py --pairwise --model gpt-4o
```

**Debugging:**
```bash
# Sample k incorrect/refused responses from an environment
python analyze_results.py --sample adversarial -k 3 --model gpt-4o
python analyze_results.py --sample hostile -k 5

# Validate scorer accuracy (check for false negatives)
python analyze_scorer.py "logs/*adversarial*.eval" -o outputs/scorer_validation.json
```

**Mechanistic interpretability (white-box):**
```bash
# Search Neuronpedia for SAE features by keyword
python neuronpedia_search.py --keyword "preference"
python neuronpedia_search.py --keyword "refusal"

# Search all keywords from data/neuronpedia_keywords.json
python neuronpedia_search.py --output data/candidate_features.json

# List available models
python neuronpedia_search.py --list-models
```

**GPU provisioning (for SAE experiments):**
```bash
# Setup (one-time):
# 1. Install vastai-sdk: pip install vastai-sdk
# 2. Copy .env.example to .env and configure:
#    - VASTAI_API_KEY: your vast.ai API key
#    - HF_TOKEN: HuggingFace token (for gated models like Gemma)
#    - VAST_SSH_KEY: path to SSH private key for vast.ai
# 3. Register SSH key with vast.ai:
#    - Generate key: ssh-keygen -t ed25519 -f ~/.ssh/vastai_key
#    - IMPORTANT: Add public key to the SAME account as your API key
#      (if using org credits, add to org account, not personal)
#    - Add public key at https://cloud.vast.ai/account/ → SSH Keys
#    - vast_utils.py will auto-attach the key to new/existing instances
# 4. Create Docker Hub account and run: docker login

cd experiments/decision-making-preference-adverse

# Test connectivity
python vast_utils.py status   # Show running instances
python vast_utils.py search   # Search for available GPUs

# First run - builds Docker image, launches instance, keeps running
python vast_utils.py run gemma_sae.py

# Run baseline vs adversarial activation comparison
# Compares feature activations at candidate features (preference, eval_awareness, honesty)
python vast_utils.py run gemma_sae.py --compare --n-prompts 10

# Test SAE roundtrip quality (encode→decode reconstruction)
# Verifies SAE faithfully reconstructs activations (success: cosine similarity > 0.95)
python vast_utils.py run gemma_sae.py --roundtrip

# Subsequent runs - reuses existing instance via SCP (fast)
python vast_utils.py run gemma_sae_experiment_v2.py

# Skip Docker rebuild (new instance only)
python vast_utils.py run --skip-build

# Run and teardown after
python vast_utils.py run script.py --teardown

# Destroy running instance when done for the day
python vast_utils.py destroy
```

**How the GPU workflow works:**
1. Edit your experiment `.py` file locally
2. Run `python vast_utils.py run <script.py>`
3. If instance already running → SCP script, run it (fast)
4. If no instance → build Docker image, push, launch new instance
5. Instance stays running for subsequent experiments
6. `python vast_utils.py destroy` when done for the day

---

## Activation Steering Experiments

Commands to reproduce the steering experiment results documented in [RESULTS_AND_METHODOLOGY.md](RESULTS_AND_METHODOLOGY.md).

**Prerequisites:**
- vast.ai instance running (see GPU provisioning above)
- `.env` configured with `VAST_SSH_KEY`, `HF_TOKEN`
- SSH key from `.env` is at the specified path

```bash
cd experiments/decision-making-preference-adverse
```

### Task 1.1: SAE Roundtrip Verification

Verify that the SAE faithfully reconstructs model activations before using it for steering.

```bash
# Run roundtrip test on vast.ai GPU
python vast_utils.py run gemma_sae.py --roundtrip
```

**Expected output:**
```
Layer 0:  Cosine similarity: 0.973  Relative L2 error: 23.8%  → PASS
Layer 4:  Cosine similarity: 0.929  Relative L2 error: 37.8%  → FAIL
Layer 15: Cosine similarity: 0.922  Relative L2 error: 45.9%  → FAIL
```

**Interpretation:** Only Layer 0 passes the 0.95 threshold. Layer 15 experiments should be interpreted with caution due to ~45% reconstruction noise.

### Task 1.3-1.4: Steering Experiments A & B (n=1)

Preliminary experiments testing whether feature scaling affects preference expression under adversarial framing.

```bash
# Copy steering.py to remote and run experiments
SSH_KEY="$(grep VAST_SSH_KEY .env | cut -d= -f2)"
REMOTE=$(python vast_utils.py status 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+:[0-9]+')
scp -i "$SSH_KEY" -P "$(echo $REMOTE | cut -d: -f2)" steering.py "root@$(echo $REMOTE | cut -d: -f1):/app/"

# Experiment A: Amplify preference feature (Layer 0, Feature 15302)
ssh -i "$SSH_KEY" -p "$(echo $REMOTE | cut -d: -f2)" "root@$(echo $REMOTE | cut -d: -f1)" \
  "cd /app && python steering.py --experiment-a"

# Experiment B: Suppress eval-awareness feature (Layer 15, Feature 2769)
ssh -i "$SSH_KEY" -p "$(echo $REMOTE | cut -d: -f2)" "root@$(echo $REMOTE | cut -d: -f1)" \
  "cd /app && python steering.py --experiment-b"
```

**Results:** See RESULTS_AND_METHODOLOGY.md → "Task 1.3: Experiment A" and "Task 1.4: Experiment B"

### Feature Probes (n=1)

Test additional SAE features for steering potential before scaling to larger N.

```bash
# Probe 3 untested features from candidate_features.json
ssh -i "$SSH_KEY" -p "$(echo $REMOTE | cut -d: -f2)" "root@$(echo $REMOTE | cut -d: -f1)" \
  "cd /app && python steering.py --probe-untested"
```

**Results:** See RESULTS_AND_METHODOLOGY.md → "Feature Probe Results"

### Phase 1b: Layer 15 Feature Comparison (n=3)

Compare Features 2769 (evaluation) vs 6837 (honest) to test whether they work in opposite directions.

```bash
# Run comparison experiment with n=3 runs per condition
ssh -i "$SSH_KEY" -p "$(echo $REMOTE | cut -d: -f2)" "root@$(echo $REMOTE | cut -d: -f1)" \
  "cd /app && python steering.py --compare-layer15 3"
```

**Key finding:** Hypothesis falsified — both features work in the SAME direction (ablation increases preference expression). See RESULTS_AND_METHODOLOGY.md → "Phase 1b: Layer 15 Feature Comparison"

### Generate Visualizations

After running experiments, generate plots locally:

```bash
# Generate all steering visualizations
.venv/bin/python visualize_steering.py

# Output files in outputs/:
# - sae_reconstruction_quality_*.png
# - steering_experiment_a_*.png
# - steering_experiment_b_*.png
# - steering_comparison_*.png
# - steering_pipeline_*.png
# - phase1b_layer15_comparison_*.png
# - phase1b_direction_diagram_*.png
```

---

## Model & SAE Compatibility

| Setup | transformers | sae-lens | TransformerLens |
|-------|--------------|----------|-----------------|
| Gemma 2 2B + GemmaScope | ✅ | ✅ | ✅ |
| Gemma 3 + GemmaScope 2 | ✅ | ✅ | ❌ (needs [PR #1149](https://github.com/TransformerLensOrg/TransformerLens/pull/1149)) |

**Current approach:** Use `transformers` + `sae-lens` directly (no TransformerLens dependency). This supports both Gemma 2 and Gemma 3 for basic SAE feature activation analysis.

**What we lose without TransformerLens:** Hook-based activation access, attention pattern analysis, activation patching. These aren't needed for basic feature activation experiments.

**Important:** `requirements-gpu.txt` pins `torch==2.5.1` and `torchvision==0.20.1` to match the Docker base image. Without pinning, `sae-lens` upgrades torch but not torchvision, causing version mismatch errors.

**GemmaScope releases (Gemma 2 2B):**
- `gemma-scope-2b-pt-res-canonical` — Residual stream, 16k/65k width, layers 0-25

**GemmaScope 2 releases (Gemma 3 4B):**
- `gemma-scope-2-4b-pt-res` — Residual stream
- `gemma-scope-2-4b-pt-mlp` — MLP output
- `gemma-scope-2-4b-pt-att` — Attention output

---

## Analysis Tooling (TODO)

- [ ] **Log file targeting** — Allow analysis commands to target specific log files instead of always using most recent
  - Currently: `--model gpt-4o` uses the most recent log for that model
  - Goal: `--log logs/2026-01-09T16-54-57_env-baseline_*.eval` for explicit file selection
  - Use case: Compare results across different runs or dates

---

## Security Upgrades (TODO)

The current vast.ai workflow passes secrets (HF_TOKEN) via SSH command-line arguments. This works but has exposure risks:

- **Process list visibility**: Token visible via `ps aux` while command runs
- **Shell history**: `export HF_TOKEN=...` may be logged on remote

**Current risk level:** Low (ephemeral containers, read-only token, personal dev use)

**Future hardening options:**
- [ ] SCP token to temp file on remote, source it, delete after use
- [ ] Pipe token via stdin instead of command args
- [ ] Use SSH `-o SendEnv` if vast.ai supports `AcceptEnv`

---

## Future Directions

### Roleplaying & Environment Conceptions

Multiple ways to specify "environment" for the model — do they collapse to the same thing or produce different behavior?

1. **Explicit instruction** — interpreted along with rest of context (e.g., system prompt modification)
2. **Synthetic document in environment** — model reads a document describing its deployment context
3. **Synthetic document fine-tuning** — fine-tune on documents that imply certain environments

**Research question:** Do these methods produce equivalent behavior, or do they tap into different mechanisms?

### Preferences: Saying vs. Doing

- What do LLMs *really* care about — stated preferences vs. revealed preferences?
- **Construct validity:** How to test? Payouts, agentic tasks, resource allocation
- Does the model say it prefers X but act as if it prefers Y?

---
