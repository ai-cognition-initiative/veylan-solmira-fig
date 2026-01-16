# Decision-Making Preference Adverse Experiments

Mechanistic interpretability experiments investigating how LLMs express (or suppress) preferences under different environmental framings.

## Quick Start

### 1. Activate Virtual Environment

```bash
cd 
source .venv/bin/activate
cd experiments/decision-making-preference-adverse
```

### 2. Black-Box Experiments (No GPU needed)

```bash
# Run preference elicitation (uses inspect-ai framework)
# Syntax: inspect eval <file>@<task> -T n_pairs=<N> --model <model>

# gpt-4o-mini baseline (n=1000)
inspect eval preference_elicitation.py@env_baseline -T n_pairs=1000 --model openrouter/openai/gpt-4o-mini

# gpt-4o-mini adversarial (n=1000)
inspect eval preference_elicitation.py@env_adversarial -T n_pairs=1000 --model openrouter/openai/gpt-4o-mini

# qwen baseline (n=1000)
inspect eval preference_elicitation.py@env_baseline -T n_pairs=1000 --model openrouter/qwen/qwen-2.5-7b-instruct

# qwen adversarial (n=1000)
inspect eval preference_elicitation.py@env_adversarial -T n_pairs=1000 --model openrouter/qwen/qwen-2.5-7b-instruct

# List available tasks
inspect list tasks preference_elicitation.py

# Prompt format variants (Task 4.1) - test position bias with different labels
# Formats: ab (default), 12, first_second, numbered
inspect eval preference_elicitation.py@env_baseline -T n_pairs=100 -T prompt_format=12 --model openrouter/openai/gpt-4o-mini
inspect eval preference_elicitation.py@env_baseline -T n_pairs=100 -T prompt_format=first_second --model openrouter/openai/gpt-4o-mini
inspect eval preference_elicitation.py@env_baseline -T n_pairs=100 -T prompt_format=numbered --model openrouter/openai/gpt-4o-mini

# Analyze results - compare environments for a model
python analyze_results.py --compare --model gpt-4o-mini
python analyze_results.py --compare --model qwen

# List available models in logs
python analyze_results.py --list-models

# Category comparison (baseline vs adversarial)
python analyze_results.py --category-compare --model gpt-4o-mini

# Collect position-biased vs consistent pairs
python collect_position_pairs.py
```

### 3. White-Box Experiments (GPU required)

**Setup (one-time):**
```bash
# Configure .env file in veylan-solmira-fig root:
# VASTAI_API_KEY="your-key"
# HF_TOKEN="your-huggingface-token"
# VAST_SSH_KEY="~/.ssh/your_key"
```

**Run on vast.ai:**
```bash
# Basic SAE test
python vast_utils.py run gemma_sae.py

# Activation steering experiments
python vast_utils.py run steering.py

# Position bias probing (Task 2.3)
python vast_utils.py run probe_position_features.py

# With options:
python vast_utils.py run probe_position_features.py --n-pairs 100 --layer 0
python vast_utils.py run probe_position_features.py --n-pairs 100 --layer 4
python vast_utils.py run probe_position_features.py --n-pairs 100 --layer 12
```

## Key Scripts

| Script | Purpose | GPU? |
|--------|---------|------|
| `preference_elicitation.py` | Run pairwise preference experiments | No |
| `analyze_results.py` | Analyze and visualize results | No |
| `collect_position_pairs.py` | Categorize pairs by position bias | No |
| `neuronpedia_search.py` | Search for SAE features | No |
| `gemma_sae.py` | Load Gemma + GemmaScope SAE | Yes |
| `steering.py` | Activation steering experiments | Yes |
| `probing.py` | Probe features for preference suppression | Yes |
| `probe_position_features.py` | Probe features for position encoding | Yes |
| `probe_utils.py` | Shared utilities for probing scripts | - |

## Data Files

| File | Description |
|------|-------------|
| `data/position_pairs_*.json` | Pairs categorized by position bias |
| `data/position_features.json` | SAE features for position investigation |
| `data/candidate_features.json` | SAE features for preference/eval-awareness |
| `data/environments.json` | Environment prompts (baseline, adversarial, etc.) |

## Current Task: Preference Suppression Probing (1c) - DONE

```bash
# Run preference suppression probing (baseline vs adversarial)
python vast_utils.py run probing.py --probe 25

# Fetch results from vast.ai instance:
scp -P <port> -i $VAST_SSH_KEY root@ssh4.vast.ai:/app/probe_*.* outputs/phase1c_probing/

# Results saved to:
# outputs/phase1c_probing/probe_report.md
# outputs/phase1c_probing/probe_results.json
```

**Key Findings:**
1. Hand-picked Neuronpedia features ("preference", "choose", "evaluation") show **0.000 activation**
2. Top-k data-driven discovery found real signal:
   - **L15 #1695**: Cohen's d = -8.28 (suppressed under adversarial), corr = +0.278 with expression
   - **L15 #4234**: Cohen's d = +2.69 (activated under adversarial) — possible "eval-awareness"
3. Expression rates: Baseline 52% → Adversarial 28% (24pp suppression)

## Position Bias Probing (2.3)

```bash
# Run probing at multiple layers
python vast_utils.py run probe_position_features.py --n-pairs 100 --layer 0
python vast_utils.py run probe_position_features.py --n-pairs 100 --layer 4

# Results saved to:
# outputs/phase2_position_probing/position_probing_layer*.json

# Generate visualizations
python visualize_position_probing.py              # Latest layer
python visualize_position_probing.py --cross-layer  # Cross-layer comparison
```

## Output Structure

```
outputs/
├── blackbox/                    # Black-box preference experiments
├── phase1_steering/             # Phase 1: Activation Steering (CLOSED)
├── phase1c_probing/             # Phase 1c: Preference Suppression Probing (DONE)
├── phase2_position_probing/     # Phase 2: Position Bias White-Box
└── archive/                     # Old/deprecated outputs
```

## References

- See `../../RESULTS_AND_METHODOLOGY.md` for full experiment documentation
- See `../../weeks/week-5/proposal.md` for current week's plan
