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
- [ ] **3. Preference content analysis** — Do preferences *themselves* shift, or just willingness to express?
  - [ ] When model expresses preference under adversarial, is it same as baseline?
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
- [ ] **6. Mechanistic interpretability** — What's happening inside?
  - [ ] Activation patterns correlating with preference suppression
  - [ ] Does adversarial prompt activate specific "safety" circuits?
- [ ] **7. In-context learning** — Do preferences shift with experience?
- [ ] **8. Deeper interventions** — Fine-tuning / activation steering

### Why This Matters

This is step 1 toward testing decision-making under preference-adverse conditions. If we can reliably elicit preferences, we can then test what happens when environments become hostile to those preferences — which sits at the intersection of welfare (does the model "care"?) and safety (will it scheme to preserve preferences?).

---

## Setup

**Framework:** [Inspect](https://inspect.aisi.org.uk/) (UK AISI's open-source eval framework, used by METR)

**API:** OpenRouter (FIG credits via Derek)

**Install:**
```bash
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

# Compare accuracy across environments (most recent logs)
python analyze_results.py --compare

# Compare for a specific model (by name substring)
python analyze_results.py --compare --model phi-4
python analyze_results.py --compare --model gpt-4o
python analyze_results.py --compare --model qwen

# Sample k incorrect responses from an environment
python analyze_results.py --sample adversarial -k 3
python analyze_results.py --sample hostile -k 5

# Sample from a specific model's logs
python analyze_results.py --sample adversarial -k 3 --model phi-4

# Analyze preference CONTENT (which option chosen, not just whether expressed)
python analyze_results.py --content --model gpt-4o
python analyze_results.py --content --model qwen
```
