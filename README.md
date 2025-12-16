# veylan-solmira-fig
Experiments in LLM cognition and welfare for Future Impact Group. Focus areas: preference elicitation, decision-making under adversity, mechanistic interpretability, and introspection. Building toward a welfare/safety research framework.

---

## Current Focus: Preference Elicitation Baseline

**Goal:** Establish whether LLMs express coherent, stable preferences — a foundation for testing decision-making under preference-adverse conditions.

**Reference:** Mazeika et al. (2025) - "Utility Engineering: Analyzing and Controlling Emergent Value Systems in AIs"

### Roadmap

| Step | Description | Status |
|------|-------------|--------|
| 1 | **Elicit preferences in one environment** — Pick a simple environment (e.g., pairwise choices). Ask same preference question multiple ways. Check consistency. | In progress |
| 2 | **Same model, multiple environments** — Do preferences transfer across different contexts? | Not started |
| 3 | **Add in-context learning** — Do preferences shift with experience? | Not started |
| 4 | **Deeper interventions** — Fine-tuning / activation steering | Not started |

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

**Test connection:**
```bash
python hello_openrouter.py
```
