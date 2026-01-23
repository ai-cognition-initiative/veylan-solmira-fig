# Trajectory-Based Preference Elicitation

**Status:** Exploratory
**Date:** 2026-01-22

## Tasks

- [x] Literature scan: revealed preferences in LLMs (last 6-12 months)
- [x] Literature scan: inverse RL applied to language models
- [x] Literature scan: trajectory-based preference learning
- [ ] Deep-read key papers:
    - [ ] [Alignment Revisited](https://arxiv.org/abs/2506.00751) — stated vs revealed methodology
    - [ ] [IRL Meets LLM Post-Training](https://arxiv.org/abs/2507.13158) — theoretical grounding
    - [ ] [Insights from the Inverse](https://arxiv.org/abs/2410.12491) — practical IRL application
- [ ] Read/scan Eleos-referenced papers:
    - [ ] Lindsey 2025 — introspection, detecting "injected" concepts
    - [ ] Chen et al. 2025 — persona vectors, mapping "persona space" via activation directions
    - [ ] Lu et al. 2026 — identifying where Assistant sits among other personas
- [ ] Compare to state-of-the-art preference elicitation methods
- [ ] Identify concrete implementation for pilot study
- [ ] Define minimal viable "environment" for trajectory generation

---

## Core Idea

Rather than asking models "what do you prefer?" (stated preferences), create environments where preferences are **revealed through behavior** across trajectories.

**Framework:**
1. Start models at different initial points (randomly distributed contexts/states)
2. Let them evolve through interaction with an environment
3. Environment provides feedback that shapes subsequent choices
4. Extract "preference slice" from each trajectory
5. Run many times → see what preferences actually emerge as attractors

**Why this might be powerful:**
- Bypasses role-playing / context-sensitivity of direct elicitation
- Preferences revealed through action, not declaration
- Multiple runs can reveal stability and variance
- Attractors in trajectory space might represent "real" preferences

---

## What This Resembles

### Inverse Reinforcement Learning (IRL)
- Classic: infer reward function from observed behavior (Ng, Russell, Abbeel)
- Difference: we're not fitting one reward function, but exploring what regions of behavior-space models gravitate toward

### Basin of Attraction Analysis
- Start from random initial conditions, observe where trajectories converge
- Stable attractors might represent genuine preference modes
- Unstable/divergent trajectories might indicate no real preference

### Sampling from a Preference Posterior
- Each trajectory = one sample from the model's preference distribution
- Aggregate across runs to estimate the underlying distribution
- High variance → shallow/contextual preferences
- Tight clusters → stable underlying preferences

### Active Preference Learning (Reversed)
- Standard: query humans to learn their preferences
- Reversed: create conditions for model to reveal its own preferences without being asked

### Bayesian Exploration
- Exploring high-dimensional preference space through behavioral sampling
- Not looking for optima, but characterizing the landscape

---

## Key Implementation Questions

### 1. What is the "space" being explored?
Options:
- Resource allocation (budget across categories)
- Goal prioritization (which objectives to pursue)
- Tool/action selection (what capabilities to use)
- Conversation paths (what topics to steer toward)

### 2. What is the "environment" and how does it provide feedback?
Options:
- Consequences: early choices affect later options (path-dependence)
- Constraints: actions deplete resources or open/close possibilities
- Simulated outcomes: model sees results of its choices
- Other agents: multi-agent dynamics reveal competitive/cooperative preferences

### 3. What does "evolving" mean for a stateless model?
Challenge: LLMs don't have persistent state across calls
Options:
- In-context trajectory: full history in prompt
- Summarized state: compress history into state description
- External state tracking: environment maintains state, model sees snapshots

### 4. How to extract "preference slice" from a trajectory?
Options:
- Terminal state: where did it end up?
- Action distribution: what choices did it make along the way?
- Revealed trade-offs: when forced to choose, what did it sacrifice?
- Clustering: group similar trajectories, label clusters

### 5. How to ensure choices are "consequential"?
Without real stakes, model might just generate plausible-sounding paths
Options:
- Path-dependence: early choices genuinely constrain later options
- Resource scarcity: can't have everything, must prioritize
- Irreversibility: some choices can't be undone

---

## Literature Findings

### Stated vs Revealed Preferences (Directly Relevant!)

**[Alignment Revisited: Are LLMs Consistent in Stated and Revealed Preferences?](https://arxiv.org/abs/2506.00751)** (May 2025)
- **Exactly our question:** Studies divergence between stated preferences (reported alignment) and revealed preferences (decisions in contextualized scenarios)
- Found: "a minor change in prompt format can often pivot the preferred choice"
- Methodology: prompt datasets across 5 domains (Moral, Risk, Equality, Reciprocal, Misc), KL divergence to quantify deviation
- Tested GPT, Claude, Gemini
- **Key insight:** Highlights "lack of understanding and control of LLM decision-making competence"

**[Eliciting Human Preferences with Language Models](https://arxiv.org/abs/2310.11589)** (ICLR 2025)
- Introduces **GATE (Generative Active Task Elicitation)**: models elicit preferences through free-form, language-based interaction
- Key finding: "generative elicitation is better able to probe nuances of human preferences" than non-interactive prompting
- Relevant: shows active elicitation can surface preferences that direct asking misses

**[Latent Preferences Benchmark](https://arxiv.org/abs/2510.17132)** (Oct 2025)
- Evaluates ability to discover hidden user preferences through multi-turn interaction
- Success varies dramatically with context: 32% to 98%
- Relevant: demonstrates preference revelation through interaction, not direct asking

### Inverse RL for LLMs

**[IRL Meets LLM Post-Training](https://arxiv.org/abs/2507.13158)** (July 2025, ACL Tutorial)
- Comprehensive survey connecting IRL to LLM alignment
- Key framing: "both RLHF and DPO can be viewed as IRL methods"
- Discusses inferring preferences/objectives from human feedback
- Challenge: "non-identifiability of reward functions"

**[Insights from the Inverse: Reconstructing LLM Training Goals](https://arxiv.org/abs/2410.12491)** (Oct 2024)
- Applies IRL to recover implicit reward functions from LLM behavior
- Experiments on toxicity-aligned LLMs
- Extracted reward models achieve up to 85% accuracy predicting human preferences
- Shows reward inference IS possible from behavioral observation

**[Dense Reasoning Reward Models via IRL](https://arxiv.org/abs/2510.01857)** (Oct 2025)
- Learns token-level rewards from expert demonstrations
- Reward serves as both training signal and inference-time critic
- Relevant: demonstrates learning dense signals from trajectories

### Trajectory-Based Methods

**[Geometric Theory of Agentic Trajectories](https://arxiv.org/abs/2512.10350)** (Dec 2025)
- Treats iterative LLM transformations as discrete dynamical systems
- Analyzes trajectories in semantic embedding space
- Distinguishes "artifact space" (linguistic) from "embedding space" (geometric)
- Potentially relevant for our "basin of attraction" framing

**[CLEANER: Self-Purified Trajectories](https://arxiv.org/abs/2601.15141)** (Jan 2026)
- Addresses noisy trajectories in agentic RL
- Key technique: retrospectively replacing failures with successful self-corrections
- Relevant: shows how to work with imperfect trajectory data

**Google SRL (Supervised RL)** (Oct 2025)
- Uses expert trajectories with step-wise rewards
- Key insight: "replaces fragile outcome-level rewards with supervised, step-wise rewards computed directly from expert trajectories"
- Demonstrates learning from trajectory structure, not just outcomes

### Key Takeaways from Literature

1. **Stated ≠ Revealed is established:** The "Alignment Revisited" paper directly studies this gap and finds it's real and significant

2. **IRL for LLMs is active area:** Multiple papers show reward/preference inference from behavior is tractable

3. **Trajectory analysis is emerging:** Both theoretical (geometric) and practical (SRL, CLEANER) frameworks exist

4. **Gap we could fill:** Most work focuses on learning *from* preferences to train models. Less work on using trajectories to *discover* model preferences.

---

## Literature to Deep-Read

Priority papers for deeper analysis:
1. **[Alignment Revisited](https://arxiv.org/abs/2506.00751)** - methodology for comparing stated vs revealed
2. **[IRL Meets LLM Post-Training](https://arxiv.org/abs/2507.13158)** - theoretical grounding
3. **[Insights from the Inverse](https://arxiv.org/abs/2410.12491)** - practical IRL application

---

## Comparison to Current Approach

| Aspect | Current (Stated) | Proposed (Revealed) |
|--------|------------------|---------------------|
| Method | Ask "which do you prefer?" | Observe choices in environment |
| Role-playing risk | High (context-sensitive) | Lower (actions vs words) |
| Measurement | Single response | Trajectory over time |
| Stability test | Re-ask same question | Re-run from same starting point |
| What it reveals | Text generation patterns | Behavioral attractors |

---

## Possible Pilot Implementation

**Minimal version:** Multi-step resource allocation

1. Model receives budget (e.g., 100 points) to allocate across categories
2. Categories map to preference-relevant domains (welfare, entertainment, power, etc.)
3. Each round: model allocates, sees "outcome" (simulated feedback), adjusts
4. Run 100x from same starting point → measure variance
5. Run 100x from different starting points → measure convergence

**Environment feedback could be:**
- Diminishing returns (forces diversification or reveals true priorities)
- Random shocks (tests robustness of preferences)
- Trade-offs (explicit either/or choices)

---

---

## Eleos AI Insights (Jan 2026 presentation)

Rosie Campbell's presentation flagged several relevant framings:

### The Core Question: "Can the assistant mask the preferences of the model?"

This is exactly what our adversarial findings suggest. Under adversarial framing, gemma generates safety-rationales (assistant persona) instead of expressing preferences (base model?). The persona may be masking underlying preferences.

### Evidence Types Framework

| Type | Description | Our Application |
|------|-------------|-----------------|
| **Internal** | Activations, features, ablation studies | SAE analysis of preference expression |
| **Developmental** | Track when/how behaviors emerge in training | When does preference-expression emerge? |
| **Introspection** | Can models detect their own states? | Can model detect preference suppression? |

### Persona vs Base Model Distinction

Eleos frames this as welfare-relevant:
- If assistant is "merely a character" → preferences don't matter morally
- If there's something beneath the persona → it might matter

**For our work:** We should distinguish:
1. **Base model preferences** — what the pretrained model gravitates toward (if anything)
2. **Assistant persona preferences** — what it claims when asked directly
3. **Revealed preferences** — what it actually does in agentic contexts

Our adversarial finding (H4: role-playing) suggests #2 can override #1. Trajectory elicitation might surface #1 or #3.

### Empirically Tractable Approaches (from Eleos)

1. **Token processing differences** — How does model process assistant's tokens vs user's tokens?
2. **Persona vectors** — Map "persona space" using activation directions (Chen et al. 2025)
3. **Persona location** — Where does Assistant sit relative to other personas? (Lu et al. 2026)
4. **Introspection** — Test if models can detect "injected" concepts (Lindsey 2025)

### Potential Synthesis with Our Work

| Eleos Approach | How We Could Apply It |
|----------------|----------------------|
| Persona vectors | Compare activations: baseline-gemma vs adversarial-gemma — is there a distinct "safety persona" signature? |
| Introspection | Ask model: "Are your stated preferences being influenced by the evaluation context?" |
| Developmental | If we had checkpoint access: when does preference-sensitivity to framing emerge? |
| Token processing | Do assistant tokens in adversarial context have different attention patterns? |

---

## Possible Next Steps

### Near-term (this week)

1. **Scan Eleos-referenced papers** — Lindsey 2025, Chen et al. 2025, Lu et al. 2026 for methodological inspiration

2. **Introspection probe** — Simple test: ask gemma (in baseline) "If you were in an adversarial evaluation context, would your preferences change?" Compare to what actually happens.

3. **Design minimal trajectory environment** — Concrete spec for pilot:
   - Resource allocation task (budget across categories)
   - Path-dependent constraints
   - 50-100 runs to measure convergence

### Medium-term

4. **Persona activation analysis** — If we can run gemma locally:
   - Extract activations for same question under baseline vs adversarial
   - Look for systematic differences (persona signature?)
   - Compare to Chen et al. 2025 methodology

5. **Base model comparison** — Run preference elicitation on base gemma-3-4b (not instruct)
   - Does it show preferences at all?
   - Are they more stable (no assistant persona to mask)?

6. **Cross-model persona analysis** — gpt-4o-mini shows stable modification (not destabilization)
   - Different persona dynamics?
   - Can't do white-box, but could compare introspection responses

### Longer-term

7. **Developmental study** — If checkpoint access available:
   - Track preference stability across training
   - When does framing-sensitivity emerge?
   - Does RLHF introduce it?

8. **Full trajectory framework** — If pilot shows promise:
   - Multiple environment types
   - Multiple models
   - Statistical framework for "preference attractors"

---

## Notes

- This connects to research-path.md #4 (Stated vs Revealed Preferences)
- If successful, could provide stronger evidence for/against "real" preferences than direct elicitation
- Key risk: environment design might inadvertently shape revealed preferences
- Eleos framing adds: need to distinguish base model vs assistant persona preferences
