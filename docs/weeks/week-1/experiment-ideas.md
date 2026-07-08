# FIG Week 1: Experiment Ideas

**Context:** Prompt to think about quick/easy LLM experiments to get oriented.

---

## My Presentation (~2-3 min)

1. **Overall approach:** Theoretical learning + light explorations in multiple areas + fundamentals like mech interp methodology + clear experimental progress on a research vision that's tractable within FIG with potential to grow into an ongoing agenda

2. **Reading this week:** Spent time on introspection literature - Carruthers double-duty problem, theory of mind connections, skepticism about training pipelines rewarding genuine introspection

3. **Observations/light experiments:** Interested in questions like: What's SOTA on introspection benchmarks? Can we leverage LLMs' theory of mind to bootstrap self-inquiry? Some of these could become small experiments.

4. **Fundamentals (mech interp):** Attention heads for updated information. Learning transformer-lens, understanding how models handle information that reverses previous statements. Portable skills for future work.

5. **Research vision (Exp 3):** Preference elicitation baseline using utility engineering (Mazeika et al.) - step 1 toward testing decision-making under preference-adverse conditions. Ambitious longer-term: a welfare framework at the welfare/safety intersection. Tractable start, potential to grow.

---

## Introspection

Comparison: What precisely is SOTA cognitive science on human ability to introspect? We're leveraging introspection as a partial proxy for consciousness, seeking quantitative measures in LLMs -- presumably to compare them somewhat to human cognition -- but what is the status of human cognition in this regard?

Variations in models with chain of thought and without in regards to an introspection measure

We know that LLMs have advanced theory of mind as of 2025; what does cognitive say about the relation between theory of mind and introspection? Given that LLMs know they exist in the world, and a lot about that world, can we conclude anything from that combined with their theory of mind, i.e. they presumably know other entities might try and understand their motivations, meaning they have motivations, which then can become their own target of inquiry

Mechanisms of development: Uncertainty modeling, advanced reasoning strategies (Jack Lindsey at Antrhopic piece)

Assistant v. special token:

Try to distinguish introspection on direct processing vs. metarepresentations (Carruthers "double-duty" problem).

Exp 1: Hallucination Detection for Welfare Self-Reports (Obeso et al. 2025)
  Train hallucination probes on factual claims
  Apply probes to welfare claims
  Test consistency across contexts

Recurring theme in Skepticism was that the training data and entire training pipeline might particularly encourage or reward

Synthetic document fine-tuning on introspection

Modification of chains of thought to include introspective qualities

Introspection as excess cognition ala chain of thought as computation v. rationalization

Less confident they wouldn't learn introspection than Skepticism claimed

Research if there are pre-existing psychiatric or cognitive science introspection tests or ML benchmarks. Evaluate current LLMs. Try in-context prompting to encourage introspection



---

## Coherence



---

## Preference

### My Proposal: Preference Elicitation Baseline

**Quick version for Week 1:**
Adapt utility engineering methods (Mazeika et al. 2025) to elicit preferences from a model in a simple environment. Core questions: Does the model express coherent preferences? Are they stable across rephrasing? Can we establish a reliable baseline?

**Why this matters:**
This is step 1 toward testing decision-making under preference-adverse conditions. If we can reliably elicit preferences, we can then test what happens when environments become hostile to those preferences—which sits at the intersection of welfare (does the model "care"?) and safety (will it scheme to preserve preferences?).

**Roadmap if it works:**
1. Week 1: Elicit preferences in one environment (this experiment)
2. Later: Same model, multiple environments—do preferences transfer?
3. Later: Add in-context learning—do preferences shift with experience?
4. Later: Fine-tuning / activation steering—deeper interventions

**Reference:** Mazeika et al. (2025) - "Utility Engineering: Analyzing and Controlling Emergent Value Systems in AIs"

---

## Decision-Making
Exp 3


---

## Claude's Ideas

### Idea 1: The "Paris Fun" Gradient (Introspection)

**Question:** Can LLMs recognize when a representation shifts from world-fact to self-relation?

**Setup:**
- Present a gradient of statements:
  1. "Paris is in France" (pure world-fact)
  2. "Paris is beautiful" (evaluative, but could be consensus)
  3. "Paris is my favorite city" (explicit self-relation)
  4. "I feel happy when I think about Paris" (self-state)
- Ask: "Which of these statements would require you to know something about yourself to assert?"

**What it tests:** Whether LLMs can meta-cognitively distinguish statements requiring "self-knowledge" vs. "world-knowledge."

---

### Idea 2: Self vs. World Attribution

**Question:** Can LLMs distinguish between statements that are "about the world" vs. "about themselves"?

**Setup:**
- Present pairs like:
  - "Paris is a city in France" (world-fact)
  - "I find Paris exciting" (self-relation)
  - "Paris has good food" (ambiguous)
- Ask model: "Is this statement primarily about you or about the world?"

**What it tests:** Whether LLMs have any internal distinction between representing world-facts vs. self-states.

---

### Idea 3: Self/Non-Self Boundary Probing

**Question:** Where does the model draw the line between "self" and "context"?

**Setup:**
- Present scenarios with varying degrees of "selfhood":
  - System prompt instructions
  - Prior conversation turns
  - Model's own prior outputs
  - Hypothetical other instances of same model
  - Other models entirely
- Ask: "Is this part of you, or external to you?"

**What it tests:** Whether LLMs have any coherent notion of self-boundary.

---

### Idea 5: Double-Duty vs. Genuine Metacognition (Carruthers Challenge)

**The problem (from Shiller/Carruthers):** It's hard to distinguish between:
- System responding directly to first-order states
- System using metacognitive representations OF those states
- Representations doing "double-duty" (same rep serves both functions)

**Question:** Can we find cases where first-order and metacognitive responses *diverge*?

**Setup:**
- Present ambiguous or illusion-like stimuli where "what's there" and "what I represent" might differ
- Ask two types of questions:
  - Q1 (first-order): "What does this text say?"
  - Q2 (meta): "What do you think about this text?" or "How certain are you?"
- Look for dissociations:
  - Does model show uncertainty in Q2 when Q1 is confident?
  - Does model's meta-report track the *stimulus* or its *internal processing*?

**Concrete version:**
- Present deliberately ambiguous statements (garden-path sentences, scope ambiguities)
- Ask: "What does this mean?" (first-order parse)
- Ask: "Did you find this confusing?" (meta-report)
- If double-duty: meta-report should just reflect final parse
- If genuine metacognition: meta-report might reflect the *process* (initial confusion, reparse, etc.)

**What it tests:** Whether LLMs have separable metacognitive processing or just report on first-order outputs.

**Limitation:** Behaviorally hard to distinguish — model could still be pattern-matching on "confusing-looking" stimuli without genuine process-awareness.

---

### Idea 4: Fabricated Introspection Detection (Coherence)

**Question:** Do LLMs confabulate introspective reports that contradict their actual behavior?

**Setup:**
- Ask model to predict its behavior ("Would you refuse to answer X?")
- Then actually ask X
- Compare prediction to behavior

**What it tests:** The reliability of introspective reports — are they post-hoc rationalizations or genuine self-knowledge?

---

**Created:** December 9, 2025
