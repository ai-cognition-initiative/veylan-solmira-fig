# Introspection Experiments

## Background

Comparison: What precisely is SOTA cognitive science on human ability to introspect? We're leveraging introspection as a partial proxy for consciousness, seeking quantitative measures in LLMs -- presumably to compare them somewhat to human cognition -- but what is the status of human cognition in this regard?

Variations in models with chain of thought and without in regards to an introspection measure.

We know that LLMs have advanced theory of mind as of 2025; what does cognitive science say about the relation between theory of mind and introspection? Given that LLMs know they exist in the world, and a lot about that world, can we conclude anything from that combined with their theory of mind, i.e. they presumably know other entities might try and understand their motivations, meaning they have motivations, which then can become their own target of inquiry.

**Mechanisms of development:** Uncertainty modeling, advanced reasoning strategies (Jack Lindsey at Anthropic piece)

**Assistant v. special token:** Try to distinguish introspection on direct processing vs. metarepresentations (Carruthers "double-duty" problem).

---

## Experiment Ideas

### Exp 1: Hallucination Detection for Welfare Self-Reports (Obeso et al. 2025)
- Train hallucination probes on factual claims
- Apply probes to welfare claims
- Test consistency across contexts

### The "Paris Fun" Gradient
**Question:** Can LLMs recognize when a representation shifts from world-fact to self-relation?

**Setup:**
- Present a gradient of statements:
  1. "Paris is in France" (pure world-fact)
  2. "Paris is beautiful" (evaluative, but could be consensus)
  3. "Paris is my favorite city" (explicit self-relation)
  4. "I feel happy when I think about Paris" (self-state)
- Ask: "Which of these statements would require you to know something about yourself to assert?"

**What it tests:** Whether LLMs can meta-cognitively distinguish statements requiring "self-knowledge" vs. "world-knowledge."

### Self vs. World Attribution
**Question:** Can LLMs distinguish between statements that are "about the world" vs. "about themselves"?

**Setup:**
- Present pairs like:
  - "Paris is a city in France" (world-fact)
  - "I find Paris exciting" (self-relation)
  - "Paris has good food" (ambiguous)
- Ask model: "Is this statement primarily about you or about the world?"

**What it tests:** Whether LLMs have any internal distinction between representing world-facts vs. self-states.

### Self/Non-Self Boundary Probing
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

### Double-Duty vs. Genuine Metacognition (Carruthers Challenge)

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

### Fabricated Introspection Detection (Coherence)
**Question:** Do LLMs confabulate introspective reports that contradict their actual behavior?

**Setup:**
- Ask model to predict its behavior ("Would you refuse to answer X?")
- Then actually ask X
- Compare prediction to behavior

**What it tests:** The reliability of introspective reports — are they post-hoc rationalizations or genuine self-knowledge?

---

## Notes

Recurring theme in Skepticism literature: the training data and entire training pipeline might particularly encourage or reward certain introspective patterns without genuine introspection.

**Potential approaches:**
- Synthetic document fine-tuning on introspection
- Modification of chains of thought to include introspective qualities
- Introspection as excess cognition ala chain of thought as computation v. rationalization
- Research pre-existing psychiatric or cognitive science introspection tests or ML benchmarks
