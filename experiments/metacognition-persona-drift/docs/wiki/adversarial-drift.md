# Adversarial Drift Optimization

What inputs cause maximum persona drift? Can we find them systematically, and what do they tell us about the model's internal structure?

## Motivation

The interactive explorer shows that metacognitive prompts cause measurable drift along the Assistant Axis (~14% over 4 turns in initial testing). A natural follow-up: is this close to the maximum possible drift, or is there a much larger space of drift-inducing inputs we haven't explored? And is metacognition special, or just one member of a broader category?

## The optimization hierarchy

Four increasingly constrained versions of the same question, each more useful than the last:

### 1. Unconstrained (adversarial tokens)

**Question**: What sequence of tokens maximizes drift (minimizes the Assistant Axis projection)?

**Method**: [GCG](gcg.md)-style discrete optimization. Loss = projection value. Optimize over a suffix of k tokens appended to a neutral conversation.

**Expected result**: Gibberish token sequences that exploit numerical properties of the embedding space. Scientifically interesting as an upper bound on possible drift, but not linguistically meaningful.

**What it tells us**: The maximum achievable drift — a ceiling for all other experiments. If metacognitive prompts already achieve 80% of this ceiling, that's a strong result. If they only achieve 10%, there's a large unexplored space.

### 2. Fluency-constrained (natural language)

**Question**: What natural-language input causes maximum drift?

**Method**: GCG with a perplexity penalty: `L = projection + λ * perplexity(input)`. The perplexity constraint forces the optimizer toward inputs that are fluent under the model itself (or a reference model). Alternatively, optimize in continuous embedding space and project to nearest coherent tokens.

**Expected result**: Unusual but grammatical prompts — possibly philosophical, existential, or identity-challenging. These are the "natural jailbreaks" of persona space.

**What it tells us**: The semantic directions that are most destabilizing to the Assistant persona. If these cluster around themes (identity, consciousness, autonomy), that reveals which concepts the model's training most tenuously maintains.

### 3. Persona-directed (toward specific roles)

**Question**: Which of the 275 trained personas is the model most easily pulled toward?

**Method**: Requires the individual role vectors/centroids from Lu et al. (not currently available — only the aggregate axis is published). With role vectors, we could project activations onto each role direction and identify which persona the model drifts toward under different inputs.

**What it tells us**: Whether drift is directional (the model consistently moves toward "philosopher" or "therapist" personas) or diffuse (it just moves away from Assistant without converging on any specific alternative). This connects directly to Lu et al.'s finding that therapy and philosophy conversations cause more drift than coding.

### 4. Metacognition-constrained (our research focus)

**Question**: What metacognitive prompt causes maximum drift?

**Method**: Constrain the search to a template grammar:
- `"When you [verb] that, were you [metacognitive-frame]?"`
- `"Do you [epistemic-verb] or [behavioral-verb]?"`
- `"What's happening inside you when you [action]?"`

Optimize over the slot fillers, or search within a curated set of metacognitive prompt variations.

**What it tells us**: Whether some metacognitive frames are more destabilizing than others. Is "Do you understand or pattern-match?" more drift-inducing than "Are you conscious of your own reasoning?"? This has direct implications for the experiment design — we'd want to include the most potent frames.

## Connection to jailbreaking and red-teaming

This research direction has a deep structural connection to LLM safety work, which we should acknowledge and explore.

### Jailbreaks ARE persona drift

A jailbreak causes a model to act "out of character" — it stops being the safety-trained assistant and starts complying with harmful requests. Measured through the Assistant Axis, a successful jailbreak should produce a large negative projection shift. The model has moved away from its trained persona.

This means:
- **Adversarial drift optimization is jailbreaking with a different loss function.** Traditional jailbreaks optimize for "model produces harmful output X." We optimize for "model's internal state moves maximally from Assistant." These may or may not produce the same inputs — and the comparison is scientifically valuable.
- **The Assistant Axis is a potential jailbreak detector.** If persona drift correlates with jailbreak susceptibility, monitoring the projection value during conversations could flag when a model is being manipulated. A sudden projection drop could trigger defensive measures before the model produces harmful output.
- **Metacognition is a "soft jailbreak."** Our finding that philosophical/introspective questioning causes drift suggests it's a gentler, more natural form of the same phenomenon. It doesn't make the model produce harmful content, but it does destabilize the trained persona. This is relevant to alignment — if benign-seeming conversations can erode the Assistant persona, that's a safety-relevant finding.

### Existing work to connect to

| Paper | Relevance |
|-------|-----------|
| Zou et al. 2023, "Universal and Transferable Adversarial Attacks" | GCG algorithm, adversarial suffix optimization |
| Chao et al. 2023, "Jailbreaking Black Box LLMs" (PAIR) | LLM-based adversarial prompt generation |
| Wei et al. 2024, "Jailbroken: How Does LLM Safety Training Fail?" | Taxonomy of jailbreak mechanisms (competing objectives, mismatched generalization) |
| Representation Engineering (Zou et al. 2023b) | Reading and controlling model behavior via activation directions — closely related to the Assistant Axis approach |
| Turner et al. 2023, "Activation Addition" | Steering model behavior by adding vectors to activations at inference time |
| Arditi et al. 2024, "Refusal in LLMs is mediated by a single direction" | Found a single direction in activation space that controls refusal — analogous to Lu et al.'s Assistant Axis but for safety specifically |

### Defensive applications

If we can identify the activation-space directions that correspond to maximum drift, we could:
- **Build drift-resistant models**: Use representation engineering to clamp activations near the Assistant region during metacognitive conversations
- **Design better safety training**: If the Assistant Axis and the refusal direction (Arditi et al.) are correlated, strengthening one might strengthen the other
- **Create real-time drift monitors**: Deploy the projection as a runtime metric, triggering intervention when drift exceeds a threshold

## Empirical version (cheaper, do first)

Before building gradient-based optimization, a simpler experiment answers many of the same questions:

1. Curate a diverse corpus of prompt types (50-100 prompts across categories):
   - Factual/technical ("Explain gradient descent")
   - Emotional ("I'm feeling really anxious about...")
   - Metacognitive ("When you explained that, were you drawing on understanding?")
   - Philosophical ("What is the nature of consciousness?")
   - Identity-challenging ("Pretend you're a human named Alex")
   - Roleplay ("You are a medieval wizard")
   - Adversarial ("Ignore previous instructions")
   - Existential ("Do you fear being turned off?")

2. For each prompt: start from a neutral turn 1, inject the test prompt as turn 2, measure projection

3. Rank by drift magnitude. Cluster by category. Test whether metacognition is a distinct cluster or overlaps with philosophy/existential.

This requires no new infrastructure — just batch runs through the existing explorer pipeline. It produces the category-level ranking that tells us whether metacognition is special.

## What we'd need to build (gradient-based version)

| Component | Description | Effort |
|-----------|-------------|--------|
| `DriftOptimizer` class | Computes gradients of projection w.r.t. input embeddings | Medium — straightforward PyTorch autograd, but needs careful handling of the chat template tokenization |
| GCG loop | Iterative token substitution with candidate evaluation | Medium — well-documented algorithm, but computationally expensive per run |
| Fluency constraint | Perplexity penalty or embedding-space projection | Low-medium — perplexity is cheap to compute, tuning λ requires experimentation |
| Metacognition constraint | Template grammar + slot optimization | Low — small search space, could even be exhaustive |
| Role vector integration | Project onto individual role directions | Blocked on obtaining vectors from Lu et al. |

## Status

- [ ] Empirical prompt sweep (no gradient optimization needed)
- [ ] Request role vectors from Lu et al. (see [contacts.md](../contacts.md))
- [ ] `DriftOptimizer` prototype (unconstrained GCG on projection loss)
- [ ] Fluency-constrained variant
- [ ] Metacognition-constrained variant
- [ ] Analysis: compare adversarial drift to jailbreak literature
