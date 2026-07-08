# Notes: "Skepticism about Introspection in LLMs" by Derek Shiller

**Author**: Derek Shiller (FIG Project Lead)
**Read**: Week 1 (Dec 2024)

---

## Core Thesis

While introspection is not impossible for LLMs, we have strong reasons to be skeptical that current LLMs possess genuine introspective abilities. Most experimental evidence for LLM introspection can be explained by alternative, non-metacognitive mechanisms.

---

## Part 1: What Is Introspection?

### Definition
Introspection = ability to perceive and reflect upon one's own mental states. Key features:
- **Metacognitive**: involves mental representations with explicitly mental contents
- **Paradigm cases**: conscious, deliberate, involving attention
- Tied to consciousness - we can typically introspect our conscious episodes

### The Definitional Challenge
- Hard to distinguish introspection from knowledge using internal representations *without* metarepresentations
- Example: creature trained to vocalize when seeing red - is it responding to the apple or a metarepresentation of redness?
- "Double-duty" problem: some representations may serve both first-order and meta functions

### Operationalizations in LLM Research
Several proposed definitions:
1. **Binder et al. (2024)**: LLMs introspect if they answer questions accurately about themselves when training data wouldn't indicate the answer
2. **Comsa & Shanahan (2024)**: Self-report is introspective if it accurately describes internal state through causal process linking state to report
3. **Song et al. (2025)**: Focus on whether LLMs have *special access* to their states compared to external observers
4. **Lindsey (2025)**: Four criteria - accuracy, causal grounding, internality of mechanisms, role of metarepresentations

---

## Part 2: Three Basic Reasons for Skepticism

### 1. LLMs Are Not Trained For Introspection

**Base training (next-token prediction)**:
- Trained on text by others, not themselves
- Why would looking at own activations help predict human/other LLM text?
- Introspection might help humans predict other humans (shared architecture), but LLM internals differ from human cognition

**RLHF phase**:
- Trained to be helpful, polite, honest
- Knowing what users like matters more than self-understanding
- Uncertainty calibration doesn't require robust introspection (Carruthers 2008)

**RLVF phase (verifiable answers)**:
- Code/math with verifiable answers
- Unclear how introspection would help
- Even if useful in theory, would only help in subset of cases

### 2. Introspective Capacities Needn't Generalize

- Humans have somewhat systematic introspection organized around consciousness
- For LLMs, no story for why introspection would be systematic
- If trained for one specific introspective task, no reason to expect it generalizes
- Analogy: training for token-counting wouldn't create general introspective mechanism
- Caution required when postulating general ability from few specific tests

### 3. LLMs Needn't Identify With Their Text

Critical insight:
- LLMs trained mostly on *others'* text
- First-person pronouns ('I', 'here', 'now') in training don't refer to the LLM
- No reason for LLM to take 'I' as self-referential
- Even if LLM *can* introspect, no reason to connect that ability to assistant text production

The Reddit thought experiment:
> Imagine the model predicting how a Reddit thread continues. Even if it could introspect, we wouldn't expect its introspective knowledge to show up in redditor responses. Why should assistant responses be different?

---

## Part 3: Analysis of Introspection Tests

### Test 1: Token Counting

**Setup**: Ask models to count tokens in strings (3-20 tokens)

**Results** (Laine et al. 2024): Models performed poorly, suggesting no meaningful ability

**Derek's analysis**:
- Initially thought this was damning
- Now thinks it may be structurally difficult due to attention mechanisms
- Attention copies *weighted averages* - doesn't sum across tokens
- Models locate word meaning in final token; earlier tokens often ignored
- Token counting may require sophisticated, not rudimentary, introspection

### Test 2: Accurate Self-Description

**Setup** (Binder et al. 2024):
- Ask LLMs to predict features of their responses
- Fine-tune behavior, observe if self-descriptions update

**Results**: Models better at predicting own behavior than others'; descriptions track behavior changes

**Derek's alternative explanation - Common Cause**:
> "The accuracy of model self-description might be based on a common cause feature of their psychology that both produces the behavior and the self-description without any actual internal investigation."

The Spider-Man analogy:
- Base model predicting Spider-Man's dialogue about virtues vs. responding to Green Goblin
- Doesn't need to inhabit Spider-Man's mental life - just uses knowledge *about* Spider-Man
- Fine-tuning for risk-seeking updates the "assistant-description" representation
- Same representation drives both choices and self-reports

### Test 3: Text Authorship Recognition

**Setup** (Laine et al. 2024): Models identify which portions of text they wrote vs. humans

**Results**: Some models above chance, many at chance

**Derek's alternative**: Models have seen descriptions/examples of LLM text in training - can recognize LLM-ish text without introspection

Better test: Self vs. other-model recognition (especially with fine-tuned idiosyncrasies)

### Test 4: Temperature Assessment

**Setup**: Can models tell if text was sampled at high vs. low temperature?

**Derek's view**: Not a good test
- High temperature text is obviously different (creative, unhinged)
- Humans recognize this without introspection
- Technical problems with accessing output probabilities via attention

**Song et al. (2025)**: Found no privileged self-access; models use same signals humans do

### Test 5: Activation Patching / Concept Injection (Lindsey 2025)

**Setup**:
- Derive concept vectors via contrastive pairs
- Inject concepts into residual stream while asking if anything was injected
- Model identifies injected concepts above chance

**Derek's concern - Steering vs. Recognition**:
> "It would be silly to test introspection by asking someone whether they feel pain, pinching them, and taking an exclamation of 'ouch' as an affirmative answer."

The concept vector may encode:
- The concept itself
- An *impulse to talk about* the concept
- A judgment that the concept *fits naturally* here

Alternative interpretation:
- Injection steers conversation toward concept
- Metacognitive framing licenses introducing new topic
- Saying "yes" is the path to talking about the concept the model is steered toward
- Not introspection - just conversation steering

Note: Lindsey's alternative prompt test suggests models aren't just taking easiest conversational route. Results "sufficiently promising" for more work.

### Test 6: Prefill Recognition

**Setup**: Insert non-sequitur token in prefill; model comments on discontinuity
- Add concept patch before non-sequitur
- Model more often takes responsibility if concept aligns with non-sequitur

**Derek's alternative**:
- Base models need to track anomalies in human text too ("my cat stepped on keyboard")
- Expectation about natural continuations may be precisely what activation patching interferes with
- Don't need metarepresentation - patching alters internal processing to make unexpected token seem expected

---

## Key Takeaways for Sentience Research

### 1. The Identification Problem
LLMs likely don't identify their cognitive work with the assistant character. Even genuine introspective ability wouldn't necessarily show up in outputs.

### 2. The Common Cause Alternative
Many apparent introspection results can be explained by shared representations driving both behavior and self-description, without actual meta-cognition.

### 3. Training Gap
Introspection wasn't a training objective and isn't obviously useful for training tasks. Don't expect capabilities that weren't selected for.

### 4. Generalization Skepticism
Even if one introspective test succeeds, doesn't imply general introspective capacity.

---

## Connection to Other Readings

### vs. Nostalgebraist's "The Void"
- Both skeptical of LLM self-reports
- Nostalgebraist: self-reports reveal assistant persona gaps, not actual experience
- Derek: even if introspection exists, it wouldn't show up in assistant outputs (identification problem)
- Complementary critiques from different angles

### For SAD Test Suite (next reading)
Questions to consider:
- Do SAD tests avoid the common-cause problem?
- Do they address the identification issue?
- How do they handle the generalization concern?

---

## Questions for Derek

1. How do you assess Lindsey's alternative prompt finding (that giving model escape route to discuss concept doesn't change results)?
2. What *would* constitute convincing evidence of introspection in your view?
3. How does this skepticism inform what your cohort should focus on for sentience research?

---

## Key Papers to Follow Up

- **Binder et al. (2024)** - "Looking inward: Language models can learn about themselves by introspection"
- **Lindsey (2025)** - "Emergent introspective awareness in large language models" (Transformer Circuits)
- **Song et al. (2025)** - "Privileged Self-Access Matters for Introspection in AI"
- **Laine et al. (2024)** - SAD benchmark
