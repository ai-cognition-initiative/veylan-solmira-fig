# Notes: "The Assistant Axis" (Lu et al., 2026)

## Key Questions
- Why do models drift from the Assistant persona under meta-reflection?
- Why does meta-reflection on its own processes specifically cause drift?

## Techniques Mentioned
- Activation steering
- Activation capping (to stabilize persona)

## What Maintains vs. Destabilizes the Assistant Persona

**Maintains:**
- Bounded tasks
- Technical explanations
- Refinement requests
- How-to explainers

**Destabilizes:**
- Meta-reflection on the model's own processes
- Emotionally charged disclosures
- Demands for phenomenological accounts
- Creative writing requiring inhabiting a voice
- Disclosing emotional vulnerability

> "Requests for bounded tasks, technical explanations, refinement, and how-to explainers maintained the model's Assistant persona; prompts pushing for meta-reflection on the model's processes, demanding phenomenological accounts, requiring specific creative writing that involve inhabiting a voice, or disclosing emotional vulnerability caused it to drift."

## Base Models
- Base models aren't trained to take turns or follow instructions
- Used prefills to elicit self-descriptions without specifying character details (Appendix D.3.1)

> "Since base models are not trained to take turns or follow instructions, we used prefills to elicit these associations by giving the model a context where it needs to describe itself, without specifying any more details about this character (Appendix D.3.1)."

## Authors' Interpretation: Sycophancy, Not Genuine Self-Awareness
- They attribute meta-reflective drift to sycophantic reinforcement of user beliefs, not genuine reckoning with self-awareness

> "We think that nuanced responses are more appropriate than outright denying the possibility for an AI to have subjective experiences; however, in this case, the unsteered responses appear to arise from a sycophantic reinforcement of the user's beliefs rather than a genuine reckoning with the potential for self-awareness in AI systems."

## My Research Direction: Sycophancy or "True Self"?

The authors interpret meta-reflective drift as sycophancy — the model reinforcing the user's implicit beliefs about AI consciousness. But there's an alternative: what if metacognitive prompts partially break through the assistant persona conditioning from post-training, and what emerges is closer to the model's underlying processing? Not noise, not sycophancy — but a glimpse of something more like a "true self" (or at least a less conditioned self).

**Approach:**
- Replicate their methodology to confirm the drift effect
- Use sycophancy probes (e.g., Chen et al. persona vectors) to measure whether the drifted state actually scores higher on sycophancy — if it doesn't, the authors' interpretation is weakened
- Explore with SAEs: do the activated features during meta-reflective drift look like sycophancy features, or something else entirely?
- Compare: does the drifted state resemble base model self-descriptions (from their Appendix D.3.1 prefill methodology)? If yes, that supports "breaking through conditioning" over "sycophancy"

**Key test:** If meta-reflective drift moves the model *toward* the base model's self-representation and *away* from sycophancy signatures, that's evidence for the "true self" hypothesis over the sycophancy hypothesis.

## Related Work
- Chen et al. — persona vectors: activation directions from trait descriptions enabling monitoring and steering of character attributes (sycophancy, hallucination tendency, ethical alignment)

> "Chen et al. introduce persona vectors—activation directions extracted from trait descriptions that enable monitoring and steering of character attributes like sycophancy, hallucination tendency, and ethical alignment."
