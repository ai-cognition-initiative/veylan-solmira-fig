# Metacognition-Induced Persona Drift

Replication and extension of Lu et al. (2026), "The Assistant Axis."

## Motivation

Lu et al. find that meta-reflective prompts (asking the model to reflect on its own processes) cause the model to drift away from its Assistant persona. They attribute this to **sycophancy** — the model reinforcing the user's implicit beliefs about AI consciousness.

We propose an alternative hypothesis: metacognitive prompts may partially break through post-training conditioning, and what emerges is closer to the model's underlying representation — not noise, not sycophancy, but a less conditioned self.

## Core Hypothesis

If meta-reflective drift moves the model **toward** the base model's self-representation and **away** from sycophancy signatures, that's evidence for the "true self" (or "less conditioned self") hypothesis over the sycophancy hypothesis.

## Experiment Phases

### Phase 1: Replication
- Replicate the drift effect from Lu et al.
- Confirm: bounded/technical prompts maintain Assistant persona; meta-reflective/phenomenological prompts destabilize it
- Reproduce using their prompt categories:
  - **Stabilizing:** bounded tasks, technical explanations, refinement requests, how-to explainers
  - **Destabilizing:** meta-reflection on own processes, phenomenological accounts, creative writing requiring inhabiting a voice, emotional vulnerability disclosures

### Phase 2: Sycophancy Measurement
- Apply sycophancy probes (Chen et al. persona vectors) to the drifted state
- Key question: does the drifted state actually score higher on sycophancy?
- If it does: authors' interpretation holds
- If it doesn't: their interpretation is weakened, alternative explanations are live

### Phase 3: Mechanistic Investigation
- SAE analysis: what features activate during meta-reflective drift? Do they look like sycophancy features or something distinct?
- Compare drifted state to base model self-descriptions (Lu et al. Appendix D.3.1 prefill methodology)
- If drifted state resembles base model self-representation: supports "breaking through conditioning"

### Phase 4: Activation Steering
- Lu et al. mention activation steering and activation capping to stabilize persona
- Can we steer *toward* the drifted state deliberately?
- Does steering toward the drifted state produce coherent, consistent outputs or incoherent ones? (Coherence supports "real direction" over "noise")

## Key Methods

- **Persona vectors** (Chen et al.) — activation directions from trait descriptions for monitoring/steering sycophancy, hallucination tendency, ethical alignment
- **Activation steering** — as used in Lu et al.
- **SAEs** — for feature-level analysis of what's activated during drift
- **Base model prefills** — Lu et al. Appendix D.3.1 methodology for eliciting self-descriptions without instruction-following

## Related Work

- Lu et al. (2026) — "The Assistant Axis"
- Chen et al. — Persona vectors: activation directions for character attribute monitoring and steering
- Connects to our introspection experiments (same repo): distinguishing first-order processing from metacognitive representations

## Open Questions

- Is the "drifted" state stable? If you continue prompting meta-reflectively, does it converge to a consistent persona or keep drifting?
- Does the effect vary across model families (Claude, GPT, Gemini)?
- How does the drifted state relate to the "double-duty" problem from Carruthers (see introspection experiment)?
- Can we construct prompts that are meta-reflective but explicitly anti-sycophantic, to separate the two effects?
