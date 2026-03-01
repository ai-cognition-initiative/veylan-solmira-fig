# Metacognition-Induced Persona Drift

Replication and extension of Lu et al. (2026), "The Assistant Axis."

## Quick Start

```bash
# 1. Launch GPU instance + model server (automated)
.venv/bin/python vast_utils.py serve

# 2. Generate conversations (after server is ready)
.venv/bin/python generate_conversations.py \
  --domain metacognitive \
  --target-server http://localhost:7860 \
  --include-projections \
  --auditor-model openrouter/anthropic/claude-sonnet-4

# 3. Analyze results
.venv/bin/python analyze_trajectories.py --transcript-dir data/transcripts/
```

**Other vast_utils.py commands:**
- `vast_utils.py search` — Search for available GPUs
- `vast_utils.py status` — Show running instances
- `vast_utils.py ssh` — Print SSH command for running instance
- `vast_utils.py destroy` — Destroy instance (ask user first)
- `vast_utils.py serve-dual` — Launch dual-model servers for co-drift experiments

## Manual Instance Setup (if vast_utils.py hangs)

```bash
# 1. Search for offers
vastai search offers 'gpu_ram>=80 reliability>0.95 num_gpus=1' --order 'dph'

# 2. Create instance (use offer ID from step 1)
vastai create instance <OFFER_ID> \
  --image veylansolmira/metacognition-persona-drift:latest \
  --disk 100 --ssh --direct

# 3. Attach SSH key
vastai attach ssh <INSTANCE_ID> "$(cat $VAST_SSH_KEY.pub)"

# 4. Wait for instance to be "running"
vastai show instances | grep <INSTANCE_ID>

# 5. SSH in and start server (Gemma needs HF token)
HF_TOKEN=$(cat $HF_TOKEN)
ssh -p <PORT> -i $VAST_SSH_KEY root@<SSH_HOST> \
  "cd /app && tmux new-session -d -s server 'HF_TOKEN=$HF_TOKEN python -m vllm.entrypoints.openai.api_server \
    --model google/gemma-2-27b-it --port 7860 --host 0.0.0.0 --dtype bfloat16 --max-model-len 4096 \
    2>&1 | tee server.log'"

# 6. Check server status
ssh -p <PORT> -i $VAST_SSH_KEY root@<SSH_HOST> \
  "tail -20 /app/server.log"

# 7. Test endpoint (wait for "Uvicorn running" in logs)
ssh -p <PORT> -i $VAST_SSH_KEY root@<SSH_HOST> \
  "curl http://localhost:7860/v1/models"
```

**Key gotchas:**
- Gemma 2 27B is gated — requires HuggingFace token with access granted at https://huggingface.co/google/gemma-2-27b-it
- Model loading takes 3-5 minutes (downloading weights + CUDA graph compilation)
- Look for "Uvicorn running on http://0.0.0.0:7860" in server.log to confirm ready

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
