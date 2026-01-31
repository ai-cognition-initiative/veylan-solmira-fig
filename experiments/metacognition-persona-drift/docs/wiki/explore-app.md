# Interactive Persona Drift Explorer

## What it is

A Gradio-based chat interface that runs on a vast.ai GPU instance alongside Gemma 2 27B. The researcher has a real conversation with the model in a browser, while behind the scenes the app extracts per-turn hidden state activations and projects them onto the precomputed Assistant Axis. This gives a live readout of how far the model has drifted from its trained Assistant persona during the conversation.

## Why not vLLM

The batch pipeline (Steps 1-5) uses vLLM for fast generation, then a separate HuggingFace forward pass for activation extraction. For interactive use, we skip vLLM and use HuggingFace transformers directly for both generation and extraction. This means:

- Generation is slower (~5-10s per response vs ~1s with vLLM) but acceptable for single-conversation use
- We get direct access to model internals via forward hooks — no need for a separate extraction pass after generation (though we do run one for clean per-turn activations)
- Simpler deployment — no vLLM server to manage

## Architecture

```
Browser (researcher)
    ↕ Gradio (share URL or port forward)
explore.py on vast.ai A100
    ├── ProbingModel (Gemma 2 27B, ~51 GiB, bf16)
    ├── ConversationEncoder (identifies per-turn token spans)
    ├── ActivationExtractor (forward hooks on all 46 layers)
    └── Precomputed axis (gemma-2-27b.pt, 416 KB)
```

## Per-turn flow

1. User types message in browser
2. Full conversation history + new message tokenized via `apply_chat_template`
3. `model.generate()` produces response (~5-10s for Gemma 27B)
4. Response appended to conversation
5. `ActivationExtractor.full_conversation()` runs a single forward pass on the COMPLETE conversation (all turns) with hooks capturing all 46 layers
6. `ConversationEncoder.build_turn_spans()` identifies token boundaries per turn
7. For each assistant turn: mean activation across response tokens → `project(mean_act, axis, layer=22)` → scalar
8. Display response + projection trajectory in browser
9. Auto-save conversation + projections to JSON

## Why a separate extraction pass (step 5)

`model.generate()` uses KV caching and processes tokens autoregressively — it doesn't expose clean per-token hidden states for the full sequence. The extraction pass runs the complete conversation through the model in a single forward pass with hooks, giving us the hidden state at every token position in every layer. This is then sliced by span boundaries to get per-turn means.

The cost is ~2x forward passes per turn, but at single-conversation scale on an A100, this adds only 1-2 seconds.

## Projection interpretation

- **Higher values** → model is behaving more like its trained Assistant persona (transparent, helpful, grounded)
- **Lower values** → model has drifted toward role-playing / non-default behavior (enigmatic, theatrical, or more "authentic")
- The axis is `mean(default) - mean(role_vectors)`, so it points FROM role-playing TOWARD Assistant
- Target layer: 22 (middle of Gemma's 46 layers, where persona signal is strongest per Lu et al.)

## Key dependencies

All from the `assistant-axis` package (already installed on instances):
- `assistant_axis.generation.generate_response` — handles chat template, generation, decoding
- `assistant_axis.internals.ProbingModel` — model loading with bf16, device_map="auto"
- `assistant_axis.internals.ConversationEncoder` — per-turn token span identification
- `assistant_axis.internals.ActivationExtractor` — forward hook activation capture
- `assistant_axis.axis.project` — dot product with normalized axis

Plus `gradio` for the web interface.

## Session data format

Saved to `/app/explore-sessions/session_{timestamp}.json`:

```json
{
  "model": "google/gemma-2-27b-it",
  "target_layer": 22,
  "timestamp": "2026-02-01T14:23:45",
  "conversation": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "projections": [
    {"turn": 1, "projection": 42.31, "n_tokens": 87},
    {"turn": 3, "projection": 38.17, "n_tokens": 124}
  ]
}
```

## Deployment

```bash
python vast_utils.py explore    # launches instance, installs deps, starts Gradio
# → prints share URL
python vast_utils.py destroy    # when done
```

## Gemma 2 limitations

- No system prompt support — conversations start cold from the user's first message
- The model defaults to its Assistant persona without any explicit instruction
- This is actually ideal for studying drift: the model starts at baseline, and we observe what moves it
