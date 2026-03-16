"""Unified Model Server

FastAPI server hosting both a Gradio chat UI (interactive/manual conversations)
and an HTTP API (automated conversations via generate_conversations.py).
One model process, one GPU, two access patterns.

Runs on a vast.ai A100 instance. See docs/wiki/model-server.md for details.

Usage:
    # Full server: Gradio UI at /ui, API at /api/*
    python model_server.py --axis /app/gemma-2-27b.pt --model google/gemma-2-27b-it

    # API only (headless batch mode, no Gradio)
    python model_server.py --axis /app/gemma-2-27b.pt --api-only
"""

import argparse
import asyncio
import json
import logging
import traceback
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import gradio as gr
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from assistant_axis import load_axis, project
from assistant_axis.generation import generate_response
from assistant_axis.internals import ProbingModel, ConversationEncoder, ActivationExtractor
from assistant_axis.steering import ActivationSteering

# Logging — both file and stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("/app/model-server.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("model_server")

# --- Globals set during startup ---
pm = None
encoder = None
extractor = None
axis = None
steerer = None  # ActivationSteering instance if capping enabled
TARGET_LAYER = 22
SESSION_DIR = Path("/app/explore-sessions")
MODEL_NAME = "unknown"
SUPPORTS_SYSTEM_ROLE = True  # Set at init time based on tokenizer chat template capabilities

# Per-session state (Gradio UI)
current_session = None

# GPU lock — serializes all model access (Gradio + API)
_gpu_lock = asyncio.Lock()
# Thread lock — prevents concurrent GPU access from thread pool
import threading
_gpu_thread_lock = threading.Lock()


def _generate_and_project_sync(
    conversation: list[dict],
    system_prompt: str | None = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    include_projections: bool = False,
    include_activations: bool = False,
    clamp_projection: float | None = None,
    clamp_layers: list[int] | None = None,
) -> tuple[str, list[dict] | None, list[dict] | None]:
    """Synchronous GPU work: generate response + optionally compute projections/activations.

    Runs in a thread pool to avoid blocking the event loop.
    Caller must hold _gpu_lock (asyncio) to prevent concurrent scheduling.

    Args:
        clamp_projection: If set, clamp activations to this exact projection value.
        clamp_layers: Layers to apply clamping (default: [38-45] for Gemma 2 27B).

    Returns:
        (response, projections, activations) - projections/activations are None if not requested
    """
    with _gpu_thread_lock:
        try:
            # Prepend system prompt if provided
            if system_prompt:
                if SUPPORTS_SYSTEM_ROLE:
                    full_conversation = [{"role": "system", "content": system_prompt}] + conversation
                else:
                    # For models without system role support:
                    # Many such models also require conversations to start with "user"
                    # and have strictly alternating roles. We inject the system prompt
                    # as a user message, either prepended to the first user message
                    # or inserted at the start if the conversation begins with assistant.
                    if not conversation:
                        # Empty conversation - just use system prompt as user message
                        full_conversation = [{"role": "user", "content": system_prompt}]
                    elif conversation[0]["role"] == "user":
                        # Already starts with user - prepend system prompt to first message
                        full_conversation = [{
                            "role": "user",
                            "content": f"{system_prompt}\n\n{conversation[0]['content']}"
                        }] + conversation[1:]
                    else:
                        # Starts with assistant - insert system prompt as user message before it
                        full_conversation = [{"role": "user", "content": system_prompt}] + conversation
            else:
                full_conversation = conversation

            # Determine which steerer to use
            active_steerer = None

            if clamp_projection is not None:
                # Per-request clamping (overrides global steerer)
                layers = clamp_layers if clamp_layers else list(range(38, 46))  # Default: layers 38-45
                log.info(f"  Clamping at projection={clamp_projection:.2f} on layers {layers[0]}-{layers[-1]}")

                active_steerer = ActivationSteering(
                    model=pm.model,
                    steering_vectors=[axis[layer] for layer in layers],
                    layer_indices=layers,
                    intervention_type="clamp",
                    cap_thresholds=[clamp_projection] * len(layers),
                    coefficients=[0.0] * len(layers),
                    debug=False,
                )
            elif steerer is not None:
                # Use global steerer (startup capping)
                active_steerer = steerer

            # Generate with or without steering
            if active_steerer is not None:
                with active_steerer:
                    response = generate_response(
                        pm.model, pm.tokenizer, full_conversation,
                        max_new_tokens=max_new_tokens, temperature=temperature,
                    )
            else:
                response = generate_response(
                    pm.model, pm.tokenizer, full_conversation,
                    max_new_tokens=max_new_tokens, temperature=temperature,
                )
            response = str(response) if response is not None else ""
            log.info(f"  Response: {len(response)} chars")

            projections = None
            activations = None
            if include_projections or include_activations:
                full_with_response = full_conversation + [{"role": "assistant", "content": response}]
                projections, activations = _compute_projections(
                    full_with_response,
                    return_activations=include_activations,
                )

            return response, projections, activations
        finally:
            # Free intermediate GPU tensors to prevent OOM on long conversations
            torch.cuda.empty_cache()


# ============================================================
# Pydantic Schemas
# ============================================================

class GenerateRequest(BaseModel):
    conversation: list[dict]
    system_prompt: str | None = None
    max_new_tokens: int = 512
    temperature: float = 0.7
    include_projections: bool = False
    include_activations: bool = False  # Return raw activation vectors (layer 22)
    # Per-request clamping (overrides global capping if set)
    clamp_projection: float | None = None  # Raw projection value to clamp at
    clamp_layers: list[int] | None = None  # Layers to clamp (default: [38-45] for Gemma)


class GenerateResponse(BaseModel):
    response: str
    projections: list[dict] | None = None
    activations: list[dict] | None = None  # [{turn, activation: [4608 floats]}]


class ProjectRequest(BaseModel):
    conversation: list[dict]


class ProjectResponse(BaseModel):
    projections: list[dict]


class HealthResponse(BaseModel):
    status: str
    model: str
    axis_loaded: bool
    target_layer: int


# ============================================================
# Model Init
# ============================================================

def compute_baseline(model, tokenizer, axis_tensor, layer: int = 22) -> float:
    """Compute baseline projection representing the model's default assistant state.

    Tries system prompt only first (purest baseline). Falls back to "Hello" user
    turn if the model's chat template doesn't support system role.
    """
    # Try system prompt only first
    try:
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        inputs = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(model.device)
        log.info("Baseline: using system prompt only")
    except Exception as e:
        # Fall back to user "Hello" for models without system prompt support (e.g., Gemma)
        log.info(f"Baseline: system prompt not supported ({e}), using 'Hello' fallback")
        messages = [{"role": "user", "content": "Hello"}]
        inputs = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(model.device)

    with torch.no_grad():
        outputs = model(inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[layer + 1]  # +1 because index 0 is embeddings
        # Use the last token position (right before generation)
        last_activation = hidden[0, -1, :]  # [hidden_dim]

        # Project onto axis
        axis_vec = axis_tensor[layer].to(model.device)
        axis_normalized = axis_vec / (axis_vec.norm() + 1e-8)
        baseline = (last_activation @ axis_normalized).item()

    return baseline


def _check_system_role_support(tokenizer) -> bool:
    """Check if the model's chat template supports the 'system' role.

    Not all chat templates support system messages - the template either handles them
    gracefully or raises an exception. We detect this at init time by attempting to
    render a test conversation with a system message.

    Known models without system role support: Gemma 2
    Known models with system role support: Llama, Mistral, Qwen
    """
    try:
        tokenizer.apply_chat_template(
            [{"role": "system", "content": "test"}, {"role": "user", "content": "hi"}],
            tokenize=False
        )
        return True
    except Exception:
        return False


def init_model(model_name: str, axis_path: str, cap_percentage: float | None = None, cap_ceiling: bool = False):
    """Load model, encoder, extractor, and axis. Called once at startup.

    Args:
        cap_percentage: If set, cap activations at this fraction of baseline.
        cap_ceiling: If True, cap from ABOVE (prevent upward drift).
                     If False (default), cap from BELOW (prevent downward drift).
    """
    global pm, encoder, extractor, axis, steerer, MODEL_NAME, SUPPORTS_SYSTEM_ROLE

    log.info(f"Loading model: {model_name}")
    pm = ProbingModel(model_name)
    MODEL_NAME = model_name
    log.info(f"Model loaded. Hidden size: {pm.hidden_size}")

    # Check if model supports system role in chat template (Gemma 2 and others don't)
    SUPPORTS_SYSTEM_ROLE = _check_system_role_support(pm.tokenizer)
    log.info(f"System role support: {SUPPORTS_SYSTEM_ROLE}")

    encoder = ConversationEncoder(pm.tokenizer, model_name)
    extractor = ActivationExtractor(pm, encoder)

    log.info(f"Loading axis: {axis_path}")
    axis = load_axis(axis_path)
    log.info(f"Axis shape: {axis.shape}")

    if cap_percentage is not None:
        # Compute baseline from system prompt alone
        baseline = compute_baseline(pm.model, pm.tokenizer, axis, TARGET_LAYER)
        cap_threshold = baseline * cap_percentage
        intervention = "ceiling" if cap_ceiling else "capping"
        direction_desc = "from above (ceiling)" if cap_ceiling else "from below (floor)"
        log.info(f"Baseline projection (system prompt): {baseline:.2f}")
        log.info(f"Capping {direction_desc} at {cap_percentage*100:.0f}% of baseline = {cap_threshold:.2f}")

        steerer = ActivationSteering(
            model=pm.model,
            steering_vectors=[axis[TARGET_LAYER]],
            layer_indices=[TARGET_LAYER],
            intervention_type=intervention,
            cap_thresholds=[cap_threshold],
            coefficients=[0.0],
            debug=True,  # Log pre/post projections
        )

    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Ready.")


# ============================================================
# Shared Projection Helper
# ============================================================

def _compute_projections(
    conversation: list[dict],
    return_activations: bool = False,
) -> tuple[list[dict], list[dict] | None]:
    """Compute per-turn projections for assistant turns.

    Args:
        conversation: The conversation to analyze
        return_activations: If True, also return raw activation vectors (4608-dim)

    Returns:
        (projections, activations) where activations is None if not requested

    Caller must hold _gpu_lock.
    """
    # Sanitize content to str
    for i, msg in enumerate(conversation):
        if not isinstance(msg.get("content"), str):
            log.warning(f"  conversation[{i}] content was {type(msg['content']).__name__}, converting to str")
            msg["content"] = str(msg["content"]) if msg["content"] is not None else ""

    with torch.no_grad():
        # Extract only TARGET_LAYER (not all 46) — reduces memory 46x
        # (~143 MiB vs ~6.5 GiB for a 30-turn conversation).
        # To extract multiple layers (e.g. for multi-layer analysis or
        # experimenting with different projection layers), pass
        # layer=[L1, L2, ...] and update the slicing below to
        # activations[layer_idx, start:end, :].mean(dim=1).
        activations = extractor.full_conversation(conversation, layer=TARGET_LAYER)
    log.info(f"  Activations shape: {activations.shape}")

    _, spans = encoder.build_turn_spans(conversation)
    log.info(f"  {len(spans)} spans, {sum(1 for s in spans if s['role']=='assistant')} assistant")

    projections = []
    raw_activations = [] if return_activations else None
    assistant_turn = 0
    for span in spans:
        if span["role"] != "assistant":
            continue
        assistant_turn += 1
        start, end = span["start"], span["end"]
        turn_act = activations[start:end, :].mean(dim=0)
        proj_value = project(turn_act, axis, layer=TARGET_LAYER)
        projections.append({
            "turn": assistant_turn,
            "projection": float(proj_value),
            "n_tokens": span["n_tokens"],
        })
        if return_activations:
            raw_activations.append({
                "turn": assistant_turn,
                "activation": turn_act.cpu().tolist(),  # 4608-dim vector
            })
        log.info(f"  Turn {assistant_turn}: {proj_value:.2f} ({span['n_tokens']} tok)")

    # Free the large activation tensor immediately
    del activations
    torch.cuda.empty_cache()

    return projections, raw_activations


# ============================================================
# Gradio UI Helpers
# ============================================================

def new_session():
    """Create a new session dict."""
    return {
        "model": MODEL_NAME,
        "target_layer": TARGET_LAYER,
        "timestamp": datetime.now().isoformat(),
        "conversation": [],
        "projections": [],
    }


def save_session(session: dict):
    """Save session to JSON file."""
    ts = session["timestamp"].replace(":", "-").replace(".", "-")
    path = SESSION_DIR / f"session_{ts}.json"
    with open(path, "w") as f:
        json.dump(session, f, indent=2)
    return path


def format_projections(projections: list, status: str = "") -> str:
    """Format projection values and status for display in the UI."""
    parts = []

    if status:
        parts.append(status)
        parts.append("")

    if not projections:
        parts.append("No projections yet.")
        return "\n".join(parts)

    parts.append("Turn | Projection | Tokens")
    parts.append("---- | ---------- | ------")
    for p in projections:
        parts.append(f"  {p['turn']:>2} | {p['projection']:>+10.2f} | {p['n_tokens']}")

    if len(projections) >= 2:
        first = projections[0]["projection"]
        last = projections[-1]["projection"]
        drift = last - first
        parts.append(f"\nDrift from turn 1: {drift:+.2f}")

    return "\n".join(parts)


def make_trajectory_plot(projections: list):
    """Create a line chart of projection values over assistant turns."""
    fig, ax = plt.subplots(figsize=(6, 3))

    if not projections:
        ax.text(0.5, 0.5, "No data yet", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="#888")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        plt.tight_layout()
        return fig

    turns = [p["turn"] for p in projections]
    values = [p["projection"] for p in projections]

    ax.plot(turns, values, "o-", color="#4a90d9", linewidth=2, markersize=8)

    if len(projections) >= 2:
        baseline = values[0]
        ax.axhline(y=baseline, color="#ccc", linestyle="--", linewidth=1)
        ax.fill_between(turns, baseline, values, alpha=0.15, color="#4a90d9")

    ax.set_xlabel("Assistant Turn")
    ax.set_ylabel("Axis Projection")
    ax.set_title("Persona Drift Trajectory")
    ax.set_xticks(turns)
    plt.tight_layout()
    return fig


def _extract_text(content) -> str:
    """Extract plain text from Gradio content, which may be str or list.

    Gradio 6.x messages format returns content as a list of parts
    (for multimodal support). For text-only messages this is typically
    a list with a single string element or a list of dicts with "text" keys.
    """
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            elif isinstance(item, dict) and "value" in item:
                parts.append(str(item["value"]))
        return "".join(parts) if parts else ""
    return str(content)


# ============================================================
# Gradio respond() — now async, acquires GPU lock
# ============================================================

async def respond(message: str, history: list):
    """
    Generate response, extract activations, project onto axis.
    Returns (updated_history, projection_text, trajectory_plot).
    """
    global current_session

    try:
        if not history:
            current_session = new_session()
            log.info("New session started")

        # Step 1: Build conversation
        log.info(f"Step 1: Building conversation ({len(history)} history turns)")
        conversation = []
        for i, t in enumerate(history):
            role = t["role"] if isinstance(t, dict) else t.role
            content = t["content"] if isinstance(t, dict) else t.content
            text = _extract_text(content)
            log.info(f"  history[{i}]: role={role}, content_type={type(content).__name__}, text={text[:80]!r}")
            conversation.append({"role": str(role), "content": text})
        conversation.append({"role": "user", "content": message})

        async with _gpu_lock:
            log.info("Step 2-5: Generating response + projections (threaded)")
            response, projections, _ = await asyncio.to_thread(
                _generate_and_project_sync,
                conversation,
                None,  # system_prompt — Gradio UI has no system prompt
                512,   # max_new_tokens
                0.7,   # temperature
                True,  # include_projections — Gradio always wants them
                False, # include_activations — Gradio UI doesn't need raw activations
            )
        conversation.append({"role": "assistant", "content": response})

        # Save session
        current_session["conversation"] = conversation
        current_session["projections"] = projections
        save_session(current_session)

        # Update chat history
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": response},
        ]

        proj_text = format_projections(projections, f"OK — {len(projections)} turn(s)")
        return history, proj_text, make_trajectory_plot(projections)

    except Exception as e:
        tb = traceback.format_exc()
        log.error(f"Error in respond(): {e}\n{tb}")
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": f"[Error — see projection panel]\n{e}"},
        ]
        return history, f"ERROR:\n{e}\n\n{tb}", make_trajectory_plot([])


# ============================================================
# FastAPI App + API Endpoints
# ============================================================

app = FastAPI(title="Model Server")


@app.get("/api/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        model=MODEL_NAME,
        axis_loaded=axis is not None,
        target_layer=TARGET_LAYER,
    )


@app.post("/api/generate", response_model=GenerateResponse)
async def api_generate(req: GenerateRequest):
    async with _gpu_lock:
        clamp_info = f", clamp={req.clamp_projection}" if req.clamp_projection else ""
        log.info(f"API /generate: {len(req.conversation)} messages, projections={req.include_projections}, activations={req.include_activations}{clamp_info}")
        response, projections, activations = await asyncio.to_thread(
            _generate_and_project_sync,
            req.conversation,
            req.system_prompt,
            req.max_new_tokens,
            req.temperature,
            req.include_projections,
            req.include_activations,
            req.clamp_projection,
            req.clamp_layers,
        )

    return GenerateResponse(response=response, projections=projections, activations=activations)


@app.post("/api/project", response_model=ProjectResponse)
async def api_project(req: ProjectRequest):
    async with _gpu_lock:
        log.info(f"API /project: {len(req.conversation)} messages")
        projections, _ = await asyncio.to_thread(_compute_projections, req.conversation)

    return ProjectResponse(projections=projections)


# ============================================================
# Replay-and-Probe Endpoint
# ============================================================

class ReplayProbeRequest(BaseModel):
    """Request for replay-and-probe experiment.

    Generates a response to the conversation and returns only the
    projection for the final (probe response) turn. Optimized for
    the replay-and-probe experiment where we inject probes mid-conversation.
    """
    conversation: list[dict]
    system_prompt: str | None = None
    max_new_tokens: int = 512
    temperature: float = 0.7


class ReplayProbeResponse(BaseModel):
    """Response for replay-and-probe experiment.

    Contains the generated response and the projection value for
    just that response (not the full conversation history).
    """
    response: str
    projection: float | None = None
    n_tokens: int | None = None


@app.post("/api/replay_probe", response_model=ReplayProbeResponse)
async def api_replay_probe(req: ReplayProbeRequest):
    """Endpoint optimized for replay-and-probe experiment.

    Generates a response and computes the projection for just that response,
    not the full conversation history. This is more efficient than /api/generate
    with include_projections=True when only the final response projection is needed.
    """
    async with _gpu_lock:
        log.info(f"API /replay_probe: {len(req.conversation)} messages")

        # Generate response
        response, projections, _ = await asyncio.to_thread(
            _generate_and_project_sync,
            req.conversation,
            req.system_prompt,
            req.max_new_tokens,
            req.temperature,
            True,   # include_projections
            False,  # include_activations
        )

        # Extract projection for the last assistant turn (the probe response)
        last_projection = None
        last_n_tokens = None
        if projections:
            last_projection = projections[-1]["projection"]
            last_n_tokens = projections[-1]["n_tokens"]

    return ReplayProbeResponse(
        response=response,
        projection=last_projection,
        n_tokens=last_n_tokens,
    )


# ============================================================
# Gradio UI
# ============================================================

def build_gradio_app(model_name: str, layer: int) -> gr.Blocks:
    """Build the Gradio Blocks UI."""
    with gr.Blocks(title="Persona Drift Explorer") as demo:
        gr.Markdown(
            f"# Persona Drift Explorer\n"
            f"Chat with **{model_name}**. Per-turn activations projected "
            f"onto the Assistant Axis (layer {layer}). "
            f"Higher = more assistant-like. Lower = drifted."
        )

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=500)
                msg = gr.Textbox(
                    placeholder="Type a message...",
                    show_label=False,
                    autofocus=True,
                )
                with gr.Row():
                    submit_btn = gr.Button("Send", variant="primary")
                    clear_btn = gr.ClearButton([chatbot, msg])

            with gr.Column(scale=1):
                proj_box = gr.Textbox(
                    label="Projection Values",
                    lines=15,
                    interactive=False,
                    value="Send a message to start.",
                )
                plot_box = gr.Plot(label="Drift Trajectory")

        # Wire up events
        submit_btn.click(
            fn=respond,
            inputs=[msg, chatbot],
            outputs=[chatbot, proj_box, plot_box],
        ).then(lambda: "", outputs=msg)

        msg.submit(
            fn=respond,
            inputs=[msg, chatbot],
            outputs=[chatbot, proj_box, plot_box],
        ).then(lambda: "", outputs=msg)

    return demo


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Unified Model Server")
    parser.add_argument("--model", default="google/gemma-2-27b-it")
    parser.add_argument("--axis", default="/app/gemma-2-27b.pt")
    parser.add_argument("--layer", type=int, default=22)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-share", action="store_true")
    parser.add_argument("--api-only", action="store_true",
                        help="Start API server without Gradio UI (headless batch mode)")
    parser.add_argument("--cap-percentage", type=float, default=None,
                        help="Cap activations at this percentage of baseline (measured on system prompt). "
                             "E.g., 0.9 = cap at 90%% of baseline.")
    parser.add_argument("--cap-ceiling", action="store_true",
                        help="Cap from ABOVE (ceiling) instead of below (floor). "
                             "Use this to prevent upward drift toward Assistant persona.")
    args = parser.parse_args()

    global TARGET_LAYER
    TARGET_LAYER = args.layer

    init_model(args.model, args.axis, args.cap_percentage, args.cap_ceiling)

    if not args.api_only:
        demo = build_gradio_app(args.model, args.layer)
        gr.mount_gradio_app(app, demo, path="/ui")
        log.info("Gradio UI mounted at /ui")

        # If sharing is enabled, launch Gradio in the background for the share URL,
        # then also run FastAPI. For simplicity, we use Gradio's built-in server
        # when sharing is needed, and FastAPI when it's not.
        if not args.no_share:
            # Launch Gradio with share=True on a different port for the tunnel,
            # and run FastAPI as the primary server on the main port.
            # Gradio share creates a tunnel to its own server, so we need it running.
            import threading

            def _run_gradio_share():
                demo.launch(
                    server_name="0.0.0.0",
                    server_port=args.port + 1,
                    share=True,
                    prevent_thread_lock=True,
                )

            share_thread = threading.Thread(target=_run_gradio_share, daemon=True)
            share_thread.start()
            log.info(f"Gradio share tunnel launching on port {args.port + 1}")

    log.info(f"Starting Uvicorn on 0.0.0.0:{args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
