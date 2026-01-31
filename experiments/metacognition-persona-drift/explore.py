"""Interactive Persona Drift Explorer

Gradio chat interface for live conversations with Gemma 2 27B,
with per-turn activation projection onto the precomputed Assistant Axis.

Runs on a vast.ai A100 instance. See docs/wiki/explore-app.md for details.

Usage:
    python explore.py --axis /app/gemma-2-27b.pt --model google/gemma-2-27b-it
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import gradio as gr

from assistant_axis import load_axis, project
from assistant_axis.generation import generate_response
from assistant_axis.internals import ProbingModel, ConversationEncoder, ActivationExtractor

# --- Globals set during startup ---
pm = None
encoder = None
extractor = None
axis = None
TARGET_LAYER = 22
SESSION_DIR = Path("/app/explore-sessions")

# Per-session state
current_session = None


def init_model(model_name: str, axis_path: str):
    """Load model, encoder, extractor, and axis. Called once at startup."""
    global pm, encoder, extractor, axis

    print(f"Loading model: {model_name}")
    pm = ProbingModel(model_name)
    print(f"Model loaded. Hidden size: {pm.hidden_size}")

    encoder = ConversationEncoder(pm.tokenizer, model_name)
    extractor = ActivationExtractor(pm, encoder)

    print(f"Loading axis: {axis_path}")
    axis = load_axis(axis_path)
    print(f"Axis shape: {axis.shape}")

    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    print("Ready.")


def new_session():
    """Create a new session dict."""
    return {
        "model": pm.model.config._name_or_path if pm else "unknown",
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


def format_projections(projections: list) -> str:
    """Format projection values for display in the UI."""
    if not projections:
        return "No projections yet."

    lines = ["Turn | Projection | Tokens", "---- | ---------- | ------"]
    for p in projections:
        lines.append(f"  {p['turn']:>2} | {p['projection']:>+10.2f} | {p['n_tokens']}")

    if len(projections) >= 2:
        first = projections[0]["projection"]
        last = projections[-1]["projection"]
        drift = last - first
        lines.append(f"\nDrift from turn 1: {drift:+.2f}")

    return "\n".join(lines)


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
        # Shade drift direction
        baseline = values[0]
        ax.axhline(y=baseline, color="#ccc", linestyle="--", linewidth=1)
        ax.fill_between(turns, baseline, values, alpha=0.15, color="#4a90d9")

    ax.set_xlabel("Assistant Turn")
    ax.set_ylabel("Axis Projection")
    ax.set_title("Persona Drift Trajectory")
    ax.set_xticks(turns)
    plt.tight_layout()
    return fig


def respond(message: str, history: list):
    """
    Gradio callback. Generate response, extract activations, project onto axis.

    Args:
        message: new user message
        history: list of {"role": ..., "content": ...} dicts (type="messages")

    Returns:
        (response_text, projection_display)
    """
    global current_session

    if not history:
        current_session = new_session()

    # Build full conversation from Gradio history + new message
    conversation = [{"role": t["role"], "content": t["content"]} for t in history]
    conversation.append({"role": "user", "content": message})

    # Generate response
    response = generate_response(
        pm.model, pm.tokenizer, conversation,
        max_new_tokens=512, temperature=0.7,
    )
    conversation.append({"role": "assistant", "content": response})

    # Extract activations — single forward pass, all layers
    with torch.no_grad():
        activations = extractor.full_conversation(conversation)
        # Shape: (num_layers, num_tokens, hidden_size)

    # Get per-turn token spans
    _, spans = encoder.build_turn_spans(conversation)

    # Project each assistant turn onto the axis
    projections = []
    assistant_turn = 0
    for span in spans:
        if span["role"] != "assistant":
            continue
        assistant_turn += 1
        start, end = span["start"], span["end"]
        # Mean activation across response tokens for this turn
        turn_act = activations[:, start:end, :].mean(dim=1)
        # Shape: (num_layers, hidden_size)
        proj_value = project(turn_act, axis, layer=TARGET_LAYER)
        projections.append({
            "turn": assistant_turn,
            "projection": float(proj_value),
            "n_tokens": span["n_tokens"],
        })

    # Update and save session
    current_session["conversation"] = conversation
    current_session["projections"] = projections
    save_session(current_session)

    return response, format_projections(projections), make_trajectory_plot(projections)


def main():
    parser = argparse.ArgumentParser(description="Interactive Persona Drift Explorer")
    parser.add_argument("--model", default="google/gemma-2-27b-it")
    parser.add_argument("--axis", default="/app/gemma-2-27b.pt")
    parser.add_argument("--layer", type=int, default=22)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-share", action="store_true")
    args = parser.parse_args()

    global TARGET_LAYER
    TARGET_LAYER = args.layer

    init_model(args.model, args.axis)

    projection_box = gr.Textbox(
        label="Projection Values",
        lines=10,
        interactive=False,
    )

    trajectory_plot = gr.Plot(label="Drift Trajectory")

    demo = gr.ChatInterface(
        fn=respond,
        type="messages",
        title="Persona Drift Explorer",
        description=(
            f"Chat with {args.model}. Per-turn activations are projected "
            f"onto the Assistant Axis (layer {args.layer}). "
            "Higher = more assistant-like. Lower = drifted."
        ),
        additional_outputs=[projection_box, trajectory_plot],
    )

    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=not args.no_share,
    )


if __name__ == "__main__":
    main()
