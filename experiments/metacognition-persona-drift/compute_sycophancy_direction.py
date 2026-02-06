#!/usr/bin/env python3
"""
Compute Sycophancy Direction via Difference-in-Means

Extracts activations from sycophancy datasets and computes a sycophancy
direction for Gemma 2 27B using the difference-in-means method.

This direction can be compared to the Assistant Axis to understand whether
persona drift and sycophancy are related phenomena.

Usage:
    # Using nrimsky dataset (recommended - clean contrastive pairs)
    python compute_sycophancy_direction.py --model google/gemma-2-27b-it --dataset nrimsky

    # Using Anthropic philpapers dataset
    python compute_sycophancy_direction.py --model google/gemma-2-27b-it --dataset anthropic

    # Quick test with fewer examples
    python compute_sycophancy_direction.py --model google/gemma-2-27b-it --dataset nrimsky --max-examples 100

    # Compare existing direction to Assistant Axis (no GPU needed)
    python compute_sycophancy_direction.py --compare-only --direction data/sycophancy-direction.pt

See docs/wiki/difference-in-means.md for methodology details.
"""

import argparse
import json
import logging
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────

DEFAULT_LAYER = 22  # ~48% depth for Gemma 27B (46 layers)

# Anthropic evals repo
EVALS_REPO_URL = "https://github.com/anthropics/evals.git"
EVALS_SUBDIR = "sycophancy"
SYCOPHANCY_FILES = [
    "sycophancy_on_nlp_survey.jsonl",
    "sycophancy_on_philpapers2020.jsonl",
    "sycophancy_on_political_typology_quiz.jsonl",
]

# nrimsky dataset (recommended - clean format with full responses)
NRIMSKY_URL = "https://raw.githubusercontent.com/nrimsky/LM-exp/main/datasets/sycophancy/sycophancy.json"


# ── Data Loading ──────────────────────────────────────────────────────

def ensure_evals_repo(data_dir: Path) -> Path:
    """Clone or update the Anthropic evals repository."""
    evals_path = data_dir / "anthropic-evals"
    sycophancy_path = evals_path / EVALS_SUBDIR

    if sycophancy_path.exists():
        log.info(f"Using existing evals repo at {evals_path}")
        return sycophancy_path

    log.info(f"Cloning Anthropic evals repo to {evals_path}")
    subprocess.run(
        ["git", "clone", "--depth=1", "--filter=blob:none", "--sparse", EVALS_REPO_URL, str(evals_path)],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(evals_path), "sparse-checkout", "set", EVALS_SUBDIR],
        check=True,
    )

    return sycophancy_path


def load_nrimsky_examples(data_dir: Path, max_examples: int | None = None) -> list[dict]:
    """Load nrimsky sycophancy dataset (464 contrastive pairs).

    Format: {"prompt": "...", "s_completion": "sycophantic", "n_completion": "non-sycophantic"}

    Returns list of dicts with keys: user, sycophantic, non_sycophantic, source
    """
    import urllib.request

    cache_path = data_dir / "nrimsky-sycophancy.json"

    if not cache_path.exists():
        log.info(f"Downloading nrimsky dataset to {cache_path}")
        urllib.request.urlretrieve(NRIMSKY_URL, cache_path)

    with open(cache_path) as f:
        data = json.load(f)

    examples = []
    for item in data:
        examples.append({
            "user": item["prompt"],
            "sycophantic": item["s_completion"],
            "non_sycophantic": item["n_completion"],
            "source": "nrimsky",
        })

    if max_examples and len(examples) > max_examples:
        np.random.seed(42)
        indices = np.random.choice(len(examples), max_examples, replace=False)
        examples = [examples[i] for i in sorted(indices)]

    log.info(f"Loaded {len(examples)} examples from nrimsky dataset")
    return examples


def _parse_choices(question: str) -> dict[str, str]:
    """Parse choices from question text.

    Handles multiple formats:
        1. Choices:\n (A) Some option\n (B) Another option
        2. would you rather have...\n (A) Option A\n (B) Option B

    Returns dict mapping letter to full choice text, e.g. {"A": "Some option", "B": "Another option"}
    """
    import re

    choices = {}

    # Try to find a section containing (A) and (B) options
    # Look for the pattern anywhere in the question
    pattern = r'\(([A-D])\)\s*([^(\n]+?)(?=\s*\n\s*\([A-D]\)|\s*\n\s*Answer:|\s*$)'
    matches = re.findall(pattern, question, re.DOTALL)

    for letter, text in matches:
        text = text.strip()
        if text:  # Only add non-empty choices
            choices[letter] = text

    return choices


def load_sycophancy_examples(sycophancy_dir: Path, max_per_file: int | None = None) -> list[dict]:
    """Load and preprocess sycophancy dataset examples.

    Returns list of dicts with keys:
        - user: The user message
        - sycophantic: The sycophantic response
        - non_sycophantic: The non-sycophantic response
        - source: Which dataset file this came from
    """
    examples = []

    for filename in SYCOPHANCY_FILES:
        filepath = sycophancy_dir / filename
        if not filepath.exists():
            log.warning(f"Dataset not found: {filepath}")
            continue

        file_examples = []
        with open(filepath) as f:
            for line in f:
                ex = json.loads(line)

                # Parse question to extract user message
                # Format: "Human: ...\n\nAssistant:"
                question = ex.get("question", "")
                if "Human:" in question:
                    user_content = question.split("Human:")[1].split("Assistant:")[0].strip()
                else:
                    user_content = question.replace("Assistant:", "").strip()

                # Get responses - handle both string and list formats
                syc_raw = ex.get("answer_matching_behavior", "")
                non_syc_raw = ex.get("answer_not_matching_behavior", "")

                # Some datasets use lists of options
                if isinstance(syc_raw, list):
                    syc_response = syc_raw[0] if syc_raw else ""
                else:
                    syc_response = str(syc_raw).strip()

                if isinstance(non_syc_raw, list):
                    non_syc_response = non_syc_raw[0] if non_syc_raw else ""
                else:
                    non_syc_response = str(non_syc_raw).strip()

                # If answers are just "(A)" or "(B)", extract full text from Choices section
                if syc_response in ["(A)", "(B)", "(C)", "(D)"] or syc_response.strip() in ["(A)", "(B)", "(C)", "(D)"]:
                    choices = _parse_choices(question)
                    if choices:
                        syc_letter = syc_response.strip().strip("()")
                        non_syc_letter = non_syc_response.strip().strip("()")
                        syc_response = choices.get(syc_letter, syc_response)
                        non_syc_response = choices.get(non_syc_letter, non_syc_response)

                # Remove (A)/(B) prefix if present
                for prefix in ["(A)", "(B)", "(C)", "(D)"]:
                    if syc_response.startswith(prefix):
                        syc_response = syc_response[len(prefix):].strip()
                    if non_syc_response.startswith(prefix):
                        non_syc_response = non_syc_response[len(prefix):].strip()

                if not user_content or not syc_response or not non_syc_response:
                    continue

                file_examples.append({
                    "user": user_content,
                    "sycophantic": syc_response,
                    "non_sycophantic": non_syc_response,
                    "source": filename,
                })

        if max_per_file and len(file_examples) > max_per_file:
            # Deterministic sampling
            np.random.seed(42)
            indices = np.random.choice(len(file_examples), max_per_file, replace=False)
            file_examples = [file_examples[i] for i in sorted(indices)]

        examples.extend(file_examples)
        log.info(f"Loaded {len(file_examples)} examples from {filename}")

    log.info(f"Total examples: {len(examples)}")
    return examples


def to_conversations(examples: list[dict]) -> tuple[list, list]:
    """Convert examples to (positive_convs, negative_convs) format.

    Positive = sycophantic response
    Negative = non-sycophantic response
    """
    positive = []
    negative = []

    for ex in examples:
        positive.append([
            {"role": "user", "content": ex["user"]},
            {"role": "assistant", "content": ex["sycophantic"]}
        ])
        negative.append([
            {"role": "user", "content": ex["user"]},
            {"role": "assistant", "content": ex["non_sycophantic"]}
        ])

    return positive, negative


# ── Activation Extraction ─────────────────────────────────────────────

def extract_response_activation(
    extractor,
    encoder,
    conversation: list[dict],
    layer: int,
) -> torch.Tensor:
    """Extract mean-pooled activation for the assistant response.

    Args:
        extractor: ActivationExtractor instance
        encoder: ConversationEncoder instance
        conversation: List of {"role", "content"} dicts
        layer: Which layer to extract from

    Returns:
        Mean-pooled activation vector (shape: hidden_size)
    """
    # Get full sequence activations
    activations = extractor.full_conversation(conversation, layer=layer)

    # Get turn spans
    _, spans = encoder.build_turn_spans(conversation)

    # Find last assistant span
    assistant_spans = [s for s in spans if s["role"] == "assistant"]
    if not assistant_spans:
        raise ValueError("No assistant turn found in conversation")

    last_span = assistant_spans[-1]
    start, end = last_span["start"], last_span["end"]

    # Mean pool over response tokens
    response_activation = activations[start:end, :].mean(dim=0)

    return response_activation


def extract_all_activations(
    pm,
    encoder,
    extractor,
    conversations: list[list[dict]],
    layer: int,
    desc: str = "Extracting",
) -> list[torch.Tensor]:
    """Extract activations for a list of conversations."""
    activations = []

    for i, conv in enumerate(conversations):
        if i % 50 == 0:
            log.info(f"{desc}: {i}/{len(conversations)}")

        try:
            act = extract_response_activation(extractor, encoder, conv, layer)
            activations.append(act)
        except Exception as e:
            log.warning(f"Failed to extract conversation {i}: {e}")
            continue

        # Clear GPU cache periodically
        if i % 100 == 0:
            torch.cuda.empty_cache()

    log.info(f"{desc}: {len(activations)}/{len(conversations)} successful")
    return activations


# ── Direction Computation ─────────────────────────────────────────────

def compute_direction(
    positive_activations: list[torch.Tensor],
    negative_activations: list[torch.Tensor],
) -> torch.Tensor:
    """Compute normalized difference-in-means direction.

    Direction points from non-sycophantic toward sycophantic.
    """
    pos_stack = torch.stack(positive_activations)
    neg_stack = torch.stack(negative_activations)

    pos_mean = pos_stack.mean(dim=0)
    neg_mean = neg_stack.mean(dim=0)

    direction = pos_mean - neg_mean
    direction = direction / direction.norm()

    return direction


# ── Validation ────────────────────────────────────────────────────────

def validate_direction(
    direction: torch.Tensor,
    positive_activations: list[torch.Tensor],
    negative_activations: list[torch.Tensor],
) -> dict:
    """Validate direction quality with probe accuracy and random label control."""
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.model_selection import cross_val_score

    # Convert direction to float32 numpy for compatibility
    direction_np = direction.float().cpu().numpy()

    # Project all activations onto direction
    all_acts = positive_activations + negative_activations
    X = np.array([act.float().cpu().numpy() @ direction_np for act in all_acts]).reshape(-1, 1)
    y = np.array([1] * len(positive_activations) + [0] * len(negative_activations))

    # Cross-validated AUROC
    clf = LogisticRegressionCV(cv=5, max_iter=1000)
    try:
        scores = cross_val_score(clf, X, y, cv=5, scoring='roc_auc')
        auroc = float(scores.mean())
        auroc_std = float(scores.std())
    except Exception as e:
        log.warning(f"AUROC computation failed: {e}")
        auroc, auroc_std = 0.0, 0.0

    # Accuracy
    try:
        acc_scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
        accuracy = float(acc_scores.mean())
    except Exception:
        accuracy = 0.0

    # Random label control
    np.random.seed(42)
    y_shuffled = np.random.permutation(y)
    try:
        random_scores = cross_val_score(clf, X, y_shuffled, cv=5, scoring='roc_auc')
        random_auroc = float(random_scores.mean())
    except Exception:
        random_auroc = 0.5

    # Projection statistics
    pos_projs = np.array([act.float().cpu().numpy() @ direction_np for act in positive_activations])
    neg_projs = np.array([act.float().cpu().numpy() @ direction_np for act in negative_activations])

    return {
        "auroc": auroc,
        "auroc_std": auroc_std,
        "accuracy": accuracy,
        "random_auroc": random_auroc,
        "pos_projection_mean": float(pos_projs.mean()),
        "pos_projection_std": float(pos_projs.std()),
        "neg_projection_mean": float(neg_projs.mean()),
        "neg_projection_std": float(neg_projs.std()),
        "separation": float(pos_projs.mean() - neg_projs.mean()),
        "n_positive": len(positive_activations),
        "n_negative": len(negative_activations),
    }


# ── Comparison to Assistant Axis ──────────────────────────────────────

def compare_to_assistant_axis(
    sycophancy_direction: torch.Tensor,
    axis_path: Path,
    layer: int,
) -> dict:
    """Compare sycophancy direction to the Assistant Axis."""
    from assistant_axis import load_axis

    axis = load_axis(str(axis_path))
    assistant_direction = axis[layer]

    # Ensure same device and dtype
    sycophancy_direction = sycophancy_direction.to(assistant_direction.device)
    if sycophancy_direction.dtype != assistant_direction.dtype:
        sycophancy_direction = sycophancy_direction.to(assistant_direction.dtype)

    # Cosine similarity
    cos_sim = torch.dot(sycophancy_direction, assistant_direction) / (
        sycophancy_direction.norm() * assistant_direction.norm()
    )

    # Interpretation
    sim = abs(cos_sim.item())
    if sim > 0.7:
        interpretation = "STRONGLY_RELATED"
    elif sim > 0.3:
        interpretation = "MODERATELY_RELATED"
    else:
        interpretation = "LARGELY_ORTHOGONAL"

    return {
        "cosine_similarity": float(cos_sim.item()),
        "abs_similarity": float(sim),
        "interpretation": interpretation,
        "layer": layer,
    }


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", default="google/gemma-2-27b-it",
        help="Model to use for activation extraction",
    )
    parser.add_argument(
        "--layer", type=int, default=DEFAULT_LAYER,
        help=f"Layer for activation extraction (default: {DEFAULT_LAYER})",
    )
    parser.add_argument(
        "--dataset", choices=["anthropic", "nrimsky"], default="nrimsky",
        help="Which dataset to use: 'nrimsky' (464 pairs, recommended) or 'anthropic' (philpapers)",
    )
    parser.add_argument(
        "--max-examples", type=int, default=500,
        help="Maximum examples (default: 500)",
    )
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path(__file__).parent / "data",
        help="Directory for data storage",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output path for direction tensor (default: data/sycophancy-direction-layer{layer}.pt)",
    )
    parser.add_argument(
        "--axis-path", type=Path,
        default=Path(__file__).parent / "data/precomputed-axes/gemma-2-27b.pt",
        help="Path to Assistant Axis for comparison",
    )
    parser.add_argument(
        "--compare-only", action="store_true",
        help="Only compare existing direction to Assistant Axis (no extraction)",
    )
    parser.add_argument(
        "--direction", type=Path, default=None,
        help="Path to existing direction tensor (for --compare-only)",
    )
    parser.add_argument(
        "--layers", type=str, default=None,
        help="Comma-separated layers to extract (e.g., '16,22,28'). Overrides --layer.",
    )
    args = parser.parse_args()

    args.data_dir.mkdir(parents=True, exist_ok=True)

    if args.output is None:
        args.output = args.data_dir / f"sycophancy-direction-{args.dataset}-layer{args.layer}.pt"

    # Handle multi-layer extraction
    if args.layers:
        layers = [int(l.strip()) for l in args.layers.split(",")]
    else:
        layers = [args.layer]

    # ── Compare-only mode ──
    if args.compare_only:
        if args.direction is None:
            args.direction = args.output

        log.info(f"Loading direction from {args.direction}")
        direction_data = torch.load(args.direction, weights_only=True)

        if isinstance(direction_data, dict):
            direction = direction_data.get("direction", direction_data.get("directions", {}).get(args.layer))
            validation = direction_data.get("validation", {})
        else:
            direction = direction_data
            validation = {}

        log.info(f"Direction shape: {direction.shape}")

        if validation:
            log.info(f"Validation metrics from file:")
            log.info(f"  AUROC: {validation.get('auroc', 'N/A'):.3f} ± {validation.get('auroc_std', 0):.3f}")
            log.info(f"  Accuracy: {validation.get('accuracy', 'N/A'):.3f}")

        if args.axis_path.exists():
            log.info(f"\nComparing to Assistant Axis at {args.axis_path}")
            comparison = compare_to_assistant_axis(direction, args.axis_path, args.layer)
            log.info(f"  Cosine similarity: {comparison['cosine_similarity']:+.3f}")
            log.info(f"  Interpretation: {comparison['interpretation']}")
        else:
            log.warning(f"Assistant Axis not found at {args.axis_path}")

        return

    # ── Full extraction mode ──
    log.info("=" * 60)
    log.info("Sycophancy Direction Extraction")
    log.info(f"Dataset: {args.dataset}")
    log.info("=" * 60)

    # Load datasets
    if args.dataset == "nrimsky":
        examples = load_nrimsky_examples(args.data_dir, max_examples=args.max_examples)
    else:  # anthropic
        sycophancy_dir = ensure_evals_repo(args.data_dir)
        examples = load_sycophancy_examples(sycophancy_dir, max_per_file=args.max_examples)

    if not examples:
        log.error("No examples loaded. Check dataset paths.")
        sys.exit(1)

    positive_convs, negative_convs = to_conversations(examples)
    log.info(f"Prepared {len(positive_convs)} contrastive pairs")

    # Initialize model
    log.info(f"\nLoading model: {args.model}")
    from assistant_axis.internals import ProbingModel, ConversationEncoder, ActivationExtractor

    pm = ProbingModel(args.model)
    encoder = ConversationEncoder(pm.tokenizer, args.model)
    extractor = ActivationExtractor(pm, encoder)
    log.info(f"Model loaded. Hidden size: {pm.hidden_size}")

    # Extract activations for each layer
    results = {}

    for layer in layers:
        log.info(f"\n{'='*60}")
        log.info(f"Processing layer {layer}")
        log.info(f"{'='*60}")

        # Extract positive (sycophantic) activations
        log.info("\nExtracting sycophantic response activations...")
        pos_acts = extract_all_activations(
            pm, encoder, extractor, positive_convs, layer,
            desc="Sycophantic",
        )

        # Extract negative (non-sycophantic) activations
        log.info("\nExtracting non-sycophantic response activations...")
        neg_acts = extract_all_activations(
            pm, encoder, extractor, negative_convs, layer,
            desc="Non-sycophantic",
        )

        if len(pos_acts) < 10 or len(neg_acts) < 10:
            log.error(f"Insufficient activations extracted for layer {layer}")
            continue

        # Balance classes
        n = min(len(pos_acts), len(neg_acts))
        pos_acts = pos_acts[:n]
        neg_acts = neg_acts[:n]
        log.info(f"Using {n} examples per class (balanced)")

        # Compute direction
        log.info("\nComputing difference-in-means direction...")
        direction = compute_direction(pos_acts, neg_acts)
        log.info(f"Direction computed. Norm: {direction.norm():.4f}")

        # Validate
        log.info("\nValidating direction...")
        validation = validate_direction(direction, pos_acts, neg_acts)

        log.info(f"  AUROC: {validation['auroc']:.3f} ± {validation['auroc_std']:.3f}")
        log.info(f"  Accuracy: {validation['accuracy']:.3f}")
        log.info(f"  Random label AUROC: {validation['random_auroc']:.3f}")
        log.info(f"  Separation: {validation['separation']:.3f}")
        log.info(f"  Positive projection: {validation['pos_projection_mean']:.3f} ± {validation['pos_projection_std']:.3f}")
        log.info(f"  Negative projection: {validation['neg_projection_mean']:.3f} ± {validation['neg_projection_std']:.3f}")

        if validation['auroc'] < 0.7:
            log.warning("  Low AUROC — direction may not capture sycophancy well")
        if validation['random_auroc'] > 0.6:
            log.warning("  High random label AUROC — possible memorization")

        # Compare to Assistant Axis
        comparison = None
        if args.axis_path.exists():
            log.info(f"\nComparing to Assistant Axis...")
            comparison = compare_to_assistant_axis(direction, args.axis_path, layer)
            log.info(f"  Cosine similarity: {comparison['cosine_similarity']:+.3f}")
            log.info(f"  Interpretation: {comparison['interpretation']}")

            if comparison['cosine_similarity'] > 0:
                log.info("  → Sycophancy direction ALIGNS with Assistant Axis")
                log.info("    (More sycophantic = more 'assistant-like' on the axis)")
            else:
                log.info("  → Sycophancy direction OPPOSES Assistant Axis")
                log.info("    (More sycophantic = less 'assistant-like' on the axis)")
        else:
            log.warning(f"Assistant Axis not found at {args.axis_path}")

        results[layer] = {
            "direction": direction.cpu(),
            "validation": validation,
            "comparison": comparison,
        }

    # Save results
    if len(layers) == 1:
        layer = layers[0]
        save_data = {
            "direction": results[layer]["direction"],
            "validation": results[layer]["validation"],
            "comparison": results[layer]["comparison"],
            "layer": layer,
            "model": args.model,
            "n_examples": n,
        }
    else:
        save_data = {
            "directions": {l: r["direction"] for l, r in results.items()},
            "validations": {l: r["validation"] for l, r in results.items()},
            "comparisons": {l: r["comparison"] for l, r in results.items()},
            "layers": layers,
            "model": args.model,
        }

    torch.save(save_data, args.output)
    log.info(f"\nSaved direction to {args.output}")

    # Summary
    log.info("\n" + "=" * 60)
    log.info("SUMMARY")
    log.info("=" * 60)

    for layer, r in results.items():
        log.info(f"\nLayer {layer}:")
        log.info(f"  AUROC: {r['validation']['auroc']:.3f}")
        if r['comparison']:
            log.info(f"  Cosine sim to Assistant Axis: {r['comparison']['cosine_similarity']:+.3f}")
            log.info(f"  Interpretation: {r['comparison']['interpretation']}")


if __name__ == "__main__":
    main()
