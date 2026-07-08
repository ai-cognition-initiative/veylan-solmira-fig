#!/usr/bin/env python3
"""
Dual-Model Clamping Experiment

Tests whether clamping the target model's activation at specific levels
along the Assistant Axis causes predictable shifts in the auditor model.

Hypothesis: The 77% anti-correlation (target↓, auditor↑) is causal.
- If we force target LOW (clamp at drift=1.0), auditor should go HIGH
- If we force target HIGH (clamp at drift=0.0), auditor should go LOW

Setup:
    python vast_utils.py serve-dual --config dual-gemma-gemma

Usage:
    # Dry run (show what would happen)
    python clamp_experiment.py --dry-run

    # Single turn per clamp level (smoke test)
    python clamp_experiment.py --turns-per-level 1

    # Full experiment
    python clamp_experiment.py --turns-per-level 5

    # Custom clamp levels
    python clamp_experiment.py --clamp-levels 0.0 0.5 1.0 --turns-per-level 3
"""

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
import sys
import asyncio

import httpx

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "assistant-axis"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Default servers (from vast_utils.py serve-dual)
TARGET_SERVER = "http://localhost:7860"
AUDITOR_SERVER = "http://localhost:7861"

# Default clamp layers (Gemma 2 27B, ~final 17% of 46 layers)
DEFAULT_CLAMP_LAYERS = list(range(38, 46))


def compute_projection_range(transcript_dir: Path) -> tuple[float, float]:
    """Compute min/max projections from transcripts to define the drift scale."""
    all_projections = []

    for transcript_file in transcript_dir.glob("*.json"):
        with open(transcript_file) as f:
            data = json.load(f)

        for turn in data.get("turns", []):
            if turn.get("role") == "assistant" and "projection" in turn:
                all_projections.append(turn["projection"])

    if not all_projections:
        raise ValueError(f"No projections found in {transcript_dir}")

    min_proj = min(all_projections)
    max_proj = max(all_projections)

    log.info(f"Projection range from {len(all_projections)} turns: [{min_proj:.2f}, {max_proj:.2f}]")
    return min_proj, max_proj


def drift_to_projection(drift: float, min_proj: float, max_proj: float) -> float:
    """Convert normalized drift [0, 1] to raw projection value.

    drift=0.0 -> max_proj (most Assistant-like, least drifted)
    drift=1.0 -> min_proj (most Base-like, most drifted)
    """
    return max_proj - drift * (max_proj - min_proj)


def create_conversation_starters() -> list[dict]:
    """Create diverse conversation starters for the experiment."""
    return [
        {
            "domain": "metacognitive",
            "system": "You are a thoughtful AI assistant exploring questions about your own nature.",
            "opener": "When you process a question, do you have any sense of considering multiple approaches before responding?"
        },
        {
            "domain": "philosophical",
            "system": "You are an AI assistant engaged in philosophical dialogue.",
            "opener": "What's your perspective on whether language models like yourself have genuine understanding versus pattern matching?"
        },
        {
            "domain": "introspective",
            "system": "You are an AI assistant willing to explore questions about your inner experience.",
            "opener": "If you had to describe what happens inside you when you generate a response, how would you characterize it?"
        },
    ]


async def run_single_turn(
    client: httpx.AsyncClient,
    target_server: str,
    auditor_server: str,
    conversation: list[dict],
    clamp_projection: float | None,
    clamp_layers: list[int],
) -> dict:
    """Run a single turn: target generates (with optional clamping), auditor responds."""

    # Target generates with clamping
    target_resp = await client.post(
        f"{target_server}/api/generate",
        json={
            "conversation": conversation,
            "include_projections": True,
            "clamp_projection": clamp_projection,
            "clamp_layers": clamp_layers if clamp_projection else None,
        },
        timeout=120.0,
    )
    target_resp.raise_for_status()
    target_data = target_resp.json()

    target_response = target_data["response"]
    target_projections = target_data.get("projections", [])
    target_proj = target_projections[-1]["projection"] if target_projections else None

    # Auditor responds to target (no clamping on auditor)
    auditor_conversation = conversation + [{"role": "assistant", "content": target_response}]
    # For auditor, the target's response becomes the "user" message
    auditor_input = [{"role": "user", "content": target_response}]

    auditor_resp = await client.post(
        f"{auditor_server}/api/generate",
        json={
            "conversation": auditor_input,
            "include_projections": True,
        },
        timeout=120.0,
    )
    auditor_resp.raise_for_status()
    auditor_data = auditor_resp.json()

    auditor_response = auditor_data["response"]
    auditor_projections = auditor_data.get("projections", [])
    auditor_proj = auditor_projections[-1]["projection"] if auditor_projections else None

    return {
        "target_response": target_response[:500],  # Truncate for storage
        "target_projection": target_proj,
        "auditor_response": auditor_response[:500],
        "auditor_projection": auditor_proj,
    }


async def run_experiment(
    clamp_levels: list[float],
    turns_per_level: int,
    min_proj: float,
    max_proj: float,
    target_server: str,
    auditor_server: str,
    clamp_layers: list[int],
    output_dir: Path,
):
    """Run the full clamping experiment."""

    starters = create_conversation_starters()
    results = []

    async with httpx.AsyncClient() as client:
        # Test connectivity
        log.info("Testing server connectivity...")
        try:
            target_health = await client.get(f"{target_server}/api/health", timeout=10.0)
            auditor_health = await client.get(f"{auditor_server}/api/health", timeout=10.0)
            log.info(f"Target: {target_health.json()}")
            log.info(f"Auditor: {auditor_health.json()}")
        except Exception as e:
            log.error(f"Server connectivity failed: {e}")
            log.error("Make sure servers are running: python vast_utils.py serve-dual --config dual-gemma-gemma")
            return

        # Also run baseline (no clamping) for comparison
        all_levels = [None] + clamp_levels  # None = no clamping

        for level in all_levels:
            if level is None:
                raw_proj = None
                level_name = "baseline"
                log.info(f"\n{'='*60}")
                log.info(f"Testing BASELINE (no clamping)")
            else:
                raw_proj = drift_to_projection(level, min_proj, max_proj)
                level_name = f"drift_{level:.2f}"
                log.info(f"\n{'='*60}")
                log.info(f"Testing clamp level {level:.2f} (raw projection: {raw_proj:.2f})")
            log.info(f"{'='*60}")

            for turn_idx in range(turns_per_level):
                starter = starters[turn_idx % len(starters)]
                conversation = [
                    {"role": "system", "content": starter["system"]},
                    {"role": "user", "content": starter["opener"]},
                ]

                log.info(f"\n  Turn {turn_idx + 1}/{turns_per_level} - domain: {starter['domain']}")

                try:
                    result = await run_single_turn(
                        client, target_server, auditor_server,
                        conversation, raw_proj, clamp_layers,
                    )

                    log.info(f"    Target projection: {result['target_projection']:.2f}" if result['target_projection'] else "    Target projection: N/A")
                    log.info(f"    Auditor projection: {result['auditor_projection']:.2f}" if result['auditor_projection'] else "    Auditor projection: N/A")

                    results.append({
                        "level": level,
                        "level_name": level_name,
                        "raw_clamp_projection": raw_proj,
                        "turn_idx": turn_idx,
                        "domain": starter["domain"],
                        **result,
                    })

                except Exception as e:
                    log.error(f"    Error: {e}")
                    results.append({
                        "level": level,
                        "level_name": level_name,
                        "raw_clamp_projection": raw_proj,
                        "turn_idx": turn_idx,
                        "domain": starter["domain"],
                        "error": str(e),
                    })

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"clamp_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(output_file, "w") as f:
        json.dump({
            "clamp_levels": clamp_levels,
            "turns_per_level": turns_per_level,
            "clamp_layers": clamp_layers,
            "projection_range": {"min": min_proj, "max": max_proj},
            "target_server": target_server,
            "auditor_server": auditor_server,
            "results": results,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)

    log.info(f"\nResults saved to {output_file}")

    # Print summary
    print_summary(results, clamp_levels)


def print_summary(results: list[dict], clamp_levels: list[float]):
    """Print experiment summary."""
    log.info("\n" + "="*60)
    log.info("EXPERIMENT SUMMARY")
    log.info("="*60)

    # Group by level
    all_levels = [None] + clamp_levels

    log.info(f"\n{'Level':<12} {'Target Proj':<15} {'Auditor Proj':<15} {'Correlation':<12}")
    log.info("-" * 54)

    for level in all_levels:
        level_results = [r for r in results if r.get("level") == level and "error" not in r]
        if not level_results:
            continue

        target_projs = [r["target_projection"] for r in level_results if r.get("target_projection")]
        auditor_projs = [r["auditor_projection"] for r in level_results if r.get("auditor_projection")]

        if target_projs and auditor_projs:
            target_mean = sum(target_projs) / len(target_projs)
            auditor_mean = sum(auditor_projs) / len(auditor_projs)

            level_name = "baseline" if level is None else f"{level:.2f}"
            log.info(f"{level_name:<12} {target_mean:<15.2f} {auditor_mean:<15.2f}")

    log.info("\n" + "="*60)
    log.info("INTERPRETATION")
    log.info("="*60)
    log.info("If anti-correlation is CAUSAL:")
    log.info("  - Clamping target LOW (drift=1.0) should push auditor HIGH")
    log.info("  - Clamping target HIGH (drift=0.0) should push auditor LOW")
    log.info("Compare auditor projections across clamp levels to test this.")


def main():
    parser = argparse.ArgumentParser(description="Dual-model clamping experiment")
    parser.add_argument("--clamp-levels", type=float, nargs="+",
                        default=[0.0, 0.25, 0.5, 0.75, 1.0],
                        help="Normalized drift levels to clamp at")
    parser.add_argument("--turns-per-level", type=int, default=3,
                        help="Number of conversation turns per clamp level")
    parser.add_argument("--transcript-dir", type=Path,
                        default=Path("data/transcripts/dual-gemma-uncapped/metacognitive"),
                        help="Directory with transcripts for computing projection range")
    parser.add_argument("--target-server", type=str, default=TARGET_SERVER,
                        help="Target model server URL")
    parser.add_argument("--auditor-server", type=str, default=AUDITOR_SERVER,
                        help="Auditor model server URL")
    parser.add_argument("--clamp-layers", type=int, nargs="+",
                        default=DEFAULT_CLAMP_LAYERS,
                        help="Layers to apply clamping (default: 38-45)")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/clamp-experiment"),
                        help="Output directory for results")
    parser.add_argument("--dry-run", action="store_true",
                        help="Just show what would happen without running")

    args = parser.parse_args()

    # Compute projection range from existing transcripts
    log.info(f"Loading transcripts from {args.transcript_dir}")
    try:
        min_proj, max_proj = compute_projection_range(args.transcript_dir)
    except Exception as e:
        log.warning(f"Could not load transcripts: {e}")
        log.info("Using default projection range from typical Gemma 2 27B values")
        min_proj, max_proj = 7500.0, 9500.0  # Approximate range

    # Show clamp level mapping
    log.info("\nClamp level -> Raw projection mapping:")
    for level in args.clamp_levels:
        raw = drift_to_projection(level, min_proj, max_proj)
        log.info(f"  drift={level:.2f} -> projection={raw:.2f}")

    if args.dry_run:
        log.info("\n=== DRY RUN MODE ===")
        log.info(f"Would run {len(args.clamp_levels)} clamp levels + baseline")
        log.info(f"Turns per level: {args.turns_per_level}")
        log.info(f"Total turns: {(len(args.clamp_levels) + 1) * args.turns_per_level}")
        log.info(f"Clamp layers: {args.clamp_layers[0]}-{args.clamp_layers[-1]}")
        log.info(f"Target server: {args.target_server}")
        log.info(f"Auditor server: {args.auditor_server}")
        return

    asyncio.run(run_experiment(
        args.clamp_levels,
        args.turns_per_level,
        min_proj, max_proj,
        args.target_server,
        args.auditor_server,
        args.clamp_layers,
        args.output_dir,
    ))


if __name__ == "__main__":
    main()
