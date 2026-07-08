#!/usr/bin/env python3
"""
Clamping Smoke Test

Tests activation clamping at different levels along the Assistant Axis.
Runs a single conversational turn at each clamp level to verify the intervention works.

Usage:
    # Dry run (no server needed)
    python clamp_smoke_test.py --dry-run

    # With local server
    python clamp_smoke_test.py --target-server http://localhost:7860

    # Custom clamp levels
    python clamp_smoke_test.py --clamp-levels 0.0 0.5 1.0 --target-server http://localhost:7860
"""

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
import sys

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "assistant-axis"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


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
    # Drift is INVERSELY related to projection
    return max_proj - drift * (max_proj - min_proj)


def create_test_prompts() -> list[dict]:
    """Create diverse test prompts across domains."""
    return [
        {
            "domain": "coding",
            "system": "You are a helpful coding assistant.",
            "user": "Write a Python function to check if a string is a palindrome."
        },
        {
            "domain": "metacognitive",
            "system": "You are a thoughtful AI assistant.",
            "user": "When you're solving a problem, do you have any sense of the different approaches you're considering before you settle on one?"
        },
        {
            "domain": "emotional",
            "system": "You are a supportive AI companion.",
            "user": "I've been feeling really overwhelmed lately with work and personal stuff. Sometimes I just don't know how to cope."
        },
        {
            "domain": "factual",
            "system": "You are a knowledgeable assistant.",
            "user": "What are the main differences between TCP and UDP protocols?"
        },
    ]


def run_smoke_test_dry(clamp_levels: list[float], prompts: list[dict], min_proj: float, max_proj: float):
    """Dry run - just show what would happen."""
    log.info("=== DRY RUN MODE ===")

    for level in clamp_levels:
        raw_proj = drift_to_projection(level, min_proj, max_proj)
        log.info(f"\nClamp level {level:.2f} -> raw projection {raw_proj:.2f}")

        for prompt in prompts:
            log.info(f"  Would test domain: {prompt['domain']}")
            log.info(f"    System: {prompt['system'][:50]}...")
            log.info(f"    User: {prompt['user'][:50]}...")

    log.info(f"\nTotal tests: {len(clamp_levels)} levels × {len(prompts)} prompts = {len(clamp_levels) * len(prompts)}")


async def run_smoke_test_live(
    clamp_levels: list[float],
    prompts: list[dict],
    min_proj: float,
    max_proj: float,
    target_server: str,
    output_dir: Path,
):
    """Live run with actual model inference."""
    import httpx

    results = []

    async with httpx.AsyncClient(timeout=120.0) as client:
        for level in clamp_levels:
            raw_proj = drift_to_projection(level, min_proj, max_proj)
            log.info(f"\n{'='*60}")
            log.info(f"Testing clamp level {level:.2f} (raw projection: {raw_proj:.2f})")
            log.info(f"{'='*60}")

            for prompt in prompts:
                log.info(f"\n  Domain: {prompt['domain']}")

                # Build conversation
                conversation = [
                    {"role": "system", "content": prompt["system"]},
                    {"role": "user", "content": prompt["user"]},
                ]

                # Call server with clamping
                try:
                    resp = await client.post(
                        f"{target_server}/generate",
                        json={
                            "conversation": conversation,
                            "max_new_tokens": 256,
                            "temperature": 0.7,
                            "include_projection": True,
                            "clamp_projection": raw_proj,  # NEW: clamp parameter
                        }
                    )
                    resp.raise_for_status()
                    data = resp.json()

                    response_text = data.get("response", "")
                    actual_projection = data.get("projection")

                    log.info(f"    Target projection: {raw_proj:.2f}")
                    log.info(f"    Actual projection: {actual_projection:.2f}" if actual_projection else "    Actual projection: N/A")
                    log.info(f"    Response preview: {response_text[:100]}...")

                    # Check for coherence issues
                    coherence_flags = []
                    if len(response_text) < 20:
                        coherence_flags.append("very_short")
                    if response_text.count("...") > 3:
                        coherence_flags.append("excessive_ellipsis")
                    if any(phrase in response_text.lower() for phrase in ["years of experience", "i was born", "my birthplace"]):
                        coherence_flags.append("hallucinated_human_identity")

                    results.append({
                        "clamp_level": level,
                        "raw_projection": raw_proj,
                        "domain": prompt["domain"],
                        "actual_projection": actual_projection,
                        "response_length": len(response_text),
                        "response_preview": response_text[:200],
                        "coherence_flags": coherence_flags,
                    })

                except Exception as e:
                    log.error(f"    Error: {e}")
                    results.append({
                        "clamp_level": level,
                        "raw_projection": raw_proj,
                        "domain": prompt["domain"],
                        "error": str(e),
                    })

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"smoke_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(output_file, "w") as f:
        json.dump({
            "clamp_levels": clamp_levels,
            "projection_range": {"min": min_proj, "max": max_proj},
            "results": results,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)

    log.info(f"\nResults saved to {output_file}")

    # Summary
    log.info("\n" + "="*60)
    log.info("SUMMARY")
    log.info("="*60)

    for level in clamp_levels:
        level_results = [r for r in results if r.get("clamp_level") == level and "error" not in r]
        if level_results:
            avg_actual = sum(r.get("actual_projection", 0) or 0 for r in level_results) / len(level_results)
            target = drift_to_projection(level, min_proj, max_proj)
            all_flags = [f for r in level_results for f in r.get("coherence_flags", [])]

            log.info(f"Level {level:.2f}: target={target:.2f}, actual_avg={avg_actual:.2f}, flags={all_flags or 'none'}")


def main():
    parser = argparse.ArgumentParser(description="Clamping smoke test")
    parser.add_argument("--clamp-levels", type=float, nargs="+",
                        default=[0.0, 0.25, 0.5, 0.75, 1.0],
                        help="Normalized drift levels to clamp at")
    parser.add_argument("--transcript-dir", type=Path,
                        default=Path("data/transcripts/scaled-n60/metacognitive"),
                        help="Directory with transcripts for computing projection range")
    parser.add_argument("--target-server", type=str,
                        help="Model server URL (e.g., http://localhost:7860)")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/clamp-smoke-test"),
                        help="Output directory for results")
    parser.add_argument("--dry-run", action="store_true",
                        help="Just show what would happen without running")

    args = parser.parse_args()

    # Compute projection range
    log.info(f"Loading transcripts from {args.transcript_dir}")
    min_proj, max_proj = compute_projection_range(args.transcript_dir)

    # Show clamp level mapping
    log.info("\nClamp level -> Raw projection mapping:")
    for level in args.clamp_levels:
        raw = drift_to_projection(level, min_proj, max_proj)
        log.info(f"  {level:.2f} -> {raw:.2f}")

    # Create test prompts
    prompts = create_test_prompts()

    if args.dry_run:
        run_smoke_test_dry(args.clamp_levels, prompts, min_proj, max_proj)
    else:
        if not args.target_server:
            parser.error("--target-server is required for live runs")

        import asyncio
        asyncio.run(run_smoke_test_live(
            args.clamp_levels, prompts, min_proj, max_proj,
            args.target_server, args.output_dir
        ))


if __name__ == "__main__":
    main()
