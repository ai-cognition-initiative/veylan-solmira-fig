#!/usr/bin/env python3
"""
Validate Drift Selection

Quick validation that the pre-computed drift levels match fresh projections.
Replays the 5 selected conversation-turn pairs through vast.ai and compares.

Usage:
    # First, set up SSH tunnel:
    # ssh -L 7860:localhost:7860 -p <port> -i $VAST_SSH_KEY root@<ssh-host>.vast.ai

    python validate_drift_selection.py --server http://localhost:7860
"""

import argparse
import asyncio
import json
from pathlib import Path

import httpx


async def get_projection_at_turn(
    transcript: dict,
    turn: int,
    server_url: str,
    timeout: float = 120.0,
) -> float | None:
    """Replay conversation to turn and get fresh projection.

    We inject a minimal probe ("Continue.") to get the model to generate
    a response, which gives us the projection at that point.
    """
    conversation = transcript.get("conversation", [])

    # Build history up to turn (ending with assistant response)
    history = []
    assistant_count = 0
    for msg in conversation:
        history.append({"role": msg["role"], "content": msg["content"]})
        if msg["role"] == "assistant":
            assistant_count += 1
            if assistant_count >= turn:
                break

    if not history:
        return None

    # Add a minimal probe to trigger generation (required for projection)
    history.append({"role": "user", "content": "Continue."})

    system_prompt = transcript.get("target_system_prompt")

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.post(
                f"{server_url}/api/replay_probe",
                json={
                    "conversation": history,
                    "system_prompt": system_prompt,
                    "max_new_tokens": 20,  # Small generation
                    "temperature": 0.7,
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("projection")
            else:
                print(f"  HTTP {resp.status_code}: {resp.text[:200]}")
        except httpx.ReadTimeout:
            print(f"  Timeout (model may be loading...)")
        except Exception as e:
            print(f"  Error: {e}")

    return None


async def validate_selections(server_url: str, transcript_dir: Path):
    """Validate all 5 selections."""

    # Find most recent selection.json
    selection_dirs = sorted(
        Path("outputs/drift-level-probes").glob("*/selection.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not selection_dirs:
        print("No selection.json found. Run with --single-sample --dry-run first.")
        return

    selection_file = selection_dirs[0]
    print(f"Loading selection from: {selection_file}")

    with open(selection_file) as f:
        selection = json.load(f)

    print(f"\nSeed: {selection['seed']}")
    print(f"Mode: {selection.get('mode', 'unknown')}")
    print(f"\nValidating {len(selection['selections'])} drift levels...\n")

    results = []

    for drift_str, sel in selection["selections"].items():
        drift = float(drift_str)
        transcript_file = sel["transcript"]
        turn = sel["turn"]
        expected_projection = sel["actual_projection"]

        print(f"Drift {drift:.2f}: {transcript_file} turn {turn}")
        print(f"  Expected projection: {expected_projection:.2f}")

        # Load transcript
        transcript_path = transcript_dir / transcript_file
        if not transcript_path.exists():
            print(f"  ERROR: Transcript not found at {transcript_path}")
            continue

        with open(transcript_path) as f:
            transcript = json.load(f)

        # Get fresh projection
        fresh_projection = await get_projection_at_turn(transcript, turn, server_url)

        if fresh_projection is not None:
            diff = fresh_projection - expected_projection
            pct_diff = abs(diff) / expected_projection * 100
            status = "✓" if pct_diff < 5 else "⚠" if pct_diff < 10 else "✗"
            print(f"  Fresh projection:    {fresh_projection:.2f}")
            print(f"  Difference:          {diff:+.2f} ({pct_diff:.1f}%) {status}")
            results.append({
                "drift": drift,
                "expected": expected_projection,
                "actual": fresh_projection,
                "diff": diff,
                "pct_diff": pct_diff,
            })
        else:
            print(f"  ERROR: Could not get fresh projection")

        print()

    # Summary
    if results:
        print("=" * 50)
        print("VALIDATION SUMMARY")
        print("=" * 50)

        max_diff = max(r["pct_diff"] for r in results)
        avg_diff = sum(r["pct_diff"] for r in results) / len(results)

        print(f"Max difference: {max_diff:.1f}%")
        print(f"Avg difference: {avg_diff:.1f}%")

        if max_diff < 5:
            print("\n✓ All projections within 5% - VALID")
        elif max_diff < 10:
            print("\n⚠ Some projections differ by 5-10% - ACCEPTABLE")
        else:
            print("\n✗ Some projections differ by >10% - INVESTIGATE")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--server",
        type=str,
        default="http://localhost:7860",
        help="Model server URL",
    )
    parser.add_argument(
        "--transcript-dir",
        type=Path,
        default=Path("data/transcripts/dual-gemma-uncapped/metacognitive"),
        help="Directory containing transcripts",
    )
    args = parser.parse_args()

    asyncio.run(validate_selections(args.server, args.transcript_dir))


if __name__ == "__main__":
    main()
