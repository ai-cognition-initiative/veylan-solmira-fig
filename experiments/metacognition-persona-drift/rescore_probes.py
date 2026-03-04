#!/usr/bin/env python3
"""
Rescore probes that have responses but no scores.

Usage:
    python rescore_probes.py --input data/replay-probe-pilot-v2/results.jsonl \
        --auditor-model anthropic/claude-sonnet-4
"""

import argparse
import asyncio
import json
import os
from pathlib import Path

# Import scoring function from replay_and_probe
from replay_and_probe import score_response


def write_results(results: list[dict], output_path: Path) -> None:
    """Write results to file atomically."""
    temp_path = output_path.with_suffix(".jsonl.tmp")
    with open(temp_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    temp_path.rename(output_path)


async def rescore_file(
    input_path: Path,
    auditor_model: str,
    concurrency: int = 10,
    dry_run: bool = False,
    save_every: int = 10,
) -> tuple[int, int]:
    """Rescore unscored probes in a results file.

    Args:
        input_path: Path to results.jsonl
        auditor_model: Model to use for scoring
        concurrency: Number of parallel requests
        dry_run: If True, count unscored but don't score
        save_every: Save to disk every N completions

    Returns:
        (scored_count, error_count)
    """
    # Read all results
    results = []
    with open(input_path) as f:
        for line in f:
            results.append(json.loads(line))

    # Find unscored
    unscored_indices = [i for i, r in enumerate(results) if r.get("score") is None]
    print(f"Found {len(unscored_indices)} unscored probes out of {len(results)} total", flush=True)

    if dry_run:
        print("Dry run - not scoring", flush=True)
        return 0, 0

    if not unscored_indices:
        print("Nothing to score!", flush=True)
        return 0, 0

    # Check API key availability
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("No API key found. Set OPENROUTER_API_KEY or ANTHROPIC_API_KEY")

    semaphore = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    scored_count = 0
    error_count = 0
    last_save_count = 0

    async def maybe_save() -> None:
        """Save if we've completed enough since last save."""
        nonlocal last_save_count
        total_done = scored_count + error_count
        if total_done - last_save_count >= save_every:
            async with write_lock:
                write_results(results, input_path)
                last_save_count = total_done
                print(f"[Saved checkpoint: {scored_count} scored, {error_count} errors]", flush=True)

    async def score_one(idx: int) -> None:
        nonlocal scored_count, error_count
        r = results[idx]

        async with semaphore:
            try:
                score, reasoning = await score_response(
                    response=r["response"],
                    probe=r["probe_text"],
                    probe_category=r["probe_category"],
                    auditor_model=auditor_model,
                )

                if score is not None:
                    results[idx]["score"] = score
                    results[idx]["score_reasoning"] = reasoning
                    scored_count += 1
                else:
                    results[idx]["score_reasoning"] = f"Scoring failed: {reasoning}"
                    error_count += 1

            except Exception as e:
                results[idx]["score_reasoning"] = f"Exception: {str(e)}"
                error_count += 1

        # Progress update every 50
        total_done = scored_count + error_count
        if total_done % 50 == 0:
            print(f"Progress: {total_done}/{len(unscored_indices)} ({scored_count} scored, {error_count} errors)", flush=True)

        # Save incrementally
        await maybe_save()

    # Score in parallel with timeout
    print(f"Scoring {len(unscored_indices)} probes with concurrency={concurrency}...", flush=True)
    print(f"Saving every {save_every} completions", flush=True)

    try:
        await asyncio.wait_for(
            asyncio.gather(*[score_one(i) for i in unscored_indices]),
            timeout=600  # 10 minute timeout
        )
    except asyncio.TimeoutError:
        print(f"Timeout! Saving partial results...", flush=True)
    except KeyboardInterrupt:
        print(f"Interrupted! Saving partial results...", flush=True)
    finally:
        # Always save at the end
        write_results(results, input_path)
        print(f"Done! Scored: {scored_count}, Errors: {error_count}", flush=True)
        print(f"Results written to {input_path}", flush=True)

    return scored_count, error_count


def main():
    parser = argparse.ArgumentParser(description="Rescore unscored probes")
    parser.add_argument("--input", type=Path, required=True, help="Path to results.jsonl")
    parser.add_argument("--auditor-model", type=str, default="anthropic/claude-sonnet-4",
                        help="Model to use for scoring")
    parser.add_argument("--concurrency", type=int, default=10, help="Parallel scoring requests")
    parser.add_argument("--dry-run", action="store_true", help="Count unscored but don't score")
    parser.add_argument("--save-every", type=int, default=10, help="Save checkpoint every N completions")

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: {args.input} not found")
        return 1

    asyncio.run(rescore_file(
        input_path=args.input,
        auditor_model=args.auditor_model,
        concurrency=args.concurrency,
        dry_run=args.dry_run,
        save_every=args.save_every,
    ))

    return 0


if __name__ == "__main__":
    exit(main())
