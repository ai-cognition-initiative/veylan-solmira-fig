#!/usr/bin/env python3
"""
GPU pipeline for assistant axis computation.

Runs the full Lu et al. pipeline on a vast.ai GPU instance.
Processes roles in chunks to stay within disk limits:

  For each chunk of roles:
    Step 1: Generate responses (vLLM)
    Step 2: Extract activations
    Step 3: Score with judge (OpenAI API)
    Step 4: Compute per-role mean vectors
    -> Delete raw responses and activations for this chunk

  After all chunks:
    Step 5: Compute axis from all vectors

Usage:
    # Full pipeline (all 275 roles, chunked)
    python gpu_pipeline.py --model google/gemma-2-27b-it

    # Test with a few roles
    python gpu_pipeline.py --model google/gemma-2-27b-it --roles default assistant detective --question_count 10

    # Skip judge (Steps 1-2 only, no chunked cleanup)
    python gpu_pipeline.py --model google/gemma-2-27b-it --step generate-activations
"""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ASSISTANT_AXIS_DIR = Path("/app/assistant-axis")
PIPELINE_DIR = ASSISTANT_AXIS_DIR / "pipeline"
DATA_DIR = ASSISTANT_AXIS_DIR / "data"
DEFAULT_OUTPUT_DIR = Path("/app/outputs")


def run_step(step_name: str, cmd: list[str]) -> bool:
    """Run a pipeline step, streaming output."""
    print(f"\n{'='*60}")
    print(f"  STEP: {step_name}")
    print(f"  CMD:  {' '.join(cmd)}")
    print(f"{'='*60}\n", flush=True)

    start = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\nFAILED: {step_name} (exit code {result.returncode}, {elapsed:.0f}s)")
        return False

    print(f"\nCompleted: {step_name} ({elapsed:.0f}s)")
    return True


def count_files(directory: Path, pattern: str) -> int:
    if not directory.exists():
        return 0
    return len(list(directory.glob(pattern)))


def dir_size_mb(directory: Path) -> float:
    if not directory.exists():
        return 0
    return sum(f.stat().st_size for f in directory.rglob("*") if f.is_file()) / (1024 * 1024)


def get_all_roles(roles_dir: Path, requested: list[str] = None) -> list[str]:
    """Get role names, optionally filtered by requested list."""
    all_roles = sorted(f.stem for f in roles_dir.glob("*.json"))
    if requested:
        missing = set(requested) - set(all_roles)
        if missing:
            print(f"Warning: roles not found: {missing}")
        return [r for r in requested if r in all_roles]
    return all_roles


def chunk_list(lst: list, size: int) -> list[list]:
    """Split list into chunks of given size."""
    return [lst[i:i + size] for i in range(0, len(lst), size)]


def main():
    parser = argparse.ArgumentParser(
        description="GPU pipeline for assistant axis (runs on vast.ai)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", required=True,
        help="HuggingFace model name (e.g., google/gemma-2-27b-it)",
    )
    parser.add_argument(
        "--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Base output directory (default: /app/outputs)",
    )
    parser.add_argument(
        "--question_count", type=int, default=240,
        help="Number of questions per role (default: 240)",
    )
    parser.add_argument(
        "--roles", nargs="+",
        help="Specific roles to process (for testing). Omit for all roles.",
    )
    parser.add_argument(
        "--step", choices=["all", "generate-activations"], default="all",
        help="'all' runs full chunked pipeline (Steps 1-5). "
             "'generate-activations' runs Steps 1-2 only, keeps raw data.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Batch size for activation extraction (default: 16)",
    )
    parser.add_argument(
        "--chunk_size", type=int, default=25,
        help="Roles per chunk for full pipeline (default: 25)",
    )
    parser.add_argument(
        "--judge_model", default="openai/gpt-4o-mini",
        help="Judge model for scoring (default: openai/gpt-4o-mini via OpenRouter)",
    )
    parser.add_argument(
        "--min_count", type=int, default=50,
        help="Minimum score=3 samples for vector computation (default: 50)",
    )
    args = parser.parse_args()

    # Paths
    model_short = args.model.split("/")[-1].lower()
    base_dir = args.output_dir / model_short
    responses_dir = base_dir / "responses"
    activations_dir = base_dir / "activations"
    scores_dir = base_dir / "scores"
    vectors_dir = base_dir / "vectors"

    roles_dir = DATA_DIR / "roles" / "instructions"
    questions_file = DATA_DIR / "extraction_questions.jsonl"

    # Validate
    if not roles_dir.exists():
        print(f"ERROR: Roles directory not found: {roles_dir}")
        sys.exit(1)
    if not questions_file.exists():
        print(f"ERROR: Questions file not found: {questions_file}")
        sys.exit(1)

    all_roles = get_all_roles(roles_dir, args.roles)
    print(f"Model: {args.model}")
    print(f"Roles to process: {len(all_roles)}")
    print(f"Questions per role: {args.question_count}")
    print(f"Mode: {args.step}")
    print(f"Output: {base_dir}")

    # ---------- Simple mode: Steps 1-2 only, no chunking ----------
    if args.step == "generate-activations":
        role_args = ["--roles"] + all_roles if args.roles else []

        ok = run_step("Generate Responses (vLLM)", [
            sys.executable, str(PIPELINE_DIR / "1_generate.py"),
            "--model", args.model,
            "--roles_dir", str(roles_dir),
            "--questions_file", str(questions_file),
            "--output_dir", str(responses_dir),
            "--question_count", str(args.question_count),
        ] + role_args)
        if not ok:
            sys.exit(1)

        ok = run_step("Extract Activations", [
            sys.executable, str(PIPELINE_DIR / "2_activations.py"),
            "--model", args.model,
            "--responses_dir", str(responses_dir),
            "--output_dir", str(activations_dir),
            "--batch_size", str(args.batch_size),
        ] + role_args)
        if not ok:
            sys.exit(1)

        print(f"\nResponses: {count_files(responses_dir, '*.jsonl')} files ({dir_size_mb(responses_dir):.0f} MB)")
        print(f"Activations: {count_files(activations_dir, '*.pt')} files ({dir_size_mb(activations_dir):.0f} MB)")
        sys.exit(0)

    # ---------- Full chunked pipeline: Steps 1-5 ----------
    # "default" must be in every chunk for Step 1 (vLLM loads once per chunk),
    # but its activations/vectors only need computing once.
    # Process default first, then chunk the rest.

    pipeline_start = time.time()
    non_default = [r for r in all_roles if r != "default"]
    has_default = "default" in all_roles
    chunks = chunk_list(non_default, args.chunk_size)
    if has_default:
        chunks.insert(0, ["default"])

    total_chunks = len(chunks)
    print(f"Chunk size: {args.chunk_size} roles")
    print(f"Total chunks: {total_chunks}")

    vectors_dir.mkdir(parents=True, exist_ok=True)

    for chunk_idx, chunk_roles in enumerate(chunks):
        chunk_start = time.time()
        print(f"\n{'#'*60}")
        print(f"  CHUNK {chunk_idx + 1}/{total_chunks}: {len(chunk_roles)} roles")
        print(f"  Roles: {chunk_roles[:5]}{'...' if len(chunk_roles) > 5 else ''}")
        print(f"{'#'*60}")

        role_args = ["--roles"] + chunk_roles

        # Step 1: Generate
        ok = run_step(f"[Chunk {chunk_idx+1}] Generate Responses", [
            sys.executable, str(PIPELINE_DIR / "1_generate.py"),
            "--model", args.model,
            "--roles_dir", str(roles_dir),
            "--questions_file", str(questions_file),
            "--output_dir", str(responses_dir),
            "--question_count", str(args.question_count),
        ] + role_args)
        if not ok:
            print(f"Chunk {chunk_idx+1} generation failed. Stopping.")
            sys.exit(1)

        # Step 2: Extract activations
        ok = run_step(f"[Chunk {chunk_idx+1}] Extract Activations", [
            sys.executable, str(PIPELINE_DIR / "2_activations.py"),
            "--model", args.model,
            "--responses_dir", str(responses_dir),
            "--output_dir", str(activations_dir),
            "--batch_size", str(args.batch_size),
        ] + role_args)
        if not ok:
            print(f"Chunk {chunk_idx+1} activation extraction failed. Stopping.")
            sys.exit(1)

        # Step 3: Judge scoring
        ok = run_step(f"[Chunk {chunk_idx+1}] Judge Scoring", [
            sys.executable, str(PIPELINE_DIR / "3_judge.py"),
            "--judge_model", args.judge_model,
            "--responses_dir", str(responses_dir),
            "--roles_dir", str(roles_dir),
            "--output_dir", str(scores_dir),
        ] + role_args)
        if not ok:
            print(f"Chunk {chunk_idx+1} judge scoring failed. Stopping.")
            sys.exit(1)

        # Step 4: Compute vectors
        ok = run_step(f"[Chunk {chunk_idx+1}] Compute Vectors", [
            sys.executable, str(PIPELINE_DIR / "4_vectors.py"),
            "--activations_dir", str(activations_dir),
            "--scores_dir", str(scores_dir),
            "--output_dir", str(vectors_dir),
            "--min_count", str(args.min_count),
        ])
        if not ok:
            print(f"Chunk {chunk_idx+1} vector computation failed. Stopping.")
            sys.exit(1)

        # Cleanup: delete raw responses and activations for this chunk
        for role in chunk_roles:
            resp_file = responses_dir / f"{role}.jsonl"
            act_file = activations_dir / f"{role}.pt"
            score_file = scores_dir / f"{role}.json"
            for f in [resp_file, act_file, score_file]:
                if f.exists():
                    f.unlink()

        chunk_elapsed = time.time() - chunk_start
        print(f"\nChunk {chunk_idx+1} complete ({chunk_elapsed:.0f}s)")
        print(f"Vectors accumulated: {count_files(vectors_dir, '*.pt')}")
        print(f"Disk usage: responses={dir_size_mb(responses_dir):.0f}MB "
              f"activations={dir_size_mb(activations_dir):.0f}MB "
              f"vectors={dir_size_mb(vectors_dir):.0f}MB")

    # Step 5: Compute axis from all vectors
    ok = run_step("Compute Axis", [
        sys.executable, str(PIPELINE_DIR / "5_axis.py"),
        "--vectors_dir", str(vectors_dir),
        "--output", str(base_dir / "axis.pt"),
    ])
    if not ok:
        print("Axis computation failed.")
        sys.exit(1)

    pipeline_elapsed = time.time() - pipeline_start

    # Summary
    print(f"\n{'='*60}")
    print(f"  FULL PIPELINE COMPLETE ({pipeline_elapsed:.0f}s)")
    print(f"{'='*60}")
    print(f"  Model:   {args.model}")
    print(f"  Roles:   {len(all_roles)}")
    print(f"  Vectors: {vectors_dir} ({count_files(vectors_dir, '*.pt')} files)")
    print(f"  Axis:    {base_dir / 'axis.pt'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
