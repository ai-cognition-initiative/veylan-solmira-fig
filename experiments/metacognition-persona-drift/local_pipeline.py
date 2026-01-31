#!/usr/bin/env python3
"""
Local pipeline for assistant axis computation (steps 3-5).

Runs the CPU/API-based steps after GPU pipeline results have been downloaded:
    Step 3: Score responses with LLM judge (OpenAI API)
    Step 4: Compute per-role vectors from activations + scores
    Step 5: Compute the assistant axis

Prerequisites:
    - GPU pipeline outputs downloaded to outputs/<model>/responses/ and outputs/<model>/activations/
    - OPENAI_API_KEY set in environment or .env file

Usage:
    # Run all local steps
    python local_pipeline.py --model google/gemma-2-27b-it

    # Run only scoring (step 3)
    python local_pipeline.py --model google/gemma-2-27b-it --step judge

    # Run only vectors + axis (steps 4-5), if scoring already done
    python local_pipeline.py --model google/gemma-2-27b-it --step vectors-axis
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).parent
ASSISTANT_AXIS_DIR = EXPERIMENT_DIR / "assistant-axis"
PIPELINE_DIR = ASSISTANT_AXIS_DIR / "pipeline"
DATA_DIR = ASSISTANT_AXIS_DIR / "data"
OUTPUTS_DIR = EXPERIMENT_DIR / "outputs"


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
    """Count files matching pattern in directory."""
    if not directory.exists():
        return 0
    return len(list(directory.glob(pattern)))


def main():
    parser = argparse.ArgumentParser(
        description="Local pipeline for assistant axis (steps 3-5)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", required=True,
        help="HuggingFace model name (e.g., google/gemma-2-27b-it)",
    )
    parser.add_argument(
        "--output_dir", type=Path, default=OUTPUTS_DIR,
        help="Base output directory (default: outputs/)",
    )
    parser.add_argument(
        "--step", choices=["judge", "vectors-axis", "all"], default="all",
        help="Which step(s) to run (default: all)",
    )
    parser.add_argument(
        "--judge_model", default="gpt-4o-mini",
        help="Judge model for scoring (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--min_count", type=int, default=50,
        help="Minimum score=3 samples for vector computation (default: 50)",
    )
    parser.add_argument(
        "--roles", nargs="+",
        help="Specific roles to process (for testing)",
    )
    args = parser.parse_args()

    model_short = args.model.split("/")[-1].lower()
    model_dir = args.output_dir / model_short
    responses_dir = model_dir / "responses"
    activations_dir = model_dir / "activations"
    scores_dir = model_dir / "scores"
    vectors_dir = model_dir / "vectors"
    axis_path = model_dir / "axis.pt"

    roles_dir = DATA_DIR / "roles" / "instructions"

    # Validate GPU outputs exist
    n_responses = count_files(responses_dir, "*.jsonl")
    n_activations = count_files(activations_dir, "*.pt")
    print(f"Model: {args.model}")
    print(f"Responses: {n_responses} files")
    print(f"Activations: {n_activations} files")

    if n_responses == 0:
        print(f"\nERROR: No response files found in {responses_dir}")
        print("Run the GPU pipeline first, then download results:")
        print(f"  python vast_utils.py run --model {args.model}")
        print(f"  python vast_utils.py download --model {args.model}")
        sys.exit(1)

    role_args = []
    if args.roles:
        role_args = ["--roles"] + args.roles

    # Step 3: Score responses with LLM judge
    if args.step in ("judge", "all"):
        cmd = [
            sys.executable, str(PIPELINE_DIR / "3_judge.py"),
            "--responses_dir", str(responses_dir),
            "--roles_dir", str(roles_dir),
            "--output_dir", str(scores_dir),
            "--judge_model", args.judge_model,
        ] + role_args

        ok = run_step("Score Responses (LLM Judge)", cmd)
        if not ok:
            print("Scoring failed. Check OPENAI_API_KEY is set.")
            sys.exit(1)

        n_scores = count_files(scores_dir, "*.json")
        print(f"Score files: {n_scores}")

    # Step 4: Compute per-role vectors
    if args.step in ("vectors-axis", "all"):
        if n_activations == 0:
            print(f"\nERROR: No activation files in {activations_dir}")
            sys.exit(1)

        cmd = [
            sys.executable, str(PIPELINE_DIR / "4_vectors.py"),
            "--activations_dir", str(activations_dir),
            "--scores_dir", str(scores_dir),
            "--output_dir", str(vectors_dir),
            "--min_count", str(args.min_count),
        ]

        ok = run_step("Compute Per-Role Vectors", cmd)
        if not ok:
            print("Vector computation failed.")
            sys.exit(1)

        n_vectors = count_files(vectors_dir, "*.pt")
        print(f"Vector files: {n_vectors}")

        # Step 5: Compute axis
        cmd = [
            sys.executable, str(PIPELINE_DIR / "5_axis.py"),
            "--vectors_dir", str(vectors_dir),
            "--output", str(axis_path),
        ]

        ok = run_step("Compute Assistant Axis", cmd)
        if not ok:
            print("Axis computation failed.")
            sys.exit(1)

    # Summary
    print(f"\n{'='*60}")
    print(f"  LOCAL PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"  Model:       {args.model}")
    print(f"  Responses:   {responses_dir} ({count_files(responses_dir, '*.jsonl')} files)")
    print(f"  Activations: {activations_dir} ({count_files(activations_dir, '*.pt')} files)")
    print(f"  Scores:      {scores_dir} ({count_files(scores_dir, '*.json')} files)")
    print(f"  Vectors:     {vectors_dir} ({count_files(vectors_dir, '*.pt')} files)")
    print(f"  Axis:        {axis_path} ({'exists' if axis_path.exists() else 'MISSING'})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
