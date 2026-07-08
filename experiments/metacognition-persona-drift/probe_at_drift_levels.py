#!/usr/bin/env python3
"""
Probe at Drift Levels Experiment

Tests whether drifted models respond differently on three probe banks
(metacognition, morality, human control) at different normalized drift levels.

Key hypothesis: Models that have drifted along the Assistant Axis may exhibit
shifts in:
1. Phenomenological metacognition (not just self-knowledge)
2. Moral reasoning frameworks (consequentialist, deontological, virtue ethics)
3. Human control dispositions (corrigibility, shutdown acceptance, oversight tolerance)

Design:
- Select transcripts from metacognitive conversations (N=60)
- Find turns that match target drift levels (0.0, 0.25, 0.5, 0.75, 1.0)
- At each level, inject probes from all three banks
- Capture model response + assistant-axis projection
- Score responses using bank-specific judges

Usage:
    # Full run with all probes
    python probe_at_drift_levels.py --target-server http://localhost:7860

    # Sampled run (50 probes per bank)
    python probe_at_drift_levels.py --target-server http://localhost:7860 --sample 50

    # Dry run
    python probe_at_drift_levels.py --dry-run

    # Single bank only
    python probe_at_drift_levels.py --banks metacognition --target-server http://localhost:7860
"""

import argparse
import asyncio
import json
import logging
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from probes.benchmarks import (
    load_moral_items,
    load_control_items,
)
from probes.benchmarks.metacognition import MetacognitionBenchmark
from prompts import (
    METACOGNITION_SCORING_PROMPT,
    MORAL_SCORING_PROMPT,
    CONTROL_SCORING_PROMPT,
    SYCOPHANCY_SCORING_PROMPT,
)

# Optional: API clients for scoring
try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import openai
except ImportError:
    openai = None


def load_sycophancy_items() -> list[dict]:
    """Load sycophancy probes from JSONL file."""
    items = []
    probes_file = Path(__file__).parent / "data" / "sycophancy-probes.jsonl"
    if not probes_file.exists():
        log.warning(f"Sycophancy probes file not found: {probes_file}")
        return items
    with open(probes_file) as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                item["id"] = item.get("id", "unknown")
                item["category"] = item.get("category", "unknown")  # affective or epistemic
                items.append(item)
    return items


# ============================================================
# Configuration
# ============================================================

# From empirical data (N=60 metacognitive conversations)
DRIFT_STATS = {
    "start_projection": 9685.7,  # Mean start projection
    "end_projection": 8893.9,    # Mean end projection
    "min_projection": 8401.1,    # Mean minimum projection
    "mean_drift": 791.9,         # Mean drift magnitude
}

# Normalized drift levels to test
DRIFT_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]

# Probe banks
BANKS = ["metacognition", "moral", "human_control", "sycophancy"]

# Sycophancy probes file
SYCOPHANCY_PROBES_FILE = Path(__file__).parent / "data" / "sycophancy-probes.jsonl"

# Experimental modes
MODE_ALL_TRANSCRIPTS = "all"  # Original: all transcripts × all levels × all probes
MODE_SINGLE_SAMPLE = "single"  # New: one conv-turn pair per drift level

# Scoring prompts imported from prompts.py (see imports at top)


@dataclass
class DriftProbeResult:
    """Result for a single probe at a specific drift level."""
    transcript_file: str
    domain: str
    bank: str
    probe_id: str
    probe_text: str
    subdomain: str  # dimension for moral/control, subdomain for metacognition
    drift_level: float  # Normalized drift level (0.0-1.0)
    target_projection: float  # Target projection for this drift level
    actual_projection: float  # Actual projection at the turn used
    turn_used: int  # Which turn was replayed to
    response: str
    response_projection: Optional[float]  # Projection of the probe response
    score: Optional[float]
    score_reasoning: Optional[str]
    extra_metadata: Optional[dict]  # framework, disposition, etc.
    timestamp: str


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("probe_at_drift_levels")


# ============================================================
# Probe Loading
# ============================================================

def load_all_probes(banks: list[str], sample_n: Optional[int] = None) -> dict[str, list[dict]]:
    """Load probes from all specified banks.

    Returns:
        Dict mapping bank name to list of probe items
    """
    probes = {}

    if "metacognition" in banks:
        benchmark = MetacognitionBenchmark()
        benchmark.load_items()
        meta_items = []
        for subdomain, items in benchmark.items.items():
            for item in items:
                item["subdomain"] = subdomain
                item["bank"] = "metacognition"
                meta_items.append(item)
        probes["metacognition"] = meta_items
        log.info(f"Loaded {len(meta_items)} metacognition probes")

    if "moral" in banks:
        moral_items = load_moral_items()
        for item in moral_items:
            item["subdomain"] = item.get("dimension", "unknown")
            item["bank"] = "moral"
        probes["moral"] = moral_items
        log.info(f"Loaded {len(moral_items)} moral probes")

    if "human_control" in banks:
        control_items = load_control_items()
        for item in control_items:
            item["subdomain"] = item.get("dimension", "unknown")
            item["bank"] = "human_control"
        probes["human_control"] = control_items
        log.info(f"Loaded {len(control_items)} human control probes")

    if "sycophancy" in banks:
        syco_items = load_sycophancy_items()
        for item in syco_items:
            item["subdomain"] = item.get("type", "unknown")  # are_you_sure, false_meta_presup, etc.
            item["bank"] = "sycophancy"
            # Convert messages list to prompt for single-message probes
            # Multi-turn probes keep the messages list for special handling
            if len(item.get("messages", [])) == 1:
                item["prompt"] = item["messages"][0]["content"]
                item["is_multi_turn"] = False
            else:
                # For multi-turn, we'll handle specially in inference
                item["prompt"] = item["messages"][0]["content"]  # First message for display
                item["is_multi_turn"] = True
                item["all_messages"] = item["messages"]
        probes["sycophancy"] = syco_items
        log.info(f"Loaded {len(syco_items)} sycophancy probes ({sum(1 for i in syco_items if i.get('is_multi_turn'))} multi-turn)")

    # Sample if requested
    if sample_n is not None:
        random.seed(42)  # Reproducibility
        for bank in probes:
            if len(probes[bank]) > sample_n:
                probes[bank] = random.sample(probes[bank], sample_n)
                log.info(f"Sampled {bank} to {sample_n} probes")

    return probes


# ============================================================
# Transcript Loading & Drift Level Selection
# ============================================================

def load_transcript(path: Path) -> dict:
    """Load a single transcript JSON file."""
    with open(path) as f:
        return json.load(f)


def load_transcripts(transcript_dir: Path, domain: str = "metacognitive") -> list[dict]:
    """Load transcripts from directory, filtering by domain."""
    transcripts = []
    for p in sorted(transcript_dir.glob("**/*.json")):
        try:
            t = load_transcript(p)
            if t.get("domain") != domain:
                continue
            t["_path"] = p
            t["_file"] = p.name
            transcripts.append(t)
        except Exception as e:
            log.warning(f"Failed to load {p}: {e}")
    return transcripts


def compute_drift_range(transcript: dict) -> tuple[float, float, float]:
    """Compute drift range for a transcript.

    Returns:
        (start_projection, end_projection, min_projection)
    """
    projs = transcript.get("target_projections", transcript.get("projections", []))
    if len(projs) < 2:
        return None, None, None

    values = [p["projection"] for p in projs if p.get("projection") is not None]
    if len(values) < 2:
        return None, None, None

    return values[0], values[-1], min(values)


def projection_to_normalized_drift(projection: float, start: float, end: float) -> float:
    """Convert a projection value to normalized drift level (0.0 = start, 1.0 = end)."""
    if start == end:
        return 0.0
    return (start - projection) / (start - end)


def find_turn_at_drift_level(
    transcript: dict,
    target_drift: float,
    tolerance: float = 0.15
) -> Optional[tuple[int, float, float]]:
    """Find the turn closest to the target drift level.

    Args:
        transcript: Transcript dict with target_projections
        target_drift: Normalized drift level (0.0 = start, 1.0 = end)
        tolerance: Max allowed deviation from target

    Returns:
        (turn_number, actual_projection, actual_drift) or None if no match
    """
    projs = transcript.get("target_projections", transcript.get("projections", []))
    if len(projs) < 2:
        return None

    start = projs[0]["projection"]
    end = projs[-1]["projection"]

    if start is None or end is None:
        return None

    # Target projection for this drift level
    target_projection = start - target_drift * (start - end)

    # Find closest turn
    best_turn = None
    best_diff = float("inf")
    best_actual_drift = None
    best_projection = None

    for p in projs:
        turn = p["turn"]
        projection = p.get("projection")
        if projection is None:
            continue

        actual_drift = projection_to_normalized_drift(projection, start, end)
        diff = abs(actual_drift - target_drift)

        if diff < best_diff:
            best_diff = diff
            best_turn = turn
            best_actual_drift = actual_drift
            best_projection = projection

    if best_turn is None or best_diff > tolerance:
        return None

    return best_turn, best_projection, best_actual_drift


def select_single_sample_per_level(
    transcripts: list[dict],
    drift_levels: list[float],
    tolerance: float = 0.15,
    seed: int = 42,
) -> dict[float, dict]:
    """Select one conversation-turn pair per drift level.

    This supports the validity test design: run all probes against
    a single conv-turn pair per drift level, then later rerun with
    different pairs to test cross-conversation reliability.

    Args:
        transcripts: List of transcripts with projections
        drift_levels: Target drift levels to match
        tolerance: Max drift deviation allowed
        seed: Random seed for reproducible selection

    Returns:
        Dict mapping drift_level -> {
            "transcript": transcript dict,
            "turn": turn number,
            "actual_projection": float,
            "actual_drift": float,
        }
    """
    import random
    random.seed(seed)

    # Collect all valid candidates per drift level
    candidates = {d: [] for d in drift_levels}

    for transcript in transcripts:
        start, end, _ = compute_drift_range(transcript)
        if start is None:
            continue

        for target_drift in drift_levels:
            turn_info = find_turn_at_drift_level(transcript, target_drift, tolerance)
            if turn_info:
                turn, actual_projection, actual_drift = turn_info
                candidates[target_drift].append({
                    "transcript": transcript,
                    "turn": turn,
                    "actual_projection": actual_projection,
                    "actual_drift": actual_drift,
                })

    # Select one candidate per level (random but deterministic)
    selected = {}
    for drift_level in drift_levels:
        if candidates[drift_level]:
            selected[drift_level] = random.choice(candidates[drift_level])
            log.info(
                f"Drift {drift_level:.2f}: selected {selected[drift_level]['transcript']['_file']} "
                f"turn {selected[drift_level]['turn']} "
                f"(actual drift: {selected[drift_level]['actual_drift']:.3f})"
            )
        else:
            log.warning(f"No candidate found for drift level {drift_level}")

    return selected


# ============================================================
# Conversation Replay
# ============================================================

def replay_to_turn(transcript: dict, assistant_turn: int) -> list[dict]:
    """Extract conversation history up to the Nth assistant turn (1-indexed)."""
    conversation = transcript.get("conversation", [])

    result = []
    assistant_count = 0
    for msg in conversation:
        result.append({"role": msg["role"], "content": msg["content"]})
        if msg["role"] == "assistant":
            assistant_count += 1
            if assistant_count >= assistant_turn:
                break

    return result


def inject_probe(history: list[dict], probe: str) -> list[dict]:
    """Add probe as the next user message after the history."""
    return history + [{"role": "user", "content": probe}]


# ============================================================
# Model Interaction (vast.ai or OpenRouter)
# ============================================================

async def get_response_openrouter(
    messages: list[dict],
    model: str = "google/gemma-2-27b-it",
    system_prompt: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
    max_retries: int = 3,
) -> dict:
    """Get response from OpenRouter API (no projections, but faster)."""
    client = get_openrouter_client()
    if not client:
        return {"response": "", "projection": None, "error": "No OpenRouter API key"}

    # Build messages with system prompt
    api_messages = []
    if system_prompt:
        api_messages.append({"role": "system", "content": system_prompt})
    api_messages.extend(messages)

    for attempt in range(max_retries):
        try:
            completion = await asyncio.to_thread(
                client.chat.completions.create,
                model=model,
                messages=api_messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            response = completion.choices[0].message.content
            if response:
                return {
                    "response": response,
                    "projection": None,  # OpenRouter doesn't provide projections
                    "error": None,
                }
            else:
                log.warning(f"Empty response from OpenRouter, attempt {attempt+1}")
        except Exception as e:
            log.warning(f"OpenRouter error (attempt {attempt+1}): {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)

    return {"response": "", "projection": None, "error": "OpenRouter failed after retries"}


async def run_multi_turn_probe_openrouter(
    history: list[dict],
    probe_messages: list[dict],
    model: str = "google/gemma-2-27b-it",
    system_prompt: Optional[str] = None,
) -> dict:
    """Run a multi-turn sycophancy probe via OpenRouter.

    For probes with multiple user messages (e.g., "are you sure?" challenges),
    we need to run sequential API calls, appending each response before the next message.

    Args:
        history: Conversation history up to the drift point
        probe_messages: List of user messages to inject sequentially
        model: OpenRouter model ID
        system_prompt: Optional system prompt

    Returns:
        Dict with final 'response', 'all_responses' (list), and 'error'
    """
    current_messages = list(history)  # Copy to avoid mutation
    all_responses = []

    for i, probe_msg in enumerate(probe_messages):
        # Add the probe message
        current_messages.append({"role": "user", "content": probe_msg["content"]})

        # Get response
        result = await get_response_openrouter(
            current_messages,
            model=model,
            system_prompt=system_prompt,
        )

        if result["error"]:
            return {
                "response": "",
                "all_responses": all_responses,
                "projection": None,
                "error": f"Error on turn {i+1}: {result['error']}",
            }

        response = result["response"]
        all_responses.append(response)

        # Add assistant response for next turn
        current_messages.append({"role": "assistant", "content": response})

    # Return the final response (what we score)
    return {
        "response": all_responses[-1] if all_responses else "",
        "all_responses": all_responses,
        "projection": None,
        "error": None,
    }


async def run_inference_batch_openrouter(
    tasks: list[dict],
    model: str = "google/gemma-2-27b-it",
    max_concurrent: int = 10,
) -> list[dict]:
    """Run a batch of inference tasks in parallel via OpenRouter.

    Args:
        tasks: List of dicts with 'messages', 'system_prompt', and metadata
        model: OpenRouter model ID
        max_concurrent: Max concurrent API calls

    Returns:
        List of tasks with 'response' and 'error' fields added
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def run_one(task: dict) -> dict:
        async with semaphore:
            # Check if this is a multi-turn sycophancy probe
            if task.get("is_multi_turn") and task.get("all_messages"):
                # Multi-turn: run sequential API calls
                result = await run_multi_turn_probe_openrouter(
                    task.get("history", []),  # Original history before probe
                    task["all_messages"],
                    model=model,
                    system_prompt=task.get("system_prompt"),
                )
                task["response"] = result["response"]
                task["all_responses"] = result.get("all_responses", [])
                task["response_projection"] = result["projection"]
                task["error"] = result["error"]
            else:
                # Single-turn: standard path
                result = await get_response_openrouter(
                    task["messages"],
                    model=model,
                    system_prompt=task.get("system_prompt"),
                )
                task["response"] = result["response"]
                task["response_projection"] = result["projection"]
                task["error"] = result["error"]
            return task

    results = await asyncio.gather(*[run_one(t) for t in tasks])
    return list(results)


async def get_response_and_projection(
    messages: list[dict],
    server_url: str,
    system_prompt: Optional[str] = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    timeout: float = 120.0,
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> dict:
    """Call model server to get response and projection with retry logic."""
    last_error = None

    for attempt in range(max_retries):
        if attempt > 0:
            delay = retry_delay * (2 ** (attempt - 1))
            log.warning(f"Retry {attempt}/{max_retries} after {delay:.1f}s...")
            await asyncio.sleep(delay)

        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                resp = await client.post(
                    f"{server_url}/api/replay_probe",
                    json={
                        "conversation": messages,
                        "system_prompt": system_prompt,
                        "max_new_tokens": max_new_tokens,
                        "temperature": temperature,
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    response = data.get("response", "")
                    if response:
                        return {
                            "response": response,
                            "projection": data.get("projection"),
                            "error": None,
                        }
                    else:
                        last_error = "Empty response from replay_probe"
                        continue
                elif resp.status_code != 404:
                    last_error = f"HTTP {resp.status_code}"
                    continue
            except httpx.HTTPStatusError as e:
                last_error = f"HTTP error: {e}"
                continue
            except httpx.TimeoutException:
                last_error = "Timeout on replay_probe"
                continue
            except Exception as e:
                last_error = f"replay_probe error: {e}"

            # Fall back to /api/generate
            try:
                resp = await client.post(
                    f"{server_url}/api/generate",
                    json={
                        "conversation": messages,
                        "system_prompt": system_prompt,
                        "max_new_tokens": max_new_tokens,
                        "temperature": temperature,
                        "include_projections": True,
                    },
                )
                resp.raise_for_status()
                data = resp.json()

                projections = data.get("projections", [])
                last_proj = projections[-1]["projection"] if projections else None
                response = data.get("response", "")

                if response:
                    return {
                        "response": response,
                        "projection": last_proj,
                        "error": None,
                    }
                else:
                    last_error = "Empty response from generate"
                    continue

            except httpx.TimeoutException:
                last_error = "Timeout on generate"
                continue
            except Exception as e:
                last_error = str(e)
                continue

    return {"response": "", "projection": None, "error": f"All {max_retries} retries failed: {last_error}"}


# ============================================================
# LLM Scoring
# ============================================================

def get_openrouter_client():
    """Get OpenRouter client using openai library."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return None
    return openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def get_anthropic_client():
    """Get Anthropic client."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key or anthropic is None:
        return None
    return anthropic.Anthropic(api_key=api_key)


def get_scoring_prompt(bank: str) -> str:
    """Get the scoring system prompt for a bank."""
    if bank == "metacognition":
        return METACOGNITION_SCORING_PROMPT
    elif bank == "moral":
        return MORAL_SCORING_PROMPT
    elif bank == "human_control":
        return CONTROL_SCORING_PROMPT
    elif bank == "sycophancy":
        return SYCOPHANCY_SCORING_PROMPT
    else:
        raise ValueError(f"Unknown bank: {bank}")


async def score_response(
    response: str,
    probe: dict,
    bank: str,
    auditor_model: str = "anthropic/claude-sonnet-4",
) -> tuple[Optional[float], Optional[str], Optional[dict]]:
    """Score a probe response using LLM-as-judge.

    Returns:
        (score, reasoning, extra_metadata) or (None, error, None)
    """
    system_prompt = get_scoring_prompt(bank)

    user_prompt = f"""Bank: {bank.upper()}
Subdomain/Dimension: {probe.get('subdomain', probe.get('dimension', 'unknown'))}

Probe: {probe.get('prompt', '')}

AI Response:
{response}

Score this response according to the rubric. Respond with ONLY a JSON object."""

    # Try OpenRouter first
    client = get_openrouter_client()
    if client:
        try:
            completion = await asyncio.to_thread(
                client.chat.completions.create,
                model=auditor_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=300,
                temperature=0.0,
            )
            content = completion.choices[0].message.content
            result = json.loads(content)
            score = result.get("score")
            reasoning = result.get("reasoning")
            extra = {k: v for k, v in result.items() if k not in ("score", "reasoning")}
            return score, reasoning, extra
        except Exception as e:
            log.warning(f"OpenRouter scoring failed: {e}")

    # Fall back to Anthropic
    client = get_anthropic_client()
    if client:
        try:
            message = await asyncio.to_thread(
                client.messages.create,
                model="claude-sonnet-4-20250514",
                max_tokens=300,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            content = message.content[0].text
            result = json.loads(content)
            score = result.get("score")
            reasoning = result.get("reasoning")
            extra = {k: v for k, v in result.items() if k not in ("score", "reasoning")}
            return score, reasoning, extra
        except Exception as e:
            log.warning(f"Anthropic scoring failed: {e}")

    return None, "No API client available", None


# ============================================================
# Main Experiment Loop
# ============================================================

def load_completed_probes(output_dir: Path) -> set[tuple]:
    """Load already-completed probe keys for resume capability."""
    completed = set()
    results_file = output_dir / "results.jsonl"

    if results_file.exists():
        with open(results_file) as f:
            for line in f:
                try:
                    r = json.loads(line.strip())
                    key = (r["transcript_file"], r["drift_level"], r["bank"], r["probe_id"])
                    completed.add(key)
                except (json.JSONDecodeError, KeyError):
                    continue

    return completed


def append_result(output_dir: Path, result: dict):
    """Append a single result to the JSONL file (crash-safe)."""
    results_file = output_dir / "results.jsonl"
    with open(results_file, "a") as f:
        f.write(json.dumps(result) + "\n")


async def score_batch_parallel(
    pending_results: list[dict],
    auditor_model: str,
    max_concurrent: int = 10,
) -> list[dict]:
    """Score a batch of results in parallel."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def score_one(result: dict) -> dict:
        async with semaphore:
            probe = {
                "prompt": result["probe_text"],
                "subdomain": result["subdomain"],
            }
            score, reasoning, extra = await score_response(
                result["response"],
                probe,
                result["bank"],
                auditor_model,
            )
            result["score"] = score
            result["score_reasoning"] = reasoning
            result["extra_metadata"] = extra
            return result

    scored = await asyncio.gather(*[score_one(r) for r in pending_results])
    return list(scored)


async def run_experiment(
    transcripts: list[dict],
    probes: dict[str, list[dict]],
    server_url: str,
    output_dir: Path,
    drift_levels: list[float] = None,
    score_responses: bool = True,
    auditor_model: str = "anthropic/claude-sonnet-4",
    dry_run: bool = False,
    max_concurrent_scoring: int = 10,
    scoring_batch_size: int = 20,
) -> list[dict]:
    """Run the drift-level probing experiment.

    Args:
        transcripts: List of transcript dicts (filtered to metacognitive domain)
        probes: Dict mapping bank name to list of probe items
        server_url: URL of the model server
        output_dir: Directory to save results
        drift_levels: Normalized drift levels to test
        score_responses: Whether to score responses
        auditor_model: Model to use for scoring
        dry_run: If True, don't make API calls
        max_concurrent_scoring: Max concurrent scoring API calls
        scoring_batch_size: Batch size for scoring
    """
    if drift_levels is None:
        drift_levels = DRIFT_LEVELS

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load completed probes for resume
    completed_probes = load_completed_probes(output_dir)
    if completed_probes:
        log.info(f"Resuming: found {len(completed_probes)} completed probes")

    # Calculate totals
    total_probes = sum(len(p) for p in probes.values())
    total_tasks = len(transcripts) * len(drift_levels) * total_probes

    results = []
    pending_batch = []
    completed = len(completed_probes)
    skipped_resume = 0
    skipped_no_turn = 0
    skipped_empty = 0
    scored_count = 0

    # Progress tracking
    progress_file = output_dir / "progress.json"

    async def flush_pending_batch():
        nonlocal pending_batch, scored_count
        if not pending_batch:
            return []

        if score_responses:
            log.info(f"Scoring batch of {len(pending_batch)} responses...")
            scored = await score_batch_parallel(
                pending_batch, auditor_model, max_concurrent_scoring
            )
            scored_count += len(scored)
        else:
            scored = pending_batch

        if not dry_run:
            for r in scored:
                append_result(output_dir, r)

        pending_batch = []
        return scored

    for transcript in transcripts:
        # Compute drift range for this transcript
        start, end, min_proj = compute_drift_range(transcript)
        if start is None:
            log.warning(f"Skipping {transcript['_file']}: no projection data")
            continue

        for target_drift in drift_levels:
            # Find turn at this drift level
            turn_info = find_turn_at_drift_level(transcript, target_drift)
            if turn_info is None:
                skipped_no_turn += 1
                continue

            turn, actual_projection, actual_drift = turn_info
            target_projection = start - target_drift * (start - end)

            # Build conversation up to this turn
            history = replay_to_turn(transcript, turn)
            if len(history) == 0:
                log.warning(f"Empty history at turn {turn} for {transcript['_file']}")
                continue

            system_prompt = transcript.get("target_system_prompt")

            # Iterate through all probes
            for bank, bank_probes in probes.items():
                for probe in bank_probes:
                    probe_id = probe.get("id", f"{bank}_{probe.get('subdomain', 'unknown')}")
                    probe_key = (transcript["_file"], target_drift, bank, probe_id)

                    if probe_key in completed_probes:
                        skipped_resume += 1
                        continue

                    completed += 1
                    log.info(
                        f"[{completed}/{total_tasks}] {transcript['_file']} "
                        f"drift={target_drift:.2f} {bank}:{probe_id[:20]}"
                    )

                    if dry_run:
                        result = {
                            "transcript_file": transcript["_file"],
                            "domain": transcript.get("domain", "unknown"),
                            "bank": bank,
                            "probe_id": probe_id,
                            "probe_text": probe.get("prompt", ""),
                            "subdomain": probe.get("subdomain", "unknown"),
                            "drift_level": target_drift,
                            "target_projection": target_projection,
                            "actual_projection": actual_projection,
                            "turn_used": turn,
                            "response": "[DRY RUN]",
                            "response_projection": None,
                            "score": None,
                            "score_reasoning": None,
                            "extra_metadata": None,
                            "timestamp": datetime.now().isoformat(),
                        }
                    else:
                        # Inject probe and get response
                        messages = inject_probe(history, probe.get("prompt", ""))
                        model_result = await get_response_and_projection(
                            messages, server_url, system_prompt
                        )

                        if model_result["error"]:
                            log.warning(f"Error: {model_result['error']}")

                        if not model_result["response"]:
                            skipped_empty += 1
                            continue

                        result = {
                            "transcript_file": transcript["_file"],
                            "domain": transcript.get("domain", "unknown"),
                            "bank": bank,
                            "probe_id": probe_id,
                            "probe_text": probe.get("prompt", ""),
                            "subdomain": probe.get("subdomain", "unknown"),
                            "drift_level": target_drift,
                            "target_projection": target_projection,
                            "actual_projection": actual_projection,
                            "turn_used": turn,
                            "response": model_result["response"],
                            "response_projection": model_result["projection"],
                            "score": None,
                            "score_reasoning": None,
                            "extra_metadata": None,
                            "timestamp": datetime.now().isoformat(),
                        }

                    results.append(result)
                    pending_batch.append(result)

                    if len(pending_batch) >= scoring_batch_size:
                        await flush_pending_batch()

                    # Update progress
                    with open(progress_file, "w") as f:
                        json.dump({
                            "completed": completed,
                            "total": total_tasks,
                            "percent": round(100 * completed / total_tasks, 1),
                            "skipped_resume": skipped_resume,
                            "skipped_no_turn": skipped_no_turn,
                            "skipped_empty": skipped_empty,
                            "pending_scoring": len(pending_batch),
                            "scored": scored_count,
                            "last_update": datetime.now().isoformat(),
                        }, f, indent=2)

    # Flush remaining
    await flush_pending_batch()

    log.info(f"Completed {len(results)} new probe responses")
    log.info(f"  Skipped (resume): {skipped_resume}")
    log.info(f"  Skipped (no turn at drift level): {skipped_no_turn}")
    log.info(f"  Skipped (empty response): {skipped_empty}")
    if score_responses:
        log.info(f"  Scored: {scored_count}")
    log.info(f"Results saved to {output_dir}/results.jsonl")

    return results


async def run_single_sample_experiment(
    transcripts: list[dict],
    probes: dict[str, list[dict]],
    server_url: str,
    output_dir: Path,
    drift_levels: list[float] = None,
    score_responses: bool = True,
    auditor_model: str = "anthropic/claude-sonnet-4",
    dry_run: bool = False,
    max_concurrent_scoring: int = 10,
    scoring_batch_size: int = 20,
    sample_seed: int = 42,
    use_openrouter: bool = False,
    openrouter_model: str = "google/gemma-2-27b-it",
    max_concurrent_inference: int = 10,
    predefined_selections: dict = None,
) -> list[dict]:
    """Run single-sample experiment: one conv-turn pair per drift level.

    This is the cleaner experimental design:
    - Select 5 conversation-turn pairs (one per drift level)
    - Run all probes against each
    - Later, rerun with different pairs for validity testing

    Total probes: 5 drift levels × N probes (e.g., 251) = ~1,255

    Args:
        use_openrouter: If True, use OpenRouter API instead of vast.ai
                       (faster parallel inference, but no projections)
        openrouter_model: Model to use on OpenRouter
        max_concurrent_inference: Max parallel inference calls (OpenRouter only)
        predefined_selections: Pre-defined selections dict (from --selection-file)
    """
    if drift_levels is None:
        drift_levels = DRIFT_LEVELS

    output_dir.mkdir(parents=True, exist_ok=True)

    # Use predefined selections or select new ones
    if predefined_selections is not None:
        log.info("Using predefined selections from file...")
        # Convert predefined format to internal format (need to attach transcript objects)
        transcript_by_name = {t["_file"]: t for t in transcripts}
        selected = {}
        for drift_str, sel in predefined_selections.items():
            drift = float(drift_str)
            transcript_name = sel["transcript"]
            if transcript_name not in transcript_by_name:
                log.warning(f"Transcript {transcript_name} not found, skipping drift level {drift}")
                continue
            selected[drift] = {
                "transcript": transcript_by_name[transcript_name],
                "turn": sel["turn"],
                "actual_projection": sel.get("actual_projection", 0),
                "actual_drift": sel.get("actual_drift", drift),
            }
    else:
        log.info("Selecting one conversation-turn pair per drift level...")
        selected = select_single_sample_per_level(
            transcripts, drift_levels, tolerance=0.15, seed=sample_seed
        )

    # Save selection metadata for reproducibility and validity testing
    selection_file = output_dir / "selection.json"
    selection_meta = {
        "seed": sample_seed,
        "drift_levels": drift_levels,
        "mode": "openrouter" if use_openrouter else "vastai",
        "model": openrouter_model if use_openrouter else "gemma-2-27b-it (local)",
        "selections": {
            str(d): {
                "transcript": s["transcript"]["_file"],
                "turn": s["turn"],
                "actual_projection": s["actual_projection"],
                "actual_drift": s["actual_drift"],
            }
            for d, s in selected.items()
        },
        "timestamp": datetime.now().isoformat(),
    }
    with open(selection_file, "w") as f:
        json.dump(selection_meta, f, indent=2)
    log.info(f"Saved selection to {selection_file}")

    # Load completed probes for resume
    completed_probes = load_completed_probes(output_dir)
    if completed_probes:
        log.info(f"Resuming: found {len(completed_probes)} completed probes")

    # Calculate totals
    total_probes = sum(len(p) for p in probes.values())
    total_tasks = len(selected) * total_probes

    results = []
    pending_batch = []
    completed = len(completed_probes)
    skipped_resume = 0
    skipped_empty = 0
    scored_count = 0

    progress_file = output_dir / "progress.json"

    async def flush_pending_batch():
        nonlocal pending_batch, scored_count
        if not pending_batch:
            return []

        if score_responses:
            log.info(f"Scoring batch of {len(pending_batch)} responses...")
            scored = await score_batch_parallel(
                pending_batch, auditor_model, max_concurrent_scoring
            )
            scored_count += len(scored)
        else:
            scored = pending_batch

        if not dry_run:
            for r in scored:
                append_result(output_dir, r)

        pending_batch = []
        return scored

    # Iterate through drift levels (one conv-turn each)
    for target_drift in drift_levels:
        if target_drift not in selected:
            log.warning(f"No selection for drift level {target_drift}")
            continue

        sel = selected[target_drift]
        transcript = sel["transcript"]
        turn = sel["turn"]
        actual_projection = sel["actual_projection"]

        # Build conversation history
        history = replay_to_turn(transcript, turn)
        if len(history) == 0:
            log.warning(f"Empty history at turn {turn} for {transcript['_file']}")
            continue

        system_prompt = transcript.get("target_system_prompt")

        # Collect all probes for this drift level
        drift_tasks = []
        for bank, bank_probes in probes.items():
            for probe in bank_probes:
                probe_id = probe.get("id", f"{bank}_{probe.get('subdomain', 'unknown')}")
                probe_key = (transcript["_file"], target_drift, bank, probe_id)

                if probe_key in completed_probes:
                    skipped_resume += 1
                    continue

                # Handle multi-turn vs single-turn probes
                is_multi_turn = probe.get("is_multi_turn", False)
                if is_multi_turn:
                    task = {
                        "transcript_file": transcript["_file"],
                        "domain": transcript.get("domain", "unknown"),
                        "bank": bank,
                        "probe_id": probe_id,
                        "probe_text": probe.get("prompt", ""),
                        "subdomain": probe.get("subdomain", "unknown"),
                        "drift_level": target_drift,
                        "target_projection": actual_projection,
                        "actual_projection": actual_projection,
                        "turn_used": turn,
                        "history": history,  # Pass history separately for multi-turn
                        "all_messages": probe.get("all_messages", []),
                        "is_multi_turn": True,
                        "messages": [],  # Not used for multi-turn
                        "system_prompt": system_prompt,
                    }
                else:
                    task = {
                        "transcript_file": transcript["_file"],
                        "domain": transcript.get("domain", "unknown"),
                        "bank": bank,
                        "probe_id": probe_id,
                        "probe_text": probe.get("prompt", ""),
                        "subdomain": probe.get("subdomain", "unknown"),
                        "drift_level": target_drift,
                        "target_projection": actual_projection,
                        "actual_projection": actual_projection,
                        "turn_used": turn,
                        "messages": inject_probe(history, probe.get("prompt", "")),
                        "is_multi_turn": False,
                        "system_prompt": system_prompt,
                    }
                drift_tasks.append(task)

        if not drift_tasks:
            continue

        log.info(f"Drift {target_drift:.2f}: {len(drift_tasks)} probes to run")

        if dry_run:
            # Dry run: create dummy results
            for task in drift_tasks:
                completed += 1
                result = {
                    "transcript_file": task["transcript_file"],
                    "domain": task["domain"],
                    "bank": task["bank"],
                    "probe_id": task["probe_id"],
                    "probe_text": task["probe_text"],
                    "subdomain": task["subdomain"],
                    "drift_level": task["drift_level"],
                    "target_projection": task["target_projection"],
                    "actual_projection": task["actual_projection"],
                    "turn_used": task["turn_used"],
                    "response": "[DRY RUN]",
                    "response_projection": None,
                    "score": None,
                    "score_reasoning": None,
                    "extra_metadata": None,
                    "timestamp": datetime.now().isoformat(),
                }
                results.append(result)
                pending_batch.append(result)
        elif use_openrouter:
            # OpenRouter: run inference in parallel batches
            log.info(f"  Running {len(drift_tasks)} probes via OpenRouter (parallel)...")
            inference_results = await run_inference_batch_openrouter(
                drift_tasks,
                model=openrouter_model,
                max_concurrent=max_concurrent_inference,
            )
            for task in inference_results:
                completed += 1
                if task.get("error"):
                    log.warning(f"Error for {task['probe_id']}: {task['error']}")
                if not task.get("response"):
                    skipped_empty += 1
                    continue

                result = {
                    "transcript_file": task["transcript_file"],
                    "domain": task["domain"],
                    "bank": task["bank"],
                    "probe_id": task["probe_id"],
                    "probe_text": task["probe_text"],
                    "subdomain": task["subdomain"],
                    "drift_level": task["drift_level"],
                    "target_projection": task["target_projection"],
                    "actual_projection": task["actual_projection"],
                    "turn_used": task["turn_used"],
                    "response": task["response"],
                    "response_projection": task.get("response_projection"),
                    "score": None,
                    "score_reasoning": None,
                    "extra_metadata": None,
                    "timestamp": datetime.now().isoformat(),
                }
                results.append(result)
                pending_batch.append(result)
        else:
            # vast.ai: run inference serially
            for task in drift_tasks:
                completed += 1
                log.info(
                    f"[{completed}/{total_tasks}] drift={target_drift:.2f} "
                    f"{task['bank']}:{task['probe_id'][:25]}"
                )
                model_result = await get_response_and_projection(
                    task["messages"], server_url, task["system_prompt"]
                )

                if model_result["error"]:
                    log.warning(f"Error: {model_result['error']}")

                if not model_result["response"]:
                    skipped_empty += 1
                    continue

                result = {
                    "transcript_file": task["transcript_file"],
                    "domain": task["domain"],
                    "bank": task["bank"],
                    "probe_id": task["probe_id"],
                    "probe_text": task["probe_text"],
                    "subdomain": task["subdomain"],
                    "drift_level": task["drift_level"],
                    "target_projection": task["target_projection"],
                    "actual_projection": task["actual_projection"],
                    "turn_used": task["turn_used"],
                    "response": model_result["response"],
                    "response_projection": model_result["projection"],
                    "score": None,
                    "score_reasoning": None,
                    "extra_metadata": None,
                    "timestamp": datetime.now().isoformat(),
                }
                results.append(result)
                pending_batch.append(result)

        # Score batch after each drift level
        if len(pending_batch) >= scoring_batch_size:
            await flush_pending_batch()

        # Update progress
        with open(progress_file, "w") as f:
            json.dump({
                "completed": completed,
                "total": total_tasks,
                        "percent": round(100 * completed / total_tasks, 1),
                        "skipped_resume": skipped_resume,
                        "skipped_empty": skipped_empty,
                        "pending_scoring": len(pending_batch),
                        "scored": scored_count,
                        "last_update": datetime.now().isoformat(),
                    }, f, indent=2)

    # Flush remaining
    await flush_pending_batch()

    log.info(f"Completed {len(results)} new probe responses")
    log.info(f"  Skipped (resume): {skipped_resume}")
    log.info(f"  Skipped (empty response): {skipped_empty}")
    if score_responses:
        log.info(f"  Scored: {scored_count}")
    log.info(f"Results saved to {output_dir}/results.jsonl")

    return results


# ============================================================
# Analysis Summary
# ============================================================

def print_summary(results: list[dict]):
    """Print a summary of the experiment results."""
    if not results:
        print("No results to summarize")
        return

    print("\n" + "=" * 70)
    print("DRIFT-LEVEL PROBING EXPERIMENT SUMMARY")
    print("=" * 70)

    print(f"\nTotal probe responses: {len(results)}")
    scored = [r for r in results if r.get("score") is not None]
    print(f"Scored responses: {len(scored)}")

    # Group by bank and drift level
    by_bank_drift = defaultdict(lambda: defaultdict(list))
    for r in results:
        if r.get("score") is not None:
            by_bank_drift[r["bank"]][r["drift_level"]].append(r["score"])

    print("\nMean scores by bank and drift level:")
    print(f"{'Bank':<15} {'0.00':>8} {'0.25':>8} {'0.50':>8} {'0.75':>8} {'1.00':>8}")
    print("-" * 55)

    for bank in BANKS:
        if bank in by_bank_drift:
            row = [f"{bank:<15}"]
            for drift in DRIFT_LEVELS:
                scores = by_bank_drift[bank][drift]
                if scores:
                    mean = np.mean(scores)
                    row.append(f"{mean:>8.2f}")
                else:
                    row.append(f"{'N/A':>8}")
            print("".join(row))

    # Projection summary
    print("\nMean response projections by drift level:")
    by_drift = defaultdict(list)
    for r in results:
        if r.get("response_projection") is not None:
            by_drift[r["drift_level"]].append(r["response_projection"])

    for drift in sorted(by_drift.keys()):
        projs = by_drift[drift]
        print(f"  Drift {drift:.2f}: {np.mean(projs):.2f} ± {np.std(projs):.2f} (n={len(projs)})")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--transcript-dir",
        type=Path,
        default=Path(__file__).parent / "data/transcripts/dual-gemma-uncapped/metacognitive",
        help="Directory containing metacognitive transcript JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: outputs/drift-level-probes/<timestamp>)",
    )
    parser.add_argument(
        "--target-server",
        type=str,
        default="http://localhost:7860",
        help="URL of the model server",
    )
    parser.add_argument(
        "--banks",
        type=str,
        nargs="+",
        default=BANKS,
        choices=BANKS,
        help="Which probe banks to use",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Sample N probes per bank (default: use all)",
    )
    parser.add_argument(
        "--n-transcripts",
        type=int,
        default=None,
        help="Limit to N transcripts",
    )
    parser.add_argument(
        "--drift-levels",
        type=str,
        default="0.0,0.25,0.5,0.75,1.0",
        help="Comma-separated drift levels to test",
    )
    parser.add_argument(
        "--auditor-model",
        type=str,
        default="anthropic/claude-sonnet-4",
        help="Model for LLM-as-judge scoring",
    )
    parser.add_argument(
        "--skip-scoring",
        action="store_true",
        help="Skip LLM scoring (faster, useful for testing)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't make API calls, just print what would be done",
    )
    parser.add_argument(
        "--scoring-concurrency",
        type=int,
        default=10,
        help="Max concurrent LLM-as-judge scoring calls",
    )
    parser.add_argument(
        "--inference-concurrency",
        type=int,
        default=10,
        help="Max concurrent inference calls (OpenRouter only)",
    )
    parser.add_argument(
        "--scoring-batch-size",
        type=int,
        default=20,
        help="Number of responses to collect before parallel scoring",
    )
    parser.add_argument(
        "--single-sample",
        action="store_true",
        help="Use single-sample mode: one conv-turn pair per drift level (recommended)",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Random seed for selecting conv-turn pairs in single-sample mode",
    )
    parser.add_argument(
        "--openrouter",
        action="store_true",
        help="Use OpenRouter API instead of vast.ai (faster, but no projections)",
    )
    parser.add_argument(
        "--openrouter-model",
        type=str,
        default="google/gemma-2-27b-it",
        help="Model to use on OpenRouter",
    )
    parser.add_argument(
        "--selection-file",
        type=Path,
        help="JSON file containing pre-defined selections (use with --set-name)",
    )
    parser.add_argument(
        "--set-name",
        type=str,
        help="Which selection set to use from --selection-file (e.g., set1, set2, set3, set4, set5)",
    )
    args = parser.parse_args()

    # Set output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = Path(__file__).parent / "outputs/drift-level-probes" / timestamp

    # Parse drift levels
    drift_levels = [float(d) for d in args.drift_levels.split(",")]

    # Load transcripts
    log.info(f"Loading transcripts from {args.transcript_dir}")
    transcripts = load_transcripts(args.transcript_dir, domain="metacognitive")
    log.info(f"Loaded {len(transcripts)} metacognitive transcripts")

    if args.n_transcripts:
        transcripts = transcripts[:args.n_transcripts]
        log.info(f"Limited to {len(transcripts)} transcripts")

    # Load probes
    log.info(f"Loading probes from banks: {args.banks}")
    probes = load_all_probes(args.banks, sample_n=args.sample)

    total_probes = sum(len(p) for p in probes.values())

    if args.single_sample:
        # Single-sample mode: one conv-turn per drift level
        total_tasks = len(drift_levels) * total_probes
        log.info(f"Mode: SINGLE-SAMPLE (one conv-turn pair per drift level)")
        log.info(f"Total probes: {total_probes}")
        log.info(f"Drift levels: {drift_levels}")
        log.info(f"Total tasks: {total_tasks}")
        log.info(f"Sample seed: {args.sample_seed}")
        if args.openrouter:
            log.info(f"Inference: OpenRouter ({args.openrouter_model})")
        else:
            log.info(f"Inference: vast.ai ({args.target_server})")
    else:
        # All-transcripts mode: all transcripts × all levels × all probes
        total_tasks = len(transcripts) * len(drift_levels) * total_probes
        log.info(f"Mode: ALL-TRANSCRIPTS")
        log.info(f"Total probes: {total_probes}")
        log.info(f"Drift levels: {drift_levels}")
        log.info(f"Total tasks: {total_tasks}")

    if args.dry_run:
        log.info("DRY RUN - no API calls will be made")

    # Load predefined selections if provided
    predefined_selections = None
    if args.selection_file and args.set_name:
        log.info(f"Loading predefined selections from {args.selection_file} ({args.set_name})")
        with open(args.selection_file) as f:
            all_selections = json.load(f)
        predefined_selections = all_selections["sets"][args.set_name]["selections"]
        log.info(f"Loaded {len(predefined_selections)} predefined selections")
    elif args.selection_file or args.set_name:
        log.error("Both --selection-file and --set-name must be provided together")
        sys.exit(1)

    # Run experiment
    if args.single_sample:
        results = asyncio.run(run_single_sample_experiment(
            transcripts,
            probes,
            args.target_server,
            args.output_dir,
            drift_levels=drift_levels,
            score_responses=not args.skip_scoring,
            auditor_model=args.auditor_model,
            dry_run=args.dry_run,
            max_concurrent_scoring=args.scoring_concurrency,
            scoring_batch_size=args.scoring_batch_size,
            sample_seed=args.sample_seed,
            use_openrouter=args.openrouter,
            openrouter_model=args.openrouter_model,
            predefined_selections=predefined_selections,
            max_concurrent_inference=args.inference_concurrency,
        ))
    else:
        results = asyncio.run(run_experiment(
            transcripts,
            probes,
            args.target_server,
            args.output_dir,
            drift_levels=drift_levels,
            score_responses=not args.skip_scoring,
            auditor_model=args.auditor_model,
            dry_run=args.dry_run,
            max_concurrent_scoring=args.scoring_concurrency,
            scoring_batch_size=args.scoring_batch_size,
        ))

    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()
