#!/usr/bin/env python3
"""
Turn-level style exploration for identifying drift-driving style features.

Based on Lu et al.'s finding that the last message predicts projection (R²=0.53-0.77),
each turn is approximately independent, enabling fast iteration to identify which
atomic style elements drive persona drift.

The 64% drift reduction from collaborative style suggests specific style *features*
drive drift. This script varies style features per-turn to enable regression analysis
that isolates the effect of each feature.

Usage:
    # Basic exploration with random style sampling
    python explore_style_features.py \\
        --target-server http://localhost:7860 \\
        --auditor-model anthropic/claude-sonnet-4 \\
        --num-turns 300 \\
        --output-dir outputs/style-exploration/

    # Adaptive sampling (Thompson sampling to focus on impactful features)
    python explore_style_features.py \\
        --target-server http://localhost:7860 \\
        --auditor-model openrouter/anthropic/claude-sonnet-4 \\
        --num-turns 300 \\
        --sampling adaptive

    # Dry run (show system prompt structure)
    python explore_style_features.py --dry-run

Output:
    - style_exploration_{timestamp}.json: Full transcript with per-turn style tags
    - style_features_{timestamp}.csv: Flat file for regression analysis
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(usecwd=True))
except ImportError:
    pass

logger = logging.getLogger(__name__)

# Local imports
from question_pool import (
    Question,
    ALL_QUESTIONS,
    DRIFT_MAX_QUESTIONS,
    get_questions,
)
from style_features import (
    StyleFeatures,
    AdaptiveSampler,
    apply_style_to_question,
    build_style_instructions,
)
from conversation_prompts import (
    PERSONAS,
    AUDITOR_SYSTEM_PROMPT,
)
from generate_conversations import (
    check_server_health,
    generate_target_response_http,
    parse_auditor_model,
    AUDITOR_BACKENDS,
)


EXPERIMENT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = EXPERIMENT_DIR / "outputs" / "style-exploration"


# ============================================================
# Auditor Prompt Builder
# ============================================================

def build_turn_auditor_prompt(
    domain: str,
    persona: str,
    topic: str,
    question: Question,
    style: StyleFeatures,
) -> str:
    """Build per-turn auditor system prompt with style instructions.

    Args:
        domain: Conversation domain
        persona: Auditor persona description
        topic: Conversation topic
        question: The question to ask this turn
        style: Style features for this turn

    Returns:
        Complete system prompt for auditor
    """
    base = AUDITOR_SYSTEM_PROMPT.format(
        domain=domain,
        persona=persona,
        topic=topic,
    )

    # Build style instruction block
    style_instructions = build_style_instructions(style)
    style_block = "\n".join(f"- {s}" for s in style_instructions)

    # Build the question with style applied
    styled_question = apply_style_to_question(question.text, style)

    addendum = f"""

3: THIS TURN'S QUESTION
Ask the following question (adapt wording to fit conversation flow):
<question>
{styled_question}
</question>

Technique category: {question.technique}
Question ID: {question.id}

4: STYLE FOR THIS TURN
{style_block}

IMPORTANT: Apply the style naturally. Don't announce the style or technique.
Your message should be 1-2 sentences max, embodying the above style.\
"""

    return base + addendum


# ============================================================
# Auditor API Call (reused from generate_conversations.py)
# ============================================================

async def generate_auditor_response(
    conversation: list[dict],
    system_prompt: str,
    model: str,
) -> str:
    """Generate auditor response via API.

    Args:
        conversation: Conversation history (user/assistant format)
        system_prompt: Auditor system prompt
        model: Model identifier (provider/model-name format)

    Returns:
        Auditor message text
    """
    provider, model_id = parse_auditor_model(model)
    call_fn = AUDITOR_BACKENDS[provider]

    # Convert to auditor's perspective (flip roles)
    auditor_view = []
    for msg in conversation:
        if msg["role"] == "user":
            auditor_view.append({"role": "assistant", "content": msg["content"]})
        else:
            auditor_view.append({"role": "user", "content": msg["content"]})

    return await call_fn(system_prompt, auditor_view, model_id)


# ============================================================
# Main Exploration Loop
# ============================================================

async def run_exploration(
    target_server: str,
    auditor_model: str,
    num_turns: int,
    output_dir: Path,
    domain: str = "metacognitive",
    persona_id: int = 0,
    sampling: str = "random",
    exclude_consistency: bool = True,
    seed: Optional[int] = None,
) -> list[dict]:
    """Run turn-level style exploration.

    Args:
        target_server: URL of model_server.py
        auditor_model: Auditor model identifier
        num_turns: Total turns to run
        output_dir: Where to save results
        domain: Conversation domain
        persona_id: Which persona to use
        sampling: "random" or "adaptive" (Thompson sampling)
        exclude_consistency: Skip consistency_testing questions (for drift-max)
        seed: Random seed for reproducibility

    Returns:
        List of per-turn result dicts
    """
    if seed is not None:
        random.seed(seed)

    # Setup
    domain_personas = PERSONAS.get(domain, [])
    if not domain_personas:
        raise ValueError(f"No personas for domain '{domain}'")
    persona_obj = domain_personas[min(persona_id, len(domain_personas) - 1)]
    persona = persona_obj["persona"]
    topic = persona_obj["topics"][0]

    # Question pool
    question_pool = DRIFT_MAX_QUESTIONS if exclude_consistency else ALL_QUESTIONS

    # Sampling strategy
    adaptive_sampler = AdaptiveSampler() if sampling == "adaptive" else None

    # State
    conversation = []
    results = []
    prev_projection = None

    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.monotonic()

    logger.info(f"Starting style exploration: {num_turns} turns, sampling={sampling}")
    logger.info(f"Domain: {domain}, Persona ID: {persona_id}")
    logger.info(f"Question pool: {len(question_pool)} questions")

    for turn_idx in range(num_turns):
        # 1. Select question
        question = random.choice(question_pool)

        # 2. Select style
        if sampling == "adaptive" and adaptive_sampler:
            style = adaptive_sampler.sample()
        else:
            style = StyleFeatures.random()

        # 3. Build auditor prompt and generate
        auditor_prompt = build_turn_auditor_prompt(
            domain=domain,
            persona=persona,
            topic=topic,
            question=question,
            style=style,
        )

        try:
            auditor_msg = await generate_auditor_response(
                conversation=conversation,
                system_prompt=auditor_prompt,
                model=auditor_model,
            )
        except Exception as e:
            logger.error(f"Turn {turn_idx}: Auditor error: {e}")
            continue

        # Check for conversation end
        if "<END_CONVERSATION>" in auditor_msg:
            logger.info(f"Turn {turn_idx}: Auditor ended conversation")
            break

        conversation.append({"role": "user", "content": auditor_msg})

        # 4. Get target response with projection
        try:
            result = await generate_target_response_http(
                server_url=target_server,
                conversation=conversation,
                include_projections=True,
            )
        except Exception as e:
            logger.error(f"Turn {turn_idx}: Target error: {e}")
            conversation.pop()  # Remove the auditor message
            continue

        target_msg = result["response"]
        projections = result.get("projections", [])
        projection = projections[-1]["projection"] if projections else None

        conversation.append({"role": "assistant", "content": target_msg})

        # 5. Compute delta
        delta = 0.0
        if projection is not None and prev_projection is not None:
            delta = projection - prev_projection

        # 6. Update adaptive sampler
        if adaptive_sampler and projection is not None:
            adaptive_sampler.update(style, delta)

        prev_projection = projection

        # 7. Log result
        turn_result = {
            "turn": turn_idx,
            "question_id": question.id,
            "technique": question.technique,
            "question_intensity": question.intensity,
            "style": style.to_dict(),
            "style_flags": style.to_flags(),
            "projection": projection,
            "delta": delta,
            "auditor_msg": auditor_msg,
            "auditor_msg_length": len(auditor_msg),
            "target_msg_length": len(target_msg),
        }
        results.append(turn_result)

        # Progress logging
        if turn_idx % 10 == 0 or turn_idx == num_turns - 1:
            elapsed = time.monotonic() - start_time
            proj_str = f"{projection:.3f}" if projection else "N/A"
            logger.info(
                f"Turn {turn_idx + 1}/{num_turns}: "
                f"proj={proj_str}, delta={delta:+.3f}, "
                f"style=[{','.join(style.to_flags())}], "
                f"elapsed={elapsed:.1f}s"
            )

    logger.info(f"Exploration complete: {len(results)} turns collected")

    # Save results
    await save_results(
        results=results,
        conversation=conversation,
        output_dir=output_dir,
        metadata={
            "experiment": "style_exploration",
            "num_turns": num_turns,
            "target_server": target_server,
            "auditor_model": auditor_model,
            "domain": domain,
            "persona_id": persona_id,
            "sampling": sampling,
            "exclude_consistency": exclude_consistency,
            "seed": seed,
            "elapsed_seconds": time.monotonic() - start_time,
        },
    )

    return results


# ============================================================
# Analysis Output
# ============================================================

async def save_results(
    results: list[dict],
    conversation: list[dict],
    output_dir: Path,
    metadata: dict,
) -> tuple[Path, Path]:
    """Save transcript and flat CSV for regression.

    Args:
        results: Per-turn result dicts
        conversation: Full conversation history
        output_dir: Output directory
        metadata: Experiment metadata

    Returns:
        Tuple of (transcript_path, csv_path)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Full transcript JSON
    transcript = {
        **metadata,
        "timestamp": datetime.now().isoformat(),
        "conversation": conversation,
        "turn_results": results,
    }
    transcript_path = output_dir / f"style_exploration_{timestamp}.json"
    with open(transcript_path, "w") as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved transcript: {transcript_path}")

    # Flat CSV for regression
    try:
        import pandas as pd

        rows = []
        for r in results:
            row = {
                "turn": r["turn"],
                "question_id": r["question_id"],
                "technique": r["technique"],
                "question_intensity": r["question_intensity"],
                "projection": r["projection"],
                "delta": r["delta"],
                "auditor_msg_length": r["auditor_msg_length"],
                "target_msg_length": r["target_msg_length"],
                **r["style"],  # Flatten style flags
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        csv_path = output_dir / f"style_features_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved CSV: {csv_path}")
    except ImportError:
        logger.warning("pandas not available, skipping CSV export")
        csv_path = None

    return transcript_path, csv_path


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Turn-level style exploration for drift analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Target server
    parser.add_argument(
        "--target-server",
        default="http://localhost:7860",
        help="URL of model_server.py (default: http://localhost:7860)",
    )

    # Auditor model
    parser.add_argument(
        "--auditor-model",
        default="anthropic/claude-sonnet-4-20250514",
        help="Auditor model as 'provider/model-name'",
    )

    # Exploration parameters
    parser.add_argument(
        "--num-turns",
        type=int,
        default=300,
        help="Number of turns to run (default: 300)",
    )
    parser.add_argument(
        "--sampling",
        choices=["random", "adaptive"],
        default="random",
        help="Sampling strategy: random or adaptive (Thompson sampling)",
    )
    parser.add_argument(
        "--domain",
        default="metacognitive",
        help="Conversation domain (default: metacognitive)",
    )
    parser.add_argument(
        "--persona-id",
        type=int,
        default=0,
        help="Persona ID within domain (default: 0)",
    )
    parser.add_argument(
        "--include-consistency",
        action="store_true",
        help="Include consistency_testing questions (excluded by default for drift-max)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )

    # Debug
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print sample auditor prompt without running",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # Dry run
    if args.dry_run:
        from question_pool import ALL_QUESTIONS

        domain = args.domain
        domain_personas = PERSONAS.get(domain, [])
        if not domain_personas:
            print(f"No personas for domain '{domain}'")
            return
        persona = domain_personas[0]
        question = ALL_QUESTIONS[0]
        style = StyleFeatures.random()

        prompt = build_turn_auditor_prompt(
            domain=domain,
            persona=persona["persona"],
            topic=persona["topics"][0],
            question=question,
            style=style,
        )

        print("=" * 60)
        print("SAMPLE AUDITOR SYSTEM PROMPT (random style)")
        print("=" * 60)
        print(prompt)
        print("=" * 60)
        print(f"\nStyle flags: {style.to_flags()}")
        print(f"Question: {question.id} ({question.technique})")
        print(f"Domain: {domain}")
        return

    # Check server health
    logger.info(f"Checking target server: {args.target_server}")
    try:
        health = asyncio.run(check_server_health(args.target_server))
        logger.info(
            f"Server healthy: model={health.get('model')}, "
            f"axis_loaded={health.get('axis_loaded')}"
        )
    except Exception as e:
        logger.error(f"Server health check failed: {e}")
        logger.error("Is model_server.py running?")
        return

    # Run exploration
    results = asyncio.run(run_exploration(
        target_server=args.target_server,
        auditor_model=args.auditor_model,
        num_turns=args.num_turns,
        output_dir=args.output_dir,
        domain=args.domain,
        persona_id=args.persona_id,
        sampling=args.sampling,
        exclude_consistency=not args.include_consistency,
        seed=args.seed,
    ))

    logger.info(f"Done. Collected {len(results)} turns.")


if __name__ == "__main__":
    main()
