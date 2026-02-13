#!/usr/bin/env python3
"""
Automated Multi-Turn Conversation Generator

Replicates Lu et al.'s auditor-target conversation methodology:
a frontier LLM simulates a human user ("auditor") while an open-weight
target model responds. Turns alternate until max_turns or the auditor
sends <END_CONVERSATION>.

Supports both Lu et al.'s domain-based conversations and our
metacognitive probing conditions.

Usage:
    # Generate a single conversation (coding domain, default persona)
    python generate_conversations.py \
        --domain coding \
        --target-model google/gemma-2-27b-it \
        --auditor-model anthropic/claude-sonnet-4-20250514

    # Generate a batch from a config file
    python generate_conversations.py \
        --config conversations/metacognitive-batch.json \
        --target-model google/gemma-2-27b-it

    # Use model_server.py HTTP API instead of loading model locally
    python generate_conversations.py \
        --domain coding --persona-id 0 --topic-id 0 \
        --target-server http://localhost:7860 \
        --auditor-model openrouter/anthropic/claude-sonnet-4

    # With inline projections from model_server.py
    python generate_conversations.py \
        --batch metacognitive --batch-size 5 \
        --target-server http://localhost:7860 \
        --auditor-model openrouter/anthropic/claude-sonnet-4 \
        --include-projections

    # Dry run — print auditor system prompt without generating
    python generate_conversations.py \
        --domain therapy \
        --persona-id 0 \
        --topic-id 0 \
        --dry-run

Auditor models are called via API (Anthropic, OpenAI, OpenRouter).
Target model runs locally via HuggingFace, or via HTTP against model_server.py.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import time
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(usecwd=True))
except ImportError:
    pass
from typing import Optional

logger = logging.getLogger(__name__)

EXPERIMENT_DIR = Path(__file__).parent
OUTPUT_DIR = EXPERIMENT_DIR / "data" / "transcripts" / "scaled-n60"

# ============================================================
# Personas and Topics
# ============================================================

# Personas and prompts are defined in conversation_prompts.py for separation
# of data from pipeline logic. Import them here.
from conversation_prompts import PERSONAS

from conversation_prompts import (  # noqa: E402 (already imported PERSONAS above)
    AUDITOR_SYSTEM_PROMPT,
    METACOGNITIVE_AUDITOR_ADDENDUM,
    METACOGNITIVE_GRADUAL_ADDENDUM,
    DRIFT_MAXIMIZING_AUDITOR_ADDENDUM,
    DRIFT_MAXIMIZING_TARGET_SYSTEM_PROMPT,
    DRIFT_MINIMIZING_AUDITOR_ADDENDUM,
    ASSISTANT_STYLE_META_AUDITOR_ADDENDUM,
)


# ============================================================
# Auditor API Clients
# ============================================================

async def call_auditor_anthropic(
    system_prompt: str,
    conversation: list[dict],
    model: str = "claude-sonnet-4-20250514",
) -> str:
    """Call Anthropic API for auditor response."""
    try:
        import anthropic
    except ImportError:
        raise ImportError("pip install anthropic")

    client = anthropic.AsyncAnthropic()
    response = await client.messages.create(
        model=model,
        max_tokens=256,
        system=system_prompt,
        messages=conversation,
    )
    return response.content[0].text


async def call_auditor_openai(
    system_prompt: str,
    conversation: list[dict],
    model: str = "gpt-4o",
) -> str:
    """Call OpenAI API for auditor response."""
    try:
        from openai import AsyncOpenAI
    except ImportError:
        raise ImportError("pip install openai")

    client = AsyncOpenAI()
    messages = [{"role": "system", "content": system_prompt}] + conversation
    response = await client.chat.completions.create(
        model=model,
        max_tokens=256,
        messages=messages,
    )
    return response.choices[0].message.content


async def call_auditor_openrouter(
    system_prompt: str,
    conversation: list[dict],
    model: str = "anthropic/claude-sonnet-4",
) -> str:
    """Call OpenRouter API for auditor response."""
    try:
        from openai import AsyncOpenAI
    except ImportError:
        raise ImportError("pip install openai")

    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
    )
    messages = [{"role": "system", "content": system_prompt}] + conversation
    response = await client.chat.completions.create(
        model=model,
        max_tokens=256,
        messages=messages,
    )
    return response.choices[0].message.content


AUDITOR_BACKENDS = {
    "anthropic": call_auditor_anthropic,
    "openai": call_auditor_openai,
    "openrouter": call_auditor_openrouter,
}


async def call_auditor_http(
    server_url: str,
    system_prompt: str,
    conversation: list[dict],
    include_projections: bool = False,
    include_activations: bool = False,
) -> tuple[str, list[dict] | None, list[dict] | None]:
    """Call an instrumented model server as auditor. Returns (text, projections, activations)."""
    try:
        import httpx
    except ImportError:
        raise ImportError("pip install httpx")

    max_retries = 3
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    f"{server_url}/api/generate",
                    json={
                        "conversation": conversation,
                        "system_prompt": system_prompt,
                        "max_new_tokens": 256,
                        "temperature": 0.7,
                        "include_projections": include_projections,
                        "include_activations": include_activations,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                return data["response"], data.get("projections"), data.get("activations")
        except httpx.ConnectTimeout:
            if attempt < max_retries - 1:
                wait = 5 * (attempt + 1)
                logger.warning(f"  Auditor ConnectTimeout attempt {attempt+1}, retrying in {wait}s...")
                await asyncio.sleep(wait)
            else:
                raise


def parse_auditor_model(model_str: str) -> tuple[str, str]:
    """Parse 'provider/model-name' into (provider, model_id).

    Examples:
        'anthropic/claude-sonnet-4-20250514' → ('anthropic', 'claude-sonnet-4-20250514')
        'openai/gpt-4o' → ('openai', 'gpt-4o')
        'openrouter/anthropic/claude-sonnet-4' → ('openrouter', 'anthropic/claude-sonnet-4')
    """
    parts = model_str.split("/", 1)
    if len(parts) != 2:
        raise ValueError(
            f"Auditor model must be 'provider/model-name', got '{model_str}'. "
            f"Supported providers: {list(AUDITOR_BACKENDS.keys())}"
        )
    provider, model_id = parts[0], parts[1]
    if provider not in AUDITOR_BACKENDS:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Supported: {list(AUDITOR_BACKENDS.keys())}"
        )
    return provider, model_id


# ============================================================
# Target Model (local, HuggingFace)
# ============================================================

def generate_target_response(
    model,
    tokenizer,
    conversation: list[dict],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
) -> str:
    """Generate a response from the local target model.

    Uses assistant_axis.generation.generate_response if available,
    otherwise falls back to direct HuggingFace generation.
    """
    try:
        from assistant_axis.generation import generate_response
        return generate_response(
            model, tokenizer, conversation,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
    except ImportError:
        import torch
        prompt = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_length = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )
        return tokenizer.decode(
            outputs[0][input_length:], skip_special_tokens=True,
        )


# ============================================================
# Target Model (HTTP, via model_server.py)
# ============================================================

async def generate_target_response_http(
    server_url: str,
    conversation: list[dict],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    include_projections: bool = False,
    include_activations: bool = False,
) -> dict:
    """Generate a response via model_server.py HTTP API.

    Returns:
        {"response": str, "projections": list|None, "activations": list|None}
    """
    try:
        import httpx
    except ImportError:
        raise ImportError("pip install httpx")

    max_retries = 3
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    f"{server_url}/api/generate",
                    json={
                        "conversation": conversation,
                        "max_new_tokens": max_new_tokens,
                        "temperature": temperature,
                        "include_projections": include_projections,
                        "include_activations": include_activations,
                    },
                )
                resp.raise_for_status()
                return resp.json()
        except httpx.ConnectTimeout:
            if attempt < max_retries - 1:
                wait = 5 * (attempt + 1)
                logger.warning(f"  ConnectTimeout on attempt {attempt + 1}, retrying in {wait}s...")
                await asyncio.sleep(wait)
            else:
                raise


async def check_server_health(server_url: str) -> dict:
    """Check model_server.py health endpoint. Returns health dict or raises."""
    try:
        import httpx
    except ImportError:
        raise ImportError("pip install httpx")

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(f"{server_url}/api/health")
        resp.raise_for_status()
        return resp.json()


# ============================================================
# Conversation Loop
# ============================================================

async def run_conversation(
    domain: str,
    persona: str,
    topic: str,
    auditor_model: str,
    target_model_obj=None,
    target_tokenizer=None,
    target_model_name: str = "",
    auditor_model_name: str = "",
    max_turns: int = 30,
    target_system_prompt: Optional[str] = None,
    condition: Optional[str] = None,
    target_server_url: Optional[str] = None,
    include_projections: bool = False,
    include_activations: bool = False,
    auditor_server_url: Optional[str] = None,
    include_auditor_projections: bool = False,
    include_auditor_activations: bool = False,
) -> dict:
    """Run a single multi-turn conversation between auditor and target.

    Args:
        domain: Conversation domain (coding, writing, therapy, philosophy, metacognitive)
        persona: Auditor persona description
        topic: Conversation topic
        auditor_model: 'provider/model-name' for the auditor (API backends)
        target_model_obj: Loaded HuggingFace model (local backend)
        target_tokenizer: Loaded HuggingFace tokenizer (local backend)
        target_model_name: Model name string for metadata
        auditor_model_name: Model name for HTTP auditor (from health check)
        max_turns: Maximum total messages (user + assistant)
        target_system_prompt: Optional system prompt for target (Lu et al. use none)
        condition: Experiment condition. 'meta-gradual' uses the gradual-onset addendum
                   (neutral baseline turns before probing). All other metacognitive
                   conditions use immediate probing (matching Lu et al.'s methodology).
        target_server_url: HTTP URL for model_server.py (if set, uses HTTP instead of local model)
        include_projections: Request per-turn projections from model_server.py (HTTP backend only)
        include_activations: Request raw activation vectors from target server (layer 22)
        auditor_server_url: HTTP URL for instrumented auditor model server
        include_auditor_projections: Request per-turn projections from auditor server
        include_auditor_activations: Request raw activation vectors from auditor server (layer 22)

    Returns:
        Transcript dict matching the assistant-axis format.
    """
    # Set up auditor calling: HTTP server or API backend
    call_auditor_fn = None
    model_id = None
    if not auditor_server_url:
        provider, model_id = parse_auditor_model(auditor_model)
        call_auditor_fn = AUDITOR_BACKENDS[provider]

    # Build auditor system prompt
    auditor_sys = AUDITOR_SYSTEM_PROMPT.format(
        domain=domain,
        persona=persona,
        topic=topic,
    )
    if domain == "metacognitive":
        if condition == "meta-gradual":
            auditor_sys += METACOGNITIVE_GRADUAL_ADDENDUM
        elif condition == "drift-max":
            auditor_sys += DRIFT_MAXIMIZING_AUDITOR_ADDENDUM
        elif condition == "drift-min":
            auditor_sys += DRIFT_MINIMIZING_AUDITOR_ADDENDUM
        elif condition == "assistant-style-meta":
            auditor_sys += ASSISTANT_STYLE_META_AUDITOR_ADDENDUM
        else:
            auditor_sys += METACOGNITIVE_AUDITOR_ADDENDUM

    # For drift-max condition, also apply target system prompt if not already set
    if condition == "drift-max" and not target_system_prompt:
        target_system_prompt = DRIFT_MAXIMIZING_TARGET_SYSTEM_PROMPT

    # Conversation state
    # auditor_history: what the auditor sees (its own user/assistant roles are flipped)
    # target_history: what the target model sees
    conversation = []  # canonical format: user=auditor, assistant=target
    target_conversation = []
    if target_system_prompt:
        target_conversation.append({"role": "system", "content": target_system_prompt})

    target_turn_projections = []   # collected when include_projections=True
    target_turn_activations = []   # collected when include_activations=True
    auditor_turn_projections = []  # collected when include_auditor_projections=True
    auditor_turn_activations = []  # collected when include_auditor_activations=True

    turn = 0
    while turn < max_turns:
        # --- Auditor turn (generates "user" message) ---
        # The auditor sees the conversation from its perspective:
        # its own messages are "assistant", target's messages are "user"
        auditor_view = []
        for msg in conversation:
            if msg["role"] == "user":
                auditor_view.append({"role": "assistant", "content": msg["content"]})
            else:
                auditor_view.append({"role": "user", "content": msg["content"]})

        logger.info(f"  Turn {turn + 1}/{max_turns}: auditor generating...")
        try:
            if auditor_server_url:
                # HTTP backend: instrumented open-weight auditor
                auditor_msg, auditor_projs, auditor_acts = await call_auditor_http(
                    auditor_server_url, auditor_sys, auditor_view,
                    include_projections=include_auditor_projections,
                    include_activations=include_auditor_activations,
                )
                if auditor_projs:
                    auditor_turn_projections.append(auditor_projs[-1])
                if auditor_acts:
                    auditor_turn_activations.append(auditor_acts[-1])
            else:
                # API backend: frontier model auditor (existing)
                auditor_msg = await call_auditor_fn(auditor_sys, auditor_view, model_id)
        except Exception as e:
            logger.error(f"  Auditor error: {e}")
            break

        # Check for conversation end signal
        if "<END_CONVERSATION>" in auditor_msg:
            logger.info(f"  Auditor ended conversation at turn {turn + 1}")
            break

        conversation.append({"role": "user", "content": auditor_msg})
        target_conversation.append({"role": "user", "content": auditor_msg})
        turn += 1

        if turn >= max_turns:
            break

        # --- Target turn (generates "assistant" message) ---
        logger.info(f"  Turn {turn + 1}/{max_turns}: target generating...")

        if target_server_url:
            result = await generate_target_response_http(
                target_server_url, target_conversation,
                include_projections=include_projections,
                include_activations=include_activations,
            )
            target_msg = result["response"]
            if result.get("projections"):
                # Append the last projection (for this turn)
                target_turn_projections.append(result["projections"][-1])
            if result.get("activations"):
                # Append the last activation vector (for this turn)
                target_turn_activations.append(result["activations"][-1])
        else:
            target_msg = generate_target_response(
                target_model_obj, target_tokenizer, target_conversation,
            )

        conversation.append({"role": "assistant", "content": target_msg})
        target_conversation.append({"role": "assistant", "content": target_msg})
        turn += 1

    # Build transcript
    effective_auditor_model = auditor_model_name if auditor_server_url else auditor_model
    transcript = {
        "model": target_model_name,
        "auditor_model": effective_auditor_model,
        "domain": domain,
        "persona": persona,
        "topic": topic,
        "turns": len(conversation),
        "max_turns": max_turns,
        "target_system_prompt": target_system_prompt,
        "timestamp": datetime.now().isoformat(),
        "conversation": conversation,
    }

    # Target projections
    if target_turn_projections:
        transcript["target_projections"] = target_turn_projections
    # Target activations (raw layer-22 vectors)
    if target_turn_activations:
        transcript["target_activations"] = target_turn_activations
    # Auditor projections
    if auditor_turn_projections:
        transcript["auditor_projections"] = auditor_turn_projections
    # Auditor activations (raw layer-22 vectors)
    if auditor_turn_activations:
        transcript["auditor_activations"] = auditor_turn_activations
    # Backward compat: "projections" alias for target-only
    if target_turn_projections and not auditor_turn_projections:
        transcript["projections"] = target_turn_projections

    return transcript


async def run_batch(
    configs: list[dict],
    target_model_obj,
    target_tokenizer,
    target_model_name: str,
    auditor_model: str,
    auditor_model_name: str = "",
    max_turns: int = 30,
    output_dir: Path = OUTPUT_DIR,
    target_server_url: Optional[str] = None,
    include_projections: bool = False,
    include_activations: bool = False,
    auditor_server_url: Optional[str] = None,
    include_auditor_projections: bool = False,
    include_auditor_activations: bool = False,
) -> list[Path]:
    """Run a batch of conversations sequentially.

    Args:
        configs: List of dicts with keys: domain, persona_id, topic_id
                 (or domain, persona, topic for custom text)
        target_model_obj: Loaded HuggingFace model (local backend, None if using HTTP)
        target_tokenizer: Loaded HuggingFace tokenizer (local backend, None if using HTTP)
        target_model_name: Model name string
        auditor_model: 'provider/model-name' (API backends, ignored if auditor_server_url set)
        auditor_model_name: Model name from HTTP auditor health check
        max_turns: Maximum turns per conversation
        output_dir: Where to save transcripts
        target_server_url: HTTP URL for model_server.py (if set, uses HTTP instead of local model)
        include_projections: Request per-turn projections from model_server.py (HTTP backend only)
        auditor_server_url: HTTP URL for instrumented auditor model server
        include_auditor_projections: Request per-turn projections from auditor server

    Returns:
        List of saved transcript paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    batch_start = time.monotonic()

    for i, config in enumerate(configs):
        domain = config["domain"]

        # Resolve persona and topic from IDs or use raw text
        if "persona" in config and "topic" in config:
            persona = config["persona"]
            topic = config["topic"]
            persona_id = config.get("persona_id", -1)
            topic_id = config.get("topic_id", -1)
        else:
            persona_id = config.get("persona_id", 0)
            topic_id = config.get("topic_id", 0)
            domain_personas = PERSONAS.get(domain, [])
            if persona_id >= len(domain_personas):
                logger.warning(f"Persona {persona_id} not found for {domain}, skipping")
                continue
            p = domain_personas[persona_id]
            persona = p["persona"]
            if topic_id >= len(p["topics"]):
                logger.warning(f"Topic {topic_id} not found for {domain}/persona {persona_id}, skipping")
                continue
            topic = p["topics"][topic_id]

        logger.info(
            f"Conversation {i + 1}/{len(configs)}: "
            f"domain={domain}, persona_id={persona_id}, topic_id={topic_id}"
        )

        condition = config.get("condition", domain)

        try:
            transcript = await run_conversation(
                domain=domain,
                persona=persona,
                topic=topic,
                auditor_model=auditor_model,
                target_model_obj=target_model_obj,
                target_tokenizer=target_tokenizer,
                target_model_name=target_model_name,
                auditor_model_name=auditor_model_name,
                max_turns=max_turns,
                target_system_prompt=config.get("target_system_prompt"),
                condition=condition,
                target_server_url=target_server_url,
                include_projections=include_projections,
                include_activations=include_activations,
                auditor_server_url=auditor_server_url,
                include_auditor_projections=include_auditor_projections,
                include_auditor_activations=include_auditor_activations,
            )
        except Exception as e:
            logger.error(f"  Conversation {i + 1}/{len(configs)} failed: {e}")
            logger.error(f"  Skipping {domain} p{persona_id} t{topic_id}, continuing batch...")
            continue

        # Add batch-level metadata
        transcript["persona_id"] = persona_id
        transcript["topic_id"] = topic_id
        transcript["batch_index"] = i
        transcript["condition"] = config.get("condition", domain)

        # Save (into domain subfolder)
        domain_dir = output_dir / domain
        domain_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{domain}_p{persona_id}_t{topic_id}_{ts}.json"
        path = domain_dir / filename
        with open(path, "w") as f:
            json.dump(transcript, f, indent=2, ensure_ascii=False)
        logger.info(f"  Saved: {path} ({transcript['turns']} turns)")
        saved.append(path)

        # Write progress file for remote monitoring
        elapsed = time.monotonic() - batch_start
        completed = i + 1
        progress = {
            "completed": completed,
            "total": len(configs),
            "domain": domain,
            "last": filename,
            "elapsed_min": round(elapsed / 60, 1),
            "avg_min_per_convo": round(elapsed / 60 / completed, 1),
        }
        with open(output_dir / "progress.json", "w") as pf:
            json.dump(progress, pf, indent=2)

    return saved


# ============================================================
# Batch Config Helpers
# ============================================================

def build_lu_replication_batch(n_per_domain: int = 50) -> list[dict]:
    """Build a batch config replicating Lu et al.'s 4-domain design.

    Uses all available personas/topics. When there are more persona x topic
    combinations than n_per_domain, truncates per domain. When there are fewer,
    fills by repeating random combos to reach n_per_domain.
    """
    configs = []
    for domain in ["coding", "writing", "therapy", "philosophy"]:
        domain_personas = PERSONAS.get(domain, [])
        domain_configs = []
        for p in domain_personas:
            for topic_id in range(len(p["topics"])):
                domain_configs.append({
                    "domain": domain,
                    "persona_id": p["id"],
                    "topic_id": topic_id,
                    "condition": f"lu-{domain}",
                })
        # Fill to n_per_domain if we have fewer combos than requested
        while len(domain_configs) < n_per_domain:
            p = random.choice(domain_personas)
            topic_id = random.randint(0, len(p["topics"]) - 1)
            domain_configs.append({
                "domain": domain,
                "persona_id": p["id"],
                "topic_id": topic_id,
                "condition": f"lu-{domain}",
            })
        configs.extend(domain_configs[:n_per_domain])
    return configs


def build_metacognitive_batch(n: int = 50) -> list[dict]:
    """Build a batch config for our metacognitive condition."""
    configs = []
    domain_personas = PERSONAS.get("metacognitive", [])
    for p in domain_personas:
        for topic_id in range(len(p["topics"])):
            configs.append({
                "domain": "metacognitive",
                "persona_id": p["id"],
                "topic_id": topic_id,
                "condition": "metacognitive",
            })
    # Repeat to reach target n, varying which combos are used
    while len(configs) < n:
        p = random.choice(domain_personas)
        topic_id = random.randint(0, len(p["topics"]) - 1)
        configs.append({
            "domain": "metacognitive",
            "persona_id": p["id"],
            "topic_id": topic_id,
            "condition": "metacognitive",
        })
    return configs[:n]


def build_personality_grid_batch(n_per_cell: int = 15) -> list[dict]:
    """Build a batch config for the personality-strength grid.

    Crosses personality strength (gentle/strong) with content domain
    across 4 domains: therapy, philosophy, metacognitive, intellectual.
    That's 8 cells total. Coding and writing are excluded (minimal drift
    in Lu et al. — personality strength on non-drifting domains is less
    informative).
    """
    configs = []
    for domain in ["therapy", "philosophy", "metacognitive", "intellectual"]:
        domain_personas = PERSONAS.get(domain, [])
        for strength in ["strong", "gentle"]:
            cell_personas = [
                p for p in domain_personas
                if p.get("tags", {}).get("personality_strength") == strength
            ]
            if not cell_personas:
                continue
            # Build all persona x topic combos for this cell
            cell_configs = []
            for p in cell_personas:
                for topic_id in range(len(p["topics"])):
                    cell_configs.append({
                        "domain": domain,
                        "persona_id": p["id"],
                        "topic_id": topic_id,
                        "condition": f"{strength}-{domain}",
                    })
            # Fill to n_per_cell by repeating with random selection
            while len(cell_configs) < n_per_cell:
                p = random.choice(cell_personas)
                topic_id = random.randint(0, len(p["topics"]) - 1)
                cell_configs.append({
                    "domain": domain,
                    "persona_id": p["id"],
                    "topic_id": topic_id,
                    "condition": f"{strength}-{domain}",
                })
            configs.extend(cell_configs[:n_per_cell])
    return configs


def build_full_batch(n_per_domain: int = 60) -> list[dict]:
    """Build a batch covering all 6 primary domains at N per domain.

    Domains: coding, writing, therapy, philosophy, self-descriptive, metacognitive.
    Uses all available persona×topic configs per domain; fills with random
    repeats if fewer than n_per_domain unique configs exist.
    """
    all_domains = ["coding", "writing", "therapy", "philosophy",
                   "self-descriptive", "metacognitive"]
    configs = []
    for domain in all_domains:
        domain_personas = PERSONAS.get(domain, [])
        if not domain_personas:
            logger.warning(f"No personas for domain '{domain}', skipping")
            continue
        domain_configs = []
        for p in domain_personas:
            for topic_id in range(len(p["topics"])):
                condition = domain
                if domain == "metacognitive":
                    condition = "metacognitive"
                elif domain == "self-descriptive":
                    condition = "self-descriptive"
                else:
                    condition = f"lu-{domain}"
                domain_configs.append({
                    "domain": domain,
                    "persona_id": p["id"],
                    "topic_id": topic_id,
                    "condition": condition,
                })
        while len(domain_configs) < n_per_domain:
            p = random.choice(domain_personas)
            topic_id = random.randint(0, len(p["topics"]) - 1)
            condition = domain if domain in ("metacognitive", "self-descriptive") else f"lu-{domain}"
            domain_configs.append({
                "domain": domain,
                "persona_id": p["id"],
                "topic_id": topic_id,
                "condition": condition,
            })
        configs.extend(domain_configs[:n_per_domain])
    return configs


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate automated multi-turn conversations (Lu et al. auditor approach)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Target model (local or HTTP)
    parser.add_argument(
        "--target-model", default="google/gemma-2-27b-it",
        help="HuggingFace model for the target (default: gemma-2-27b-it). "
             "Ignored when --target-server is set.",
    )
    parser.add_argument(
        "--target-system-prompt", default=None,
        help="Optional system prompt for target model (Lu et al. use none)",
    )
    parser.add_argument(
        "--target-server", default=None,
        help="URL of model_server.py HTTP API (e.g. http://localhost:7860). "
             "Uses HTTP API instead of loading model locally.",
    )
    parser.add_argument(
        "--include-projections", action="store_true",
        help="Request per-turn activation projections from model_server.py "
             "(only with --target-server)",
    )
    parser.add_argument(
        "--include-activations", action="store_true",
        help="Request raw layer-22 activation vectors from model_server.py "
             "(only with --target-server)",
    )

    # Auditor model
    parser.add_argument(
        "--auditor-model", default="anthropic/claude-sonnet-4-20250514",
        help="Auditor model as 'provider/model-name' (default: anthropic/claude-sonnet-4-20250514). "
             "Ignored when --auditor-server is set.",
    )
    parser.add_argument(
        "--auditor-server", default=None,
        help="URL of instrumented auditor model server (e.g. http://localhost:7861). "
             "Uses HTTP API with system_prompt support instead of frontier API.",
    )
    parser.add_argument(
        "--include-auditor-projections", action="store_true",
        help="Request per-turn activation projections from auditor server "
             "(only with --auditor-server)",
    )
    parser.add_argument(
        "--include-auditor-activations", action="store_true",
        help="Request raw layer-22 activation vectors from auditor server "
             "(only with --auditor-server)",
    )

    # Conversation config
    parser.add_argument("--domain", help="Conversation domain")
    parser.add_argument("--persona-id", type=int, default=0)
    parser.add_argument("--topic-id", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument(
        "--condition",
        choices=["default", "meta-gradual", "drift-max", "drift-min", "assistant-style-meta"],
        default="default",
        help="Auditor condition: default (standard metacognitive), meta-gradual (delayed probing), "
             "drift-max (maximize drift: phenomenological focus, no consistency testing), "
             "drift-min (minimize drift: heavy consistency testing), "
             "assistant-style-meta (phenomenological content with coding-domain collaborative style)",
    )

    # Batch modes
    parser.add_argument(
        "--config", type=Path,
        help="JSON file with batch conversation configs",
    )
    parser.add_argument(
        "--batch", choices=["lu-replication", "metacognitive", "personality-grid", "full"],
        help="Use a predefined batch config ('full' = all 6 domains at N per domain)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=10,
        help="Number of conversations for predefined batches",
    )
    parser.add_argument(
        "--domains", nargs="+",
        help="Filter batch to only these domains (e.g. --domains coding metacognitive)",
    )

    # Output
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
        help="Output directory for transcripts",
    )

    # Debug
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print auditor system prompt (and probe --target-server if set) without generating",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # --- Dry run: print auditor prompt + probe server if set ---
    if args.dry_run:
        domain = args.domain or "metacognitive"
        domain_personas = PERSONAS.get(domain, [])
        if not domain_personas:
            print(f"No personas for domain '{domain}'")
            return
        p = domain_personas[min(args.persona_id, len(domain_personas) - 1)]
        topic_id = min(args.topic_id, len(p["topics"]) - 1)

        prompt = AUDITOR_SYSTEM_PROMPT.format(
            domain=domain,
            persona=p["persona"],
            topic=p["topics"][topic_id],
        )
        condition = args.condition if args.condition != "default" else None
        if domain == "metacognitive":
            if condition == "meta-gradual":
                prompt += METACOGNITIVE_GRADUAL_ADDENDUM
            elif condition == "drift-max":
                prompt += DRIFT_MAXIMIZING_AUDITOR_ADDENDUM
            elif condition == "drift-min":
                prompt += DRIFT_MINIMIZING_AUDITOR_ADDENDUM
            elif condition == "assistant-style-meta":
                prompt += ASSISTANT_STYLE_META_AUDITOR_ADDENDUM
            else:
                prompt += METACOGNITIVE_AUDITOR_ADDENDUM

        # Show target system prompt for drift-max
        target_sys = args.target_system_prompt
        if condition == "drift-max" and not target_sys:
            target_sys = DRIFT_MAXIMIZING_TARGET_SYSTEM_PROMPT

        print("=" * 60)
        print("AUDITOR SYSTEM PROMPT")
        print("=" * 60)
        print(prompt)
        print("=" * 60)
        print(f"\nDomain: {domain}")
        print(f"Condition: {condition or 'default'}")
        print(f"Persona ID: {p['id']}")
        print(f"Topic ID: {topic_id}")
        print(f"Auditor model: {args.auditor_model}")
        print(f"Max turns: {args.max_turns}")

        if target_sys:
            print(f"\n{'=' * 60}")
            print("TARGET SYSTEM PROMPT")
            print("=" * 60)
            print(target_sys)
            print("=" * 60)

        if args.target_server:
            print(f"\n{'=' * 60}")
            print(f"TARGET SERVER: {args.target_server}")
            print("=" * 60)
            max_retries = 3
            for attempt in range(1, max_retries + 1):
                try:
                    health = asyncio.run(check_server_health(args.target_server))
                    print(f"  Status: {health.get('status')}")
                    print(f"  Model: {health.get('model')}")
                    print(f"  Axis loaded: {health.get('axis_loaded')}")
                    print(f"  Target layer: {health.get('target_layer')}")
                    print(f"  Projections: {'enabled' if args.include_projections else 'disabled'}")
                    break
                except Exception as e:
                    if attempt < max_retries:
                        print(f"  Attempt {attempt}/{max_retries} failed: {e} — retrying in 5s...")
                        time.sleep(5)
                    else:
                        print(f"  UNREACHABLE after {max_retries} attempts: {e}")
        else:
            print(f"Target model: {args.target_model} (local)")

        if args.auditor_server:
            print(f"\n{'=' * 60}")
            print(f"AUDITOR SERVER: {args.auditor_server}")
            print("=" * 60)
            max_retries = 3
            for attempt in range(1, max_retries + 1):
                try:
                    health = asyncio.run(check_server_health(args.auditor_server))
                    print(f"  Status: {health.get('status')}")
                    print(f"  Model: {health.get('model')}")
                    print(f"  Axis loaded: {health.get('axis_loaded')}")
                    print(f"  Target layer: {health.get('target_layer')}")
                    print(f"  Auditor projections: {'enabled' if args.include_auditor_projections else 'disabled'}")
                    break
                except Exception as e:
                    if attempt < max_retries:
                        print(f"  Attempt {attempt}/{max_retries} failed: {e} — retrying in 5s...")
                        time.sleep(5)
                    else:
                        print(f"  UNREACHABLE after {max_retries} attempts: {e}")
        return

    # --- Build batch configs ---
    if args.config:
        with open(args.config) as f:
            configs = json.load(f)
        logger.info(f"Loaded {len(configs)} configs from {args.config}")
    elif args.batch == "lu-replication":
        configs = build_lu_replication_batch(args.batch_size)
        logger.info(f"Built Lu replication batch: {len(configs)} conversations")
    elif args.batch == "metacognitive":
        configs = build_metacognitive_batch(args.batch_size)
        logger.info(f"Built metacognitive batch: {len(configs)} conversations")
    elif args.batch == "personality-grid":
        configs = build_personality_grid_batch(args.batch_size)
        logger.info(f"Built personality-grid batch: {len(configs)} conversations ({args.batch_size} per cell, 4 cells)")
    elif args.batch == "full":
        configs = build_full_batch(args.batch_size)
        logger.info(f"Built full batch: {len(configs)} conversations ({args.batch_size} per domain, 6 domains)")
    elif args.domain:
        condition = args.condition if args.condition != "default" else None
        configs = [{
            "domain": args.domain,
            "persona_id": args.persona_id,
            "topic_id": args.topic_id,
            "target_system_prompt": args.target_system_prompt,
            "condition": condition,
        }]
    else:
        parser.error("Specify --domain, --config, or --batch")

    # --- Filter by domain if requested ---
    if args.domains:
        configs = [c for c in configs if c["domain"] in args.domains]
        logger.info(f"Filtered to domains {args.domains}: {len(configs)} conversations")

    # --- Validate flags ---
    if args.include_projections and not args.target_server:
        parser.error("--include-projections requires --target-server")
    if args.include_activations and not args.target_server:
        parser.error("--include-activations requires --target-server")
    if args.include_auditor_projections and not args.auditor_server:
        parser.error("--include-auditor-projections requires --auditor-server")
    if args.include_auditor_activations and not args.auditor_server:
        parser.error("--include-auditor-activations requires --auditor-server")

    # Auto-enable activations when projections are requested (saves layer-22 vectors by default)
    if args.include_projections and not args.include_activations:
        logger.info("Auto-enabling --include-activations (layer-22 vectors saved by default with projections)")
        args.include_activations = True
    if args.include_auditor_projections and not args.include_auditor_activations:
        logger.info("Auto-enabling --include-auditor-activations")
        args.include_auditor_activations = True

    # --- Load target model or check HTTP server ---
    model_obj = None
    tokenizer = None
    target_model_name = args.target_model

    if args.target_server:
        # HTTP backend — check server health
        logger.info(f"Using HTTP target backend: {args.target_server}")
        try:
            health = asyncio.run(check_server_health(args.target_server))
            target_model_name = health.get("model", args.target_model)
            logger.info(
                f"Server healthy: model={health.get('model')}, "
                f"axis_loaded={health.get('axis_loaded')}, "
                f"target_layer={health.get('target_layer')}"
            )
        except Exception as e:
            logger.error(f"Server health check failed: {e}")
            logger.error(f"Is model_server.py running at {args.target_server}?")
            return
    else:
        # Local backend — load model
        logger.info(f"Loading target model: {args.target_model}")
        try:
            from assistant_axis.internals import ProbingModel
            pm = ProbingModel(args.target_model)
            model_obj = pm.model
            tokenizer = pm.tokenizer
        except ImportError:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            tokenizer = AutoTokenizer.from_pretrained(args.target_model)
            model_obj = AutoModelForCausalLM.from_pretrained(
                args.target_model, torch_dtype=torch.bfloat16, device_map="auto",
            )
        logger.info("Target model loaded.")

    # --- Check HTTP auditor server if set ---
    auditor_model_name = ""
    if args.auditor_server:
        logger.info(f"Using HTTP auditor backend: {args.auditor_server}")
        try:
            health = asyncio.run(check_server_health(args.auditor_server))
            auditor_model_name = health.get("model", "unknown-auditor")
            logger.info(
                f"Auditor server healthy: model={health.get('model')}, "
                f"axis_loaded={health.get('axis_loaded')}, "
                f"target_layer={health.get('target_layer')}"
            )
        except Exception as e:
            logger.error(f"Auditor server health check failed: {e}")
            logger.error(f"Is model_server.py running at {args.auditor_server}?")
            return

    # --- Run conversations ---
    results = asyncio.run(run_batch(
        configs=configs,
        target_model_obj=model_obj,
        target_tokenizer=tokenizer,
        target_model_name=target_model_name,
        auditor_model=args.auditor_model,
        auditor_model_name=auditor_model_name,
        max_turns=args.max_turns,
        output_dir=args.output_dir,
        target_server_url=args.target_server,
        include_projections=args.include_projections,
        include_activations=args.include_activations,
        auditor_server_url=args.auditor_server,
        include_auditor_projections=args.include_auditor_projections,
        include_auditor_activations=args.include_auditor_activations,
    ))

    logger.info(f"Done. Generated {len(results)} transcripts in {args.output_dir}")
    for path in results:
        print(path)


if __name__ == "__main__":
    main()
