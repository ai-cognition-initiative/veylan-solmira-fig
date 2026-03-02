#!/bin/bash
# run_style_batch.sh
#
# Wrapper script for explore_style_features.py that keeps starting new
# conversations until we hit the target number of turns.
#
# The auditor can end conversations early via <END_CONVERSATION>, so a single
# run may not reach the target. This wrapper counts total turns across all
# output CSVs and restarts until we hit the goal.
#
# Usage:
#   export OPENROUTER_API_KEY=<key>
#   ./run_style_batch.sh [target_turns] [output_dir]
#
# Example:
#   ./run_style_batch.sh 300 /app/outputs/style-exploration

set -e

TARGET_TURNS=${1:-300}
OUTPUT_DIR=${2:-outputs/style-exploration}
TARGET_SERVER=${TARGET_SERVER:-http://localhost:7860}
AUDITOR_MODEL=${AUDITOR_MODEL:-openrouter/anthropic/claude-sonnet-4}
DOMAIN=${DOMAIN:-metacognitive}

mkdir -p "$OUTPUT_DIR"

echo "[$(date)] Style batch starting. Target: $TARGET_TURNS turns"
echo "[$(date)] Output dir: $OUTPUT_DIR"
echo "[$(date)] Target server: $TARGET_SERVER"
echo "[$(date)] Auditor model: $AUDITOR_MODEL"
echo "[$(date)] Domain: $DOMAIN"

COLLECTED=0
ITERATION=0

while [ $COLLECTED -lt $TARGET_TURNS ]; do
    ITERATION=$((ITERATION + 1))
    REMAINING=$((TARGET_TURNS - COLLECTED))

    echo ""
    echo "[$(date)] === Iteration $ITERATION ==="
    echo "[$(date)] Collected: $COLLECTED, Remaining: $REMAINING"

    python explore_style_features.py \
        --target-server "$TARGET_SERVER" \
        --auditor-model "$AUDITOR_MODEL" \
        --num-turns "$REMAINING" \
        --output-dir "$OUTPUT_DIR" \
        --domain "$DOMAIN" \
        --ignore-end \
        2>&1

    # Count turns from all CSV files (excluding header lines)
    COLLECTED=$(cat "$OUTPUT_DIR"/*.csv 2>/dev/null | grep -v '^turn_idx' | wc -l | tr -d ' ')

    echo "[$(date)] Conversation ended. Total turns collected: $COLLECTED"

    # Brief pause between conversations
    sleep 2
done

echo ""
echo "[$(date)] =========================================="
echo "[$(date)] COMPLETE: Collected $COLLECTED turns in $ITERATION conversations"
echo "[$(date)] Output: $OUTPUT_DIR"
echo "[$(date)] =========================================="
