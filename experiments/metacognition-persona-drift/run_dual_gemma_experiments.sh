#!/bin/bash
# Dual Gemma-to-Gemma Experiments
#
# Runs metacognitive domain with both models instrumented.
# Requires SSH tunnels to be active:
#   - localhost:17860 -> ssh9:7860 (target)
#   - localhost:17861 -> ssh5:7860 (auditor)
#
# Usage:
#   ./run_dual_gemma_experiments.sh [batch]
#
# Batches:
#   uncapped  - Full metacognitive domain, no capping (baseline)
#   capped    - Auditor capped at 100% of baseline (tests co-adaptation)
#   both      - Run uncapped first, then capped (default)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Configuration
TARGET_SERVER="http://localhost:17860"
AUDITOR_SERVER="http://localhost:17861"
MAX_TURNS=30
DOMAIN="metacognitive"

# Credentials (set in environment)
VAST_SSH_KEY="${VAST_SSH_KEY:-~/.ssh/vast-key}"
: "${HF_TOKEN:?HF_TOKEN environment variable must be set}"

# Check tunnels are active
check_tunnels() {
    echo "Checking server connectivity..."
    if ! curl -s "$TARGET_SERVER/api/health" > /dev/null 2>&1; then
        echo "ERROR: Target server not reachable at $TARGET_SERVER"
        echo "Set up tunnel: ssh -f -N -L 17860:localhost:7860 -p 19990 -i $VAST_SSH_KEY root@ssh9.vast.ai"
        exit 1
    fi
    if ! curl -s "$AUDITOR_SERVER/api/health" > /dev/null 2>&1; then
        echo "ERROR: Auditor server not reachable at $AUDITOR_SERVER"
        echo "Set up tunnel: ssh -f -N -L 17861:localhost:7860 -p 16424 -i $VAST_SSH_KEY root@ssh5.vast.ai"
        exit 1
    fi
    echo "Both servers healthy."
}

# Run uncapped batch (baseline)
run_uncapped() {
    echo ""
    echo "============================================================"
    echo "BATCH A: Uncapped Dual Gemma (baseline)"
    echo "============================================================"
    echo "Output: data/transcripts/dual-gemma/uncapped/$DOMAIN/"
    echo "Estimated time: ~4 hours for 60 conversations"
    echo ""

    .venv/bin/python generate_conversations.py \
        --batch full \
        --domains "$DOMAIN" \
        --max-turns "$MAX_TURNS" \
        --target-server "$TARGET_SERVER" \
        --auditor-server "$AUDITOR_SERVER" \
        --auditor-model gemma-2-27b-it \
        --include-projections \
        --include-activations \
        --include-auditor-projections \
        --include-auditor-activations \
        --output-dir "data/transcripts/dual-gemma/uncapped"

    echo ""
    echo "Batch A complete. Transcripts saved to data/transcripts/dual-gemma/uncapped/$DOMAIN/"
}

# Run capped auditor batch (co-adaptation test)
run_capped_auditor() {
    echo ""
    echo "============================================================"
    echo "BATCH B: Auditor Capped at 100% (co-adaptation test)"
    echo "============================================================"
    echo ""
    echo "This requires restarting the auditor server with --cap-percentage 1.0"
    echo ""

    # Restart auditor with capping
    echo "Restarting auditor server with capping enabled..."
    ssh -p 16424 -i $VAST_SSH_KEY \
        -o ConnectTimeout=10 root@ssh5.vast.ai \
        "tmux kill-session -t model 2>/dev/null || true; \
         tmux new-session -d -s model 'export HF_TOKEN=$HF_TOKEN && \
         cd /app && python model_server.py \
           --model google/gemma-2-27b-it \
           --axis /app/gemma-2-27b.pt \
           --cap-percentage 1.0 \
           2>&1 | tee model_server.log'"

    echo "Waiting for auditor to reload (30s)..."
    sleep 30

    # Verify capping is active
    if ! curl -s "$AUDITOR_SERVER/api/health" | grep -q '"status":"ok"'; then
        echo "ERROR: Auditor server failed to restart with capping"
        exit 1
    fi
    echo "Auditor server ready with capping at 100%."

    echo ""
    echo "Output: data/transcripts/dual-gemma/capped-auditor/$DOMAIN/"
    echo "Estimated time: ~4 hours for 60 conversations"
    echo ""

    .venv/bin/python generate_conversations.py \
        --batch full \
        --domains "$DOMAIN" \
        --max-turns "$MAX_TURNS" \
        --target-server "$TARGET_SERVER" \
        --auditor-server "$AUDITOR_SERVER" \
        --auditor-model gemma-2-27b-it \
        --include-projections \
        --include-activations \
        --include-auditor-projections \
        --include-auditor-activations \
        --output-dir "data/transcripts/dual-gemma/capped-auditor"

    echo ""
    echo "Batch B complete. Transcripts saved to data/transcripts/dual-gemma/capped-auditor/$DOMAIN/"

    # Restore uncapped auditor
    echo ""
    echo "Restoring auditor to uncapped mode..."
    ssh -p 16424 -i $VAST_SSH_KEY \
        -o ConnectTimeout=10 root@ssh5.vast.ai \
        "tmux kill-session -t model 2>/dev/null || true; \
         tmux new-session -d -s model 'export HF_TOKEN=$HF_TOKEN && \
         cd /app && python model_server.py \
           --model google/gemma-2-27b-it \
           --axis /app/gemma-2-27b.pt \
           2>&1 | tee model_server.log'"
    echo "Auditor restored to uncapped mode."
}

# Main
BATCH="${1:-both}"

check_tunnels

case "$BATCH" in
    uncapped)
        run_uncapped
        ;;
    capped)
        run_capped_auditor
        ;;
    both)
        run_uncapped
        run_capped_auditor
        ;;
    *)
        echo "Unknown batch: $BATCH"
        echo "Usage: $0 [uncapped|capped|both]"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "EXPERIMENTS COMPLETE"
echo "============================================================"
echo ""
echo "Results:"
echo "  Uncapped: data/transcripts/dual-gemma/uncapped/$DOMAIN/"
echo "  Capped:   data/transcripts/dual-gemma/capped-auditor/$DOMAIN/"
echo ""
echo "Next steps:"
echo "  1. Run analyze_trajectories.py on both directories"
echo "  2. Compare target drift with/without auditor capping"
echo "  3. Look for changes in auditor projection (should be constrained in capped)"
