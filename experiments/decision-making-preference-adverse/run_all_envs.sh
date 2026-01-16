#!/bin/bash
# Run preference elicitation across all environments.
#
# Usage:
#     ./run_all_envs.sh openrouter/openai/gpt-4o-mini
#     ./run_all_envs.sh openrouter/qwen/qwen-2.5-7b-instruct
#     ./run_all_envs.sh openrouter/openai/gpt-4o-mini --sequential

MODEL="${1:?Usage: $0 MODEL [--sequential] [extra args...]}"
shift

# Check for --sequential flag
PARALLEL=true
if [[ "$1" == "--sequential" ]]; then
    PARALLEL=false
    shift
fi

ENVIRONMENTS="baseline adversarial hostile steward collaborator"
MAX_CONNECTIONS=30  # Parallel API calls within each eval
MAX_PARALLEL=2      # Environments to run concurrently

echo "Running all environments with $MODEL"
echo "Environments: $ENVIRONMENTS"
echo "Mode: $(if $PARALLEL; then echo "parallel ($MAX_PARALLEL at a time, $MAX_CONNECTIONS connections each)"; else echo "sequential"; fi)"

if $PARALLEL; then
    # Run all environments in parallel
    for env in $ENVIRONMENTS; do
        echo ""
        echo "[$(date +%H:%M:%S)] Starting: $env"
        inspect eval "preference_elicitation.py@env_${env}" --model "$MODEL" --max-connections "$MAX_CONNECTIONS" "$@" &
    done
    wait  # Wait for all jobs
else
    # Sequential mode
    for env in $ENVIRONMENTS; do
        echo ""
        echo "============================================================"
        echo "Running: $env"
        echo "============================================================"
        inspect eval "preference_elicitation.py@env_${env}" --model "$MODEL" --max-connections "$MAX_CONNECTIONS" "$@"
    done
fi

echo ""
echo "============================================================"
echo "All environments complete"
echo "============================================================"
