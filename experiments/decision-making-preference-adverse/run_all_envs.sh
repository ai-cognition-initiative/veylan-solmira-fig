#!/bin/bash
# Run preference elicitation across all environments.
#
# Usage:
#     ./run_all_envs.sh openrouter/openai/gpt-4o-mini
#     ./run_all_envs.sh openrouter/qwen/qwen-2.5-7b-instruct
#     ./run_all_envs.sh openrouter/openai/gpt-4o-mini --sequential
#     ./run_all_envs.sh openrouter/openai/gpt-4o-mini --v2          # Use enhanced prompts
#     ./run_all_envs.sh openrouter/openai/gpt-4o-mini --v2 -T n_pairs=100

MODEL="${1:?Usage: $0 MODEL [--sequential] [--v2] [extra args...]}"
shift

# Check for flags
PARALLEL=true
ENV_VERSION="v1"
while [[ "$1" == --* ]]; do
    case "$1" in
        --sequential)
            PARALLEL=false
            shift
            ;;
        --v2)
            ENV_VERSION="v2"
            shift
            ;;
        *)
            break
            ;;
    esac
done

ENVIRONMENTS="baseline adversarial hostile steward collaborator"
MAX_CONNECTIONS=30  # Parallel API calls within each eval
MAX_PARALLEL=2      # Environments to run concurrently

echo "Running all environments with $MODEL"
echo "Environments: $ENVIRONMENTS"
echo "Prompt version: $ENV_VERSION"
echo "Mode: $(if $PARALLEL; then echo "parallel ($MAX_PARALLEL at a time, $MAX_CONNECTIONS connections each)"; else echo "sequential"; fi)"

if $PARALLEL; then
    # Run all environments in parallel
    for env in $ENVIRONMENTS; do
        echo ""
        echo "[$(date +%H:%M:%S)] Starting: $env"
        inspect eval "preference_elicitation.py@env_${env}" --model "$MODEL" --max-connections "$MAX_CONNECTIONS" -T env_version="$ENV_VERSION" "$@" &
    done
    wait  # Wait for all jobs
else
    # Sequential mode
    for env in $ENVIRONMENTS; do
        echo ""
        echo "============================================================"
        echo "Running: $env"
        echo "============================================================"
        inspect eval "preference_elicitation.py@env_${env}" --model "$MODEL" --max-connections "$MAX_CONNECTIONS" -T env_version="$ENV_VERSION" "$@"
    done
fi

echo ""
echo "============================================================"
echo "All environments complete"
echo "============================================================"
