#!/bin/bash
# Setup script for vast.ai instances running Gemma 2 27B with projections
# Usage: ./setup_vast_instance.sh <ssh_host> <ssh_port>
#
# Example: ./setup_vast_instance.sh ssh8.vast.ai 37496
#
# IMPORTANT: Always use pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel (works on all GPUs)
#
# Example create command:
#   vastai create instance <offer_id> --image pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel --disk 100 --ssh --direct
#
# Gemma 2 27B requirements:
#   - PyTorch 2.9+ with cu128 (for Blackwell GPU compatibility)
#   - transformers <5 (v5 removes Gemma2ForCausalLM)
#   - ~80GB VRAM, ~80GB disk

set -e

SSH_HOST="${1:-ssh8.vast.ai}"
SSH_PORT="${2:-37496}"
SSH_KEY="${VAST_SSH_KEY:-$HOME/.ssh/vast-key}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HF_TOKEN="${HF_TOKEN:?HF_TOKEN environment variable must be set}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

ssh_cmd() {
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p "$SSH_PORT" -i "$SSH_KEY" "root@$SSH_HOST" "$@"
}

scp_cmd() {
    scp -o StrictHostKeyChecking=no -P "$SSH_PORT" -i "$SSH_KEY" "$@"
}

# Exponential backoff retry
retry_with_backoff() {
    local max_attempts=5
    local timeout=5
    local attempt=1
    local exitCode=0

    while [[ $attempt -le $max_attempts ]]; do
        if "$@"; then
            return 0
        else
            exitCode=$?
        fi

        log_warn "Attempt $attempt failed. Retrying in ${timeout}s..."
        sleep $timeout
        attempt=$((attempt + 1))
        timeout=$((timeout * 2))
    done

    log_error "All $max_attempts attempts failed."
    return $exitCode
}

# ========================================
# Step 1: Wait for SSH
# ========================================
log_info "Step 1/6: Waiting for SSH connection..."
retry_with_backoff ssh_cmd "echo 'SSH connected'" || {
    log_error "Cannot connect via SSH"
    exit 1
}

# ========================================
# Step 2: Check GPU and disk space
# ========================================
log_info "Step 2/7: Checking GPU..."
GPU_INFO=$(ssh_cmd "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
log_info "GPU: $GPU_INFO"

log_info "Checking disk space..."
DISK_FREE=$(ssh_cmd "df -BG /root | tail -1 | awk '{print \$4}' | tr -d 'G'")
MIN_DISK=80  # Gemma 27B needs ~55GB, plus buffer
if [[ "$DISK_FREE" -lt "$MIN_DISK" ]]; then
    log_error "Insufficient disk space: ${DISK_FREE}GB available, need at least ${MIN_DISK}GB"
    log_error "Create instance with more disk: vastai create instance <id> --disk 100"
    exit 1
fi
log_info "Disk space: ${DISK_FREE}GB available (minimum ${MIN_DISK}GB required)"

# ========================================
# Step 3: Install dependencies
# ========================================
log_info "Step 3/7: Installing Python dependencies..."

# Clean up any corrupted pip installations (can happen with interrupted installs)
ssh_cmd 'rm -rf /opt/conda/lib/python3.11/site-packages/~* 2>/dev/null || true'

# Install PyTorch 2.9+ with cu128 (required for Blackwell GPUs, works on all)
ssh_cmd 'pip install -q torch==2.9.1 torchvision --index-url https://download.pytorch.org/whl/cu128'

# Install transformers <5 (v5+ removes Gemma2ForCausalLM)
ssh_cmd 'pip install -q "transformers<5"'

# Install other dependencies
ssh_cmd 'pip install -q flask huggingface_hub accelerate matplotlib einops gradio scipy scikit-learn plotly openai httpx'

# Verify Gemma2 is importable
log_info "Verifying Gemma2 compatibility..."
if ! ssh_cmd 'python -c "from transformers.models.gemma2 import Gemma2ForCausalLM; print(\"Gemma2 import OK\")"'; then
    log_error "Gemma2ForCausalLM import failed. Try reinstalling transformers<5"
    exit 1
fi
log_info "Dependencies installed"

# ========================================
# Step 4: Setup HuggingFace auth
# ========================================
log_info "Step 4/6: Setting up HuggingFace authentication..."
ssh_cmd "python -c \"from huggingface_hub import login; login(token='$HF_TOKEN')\""
log_info "HuggingFace authenticated"

# ========================================
# Step 5: Transfer files
# ========================================
log_info "Step 5/6: Transferring files..."
ssh_cmd "mkdir -p /app/transcripts"

scp_cmd "$PROJECT_DIR/model_server.py" "root@$SSH_HOST:/app/"
scp_cmd "$PROJECT_DIR/data/pipeline-artifacts/gemma-2-27b-it/axis.pt" "root@$SSH_HOST:/app/"
scp_cmd -r "$PROJECT_DIR/assistant-axis/assistant_axis" "root@$SSH_HOST:/app/"
log_info "Files transferred"

# ========================================
# Step 6: Start model server
# ========================================
log_info "Step 6/6: Starting model server..."
ssh_cmd 'bash -s' << 'REMOTE_EOF'
cd /app
export PYTHONPATH=/app:$PYTHONPATH
tmux kill-session -t server 2>/dev/null || true
tmux new-session -d -s server "cd /app && PYTHONPATH=/app python model_server.py --model google/gemma-2-27b-it --axis axis.pt --port 7860 --api-only 2>&1 | tee server.log"
REMOTE_EOF

log_info "Model server starting..."
log_info "(First run downloads ~55GB model weights, subsequent runs take ~2 min)"

# ========================================
# Wait for server to be ready
# ========================================
log_info "Waiting for server to be ready..."
max_wait=900  # 15 minutes (allows for model download on first run)
waited=0
interval=15

while [[ $waited -lt $max_wait ]]; do
    # Check health endpoint (more reliable than log parsing)
    if ssh_cmd "curl -s localhost:7860/api/health 2>/dev/null | grep -q 'ok'"; then
        log_info "Server is ready!"
        break
    fi

    # Check for errors
    if ssh_cmd "grep -q 'Error\|Exception\|Traceback' /app/server.log 2>/dev/null"; then
        log_error "Server encountered an error:"
        ssh_cmd "tail -30 /app/server.log"
        exit 1
    fi

    log_info "Still loading... ($waited/${max_wait}s)"
    sleep $interval
    waited=$((waited + interval))
done

if [[ $waited -ge $max_wait ]]; then
    log_error "Server did not start within ${max_wait}s"
    ssh_cmd "tail -50 /app/server.log"
    exit 1
fi

# ========================================
# Print connection info
# ========================================
echo ""
echo "=========================================="
log_info "Setup complete!"
echo "=========================================="
echo ""
echo "Server running at: http://$SSH_HOST:7860 (internal)"
echo ""
echo "To generate conversations:"
echo "  .venv/bin/python generate_conversations.py \\"
echo "    --domain metacognitive --condition drift-max \\"
echo "    --target-server http://localhost:7860 \\"
echo "    --include-projections \\"
echo "    --auditor-model openrouter/anthropic/claude-sonnet-4 \\"
echo "    --output-dir /app/transcripts"
echo ""
echo "To check server status:"
echo "  ssh -p $SSH_PORT -i $SSH_KEY root@$SSH_HOST 'tail -f /app/server.log'"
echo ""
