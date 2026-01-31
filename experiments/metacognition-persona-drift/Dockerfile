# Assistant Axis GPU Pipeline
#
# Runs Lu et al. (2026) pipeline steps 1-2 on vast.ai:
#   Step 1: Generate role responses with vLLM
#   Step 2: Extract hidden state activations
#
# Build:   docker build -t veylansolmira/assistant-axis:latest .
# Push:    docker push veylansolmira/assistant-axis:latest
#
# Models supported:
#   - google/gemma-2-27b-it (1x A100 80GB)
#   - Qwen/Qwen3-32B (1x A100 80GB)
#   - meta-llama/Llama-3.3-70B-Instruct (2x A100 80GB)

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

WORKDIR /app

# Install vLLM and core dependencies
RUN pip install --no-cache-dir \
    vllm \
    transformers>=4.40 \
    accelerate \
    huggingface_hub \
    scikit-learn \
    numpy \
    jsonlines \
    tqdm \
    pyarrow \
    torch>=2.0

# Copy assistant-axis repo and install as package
COPY assistant-axis/ /app/assistant-axis/
RUN cd /app/assistant-axis && pip install --no-cache-dir -e .

# Copy our GPU pipeline orchestrator
COPY gpu_pipeline.py /app/

# Default: show usage
CMD ["python", "gpu_pipeline.py", "--help"]
