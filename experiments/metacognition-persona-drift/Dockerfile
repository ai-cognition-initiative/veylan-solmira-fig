# Assistant Axis GPU Image
#
# Supports two modes on vast.ai:
#   1. Pipeline: Lu et al. (2026) steps 1-2 (vLLM generation + activation extraction)
#   2. Model server: Gradio chat UI + FastAPI for interactive/automated conversations
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

# Install all dependencies (pipeline + model server)
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
    torch>=2.0 \
    gradio \
    matplotlib \
    fastapi \
    uvicorn \
    httpx \
    openai \
    python-dotenv

# Copy assistant-axis repo and install as package
COPY assistant-axis/ /app/assistant-axis/
RUN cd /app/assistant-axis && pip install --no-cache-dir -e .

# Copy pipeline orchestrator and model server
COPY gpu_pipeline.py /app/
COPY model_server.py /app/
COPY generate_conversations.py /app/

# Precomputed axes
COPY data/precomputed-axes/ /app/precomputed-axes/

# Session storage
RUN mkdir -p /app/explore-sessions

# Default: show usage
CMD ["python", "gpu_pipeline.py", "--help"]
