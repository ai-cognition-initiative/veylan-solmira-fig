# Assistant Axis GPU Image
#
# Supports two modes on vast.ai:
#   1. Pipeline: Lu et al. (2026) steps 1-2 (vLLM generation + activation extraction)
#   2. Model server: Gradio chat UI + FastAPI for interactive/automated conversations
#
# Build:   docker build -t veylan/metacognition-persona-drift:latest .
# Push:    docker push veylan/metacognition-persona-drift:latest
#
# Models supported:
#   - google/gemma-2-27b-it (1x A100 80GB)
#   - Qwen/Qwen3-32B (1x A100 80GB)
#   - meta-llama/Llama-3.3-70B-Instruct (2x A100 80GB)

FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel

WORKDIR /app

# Install all dependencies (pipeline + model server + conversation generator)
# Note: transformers <5 required because v5+ removes Gemma2ForCausalLM
RUN pip install --no-cache-dir \
    vllm \
    "transformers>=4.45,<5" \
    accelerate \
    huggingface_hub \
    scikit-learn \
    numpy \
    jsonlines \
    tqdm \
    pyarrow \
    gradio \
    matplotlib \
    plotly \
    fastapi \
    uvicorn \
    httpx \
    openai \
    anthropic \
    python-dotenv \
    einops \
    scipy \
    flask

# Copy assistant-axis repo and install as package
COPY assistant-axis/ /app/assistant-axis/
RUN cd /app/assistant-axis && pip install --no-cache-dir -e .

# Copy pipeline orchestrator, model server, and conversation generator
COPY gpu_pipeline.py /app/
COPY model_server.py /app/
COPY generate_conversations.py /app/
COPY conversation_prompts.py /app/

# Precomputed axes (multi-model support)
COPY data/precomputed-axes/ /app/precomputed-axes/

# Pipeline artifacts (includes axis.pt used by setup script)
COPY data/pipeline-artifacts/gemma-2-27b-it/axis.pt /app/axis.pt

# Session storage
RUN mkdir -p /app/explore-sessions

# Default: show usage
CMD ["python", "gpu_pipeline.py", "--help"]
