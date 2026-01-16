# GemmaScope SAE Environment for Mechanistic Interpretability
#
# Build:   docker build -t veylansolmira/gemma-sae:latest .
# Push:    docker push veylansolmira/gemma-sae:latest
#
# Note: Update experiment .py files and rebuild to update experiment code.
# Layer caching makes this fast (~seconds) after initial build.

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# Copy and install requirements (cached unless requirements-gpu.txt changes)
COPY requirements-gpu.txt /app/
RUN pip install --no-cache-dir -r requirements-gpu.txt

# Copy experiment code (rebuilds fast when files change)
COPY gemma_sae.py /app/
COPY steering.py /app/
COPY probe_clt.py /app/

# Default command - can be overridden at runtime
CMD ["python", "gemma_sae.py"]
