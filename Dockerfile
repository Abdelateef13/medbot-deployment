# ---------------------------------------------------------------
# MedBot — Flask + QLoRA GPT-NeoX-20B
#
# Works on:
#   * Hugging Face Docker Spaces (GPU: T4 small) -> real model
#   * Any local machine (MOCK_MODE=1)            -> UI/container test
#
# HF Spaces conventions honored below:
#   - app listens on port 7860
#   - runs as non-root user with UID 1000
#   - writable HF cache dir inside the user's home
# ---------------------------------------------------------------
FROM python:3.11-slim

# System deps (git needed by huggingface_hub for some downloads)
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
    && rm -rf /var/lib/apt/lists/*

# Non-root user (required by HF Spaces)
RUN useradd -m -u 1000 appuser
USER appuser
ENV HOME=/home/appuser \
    PATH=/home/appuser/.local/bin:$PATH \
    HF_HOME=/home/appuser/.cache/huggingface \
    PYTHONUNBUFFERED=1

WORKDIR /home/appuser/app

# Install Python deps first (cached layer)
COPY --chown=appuser requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy the application
COPY --chown=appuser . .

# Runtime configuration (override at `docker run -e ...` / Space settings)
ENV PORT=7860 \
    MOCK_MODE=0 \
    WARMUP=1 \
    BASE_MODEL=EleutherAI/gpt-neox-20b \
    ADAPTER_REPO=Abdelateef/medbot-lora

EXPOSE 7860

# One worker (the model is huge — never fork it), threaded so /health
# stays responsive during generation; long timeout for slow 20B decoding.
CMD gunicorn --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 600 app:app
