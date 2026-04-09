FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps
COPY requirements.txt .
RUN pip install --no-cache-dir fastapi uvicorn[standard] python-multipart httpx faster-whisper piper-tts && \
    pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# App code + frontend
COPY server.py .
COPY static/ static/

# Download Piper voice at build time so first request is instant
RUN mkdir -p /app/voices && \
    curl -L -o /app/voices/en_US-lessac-medium.onnx \
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx" && \
    curl -L -o /app/voices/en_US-lessac-medium.onnx.json \
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"

ENV VOICES_DIR=/app/voices
ENV PORT=8080

EXPOSE 8080

CMD ["python", "server.py"]
