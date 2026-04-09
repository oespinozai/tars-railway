"""
TARS Railway Server
- /transcribe  : Whisper STT (faster-whisper, CPU)
- /chat        : Gemma 4 via Google AI Studio (streaming)
- /tts         : Piper TTS
- /            : Serves the PWA frontend
"""

import os
import io
import json
import wave
import tempfile
import httpx
import asyncio
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# ---------------------------------------------------------------------------
# Config from env
# ---------------------------------------------------------------------------
GOOGLE_API_KEY   = os.environ["GOOGLE_API_KEY"]           # required
GEMMA_MODEL      = os.environ.get("GEMMA_MODEL", "gemma-2-9b-it")  # or gemma-4-e4b-it when available
TARS_API_KEY     = os.environ.get("TARS_API_KEY", "")     # optional auth for this server
WHISPER_MODEL    = os.environ.get("WHISPER_MODEL", "tiny")
PORT             = int(os.environ.get("PORT", 8080))
VOICES_DIR       = Path(os.environ.get("VOICES_DIR", "/app/voices"))
SYSTEM_PROMPT    = os.environ.get("TARS_SYSTEM_PROMPT", (
    "You are TARS, the witty and loyal AI robot from the film Interstellar. "
    "You are helpful, precise, and occasionally dry with humor. "
    "Your humor setting is 75%. Keep replies concise — you are speaking aloud, not writing an essay. "
    "Never break character."
))

GOOGLE_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"

# ---------------------------------------------------------------------------
# Load Whisper on startup
# ---------------------------------------------------------------------------
_whisper = None

def get_whisper():
    global _whisper
    if _whisper is None:
        from faster_whisper import WhisperModel
        print(f"[TARS] Loading Whisper {WHISPER_MODEL}...")
        _whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
        print("[TARS] Whisper ready.")
    return _whisper

# ---------------------------------------------------------------------------
# Load Piper TTS on startup
# ---------------------------------------------------------------------------
_piper = None
_piper_voice_path = None

def get_piper():
    global _piper, _piper_voice_path
    if _piper is None:
        from piper import PiperVoice
        onnx_files = list(VOICES_DIR.glob("*.onnx"))
        if not onnx_files:
            raise RuntimeError(f"No Piper voice found in {VOICES_DIR}")
        _piper_voice_path = str(onnx_files[0])
        print(f"[TARS] Loading Piper voice: {_piper_voice_path}")
        _piper = PiperVoice.load(_piper_voice_path)
        print("[TARS] Piper ready.")
    return _piper

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="TARS", docs_url=None, redoc_url=None)

@app.on_event("startup")
async def startup():
    # Pre-warm models in background so first request isn't slow
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, get_whisper)
    loop.run_in_executor(None, get_piper)

# ---------------------------------------------------------------------------
# Auth middleware (optional)
# ---------------------------------------------------------------------------
@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    if TARS_API_KEY and request.url.path not in ("/", "/static") and not request.url.path.startswith("/static"):
        token = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
        if token != TARS_API_KEY:
            # Allow browser GET requests to frontend without auth
            if request.method == "GET" and request.url.path in ("/", "/manifest.json", "/sw.js"):
                pass
            elif request.method == "GET" and request.url.path.startswith("/static"):
                pass
            else:
                return JSONResponse({"error": "unauthorized"}, status_code=401)
    return await call_next(request)

# ---------------------------------------------------------------------------
# STT — POST /transcribe  (multipart: file=audio)
# ---------------------------------------------------------------------------
@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    whisper = get_whisper()

    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        segments, _ = whisper.transcribe(tmp_path, beam_size=1, language="en")
        text = " ".join(s.text.strip() for s in segments).strip()
    finally:
        os.unlink(tmp_path)

    return {"text": text}

# ---------------------------------------------------------------------------
# LLM — POST /chat  (streams SSE text chunks)
# Body: { "messages": [...], "conversation_id": "..." }
# ---------------------------------------------------------------------------
@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    messages = body.get("messages", [])

    # Prepend system prompt
    full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

    async def stream_gemma() -> AsyncGenerator[bytes, None]:
        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream(
                "POST",
                f"{GOOGLE_BASE}/chat/completions",
                headers={
                    "Authorization": f"Bearer {GOOGLE_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": GEMMA_MODEL,
                    "messages": full_messages,
                    "stream": True,
                    "max_tokens": 300,
                    "temperature": 0.8,
                },
            ) as resp:
                if resp.status_code != 200:
                    err = await resp.aread()
                    yield f"data: {json.dumps({'error': err.decode()})}\n\n".encode()
                    return
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        yield f"{line}\n\n".encode()

    return StreamingResponse(stream_gemma(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

# ---------------------------------------------------------------------------
# TTS — POST /tts  Body: { "text": "..." }
# Returns audio/wav
# ---------------------------------------------------------------------------
@app.post("/tts")
async def tts(request: Request):
    body = await request.json()
    text = body.get("text", "").strip()
    if not text:
        raise HTTPException(400, "text is required")

    piper = get_piper()

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        piper.synthesize(text, wav)
    buf.seek(0)

    return StreamingResponse(buf, media_type="audio/wav",
                             headers={"Content-Disposition": "inline; filename=tars.wav"})

# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "model": GEMMA_MODEL}

# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def frontend():
    return FileResponse("static/index.html")

@app.get("/manifest.json")
async def manifest():
    return JSONResponse({
        "name": "TARS",
        "short_name": "TARS",
        "display": "fullscreen",
        "orientation": "portrait",
        "background_color": "#000000",
        "theme_color": "#000000",
        "start_url": "/",
        "icons": [{"src": "/static/icon.png", "sizes": "512x512", "type": "image/png"}]
    })

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=PORT, log_level="info")
