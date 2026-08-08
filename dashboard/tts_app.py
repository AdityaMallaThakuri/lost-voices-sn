"""
tts_app.py — standalone FastAPI app serving TTS synthesis only.

Environment separation, not a stylistic choice: dashboard/app.py runs in the
`myenv` conda env, which is deliberately transformers-free for the MLM/CLM/NMT
services (see CLAUDE.md, tts_service.py's docstring). TTS needs transformers,
so it cannot be imported into that process — it runs here, in its own process,
under the separate `tts_env` conda env. Run with:

    conda activate tts_env
    cd dashboard
    uvicorn tts_app:app --reload --port 8001

The frontend therefore talks to TWO backend ports: 8000 for MLM/CLM/translation
(app.py) and 8001 for TTS (this file).
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

from tts_service import BaseModelService, TTSService

app = FastAPI(title="Lost Voices TTS")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading sunuwarTTS checkpoint-5500...")
tts = TTSService()
print(f"Loaded. sample_rate={tts.sample_rate}")

print("Loading facebook/mms-tts-mai (base checkpoint, pre-fine-tuning)...")
base_tts = BaseModelService()
print(f"Loaded. sample_rate={base_tts.sample_rate}")


class TTSRequest(BaseModel):
    text: str


@app.post("/api/tts/synthesize")
def api_tts_synthesize(body: TTSRequest):
    try:
        wav_bytes = tts.synthesize_wav_bytes(body.text)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return Response(content=wav_bytes, media_type="audio/wav")


@app.post("/api/tts/synthesize-base")
def api_tts_synthesize_base(body: TTSRequest):
    try:
        wav_bytes = base_tts.synthesize_wav_bytes(body.text)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return Response(content=wav_bytes, media_type="audio/wav")
