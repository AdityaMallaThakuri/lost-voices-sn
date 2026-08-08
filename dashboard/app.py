"""
Lost Voices dashboard — FastAPI app serving the NLP demo, TTS showcase,
and Resources page from one process. Run with:

    uvicorn app:app --reload --port 8000

(run from inside the dashboard/ directory, using an environment with
torch, sentencepiece, fastapi, uvicorn, jinja2 installed — e.g. the
project's `myenv` conda env).
"""

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from model_service import MLMService, DEMO_SENTENCES
from clm_service import CLMService
from nmt_service import NMTService
from resources_data import DELIVERABLES, OUR_PAPER

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Lost Voices Dashboard")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

print("Loading SunuwarBERT-small checkpoint...")
mlm = MLMService()
print(f"Loaded. {mlm.param_count:,} parameters.")

print("Loading SunuwarCLM-small checkpoint...")
clm = CLMService()
print(f"Loaded. {clm.param_count:,} parameters.")

print("Loading sunuwarNMT-small checkpoint...")
nmt = NMTService()
print(f"Loaded. {nmt.param_count:,} parameters.")


def load_tts_manifest() -> list:
    manifest_path = BASE_DIR / "static" / "audio" / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    return []


@app.get("/")
def root():
    return RedirectResponse(url="/nlp")


@app.get("/nlp")
def nlp_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="nlp.html",
        context={
            "active": "nlp",
            "model_info": mlm.info(),
            "demo_sentences": DEMO_SENTENCES,
        },
    )


@app.get("/tts")
def tts_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="tts.html",
        context={
            "active": "tts",
            "samples": load_tts_manifest(),
        },
    )


@app.get("/resources")
def resources_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="resources.html",
        context={
            "active": "resources",
            "deliverables": DELIVERABLES,
            "our_paper": OUR_PAPER,
        },
    )


class PredictRequest(BaseModel):
    sentence: str


@app.post("/api/predict")
def api_predict(body: PredictRequest):
    try:
        return mlm.predict(body.sentence)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/perplexity")
def api_perplexity():
    return mlm.perplexity()


class CLMGenerateRequest(BaseModel):
    prompt: str
    num_new_tokens: int = 15


@app.post("/api/clm/generate")
def api_clm_generate(body: CLMGenerateRequest):
    try:
        continuation = clm.generate(body.prompt, num_new_tokens=body.num_new_tokens)
        return {"prompt": body.prompt, "continuation": continuation}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


class TranslateRequest(BaseModel):
    text: str
    direction: str


@app.post("/api/translate")
def api_translate(body: TranslateRequest):
    try:
        translation = nmt.translate(body.text, body.direction)
        return {"text": body.text, "direction": body.direction, "translation": translation}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
