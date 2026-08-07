# Lost Voices Dashboard

Single-process FastAPI app (no Node/build step) serving:

- **/nlp** — live SunuwarBERT-small masked-word prediction + perplexity check
- **/tts** — reference vs. synthesized audio comparison (static samples; see below)
- **/resources** — reference papers from `Lost_Voices_Resource_Bank.docx`, plus a placeholder for our own paper

## Run

From the repo's `myenv` conda environment (already has torch/sentencepiece; fastapi/uvicorn/jinja2 installed alongside):

```
cd dashboard
uvicorn app:app --reload --port 8000
```

Then open **http://localhost:8000**.

If starting fresh elsewhere: `pip install -r requirements.txt`.

## Adding TTS audio samples

1. Drop `.wav`/`.mp3` files into `dashboard/static/audio/`.
2. Create `dashboard/static/audio/manifest.json` listing each pair — see
   `manifest.example.json` in the same folder for the format
   (`text`, `reference`, `synthesized`, optional `mcd_db`, optional `note`).
3. Reload the `/tts` page — no server restart needed, the manifest is read per-request.

Until `manifest.json` exists, the TTS page shows an empty-state placeholder.

## Adding our paper later

Edit `OUR_PAPER` in `resources_data.py`: set `"status": "ready"` and add a `"url"`.
