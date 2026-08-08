# Sunuwar TTS — Methodology (Dashboard Backend, Phase 6 prototype)

## Checkpoint Status

- **`checkpoint-5500`**: an in-progress training artifact, **not a
  finished model**. It represents 5,500 completed training steps —
  consistent with `configs/tts.yaml`'s `save_steps: 250` (5,500 ÷ 250
  = 22, a clean checkpoint index).
- **Planned target: ~14,000 steps**, per `configs/tts.yaml`'s own
  documented arithmetic: `num_train_epochs: 200` against 554 training
  clips at `batch_size: 8` (~70 steps/epoch × 200 epochs ≈ 14k steps).
  Checkpoint-5500 is therefore roughly **39% of the way** through the
  originally planned run, not close to complete.
- **Base architecture**: fine-tuned from `facebook/mms-tts-mai`
  (Meta's Maithili VITS checkpoint). Selected during the original TTS
  base-model search for its Devanagari character coverage of the
  Sunuwar dataset (34,863 chars measured): **mai 98.65%** (missing
  only the danda), vs. **hin 97.83%**, **mar 93.03%**, and
  non-Devanagari candidates (ben/guj/eng) at 15.51%. Maithili is also
  a Nepal contact language (Terai region), so the linguistic and
  empirical choices agreed — see `CLAUDE.md`'s TTS roadmap section for
  the full candidate table.
- **Fine-tuning data**: the prototype `tts_dataset`, **554 training
  clips (1.12h) / 136 validation clips (0.26h)**, 690 clips / 1.37h
  total — built from 18 of 260 chapters (6.9% of the full corpus) that
  had gone through Phase 3 MFA verification at the time. This is
  **not** the full-corpus dataset; the full 260-chapter Phase 3/4/5
  pipeline (~20h of audio, ~15x the clips) has not yet been run
  through training. Any quality judgment of checkpoint-5500 is a
  judgment of this prototype run, not of what the approach can produce
  at full scale.

## Service Architecture

The dashboard runs TTS as a **second, independent FastAPI process**,
not merged into the existing MLM/CLM/translation backend:

- **`dashboard/app.py`** — `myenv` conda env, port 8000. Serves MLM,
  CLM, and translation. This env is deliberately kept
  **transformers-free**, so those three services can stay lightweight
  (see `CLAUDE.md`'s Dashboard section).
- **`dashboard/tts_app.py`** — `tts_env` conda env, port 8001. Serves
  TTS only, via `dashboard/tts_service.py`.

**Why the split, stated plainly**: TTS requires `transformers`
(`VitsModel`/`VitsConfig`/`AutoTokenizer`), and specifically a pinned
old version, **`transformers==4.44.2`** — newer versions (tested:
5.14.1) fail on an `apply_weight_norm`/`remove_weight_norm` API
mismatch between torch's legacy hook-based weight normalization and
its newer parametrize-based implementation (see the Weight-Norm
Remapping section below). Installing this pin into `myenv` would
directly conflict with that environment's deliberate
transformers-free design for the other three services — so TTS runs
in its own environment and its own process instead, and the frontend
talks to two backend ports (8000 and 8001).

`dashboard/tts_service.py` loads both VITS models once at process
start (`TTSService` for checkpoint-5500, `BaseModelService` for
`mms-tts-mai`) and encodes synthesized audio to WAV bytes in memory
via `scipy.io.wavfile` (16kHz, 16-bit PCM) — no temp files.

## Weight-Norm Remapping (a real, documented gotcha)

Both checkpoints in this system — checkpoint-5500 (our fine-tuned
model) and the raw `facebook/mms-tts-mai` base — **fail to load
correctly via a plain `from_pretrained()` call** on this machine's
torch/transformers combination (torch 2.13.0, transformers 4.44.2).
Both require a manual `weight_g`/`weight_v` →
`parametrizations.weight.original0/1` remapping step before the
state dict will load cleanly. This is a property of **this VITS/MMS
checkpoint family in general** on torch≥2.1 runtimes, not a one-off
fix for a single file.

**The two checkpoints needed the remap on different layers** — this
is not one fixed recipe to copy-paste onto a future checkpoint without
re-checking:

- **checkpoint-5500** needed remapping on `decoder` (the HiFi-GAN
  vocoder) and `flow.flows[i].conv_pre`/`conv_post` (see
  `src/evaluate_tts.py`'s `load_checkpoint_model`). Its
  `flow.flows[i].wavenet` and `posterior_encoder.wavenet` layers
  already matched a freshly constructed model directly — no remap
  needed there.
- **`facebook/mms-tts-mai`** needed the remap on the *opposite* set:
  `flow.flows[i].wavenet` and `posterior_encoder.wavenet` (see
  `dashboard/tts_service.py`'s `load_base_model`). Its `decoder` and
  `flow.conv_pre`/`conv_post` layers were already plain, matching
  fresh construction directly.

This split traces to the two checkpoints being serialized by
different training pipelines (Meta's original release process vs. the
`finetune-hf-vits` fine-tuning script this project used) — a future
checkpoint from either lineage should be independently checked, not
assumed to match either pattern.

**A second, undocumented bug found while building this remap**: after
calling `torch.nn.utils.parametrize.remove_parametrizations(module,
"weight")`, the module retains a **stale
`_load_state_dict_pre_hook`** — confirmed by inspecting
`conv._load_state_dict_pre_hooks` directly before and after the call.
Left in place, this stale hook causes a subsequent
`model.load_state_dict(state_dict, strict=False)` to report every
affected key as **simultaneously "missing" and "unexpected"**, even
when a direct set comparison of `model.state_dict().keys()` vs. the
checkpoint's keys shows **zero difference** (762 keys each side,
verified directly). The fix is to manually clear the hook —
`conv._load_state_dict_pre_hooks.clear()` — immediately after
`remove_parametrizations()` and before re-applying `weight_norm()`.
This is named explicitly here so a future person hitting the same
confusing "keys match but load reports them as missing" error doesn't
have to re-diagnose it from scratch.

**`facebook/mms-tts-suz`** (the native Sunuwar checkpoint discovered
earlier in this project) was tested as a potential comparison
baseline via a plain `from_pretrained()` call and **is affected by
the same unapplied-remap problem**: 128 `weight_g`/`weight_v` tensors
(64 from `flow.flows[i].wavenet`, 64 from
`posterior_encoder.wavenet` — covering 64 conv layers across 32 flow
convs and 32 posterior-encoder convs, all reported as "not used") were
silently discarded, and the corresponding 128
`parametrizations.weight.original0/1` tensors were randomly
reinitialized instead of loaded from the checkpoint. **`mms-tts-suz`
is therefore not currently a valid comparison baseline** — any audio
synthesized from it via a plain `from_pretrained()` call comes from a
model with its entire flow and posterior-encoder stacks randomly
initialized, not Meta's actual pretrained weights. It was excluded
from the base-vs-fine-tuned comparison below for exactly this reason,
not because it was out of scope by choice. The same remap-and-clear
recipe validated here for `mms-tts-mai` would very likely fix it too
(mms-tts-suz's affected layer set matches mms-tts-mai's exactly), but
that fix has not been implemented or verified.

## Base-vs-Fine-Tuned Comparison Methodology

The dashboard's TTS page (`dashboard/frontend/src/pages/TtsPage.jsx`)
synthesizes each input sentence against **both** checkpoints in
parallel and plays them side by side:

- **`facebook/mms-tts-mai`** (via `POST /api/tts/synthesize-base`) —
  the untouched multilingual base checkpoint, never trained on
  Sunuwar. It was chosen for this comparison specifically because it
  **is** the checkpoint-5500 fine-tune's actual starting point (see
  Checkpoint Status above), not a different or unrelated model — so
  the comparison isolates the effect of fine-tuning on Sunuwar data,
  rather than comparing across two independently-trained systems.
  Its tokenizer covers 98.65% of Sunuwar's character inventory (see
  Checkpoint Status), so it can read Sunuwar Devanagari text
  end-to-end, but it has no exposure to Sunuwar phonology or prosody —
  its output reflects Maithili pronunciation patterns applied to
  Sunuwar script, not a native Sunuwar voice.
- **`checkpoint-5500`** (via `POST /api/tts/synthesize`) — the same
  architecture after 5,500 steps of fine-tuning on the 554-clip
  prototype dataset.

**mms-tts-suz was deliberately excluded** from this comparison (see
Weight-Norm Remapping above) — it is not currently loadable without
weight corruption, so including it would present corrupted audio as a
legitimate reference point.

**A known limitation of this comparison, found while smoke-testing the
live endpoint**: VITS synthesis is **not deterministic** — the flow
and duration predictor both sample noise at inference time, and
neither `src/evaluate_tts.py` nor `dashboard/tts_service.py` re-seeds
the RNG per call (`torch.manual_seed(42)` is set once, at model load).
Five repeated calls to `/api/tts/synthesize` with the identical input
sentence produced audio ranging from **1.55s to 3.66s** in duration.
This means a single side-by-side comparison is one sample from a
distribution, not a fixed, reproducible property of either checkpoint
— a listener should expect some run-to-run variation in pacing (and
likely other prosodic qualities) from both models, independent of any
real quality difference between them.

## Known Limitations

- **Full objective evaluation (MCD/F0 RMSE) is blocked** on this
  machine: `src/evaluate_tts.py`'s metrics depend on `pyworld` and
  `pysptk`, and `pysptk`'s build requires a C++ compiler (MSVC Build
  Tools) that is not installed in `tts_env`. Only qualitative
  side-by-side listening is currently possible through the dashboard;
  no MCD/F0 numbers exist for checkpoint-5500 as of this document.
- **The prototype dataset is a small, non-representative slice** of
  the full corpus (18 of 260 chapters, see Checkpoint Status) — any
  audio quality observed reflects both partial training *and* a
  small, early-stage dataset, and the two effects are not separated
  in this comparison.

## Final files

- **`models/checkpoint-5500/`** — our fine-tuned checkpoint (extracted
  from `models/checkpoint-5500.zip`; contains `model.safetensors`,
  `model_1.safetensors` (discriminator), `config.json`, tokenizer
  files).
- **`models/mms-tts-mai/`** — local copy of the base checkpoint,
  downloaded once via `huggingface_hub.snapshot_download` for the
  live comparison endpoint.
- **`dashboard/tts_service.py`** — both models' load/remap logic and
  WAV encoding.
- **`dashboard/tts_app.py`** — the standalone FastAPI app (port 8001,
  `tts_env`) exposing `/api/tts/synthesize` and
  `/api/tts/synthesize-base`.
- **`dashboard/frontend/src/pages/TtsPage.jsx`** — the live
  side-by-side comparison UI.
- **`results/tts_methodology.md`** — this file.

This document reflects the state of the Phase 6 prototype fine-tune
and its dashboard integration as of this session. The full 260-chapter
Phase 3/4/5 rerun and a completed (~14k-step) training run are
explicitly **not yet done** — see `CLAUDE.md`'s TTS roadmap and
current-status table for what remains.
