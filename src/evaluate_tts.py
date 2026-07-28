"""
evaluate_tts.py — objective TTS evaluation: MCD (Mel-Cepstral Distortion) and
F0 RMSE between synthesized speech and the real reference recording, for
every clip in the held-out validation set.

Why these two, not WER
-----------------------
Whisper cannot transcribe Sunuwar (verified empirically — see CLAUDE.md), so
a WER-based intelligibility score is not trustworthy here. MCD and F0 RMSE
are standard objective TTS metrics that only need the reference audio you
already have for every validation sentence — no ASR model involved. Real
naturalness/intelligibility judgment still needs a Sunuwar speaker (roadmap
Phase 7); these are a cheap automated proxy in the meantime, same spirit as
using MFA's self-alignment confidence.

Checkpoint loading
------------------
Mid-training checkpoints (`checkpoint-NNNN/`) are Accelerate `save_state()`
snapshots: they hold only `model.safetensors`, not a `config.json` or
tokenizer — those live once in the top-level `output_dir`. Training also
applies `torch.nn.utils.weight_norm` to the decoder and flow conv layers for
GAN stability, so a plain `from_pretrained` on the checkpoint silently
random-initialises those layers instead of erroring (same class of bug as
the base-checkpoint weight-norm remap in notebooks/finetune_tts_colab.ipynb).
`load_checkpoint_model` below re-applies weight_norm before loading, then
removes it again afterward for clean inference.

Usage (run in Colab, where torch/transformers are already installed):
    !pip install -q librosa pyworld pysptk
    python src/evaluate_tts.py configs/eval_tts.yaml
"""

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

SEED = 42


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------

def load_checkpoint_model(output_dir: str, checkpoint_dir: str):
    import torch
    from safetensors.torch import load_file
    from torch.nn.utils import weight_norm
    from transformers import AutoTokenizer, VitsConfig, VitsModel

    torch.manual_seed(SEED)

    config = VitsConfig.from_pretrained(output_dir)
    model = VitsModel(config)
    tokenizer = AutoTokenizer.from_pretrained(output_dir)

    model.decoder.apply_weight_norm()
    for flow_step in model.flow.flows:
        weight_norm(flow_step.conv_pre, name="weight")
        weight_norm(flow_step.conv_post, name="weight")

    state_dict = load_file(f"{checkpoint_dir}/model.safetensors")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    assert not missing and not unexpected, (
        f"checkpoint weights did not load cleanly — "
        f"missing={missing} unexpected={unexpected}"
    )

    from torch.nn.utils import remove_weight_norm
    model.decoder.remove_weight_norm()
    for flow_step in model.flow.flows:
        remove_weight_norm(flow_step.conv_pre)
        remove_weight_norm(flow_step.conv_post)

    model.eval()
    return model, tokenizer


def synthesize(model, tokenizer, text: str):
    import torch

    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        # waveform shape is (batch, samples, 1) — squeeze or it plays as
        # silent/broken audio in downstream tools (see memory notes).
        wav = model(**inputs).waveform[0].squeeze(-1).cpu().numpy()
    return wav


# --------------------------------------------------------------------------
# MCD (Mel-Cepstral Distortion) + F0 RMSE, DTW-aligned
# --------------------------------------------------------------------------
# Reference and synthesized audio are different lengths (the model's own
# rhythm/pacing won't exactly match the narrator's), so frames are matched
# with dynamic time warping on mel-cepstral coefficients (mgc) before
# computing either metric — the same warping path is reused for F0 so both
# numbers come from one alignment.
#
# mgc, not raw librosa MFCC: the classic MCD dB formula's constant
# (10/ln(10)*sqrt(2)) is calibrated for true mel-cepstral coefficients as
# extracted via a vocoder analysis (WORLD + SPTK's mel-cepstral conversion),
# which are small-magnitude (~0.01-1). Plain librosa MFCC (DCT of the log-mel
# spectrogram) is a different, much larger-magnitude quantity — plugging that
# into the same formula inflates the number 10-50x with no real calibration,
# confirmed empirically: two DIFFERENT real reference recordings compared
# against each other came out at ~550 "dB" with the librosa-MFCC version,
# which is impossible for genuine speech and would have been silently
# reported as if it were a real MCD. Use pyworld (F0 + spectral envelope)
# and pysptk (envelope -> mgc) instead, matching what ESPnet/ParallelWaveGAN
# eval scripts actually do.

MCEP_DIM = 24  # standard mel-cepstral order for MCD (excludes c0)


def _mcd_const():
    import numpy as np
    return 10.0 / np.log(10.0) * np.sqrt(2.0)


def extract_world_features(wav, sr: int):
    """F0 contour + WORLD spectral envelope, shared by MCD and F0 RMSE."""
    import numpy as np
    import pyworld as pw

    x = np.ascontiguousarray(wav.astype(np.float64))
    f0, time_axis = pw.dio(x, sr)
    f0 = pw.stonemask(x, f0, time_axis, sr)
    _, spectral_envelope, _ = pw.wav2world(x, sr)
    return f0, spectral_envelope


def envelope_to_mgc(spectral_envelope, sr: int):
    """WORLD spectral envelope -> mel-cepstral coefficients, c0 dropped, shape (D, T)."""
    import pysptk

    alpha = pysptk.util.mcepalpha(sr)
    mgc = pysptk.sp2mc(spectral_envelope, MCEP_DIM, alpha)
    return mgc[:, 1:].T  # drop c0 (energy), transpose to (D, T) for DTW


def dtw_align(ref_feat, syn_feat):
    """Align two (D, T) feature sequences with DTW; return matched frame index pairs, chronological order."""
    import librosa

    _, wp = librosa.sequence.dtw(X=ref_feat, Y=syn_feat, metric="euclidean")
    return wp[::-1]  # librosa returns the path end-to-start


def compute_mcd(ref_mgc, syn_mgc, path) -> float:
    import numpy as np

    diffs = ref_mgc[:, path[:, 0]] - syn_mgc[:, path[:, 1]]
    per_frame_db = _mcd_const() * np.sqrt(np.sum(diffs ** 2, axis=0))
    return float(np.mean(per_frame_db))


def compute_f0_rmse(ref_f0, syn_f0, path) -> float:
    import numpy as np

    ref_idx = np.clip(path[:, 0], 0, len(ref_f0) - 1)
    syn_idx = np.clip(path[:, 1], 0, len(syn_f0) - 1)
    ref_aligned = ref_f0[ref_idx]
    syn_aligned = syn_f0[syn_idx]

    voiced = (ref_aligned > 0) & (syn_aligned > 0)
    if not np.any(voiced):
        return float("nan")
    return float(np.sqrt(np.mean((ref_aligned[voiced] - syn_aligned[voiced]) ** 2)))


# --------------------------------------------------------------------------

def main() -> None:
    import numpy as np
    import yaml

    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/eval_tts.yaml"
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))

    log(f"loading checkpoint: {cfg['checkpoint_dir']}")
    model, tokenizer = load_checkpoint_model(cfg["output_dir"], cfg["checkpoint_dir"])
    sr = model.config.sampling_rate

    val_dir = Path(cfg["dataset_dir"]) / "validation"
    rows = list(csv.DictReader(open(val_dir / "metadata.csv", encoding="utf-8")))

    import librosa

    results = []
    for row in rows:
        ref_wav, _ = librosa.load(val_dir / row["file_name"], sr=sr)
        syn_wav = synthesize(model, tokenizer, row["text"])

        ref_f0, ref_env = extract_world_features(ref_wav, sr)
        syn_f0, syn_env = extract_world_features(syn_wav, sr)
        ref_mgc = envelope_to_mgc(ref_env, sr)
        syn_mgc = envelope_to_mgc(syn_env, sr)
        path = dtw_align(ref_mgc, syn_mgc)

        mcd = compute_mcd(ref_mgc, syn_mgc, path)
        f0_rmse = compute_f0_rmse(ref_f0, syn_f0, path)
        results.append({
            "file_name": row["file_name"],
            "mcd_db": round(mcd, 4),
            "f0_rmse_hz": round(f0_rmse, 4) if not np.isnan(f0_rmse) else None,
        })
        log(f"{row['file_name']:<28} MCD={mcd:6.3f} dB   F0 RMSE={f0_rmse:6.2f} Hz")

    mcd_values = [r["mcd_db"] for r in results]
    f0_values = [r["f0_rmse_hz"] for r in results if r["f0_rmse_hz"] is not None]

    summary = {
        "checkpoint": cfg["checkpoint_dir"],
        "n_clips": len(results),
        "mean_mcd_db": round(float(np.mean(mcd_values)), 4),
        "mean_f0_rmse_hz": round(float(np.mean(f0_values)), 4) if f0_values else None,
        "per_clip": results,
    }

    out_path = Path(cfg["output_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    log(f"mean MCD:     {summary['mean_mcd_db']:.3f} dB  "
        f"(lower is better; under ~6 dB is generally considered good quality)")
    if f0_values:
        log(f"mean F0 RMSE: {summary['mean_f0_rmse_hz']:.2f} Hz")
    else:
        log("mean F0 RMSE: n/a (no voiced overlap found across any clip)")
    log(f"results saved to {out_path}")


if __name__ == "__main__":
    main()
