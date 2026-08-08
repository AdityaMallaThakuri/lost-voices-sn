"""
tts_service.py — loads two VITS checkpoints once and exposes text -> WAV-bytes
synthesis for the FastAPI TTS routes:

- TTSService: our fine-tuned Sunuwar checkpoint (checkpoint-5500). Wraps
  src/evaluate_tts.py's load_checkpoint_model()/synthesize() (no behavior
  changes).
- BaseModelService: facebook/mms-tts-mai, the pretrained checkpoint
  checkpoint-5500 was fine-tuned FROM (see CLAUDE.md's TTS roadmap). Serving
  its output alongside ours is a before/after fine-tuning comparison, not a
  claim that Maithili is a native Sunuwar voice.

Both follow the same load-once-in-__init__ pattern already used by
model_service.py/clm_service.py/nmt_service.py.

Environment note (see the TTS backend-readiness investigation): this module
imports `transformers`, which the dashboard's `myenv` conda env deliberately
does NOT have — myenv was kept transformers-free so the MLM/CLM/NMT services
could stay lightweight (see CLAUDE.md's Dashboard section). transformers also
needs a specific old pin (==4.44.2) to load checkpoint-5500's weight-norm
layout correctly (newer versions hit the apply_weight_norm/remove_weight_norm
API mismatch documented for the Colab fine-tune). Rather than touch myenv, TTS
runs in its own `tts_env` conda env, so this module CANNOT be imported into
dashboard/app.py's process — it is served by its own FastAPI app (tts_app.py)
on its own port.

mms-tts-mai's raw safetensors need a DIFFERENT weight-norm remap than
checkpoint-5500's own (see load_base_model's docstring) — the two checkpoints
were serialized by different training pipelines and split the weight-norm
decomposition across different layers.
"""

import io
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.io import wavfile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluate_tts import load_checkpoint_model, synthesize  # noqa: E402

BASE_MODEL_DIR = PROJECT_ROOT / "models" / "mms-tts-mai"


def load_base_model(model_dir: str):
    """Load a raw Meta-released VITS checkpoint (mms-tts-mai / mms-tts-suz format).

    Different bug than checkpoint-5500's: `VitsWaveNet.__init__` (used inside
    `flow.flows[i].wavenet` and `posterior_encoder.wavenet`) unconditionally
    applies NEW-style parametrized weight_norm at construction time on
    torch>=2.1, but this checkpoint's safetensors store those specific layers
    under the OLD split naming (weight_g/weight_v) -- the opposite split from
    checkpoint-5500, whose decoder/flow conv_pre/post need the old-style
    remap instead (see evaluate_tts.py's load_checkpoint_model) while its
    wavenet layers already match fresh construction directly.

    torch.nn.utils.parametrize.remove_parametrizations() also leaves a stale
    `_load_state_dict_pre_hook` behind on the conv module -- undocumented
    torch behavior, confirmed by inspecting `conv._load_state_dict_pre_hooks`
    before/after removal. Left in place, that stale hook intercepts the
    later load_state_dict() call and silently reports every remapped layer as
    both missing and unexpected, even though the plain key sets match
    exactly. It must be cleared manually.
    """
    import torch
    from safetensors.torch import load_file
    from torch.nn.utils import remove_weight_norm, weight_norm
    from torch.nn.utils.parametrize import remove_parametrizations
    from transformers import AutoTokenizer, VitsConfig, VitsModel

    config = VitsConfig.from_pretrained(model_dir)
    model = VitsModel(config)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    wavenets = [flow_step.wavenet for flow_step in model.flow.flows] + [model.posterior_encoder.wavenet]
    convs = [layer for wn in wavenets for layer in list(wn.in_layers) + list(wn.res_skip_layers)]
    for conv in convs:
        remove_parametrizations(conv, "weight")
        conv._load_state_dict_pre_hooks.clear()
        weight_norm(conv, name="weight")

    state_dict = load_file(f"{model_dir}/model.safetensors")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    assert not missing and not unexpected, (
        f"base checkpoint weights did not load cleanly — "
        f"missing={missing} unexpected={unexpected}"
    )

    for conv in convs:
        remove_weight_norm(conv)

    model.eval()
    return model, tokenizer


def _wav_bytes(model, sample_rate: int, tokenizer, text: str) -> bytes:
    if not text.strip():
        raise ValueError("Text must not be empty.")

    wav = synthesize(model, tokenizer, text)
    pcm16 = (np.clip(wav, -1.0, 1.0) * 32767).astype(np.int16)

    buffer = io.BytesIO()
    wavfile.write(buffer, sample_rate, pcm16)
    return buffer.getvalue()


class TTSService:
    def __init__(self):
        config_path = PROJECT_ROOT / "configs" / "eval_tts.yaml"
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        output_dir = PROJECT_ROOT / config["output_dir"]
        checkpoint_dir = PROJECT_ROOT / config["checkpoint_dir"]
        self.model, self.tokenizer = load_checkpoint_model(str(output_dir), str(checkpoint_dir))
        self.sample_rate = self.model.config.sampling_rate

    def synthesize_wav_bytes(self, text: str) -> bytes:
        return _wav_bytes(self.model, self.sample_rate, self.tokenizer, text)


class BaseModelService:
    """facebook/mms-tts-mai -- the pretrained checkpoint fine-tuned into checkpoint-5500."""

    def __init__(self):
        self.model, self.tokenizer = load_base_model(str(BASE_MODEL_DIR))
        self.sample_rate = self.model.config.sampling_rate

    def synthesize_wav_bytes(self, text: str) -> bytes:
        return _wav_bytes(self.model, self.sample_rate, self.tokenizer, text)
