"""
train_tts.py — Phase 6 of the TTS roadmap: fine-tune a VITS/MMS-TTS checkpoint
on the Sunuwar single-speaker dataset built by build_tts_dataset.py.

Why this is a wrapper and not a training loop
---------------------------------------------
VITS is a GAN: training needs a multi-period discriminator, a mel
reconstruction loss, a KL term and the monotonic alignment search. Mainline
`transformers` ships `VitsModel` for **inference only** — the released
checkpoints carry no discriminator weights and the class exposes no training
loss. Re-implementing that is a large, easy-to-get-subtly-wrong job.

So, exactly like align.py wraps the `mfa` CLI, this wraps the community
reference implementation (ylacombe/finetune-hf-vits): we own the config, the
text/vocabulary adaptation and the dataset contract; it owns the GAN training
step. Two prerequisites, both handled by notebooks/finetune_tts_colab.ipynb:

  1. the repo is cloned and its `monotonic_align` Cython extension is built,
  2. the base checkpoint has been converted to carry a discriminator
     (`convert_original_discriminator_checkpoint.py --language_code nep`).

Text/vocabulary adaptation
--------------------------
MMS-TTS tokenizers are character-based, so a character the checkpoint has
never seen becomes `<unk>` — silently. Sunuwar text is 3.5% ZWJ (U+200D) by
character count, and the Nepali checkpoint may well not have it. `--preflight`
reports the coverage gap; `adapt_text_to_vocab` then rewrites the dataset's
metadata.csv to stay inside the supported inventory before training.

Dropping ZWJ is phonemically safe: in Devanagari a ZWJ following a virama
(U+094D) only selects the half-form glyph over the ligature. The virama
itself is what encodes the conjunct, and it is kept. This is a rendering
distinction, not a pronunciation one.

Usage:
    python src/train_tts.py configs/tts.yaml --preflight   # report only
    python src/train_tts.py configs/tts.yaml               # adapt + train
"""

import csv
import json
import os
import random
import subprocess
import sys
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path

import yaml

SEED = 42
random.seed(SEED)

ZWJ = "‍"
VIRAMA = "्"
SPLITS = ("train", "validation")


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


# --------------------------------------------------------------------------
# Vocabulary preflight
# --------------------------------------------------------------------------

def load_vocab(model_name_or_path: str):
    """Character inventory of the base checkpoint's tokenizer."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    vocab = set(tokenizer.get_vocab())
    return tokenizer, vocab


def dataset_charset(dataset_dir: Path) -> Counter:
    chars = Counter()
    for split in SPLITS:
        meta = dataset_dir / split / "metadata.csv"
        if not meta.is_file():
            continue
        with meta.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                chars.update(row["text"])
    return chars


def preflight(dataset_dir: Path, model_name_or_path: str, is_uroman: bool = False):
    """Compare dataset characters against the tokenizer vocabulary."""
    tokenizer, vocab = load_vocab(model_name_or_path)
    chars = dataset_charset(dataset_dir)
    total = sum(chars.values())

    lowercase = getattr(tokenizer, "do_lower_case", False)
    log(f"tokenizer                : {model_name_or_path}")
    log(f"  vocab size             : {len(vocab)}")
    log(f"  is_uroman              : {getattr(tokenizer, 'is_uroman', is_uroman)}")
    log(f"  do_lower_case          : {lowercase}")
    log(f"dataset characters       : {len(chars)} distinct, {total} total")

    missing = {c: n for c, n in chars.items() if c not in vocab and not c.isspace()}
    covered = total - sum(missing.values())
    log(f"  covered by vocab       : {covered}/{total} ({covered / total:.2%})")

    if not missing:
        log("  MISSING                : none — no text adaptation needed")
    else:
        log(f"  MISSING                : {len(missing)} distinct characters")
        for c, n in sorted(missing.items(), key=lambda kv: -kv[1]):
            name = unicodedata.name(c, "<unnamed>")
            log(f"      U+{ord(c):04X} {name:<44} {n:>6}  ({n / total:.2%})")
    return missing


# --------------------------------------------------------------------------
# Text adaptation
# --------------------------------------------------------------------------

def adapt_text(text: str, missing: dict, policy: dict) -> str:
    """Bring one transcript inside the tokenizer's character inventory."""
    out = []
    for ch in text:
        if ch not in missing:
            out.append(ch)
            continue
        replacement = policy.get(ch, policy.get("__default__", ""))
        out.append(replacement)
    return " ".join("".join(out).split())


def build_policy(missing: dict, cfg: dict) -> dict:
    """Decide what to substitute for each unsupported character."""
    policy = {"__default__": ""}

    # ZWJ: drop it. The virama it follows already encodes the conjunct, so
    # this changes glyph shaping only, never pronunciation.
    if ZWJ in missing:
        policy[ZWJ] = ""

    # Sentence-final danda: map to a full stop if the vocab has one, so the
    # model keeps a phrase-final cue instead of losing the boundary entirely.
    for danda in ("।", "॥"):
        if danda in missing:
            policy[danda] = cfg.get("danda_replacement", "")

    policy.update(cfg.get("character_overrides", {}) or {})
    return policy


def apply_adaptation(dataset_dir: Path, missing: dict, policy: dict) -> None:
    """Rewrite each split's metadata.csv in place. Audio is untouched.

    In Colab the dataset directory is an unzipped throwaway copy, so editing
    it in place is safe; the canonical dataset in the repo is not modified.
    """
    for split in SPLITS:
        meta = dataset_dir / split / "metadata.csv"
        if not meta.is_file():
            continue
        with meta.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        changed = 0
        emptied = []
        for row in rows:
            new = adapt_text(row["text"], missing, policy)
            if new != row["text"]:
                changed += 1
            if not new.strip():
                emptied.append(row["file_name"])
            row["text"] = new

        if emptied:
            raise SystemExit(
                f"{split}: adaptation emptied {len(emptied)} transcripts "
                f"(e.g. {emptied[:3]}) — refusing to write a broken dataset"
            )

        with meta.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["file_name", "text"], extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        log(f"  {split:<11} {changed}/{len(rows)} transcripts rewritten")


# --------------------------------------------------------------------------
# finetune-hf-vits config + launch
# --------------------------------------------------------------------------

def build_training_json(cfg: dict, dataset_dir: Path, out_path: Path) -> dict:
    """Translate our YAML into the JSON run_vits_finetuning.py expects."""
    training = {
        "project_name": cfg.get("project_name", "sunuwar-tts"),
        "model_name_or_path": cfg["discriminator_checkpoint"],
        "output_dir": cfg["output_dir"],
        "overwrite_output_dir": True,

        # The trainer has no `data_dir` field (verified by introspecting its
        # dataclasses), so the local directory goes straight into
        # `dataset_name`: load_dataset() on a path auto-detects the audiofolder
        # layout and yields the train/ and validation/ splits by itself. This
        # also keeps the licensed audio local — nothing is uploaded to the Hub.
        "dataset_name": str(dataset_dir),
        "dataset_config_name": None,
        "audio_column_name": "audio",
        "text_column_name": "text",
        "train_split_name": "train",
        "eval_split_name": "validation",

        "seed": SEED,
        "do_train": True,
        "do_eval": True,
        "per_device_train_batch_size": cfg.get("batch_size", 8),
        "per_device_eval_batch_size": cfg.get("eval_batch_size", 8),
        "gradient_accumulation_steps": cfg.get("gradient_accumulation_steps", 1),
        "learning_rate": cfg.get("learning_rate", 2e-4),
        "num_train_epochs": cfg.get("num_train_epochs", 200),
        "warmup_ratio": cfg.get("warmup_ratio", 0.0),
        "lr_scheduler_type": cfg.get("lr_scheduler_type", "linear"),
        "fp16": cfg.get("fp16", True),
        "group_by_length": False,

        "do_step_schedule_per_epoch": True,
        "max_duration_in_seconds": cfg.get("max_duration_in_seconds", 20.0),
        "min_duration_in_seconds": cfg.get("min_duration_in_seconds", 1.0),
        "preprocessing_num_workers": cfg.get("preprocessing_num_workers", 2),
        "dataloader_num_workers": cfg.get("dataloader_num_workers", 2),

        # GAN loss weights — the reference recipe's defaults
        "weight_disc": cfg.get("weight_disc", 3.0),
        "weight_fmaps": cfg.get("weight_fmaps", 1.0),
        "weight_gen": cfg.get("weight_gen", 1.0),
        "weight_kl": cfg.get("weight_kl", 1.5),
        "weight_duration": cfg.get("weight_duration", 1.0),
        "weight_mel": cfg.get("weight_mel", 35.0),

        # Synthesised periodically during training so you can hear progress
        # instead of only watching the loss.
        "full_generation_sample_text": cfg.get("sample_text", ""),

        "logging_steps": cfg.get("logging_steps", 10),
        "eval_steps": cfg.get("eval_steps", 50),
        "save_steps": cfg.get("save_steps", 500),
        "save_total_limit": cfg.get("save_total_limit", 2),
        "report_to": cfg.get("report_to", []),
        "push_to_hub": False,
    }

    # Single narrator: no speaker embeddings anywhere in this corpus.
    training["speaker_id_column_name"] = None
    training["override_speaker_embeddings"] = False
    training["filter_on_speaker_id"] = None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(training, indent=2), encoding="utf-8")
    return training


def launch(cfg: dict, json_path: Path) -> None:
    repo = Path(cfg["finetune_repo_dir"])
    script = repo / "run_vits_finetuning.py"
    if not script.is_file():
        raise SystemExit(
            f"{script} not found — clone ylacombe/finetune-hf-vits and build its "
            f"monotonic_align extension first (see notebooks/finetune_tts_colab.ipynb)"
        )

    cmd = ["accelerate", "launch", str(script), str(json_path.resolve())]
    log("Running: " + " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(repo))
    if result.returncode != 0:
        raise SystemExit(f"training failed with exit code {result.returncode}")
    log(f"training complete — checkpoint in {cfg['output_dir']}")


def main(config_path: str, preflight_only: bool) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    dataset_dir = Path(cfg["dataset_dir"]).resolve()
    if not (dataset_dir / "train" / "metadata.csv").is_file():
        raise SystemExit(f"no audiofolder dataset at {dataset_dir} — run build_tts_dataset.py")

    base = cfg.get("base_checkpoint", "facebook/mms-tts-nep")
    missing = preflight(dataset_dir, base)

    if preflight_only:
        log("preflight only — nothing written")
        return

    if missing:
        if not cfg.get("adapt_text_to_vocab", True):
            raise SystemExit(
                "dataset contains characters the tokenizer cannot represent and "
                "adapt_text_to_vocab is false — they would become <unk>"
            )
        policy = build_policy(missing, cfg)
        log("adapting transcripts to the tokenizer inventory:")
        for ch in sorted(missing, key=lambda c: -missing[c]):
            shown = policy.get(ch, policy["__default__"]) or "<deleted>"
            log(f"      U+{ord(ch):04X} -> {shown}")
        apply_adaptation(dataset_dir, missing, policy)

    json_path = Path(cfg.get("training_json", "configs/_tts_finetune.json"))
    build_training_json(cfg, dataset_dir, json_path)
    log(f"training config written  : {json_path}")

    launch(cfg, json_path)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    if len(args) != 1:
        print("Usage: python src/train_tts.py <config.yaml> [--preflight]")
        sys.exit(1)
    main(args[0], preflight_only="--preflight" in flags)
