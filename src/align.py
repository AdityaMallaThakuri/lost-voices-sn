"""
align.py — thin subprocess wrapper around Montreal Forced Aligner.

MFA is a standalone CLI tool (installed via conda, not pip) — this script
just constructs and runs the right `mfa train` / `mfa align` command from
a config, the same way preprocess_audio.py wraps ffmpeg. Requires `mfa` to
already be on PATH (conda install -c conda-forge montreal-forced-aligner).

mode: train — trains a new acoustic model from the corpus AND produces
      alignments (TextGrids) as a side effect. Use this for the initial
      full-corpus, chapter-level pass (Phase 2, Step 2).
mode: align — aligns a corpus against an already-trained acoustic model,
      no training. Use this to re-align cut segments against the model
      trained in the `train` pass (Phase 2, Step 4).

Usage: python src/align.py configs/align_train.yaml
       python src/align.py configs/align_segments.yaml
"""

import subprocess
import sys
from pathlib import Path

import yaml


def build_command(cfg: dict) -> list:
    mode = cfg["mode"]
    corpus_dir = cfg["corpus_dir"]
    dictionary_path = cfg["dictionary_path"]
    output_dir = cfg["output_textgrid_dir"]
    num_jobs = str(cfg.get("num_jobs", 4))

    if mode == "train":
        cmd = [
            "mfa", "train",
            corpus_dir,
            dictionary_path,
            cfg["output_model_path"],
            "--output_directory", output_dir,
            "--num_jobs", num_jobs,
        ]
    elif mode == "align":
        cmd = [
            "mfa", "align",
            corpus_dir,
            dictionary_path,
            cfg["model_path"],
            output_dir,
            "--num_jobs", num_jobs,
        ]
    else:
        raise ValueError(f"mode must be 'train' or 'align', got {mode!r}")

    if cfg.get("clean", True):
        cmd.append("--clean")

    if cfg.get("temporary_directory"):
        cmd.extend(["--temporary_directory", cfg["temporary_directory"]])

    return cmd


def main(config_path: str) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))

    Path(cfg["output_textgrid_dir"]).mkdir(parents=True, exist_ok=True)
    if cfg["mode"] == "train":
        Path(cfg["output_model_path"]).parent.mkdir(parents=True, exist_ok=True)

    cmd = build_command(cfg)
    print("Running:", " ".join(cmd))

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nmfa {cfg['mode']} failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    print(f"\nmfa {cfg['mode']} completed.")
    print(f"TextGrids written to: {cfg['output_textgrid_dir']}")
    if cfg["mode"] == "train":
        print(f"Acoustic model saved to: {cfg['output_model_path']}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/align.py <config.yaml>")
        sys.exit(1)
    main(sys.argv[1])
