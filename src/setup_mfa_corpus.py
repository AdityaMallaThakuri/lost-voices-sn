"""
setup_mfa_corpus.py — assemble MFA 3.x corpus directory for a pilot book.

MFA expects:  corpus_dir/speaker_name/audio.wav + audio.txt (same stem)

Usage: python src/setup_mfa_corpus.py configs/mfa_corpus.yaml
"""

import shutil
import sys
from pathlib import Path

import yaml


def main(config_path: str) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))

    wav_dir   = Path(cfg["wav_dir"])
    text_dir  = Path(cfg["text_dir"])
    corpus_dir = Path(cfg["corpus_dir"])
    book      = cfg["pilot_book"]
    n_chapters = int(cfg["pilot_chapters"])

    speaker_dir = corpus_dir / "speaker1"
    speaker_dir.mkdir(parents=True, exist_ok=True)

    wav_copied  = 0
    txt_copied  = 0
    missing     = 0

    for ch in range(1, n_chapters + 1):
        stem = f"{book}_{ch:03d}"

        src_wav = wav_dir  / f"{stem}.wav"
        src_txt = text_dir / f"{stem}.txt"

        wav_ok = src_wav.exists()
        txt_ok = src_txt.exists()

        if not wav_ok or not txt_ok:
            missing += 1
            if not wav_ok:
                print(f"MISSING WAV: {src_wav}")
            if not txt_ok:
                print(f"MISSING TXT: {src_txt}")
            continue

        shutil.copy2(src_wav, speaker_dir / f"{stem}.wav")
        shutil.copy2(src_txt, speaker_dir / f"{stem}.txt")
        wav_copied += 1
        txt_copied += 1

    print(f"\nCorpus: {speaker_dir}")
    print(f"WAV files copied : {wav_copied}")
    print(f"TXT files copied : {txt_copied}")
    print(f"Missing pairs    : {missing}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/setup_mfa_corpus.py configs/mfa_corpus.yaml")
        sys.exit(1)
    main(sys.argv[1])
