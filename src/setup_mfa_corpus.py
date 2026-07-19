"""
setup_mfa_corpus.py — assemble MFA 3.x corpus directory from the full
27-book NT corpus (chapter-level utterances, not pre-chunked — alignment
happens on whole chapters, segmentation happens afterwards from the
resulting TextGrids; see CLAUDE.md "TTS roadmap").

MFA expects:  corpus_dir/speaker_name/utterance.wav + utterance.txt (same stem)

Usage: python src/setup_mfa_corpus.py configs/mfa_corpus.yaml
"""

import sys
import shutil
from pathlib import Path

import yaml


def main(config_path: str) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))

    wav_dir    = Path(cfg["wav_dir"])
    text_dir   = Path(cfg["text_dir"])
    corpus_dir = Path(cfg["corpus_dir"])
    book_map   = cfg["book_map"]

    speaker_dir = corpus_dir / "speaker1"
    speaker_dir.mkdir(parents=True, exist_ok=True)

    copied  = 0
    missing = 0

    for code, n_chapters in book_map.items():
        for ch in range(1, int(n_chapters) + 1):
            stem = f"{code}_{ch:03d}"

            src_wav = wav_dir / f"{stem}.wav"
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
            copied += 1

    total_expected = sum(int(n) for n in book_map.values())
    print(f"\nCorpus: {speaker_dir}")
    print(f"Books              : {len(book_map)}")
    print(f"Chapters expected  : {total_expected}")
    print(f"Chapter pairs copied: {copied}")
    print(f"Missing pairs      : {missing}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/setup_mfa_corpus.py configs/mfa_corpus.yaml")
        sys.exit(1)
    main(sys.argv[1])
