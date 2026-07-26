"""
fix_intro_offset.py — correct the text/audio offset caused by the narrator's
spoken book/chapter intro at the start of each chapter recording.

segment_from_pauses.py assumed the audio contains exactly n_sentences spoken
chunks (matching the danda-split text count), but the narrator's un-transcribed
intro announcement steals two pause-slots meant for real sentence boundaries.
Confirmed by listening (see CLAUDE.md-adjacent conversation, 2026-07-25):
seg_001.txt's real audio is in seg_003.wav, seg_002.txt's is in seg_004.wav,
and so on for every chapter.

Fix, per chapter, per speaker:
  1. Delete seg_001.wav and seg_002.wav (intro speech, no text).
  2. Shift every remaining wav down by 2: seg_{i:03d}.wav -> seg_{i-2:03d}.wav
     for i = 3..N, processed high-to-low to avoid overwrite collisions.
  3. Delete seg_{N-1:03d}.txt and seg_{N:03d}.txt (no wav left to pair with).

Usage: python src/fix_intro_offset.py <mfa_corpus_segments_dir>
       (defaults to data/processed/audio/mfa_corpus_segments)
"""

import re
import sys
from collections import defaultdict
from pathlib import Path

SEG_RE = re.compile(r"^(?P<chapter>.+)_seg_(?P<num>\d{3})$")


def fix_speaker_dir(speaker_dir: Path) -> None:
    chapters = defaultdict(list)
    for wav_path in speaker_dir.glob("*.wav"):
        m = SEG_RE.match(wav_path.stem)
        if not m:
            print(f"  SKIP (unexpected name): {wav_path.name}")
            continue
        chapters[m.group("chapter")].append(int(m.group("num")))

    for chapter, nums in sorted(chapters.items()):
        nums.sort()
        n = len(nums)
        if n != nums[-1]:
            print(f"  SKIP {chapter}: non-contiguous segment numbers {nums}")
            continue
        if n < 3:
            print(f"  SKIP {chapter}: only {n} segments, can't drop 2")
            continue

        (speaker_dir / f"{chapter}_seg_001.wav").unlink()
        (speaker_dir / f"{chapter}_seg_002.wav").unlink()

        for i in range(3, n + 1):
            src = speaker_dir / f"{chapter}_seg_{i:03d}.wav"
            dst = speaker_dir / f"{chapter}_seg_{i - 2:03d}.wav"
            src.rename(dst)

        (speaker_dir / f"{chapter}_seg_{n - 1:03d}.txt").unlink()
        (speaker_dir / f"{chapter}_seg_{n:03d}.txt").unlink()

        print(f"  {chapter}: {n} -> {n - 2} segments")


def main(corpus_dir: str) -> None:
    corpus_path = Path(corpus_dir)
    speaker_dirs = sorted(p for p in corpus_path.iterdir() if p.is_dir())
    if not speaker_dirs:
        print(f"No speaker subdirectories found under {corpus_path}")
        return

    for speaker_dir in speaker_dirs:
        print(f"Speaker: {speaker_dir.name}")
        fix_speaker_dir(speaker_dir)


if __name__ == "__main__":
    corpus_dir = sys.argv[1] if len(sys.argv) > 1 else "data/processed/audio/mfa_corpus_segments"
    main(corpus_dir)
