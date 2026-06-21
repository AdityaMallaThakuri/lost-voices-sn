"""
split_audio_chunks.py — split chapter WAVs into 15-25 s chunks for MFA.

MFA 3.x requires utterances under 30 s.  Text is assigned proportionally:
each chunk receives a slice of the chapter words sized by its share of
total chapter duration.  This keeps the word-count per chunk in range so
Kaldi can align them.

Usage: python src/split_audio_chunks.py configs/mfa_corpus.yaml
"""

import shutil
import sys
from pathlib import Path

import yaml
from pydub import AudioSegment
from pydub.silence import split_on_silence

TARGET_MIN_MS = 15_000
TARGET_MAX_MS = 25_000
MIN_WORDS = 3


def group_chunks(segments: list[AudioSegment]) -> list[AudioSegment]:
    """Greedily merge silence-split segments into 15-25 s groups."""
    groups: list[AudioSegment] = []
    current: list[AudioSegment] = []
    current_ms = 0

    for seg in segments:
        seg_ms = len(seg)
        if current and current_ms >= TARGET_MIN_MS and current_ms + seg_ms > TARGET_MAX_MS:
            groups.append(sum(current, AudioSegment.empty()))
            current = [seg]
            current_ms = seg_ms
        else:
            current.append(seg)
            current_ms += seg_ms

    if current:
        groups.append(sum(current, AudioSegment.empty()))

    return groups


def assign_words(
    words: list[str], groups: list[AudioSegment]
) -> list[tuple[AudioSegment, str]]:
    """
    Assign each group a proportional word slice based on duration fraction.
    Returns (group, text) pairs; groups with < MIN_WORDS are dropped.
    """
    total_ms = sum(len(g) for g in groups)
    total_words = len(words)

    pairs: list[tuple[AudioSegment, str]] = []
    offset = 0

    for i, group in enumerate(groups):
        is_last = i == len(groups) - 1

        if is_last:
            chunk_words = words[offset:]
        else:
            fraction = len(group) / total_ms
            count = round(total_words * fraction)
            # Always advance at least 1 word so we don't stall
            count = max(count, 1)
            # Don't overshoot — leave at least 1 word for remaining chunks
            remaining_chunks = len(groups) - i - 1
            count = min(count, len(words) - offset - remaining_chunks)
            chunk_words = words[offset : offset + count]
            offset += count

        if len(chunk_words) < MIN_WORDS:
            continue

        pairs.append((group, " ".join(chunk_words)))

    return pairs


def process_chapter(wav_path: Path, txt_path: Path, out_dir: Path) -> list[float]:
    """Split one chapter WAV and write proportionally-assigned chunk TXTs.

    Returns list of written chunk durations in seconds.
    """
    audio = AudioSegment.from_wav(str(wav_path))
    transcript = txt_path.read_text(encoding="utf-8").strip()
    words = transcript.split()

    segments = split_on_silence(
        audio,
        min_silence_len=500,
        silence_thresh=-40,
        keep_silence=200,
    )
    if not segments:
        segments = [audio]

    groups = group_chunks(segments)
    pairs = assign_words(words, groups)

    stem = wav_path.stem
    durations: list[float] = []

    for i, (group, text) in enumerate(pairs, start=1):
        chunk_stem = f"{stem}_chunk_{i:03d}"
        group.export(str(out_dir / f"{chunk_stem}.wav"), format="wav")
        (out_dir / f"{chunk_stem}.txt").write_text(text, encoding="utf-8")
        durations.append(len(group) / 1000.0)

    return durations


def main(config_path: str) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))

    corpus_dir = Path(cfg["corpus_dir"]) / "speaker1"
    chunk_dir  = Path(cfg["chunk_dir"])  / "speaker1"

    # Wipe and recreate output dir for a clean run
    if chunk_dir.exists():
        shutil.rmtree(chunk_dir)
    chunk_dir.mkdir(parents=True)

    wav_files = sorted(corpus_dir.glob("*.wav"))
    if not wav_files:
        print(f"No WAV files found in {corpus_dir}")
        sys.exit(1)

    all_durations: list[float] = []
    chapters_processed = 0

    for wav_path in wav_files:
        txt_path = corpus_dir / (wav_path.stem + ".txt")
        if not txt_path.exists():
            print(f"MISSING TXT: {txt_path} — skipping")
            continue

        durations = process_chapter(wav_path, txt_path, chunk_dir)
        all_durations.extend(durations)
        chapters_processed += 1
        print(f"  {wav_path.name}: {len(durations)} chunks")

    if not all_durations:
        print("No chunks created.")
        return

    mean_s = sum(all_durations) / len(all_durations)
    print()
    print(f"Chapters processed : {chapters_processed}")
    print(f"Total chunks       : {len(all_durations)}")
    print(f"Mean duration      : {mean_s:.1f} s")
    print(f"Min duration       : {min(all_durations):.1f} s")
    print(f"Max duration       : {max(all_durations):.1f} s")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/split_audio_chunks.py configs/mfa_corpus.yaml")
        sys.exit(1)
    main(sys.argv[1])
