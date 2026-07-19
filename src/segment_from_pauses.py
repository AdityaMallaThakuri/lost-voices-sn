"""
segment_from_pauses.py — split chapter audio into per-sentence segments
using ranked silence-duration boundaries, replacing the proportional
word-count guessing in split_audio_chunks.py (see CLAUDE.md "TTS roadmap").

Method: detect every candidate pause in the audio (permissive threshold,
over-detects), then keep only the (sentence_count - 1) longest pauses as
real sentence boundaries — validated against a 40-chapter sample where
this produced plausible single-sentence durations (5-10s mean) versus the
old method's proportional guess, which never matched sentence count within
even a loose tolerance.

Each segment's text is the literal sentence it corresponds to (from danda
splitting) — not estimated. Segments with implausible duration are flagged
low-confidence for Phase 3 to filter, not silently dropped.

Usage: python src/segment_from_pauses.py configs/segment_from_pauses.yaml
"""

import csv
import re
import sys
from pathlib import Path

from pydub import AudioSegment
from pydub.silence import detect_silence
import yaml

SENTENCE_SPLIT = re.compile(r"(?<=[।॥])")


def split_sentences(text: str) -> list:
    parts = SENTENCE_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]


def find_boundaries(audio: AudioSegment, n_sentences: int, min_silence_len_ms: int,
                     silence_thresh_db: float) -> tuple:
    """Returns (cut_points_ms, status) where status explains any fallback used."""
    if n_sentences <= 1:
        return [0, len(audio)], "single_sentence"

    silences = detect_silence(audio, min_silence_len=min_silence_len_ms, silence_thresh=silence_thresh_db)
    n_needed = n_sentences - 1

    if len(silences) < n_needed:
        return [0, len(audio)], "insufficient_pauses"

    ranked = sorted(silences, key=lambda s: s[1] - s[0], reverse=True)
    chosen = ranked[:n_needed]
    chosen.sort(key=lambda s: s[0])

    cut_points = [0] + [(s[0] + s[1]) // 2 for s in chosen] + [len(audio)]
    return cut_points, "ranked_ok"


def process_chapter(wav_path: Path, text: str, cfg: dict, speaker_dir: Path) -> list:
    """Returns list of dicts, one per segment written, for the report."""
    sentences = split_sentences(text)
    n_sentences = len(sentences)

    audio = AudioSegment.from_wav(str(wav_path))
    cut_points, status = find_boundaries(
        audio, n_sentences, cfg["min_silence_len_ms"], cfg["silence_thresh_db"]
    )

    stem = wav_path.stem
    rows = []

    if status != "ranked_ok":
        # Can't reliably split — write the whole chapter as one flagged segment
        # rather than silently guessing a bad split.
        seg_path_stem = f"{stem}_seg_001"
        audio.export(str(speaker_dir / f"{seg_path_stem}.wav"), format="wav")
        (speaker_dir / f"{seg_path_stem}.txt").write_text(text, encoding="utf-8")
        rows.append({
            "chapter": stem, "segment": seg_path_stem,
            "duration_s": round(len(audio) / 1000.0, 2),
            "n_sentences_in_chapter": n_sentences,
            "confidence": "low", "reason": status,
        })
        return rows

    for i, sentence in enumerate(sentences):
        start_ms, end_ms = cut_points[i], cut_points[i + 1]
        duration_s = (end_ms - start_ms) / 1000.0
        seg_path_stem = f"{stem}_seg_{i+1:03d}"

        segment_audio = audio[start_ms:end_ms]
        segment_audio.export(str(speaker_dir / f"{seg_path_stem}.wav"), format="wav")
        (speaker_dir / f"{seg_path_stem}.txt").write_text(sentence, encoding="utf-8")

        low_conf = not (cfg["min_confident_duration_s"] <= duration_s <= cfg["max_confident_duration_s"])
        rows.append({
            "chapter": stem, "segment": seg_path_stem,
            "duration_s": round(duration_s, 2),
            "n_sentences_in_chapter": n_sentences,
            "confidence": "low" if low_conf else "high",
            "reason": "duration_outlier" if low_conf else "",
        })

    return rows


def main(config_path: str) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))

    wav_dir = Path(cfg["wav_dir"])
    text_dir = Path(cfg["text_dir"])
    speaker_dir = Path(cfg["output_corpus_dir"]) / "speaker1"
    speaker_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    chapters_processed = 0
    chapters_fallback = 0

    for wav_path in sorted(wav_dir.glob("*.wav")):
        txt_path = text_dir / (wav_path.stem + ".txt")
        if not txt_path.exists():
            print(f"MISSING TXT: {txt_path} — skipping")
            continue

        text = txt_path.read_text(encoding="utf-8")
        rows = process_chapter(wav_path, text, cfg, speaker_dir)
        all_rows.extend(rows)
        chapters_processed += 1
        if len(rows) == 1 and rows[0]["reason"] in ("single_sentence", "insufficient_pauses"):
            chapters_fallback += 1

        print(f"  {wav_path.stem}: {len(rows)} segments")

    report_path = Path(cfg["report_path"])
    with open(report_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "chapter", "segment", "duration_s", "n_sentences_in_chapter", "confidence", "reason"
        ])
        writer.writeheader()
        writer.writerows(all_rows)

    total_segments = len(all_rows)
    low_conf = sum(1 for r in all_rows if r["confidence"] == "low")
    durations = [r["duration_s"] for r in all_rows]

    print()
    print("--- Segmentation summary ---")
    print(f"Chapters processed      : {chapters_processed}")
    print(f"Chapters using fallback  : {chapters_fallback} (whole-chapter, not sentence-split)")
    print(f"Total segments written   : {total_segments}")
    print(f"Low-confidence segments  : {low_conf} ({100*low_conf/total_segments:.1f}%)")
    print(f"Mean duration            : {sum(durations)/len(durations):.1f}s")
    print(f"Min / Max duration       : {min(durations):.1f}s / {max(durations):.1f}s")
    print(f"Report written to        : {report_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/segment_from_pauses.py configs/segment_from_pauses.yaml")
        sys.exit(1)
    main(sys.argv[1])
