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

Book-intro handling (added after the transcript-bug fix, see CLAUDE.md):
every chapter's audio opens with the narrator reading two header lines
that clean_transcripts.py strips before this script ever sees the text —
a book-name announcement ("मत्ती।") and a chapter-number announcement
("१।"). Both consume a pause-slot, which is why the original version of
this script (before this fix) was systematically off by 2 segments,
previously patched after the fact by fix_intro_offset.py deleting the
first two segments of every chapter outright.

Now handled at the source instead: clean_transcripts.py recovers the
book-name line as real (not guessed) text and writes it to
mfa_intro/<chapter>.txt. This script still reserves 2 leading pause-slots
per chapter (both are real, audible pauses), assigns the recovered
book-name text to the first as a genuine kept segment (`_seg_book`), and
excludes the second from the corpus entirely rather than fabricating a
transcript for it — the chapter-number line only ever wrote the bare
digit, not the Sunuwar number word actually spoken, so there's no honest
transcript to give it. fix_intro_offset.py is no longer needed or run.

Usage: python src/segment_from_pauses.py configs/segment_from_pauses.yaml
"""

import csv
import re
import sys
from pathlib import Path

from pydub import AudioSegment
from pydub.silence import detect_silence, detect_leading_silence
import yaml

SENTENCE_SPLIT = re.compile(r"(?<=[।॥])")


def trim_edge_silence(audio: AudioSegment, silence_thresh_db: float) -> AudioSegment:
    """Strip dead air at the very start/end of the recording before pause
    detection runs. Without this, some chapters' first ranked pause boundary
    falls on pre-speech silence rather than the book-name announcement,
    which was caught by a 3-chapter pilot: MAT_001 and REV_021 both produced
    a ~0.3s, -inf dBFS "book intro" segment — pure silence, not a spoken
    word — while 1CO_001 (no leading dead air) produced a plausible one."""
    start_trim = detect_leading_silence(audio, silence_threshold=silence_thresh_db)
    end_trim = detect_leading_silence(audio.reverse(), silence_threshold=silence_thresh_db)
    return audio[start_trim: len(audio) - end_trim]


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


def process_chapter(wav_path: Path, text: str, intro_text: str, cfg: dict, speaker_dir: Path) -> list:
    """Returns list of dicts, one per segment (written or intentionally
    excluded), for the report."""
    sentences = split_sentences(text)
    n_sentences = len(sentences)
    n_leading = 2 if intro_text else 0  # book-name + chapter-number announcements
    n_total_chunks = n_leading + n_sentences

    audio = AudioSegment.from_wav(str(wav_path))
    audio = trim_edge_silence(audio, cfg["silence_thresh_db"])
    cut_points, status = find_boundaries(
        audio, n_total_chunks, cfg["min_silence_len_ms"], cfg["silence_thresh_db"]
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
            "chapter": stem, "segment": seg_path_stem, "type": "whole_chapter_fallback",
            "duration_s": round(len(audio) / 1000.0, 2),
            "n_sentences_in_chapter": n_sentences,
            "confidence": "low", "reason": status,
        })
        return rows

    def _duration_confidence(duration_s: float) -> tuple:
        low_conf = not (cfg["min_confident_duration_s"] <= duration_s <= cfg["max_confident_duration_s"])
        return ("low" if low_conf else "high"), ("duration_outlier" if low_conf else "")

    if n_leading:
        # Chunk 0: book-name announcement — known text, kept as a real segment.
        start_ms, end_ms = cut_points[0], cut_points[1]
        duration_s = (end_ms - start_ms) / 1000.0
        seg_path_stem = f"{stem}_seg_book"
        audio[start_ms:end_ms].export(str(speaker_dir / f"{seg_path_stem}.wav"), format="wav")
        (speaker_dir / f"{seg_path_stem}.txt").write_text(intro_text, encoding="utf-8")
        confidence, reason = _duration_confidence(duration_s)
        rows.append({
            "chapter": stem, "segment": seg_path_stem, "type": "book_intro",
            "duration_s": round(duration_s, 2),
            "n_sentences_in_chapter": n_sentences,
            "confidence": confidence, "reason": reason,
        })

        # Chunk 1: chapter-number announcement — real audio, but the source
        # text only ever gave us the bare digit, not the Sunuwar number word
        # actually spoken. No honest transcript to assign, so this chunk is
        # excluded from the corpus (no wav/txt written) rather than guessed.
        start_ms, end_ms = cut_points[1], cut_points[2]
        rows.append({
            "chapter": stem, "segment": f"{stem}_seg_chapter_excluded", "type": "chapter_number_excluded",
            "duration_s": round((end_ms - start_ms) / 1000.0, 2),
            "n_sentences_in_chapter": n_sentences,
            "confidence": "excluded", "reason": "chapter_number_text_unknown",
        })

    for i, sentence in enumerate(sentences):
        start_ms, end_ms = cut_points[n_leading + i], cut_points[n_leading + i + 1]
        duration_s = (end_ms - start_ms) / 1000.0
        seg_path_stem = f"{stem}_seg_{i+1:03d}"

        segment_audio = audio[start_ms:end_ms]
        segment_audio.export(str(speaker_dir / f"{seg_path_stem}.wav"), format="wav")
        (speaker_dir / f"{seg_path_stem}.txt").write_text(sentence, encoding="utf-8")

        confidence, reason = _duration_confidence(duration_s)
        rows.append({
            "chapter": stem, "segment": seg_path_stem, "type": "sentence",
            "duration_s": round(duration_s, 2),
            "n_sentences_in_chapter": n_sentences,
            "confidence": confidence, "reason": reason,
        })

    return rows


def main(config_path: str) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))

    wav_dir = Path(cfg["wav_dir"])
    text_dir = Path(cfg["text_dir"])
    intro_text_dir = Path(cfg["intro_text_dir"])
    speaker_dir = Path(cfg["output_corpus_dir"]) / "speaker1"
    speaker_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    chapters_processed = 0
    chapters_fallback = 0
    chapters_missing_intro = 0

    for wav_path in sorted(wav_dir.glob("*.wav")):
        txt_path = text_dir / (wav_path.stem + ".txt")
        if not txt_path.exists():
            print(f"MISSING TXT: {txt_path} — skipping")
            continue

        text = txt_path.read_text(encoding="utf-8")

        intro_path = intro_text_dir / (wav_path.stem + ".txt")
        intro_text = intro_path.read_text(encoding="utf-8").strip() if intro_path.exists() else ""
        if not intro_text:
            chapters_missing_intro += 1

        rows = process_chapter(wav_path, text, intro_text, cfg, speaker_dir)
        all_rows.extend(rows)
        chapters_processed += 1
        if len(rows) == 1 and rows[0]["reason"] in ("single_sentence", "insufficient_pauses"):
            chapters_fallback += 1

        print(f"  {wav_path.stem}: {len(rows)} rows ({'no intro found' if not intro_text else 'intro ok'})")

    report_path = Path(cfg["report_path"])
    with open(report_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "chapter", "segment", "type", "duration_s", "n_sentences_in_chapter", "confidence", "reason"
        ])
        writer.writeheader()
        writer.writerows(all_rows)

    written_rows = [r for r in all_rows if r["confidence"] != "excluded"]
    excluded_rows = [r for r in all_rows if r["confidence"] == "excluded"]
    total_segments = len(written_rows)
    low_conf = sum(1 for r in written_rows if r["confidence"] == "low")
    durations = [r["duration_s"] for r in written_rows]

    print()
    print("--- Segmentation summary ---")
    print(f"Chapters processed         : {chapters_processed}")
    print(f"Chapters missing intro text: {chapters_missing_intro} (no leading pause-slots reserved for these)")
    print(f"Chapters using fallback    : {chapters_fallback} (whole-chapter, not sentence-split)")
    print(f"Segments written (wav+txt) : {total_segments}")
    print(f"  of which book-intro      : {sum(1 for r in written_rows if r['type'] == 'book_intro')}")
    print(f"Chapter-number chunks excluded (audio only, no honest transcript): {len(excluded_rows)}")
    print(f"Low-confidence segments    : {low_conf} ({100*low_conf/total_segments:.1f}%)")
    print(f"Mean duration              : {sum(durations)/len(durations):.1f}s")
    print(f"Min / Max duration         : {min(durations):.1f}s / {max(durations):.1f}s")
    print(f"Report written to          : {report_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/segment_from_pauses.py configs/segment_from_pauses.yaml")
        sys.exit(1)
    main(sys.argv[1])
