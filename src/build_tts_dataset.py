"""
build_tts_dataset.py — Phase 5 of the TTS roadmap: turn QC-passed aligned
segments into a training-ready single-speaker TTS dataset.

Reads the verdicts from Phase 4 (`results/qc_report.csv`), and for every
segment that passed:
  * trims the clip to the MFA-derived speech span (plus a small pad) so the
    model is not asked to learn the narrator's variable pre-roll silence,
  * copies the trimmed audio into `data/processed/tts_dataset/wavs/`,
  * NFC-normalises the transcript (ZWJ U+200D preserved — it is part of
    Sunuwar Devanagari orthography, never strip it).

Audio is read and written with the stdlib `wave` module: the corpus is
already 16 kHz mono 16-bit PCM out of preprocess_audio.py, so trimming is a
byte-range slice and needs no resampling library. The script asserts the
format rather than silently mis-slicing.

Validation is held out by **whole chapter**, never by random segment. Bible
verses repeat phrasing heavily across adjacent verses, so a random split
leaks near-duplicate text from train into val and makes the val loss
meaningless.

Output layout is HuggingFace `audiofolder`-compatible, so Phase 6 can do
`load_dataset("audiofolder", data_dir=...)` and get the splits for free:

    tts_dataset/
        metadata.csv              full manifest (all splits, extra columns)
        train/metadata.csv        file_name,text
        train/<segment>.wav
        validation/metadata.csv
        validation/<segment>.wav

Usage: python src/build_tts_dataset.py configs/tts_dataset.yaml
"""

import csv
import random
import shutil
import sys
import unicodedata
import wave
from collections import Counter
from datetime import datetime
from pathlib import Path

import yaml

from clean_transcripts import DIGIT_TOKEN

SEED = 42
random.seed(SEED)

EXPECTED_CHANNELS = 1
EXPECTED_SAMPWIDTH = 2  # 16-bit PCM


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def normalise_text(raw: str, strip_numerals: bool = True):
    """NFC-normalise and collapse whitespace. ZWJ (U+200D) is left intact.

    Returns (text, n_numeral_tokens_removed).

    The segment transcripts inherited mangled cross-reference remnants —
    fragments like "१कोरिन्‍" and bare verse numerals — from the regex bug
    since fixed in clean_transcripts.py. The narrator never speaks them and
    the G2P dictionary has no entry for any of them, so they are dropped here
    too: the segments themselves were cut before the fix, and re-deriving them
    would mean re-running Phase 3 alignment. Removing an unspoken token can
    only improve text/audio agreement.
    """
    text = unicodedata.normalize("NFC", raw)
    removed = 0
    if strip_numerals:
        removed = len(DIGIT_TOKEN.findall(text))
        text = DIGIT_TOKEN.sub(" ", text)
    return " ".join(text.split()), removed


def trim_wav(src: Path, dst: Path, begin_s: float, end_s: float,
             pad_s: float, sample_rate: int) -> float:
    """Write src[begin-pad : end+pad] to dst. Returns the written duration."""
    with wave.open(str(src), "rb") as w:
        if w.getnchannels() != EXPECTED_CHANNELS or w.getsampwidth() != EXPECTED_SAMPWIDTH:
            raise ValueError(
                f"{src.name}: expected mono 16-bit, got "
                f"{w.getnchannels()}ch/{w.getsampwidth() * 8}-bit"
            )
        if w.getframerate() != sample_rate:
            raise ValueError(
                f"{src.name}: expected {sample_rate} Hz, got {w.getframerate()} Hz"
            )
        n_frames = w.getnframes()
        start = max(0, int((begin_s - pad_s) * sample_rate))
        stop = min(n_frames, int((end_s + pad_s) * sample_rate))
        if stop <= start:
            raise ValueError(f"{src.name}: empty trim range {start}:{stop}")
        w.setpos(start)
        frames = w.readframes(stop - start)

    with wave.open(str(dst), "wb") as out:
        out.setnchannels(EXPECTED_CHANNELS)
        out.setsampwidth(EXPECTED_SAMPWIDTH)
        out.setframerate(sample_rate)
        out.writeframes(frames)

    return (stop - start) / sample_rate


def choose_val_chapters(chapters, cfg):
    """Explicit list wins; otherwise sample chapters (seeded) to hit the fraction."""
    explicit = cfg.get("val_chapters")
    if explicit:
        unknown = set(explicit) - set(chapters)
        if unknown:
            raise ValueError(f"val_chapters not present in the QC report: {sorted(unknown)}")
        return set(explicit)

    fraction = cfg.get("val_fraction", 0.1)
    ordered = sorted(chapters)
    n_val = max(1, round(len(ordered) * fraction))
    return set(random.Random(SEED).sample(ordered, n_val))


def main(config_path: str) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))

    sample_rate = cfg.get("sample_rate", 16000)
    pad_s = cfg.get("trim_pad_s", 0.1)
    out_dir = Path(cfg["output_dir"])
    # "validation" (not "val") — that is the directory name the HuggingFace
    # audiofolder loader recognises as a split.
    split_dirs = {"train": out_dir / "train", "validation": out_dir / "validation"}

    if out_dir.exists() and cfg.get("clean", True):
        log(f"clearing existing {out_dir}")
        shutil.rmtree(out_dir)
    for path in split_dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    with Path(cfg["qc_report"]).open(encoding="utf-8") as f:
        passed = [r for r in csv.DictReader(f) if r["qc_pass"] == "True"]
    log(f"QC-passed segments       : {len(passed)}")
    if not passed:
        raise SystemExit("nothing passed QC — check thresholds in the Phase 4 config")

    chapters = {r["chapter"] for r in passed}
    val_chapters = choose_val_chapters(chapters, cfg)
    log(f"chapters                 : {len(chapters)} "
        f"({len(val_chapters)} held out for validation)")
    log(f"validation chapters      : {', '.join(sorted(val_chapters))}")

    strip_numerals = cfg.get("strip_unspoken_numerals", True)
    min_words = cfg.get("min_words", 3)

    rows, skipped = [], []
    numeral_tokens = 0
    numeral_segments = 0
    for rec in sorted(passed, key=lambda r: r["segment"]):
        src_wav = Path(rec["wav_path"])
        src_txt = Path(rec["txt_path"])
        if not src_wav.is_file() or not src_txt.is_file():
            skipped.append((rec["segment"], "source file missing"))
            continue

        text, removed = normalise_text(src_txt.read_text(encoding="utf-8"), strip_numerals)
        if removed:
            numeral_tokens += removed
            numeral_segments += 1
        if not text:
            skipped.append((rec["segment"], "empty transcript"))
            continue
        if len(text.split()) < min_words:
            skipped.append((rec["segment"], f"under {min_words} words after cleaning"))
            continue

        split = "validation" if rec["chapter"] in val_chapters else "train"
        dst_wav = split_dirs[split] / f"{rec['segment']}.wav"
        try:
            duration = trim_wav(
                src_wav, dst_wav,
                float(rec["speech_begin_s"]), float(rec["speech_end_s"]),
                pad_s, sample_rate,
            )
        except ValueError as exc:
            skipped.append((rec["segment"], str(exc)))
            continue

        rows.append({
            "file_name": f"{rec['segment']}.wav",   # relative to its split dir
            "text": text,
            "segment": rec["segment"],
            "chapter": rec["chapter"],
            "book": rec["book"],
            "duration_s": round(duration, 3),
            "split": split,
        })

    if strip_numerals:
        log(f"unspoken numerals cut    : {numeral_tokens} tokens "
            f"in {numeral_segments} segments")

    if skipped:
        log(f"skipped while writing    : {len(skipped)}")
        for segment, reason in skipped[:10]:
            log(f"    {segment}: {reason}")

    # The top-level metadata.csv is the full manifest (both splits, extra
    # provenance columns) for our own analysis; the per-split ones are the
    # plain two-column form the audiofolder loader expects.
    full_fields = ["file_name", "text", "segment", "chapter", "book", "duration_s", "split"]
    with (out_dir / "metadata.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=full_fields)
        writer.writeheader()
        writer.writerows(rows)

    for split, path in split_dirs.items():
        subset = [r for r in rows if r["split"] == split]
        with (path / "metadata.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["file_name", "text"], extrasaction="ignore")
            writer.writeheader()
            writer.writerows(subset)

    # ---- stats -----------------------------------------------------------
    def report(label, subset):
        if not subset:
            log(f"{label:<10} EMPTY")
            return
        hours = sum(r["duration_s"] for r in subset) / 3600
        durations = sorted(r["duration_s"] for r in subset)
        tokens = [w for r in subset for w in r["text"].split()]
        log(f"{label:<10} {len(subset):>5} clips  {hours:>5.2f}h  "
            f"median {durations[len(durations) // 2]:>5.2f}s  "
            f"{len(tokens):>6} tokens  {len(set(tokens)):>5} types")

    log("-" * 72)
    report("total", rows)
    for split in split_dirs:
        report(split, [r for r in rows if r["split"] == split])

    chars = Counter(c for r in rows for c in r["text"] if not c.isspace())
    zwj = chars.get("‍", 0)
    log(f"distinct characters      : {len(chars)} (ZWJ occurrences: {zwj})")

    charset_path = out_dir / "charset.txt"
    charset_path.write_text(
        "\n".join(f"U+{ord(c):04X}\t{n}" for c, n in chars.most_common()),
        encoding="utf-8",
    )
    log(f"dataset written          : {out_dir}")
    log(f"  metadata.csv, charset.txt, train/, validation/")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/build_tts_dataset.py <config.yaml>")
        sys.exit(1)
    main(sys.argv[1])
