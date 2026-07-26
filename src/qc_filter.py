"""
qc_filter.py — Phase 4 of the TTS roadmap: quality-control filtering of
MFA-aligned segments before they become TTS training data.

Consumes, per alignment batch:
  * MFA's own per-utterance `alignment_analysis.csv` (written into the MFA
    temporary directory under <corpus_name>/<stage>_ali/) — this is where
    overall_log_likelihood, phone_duration_deviation and snr come from. No
    custom acoustic scoring is needed; these are MFA's built-in outputs.
  * The TextGrids MFA emitted. A missing TextGrid is how a failed alignment
    shows up (MFA still writes an analysis row, but with empty likelihoods).
    The "words" tier also gives the true speech span, so we can measure
    leading/trailing/internal silence.
  * The Phase 2 `segmentation_report.csv` duration-outlier flag, carried
    through as advisory context.

Emits one row per segment with every metric plus a pass/fail verdict and the
list of reasons it failed, so thresholds can be re-tuned without re-running
alignment. Nothing is deleted here — Phase 5 (build_tts_dataset.py) is what
reads the verdicts and materialises the dataset.

Usage: python src/qc_filter.py configs/qc_filter.yaml
"""

import csv
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import yaml

SEED = 42
random.seed(SEED)


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


# --------------------------------------------------------------------------
# TextGrid parsing
# --------------------------------------------------------------------------

def parse_words_tier(path: Path):
    """Return [(xmin, xmax, text), ...] for the 'words' tier of a TextGrid.

    MFA writes long-form ooTextFile TextGrids with one `key = value` per line,
    so a small line-oriented parser is enough — no praat library needed, which
    keeps this runnable in a bare Colab runtime.
    """
    intervals = []
    in_words = False
    xmin = xmax = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith('name = '):
            in_words = line == 'name = "words"'
            continue
        if not in_words:
            continue
        if line.startswith("xmin = "):
            xmin = float(line[len("xmin = "):])
        elif line.startswith("xmax = "):
            xmax = float(line[len("xmax = "):])
        elif line.startswith("text = "):
            text = line[len("text = "):].strip().strip('"')
            if xmin is not None and xmax is not None:
                intervals.append((xmin, xmax, text))
            xmin = xmax = None
    return intervals


def silence_metrics(intervals, total_duration):
    """Leading / trailing / longest-internal silence, and the speech span.

    Returns None if the tier has no spoken word at all (a degenerate
    alignment we always want to drop).
    """
    spoken = [(a, b) for a, b, t in intervals if t]
    if not spoken:
        return None
    speech_begin = spoken[0][0]
    speech_end = spoken[-1][1]
    internal = 0.0
    for (_, prev_end), (next_begin, _) in zip(spoken, spoken[1:]):
        internal = max(internal, next_begin - prev_end)
    return {
        "speech_begin_s": round(speech_begin, 3),
        "speech_end_s": round(speech_end, 3),
        "speech_duration_s": round(speech_end - speech_begin, 3),
        "leading_silence_s": round(speech_begin, 3),
        "trailing_silence_s": round(max(0.0, total_duration - speech_end), 3),
        "max_internal_silence_s": round(internal, 3),
        "n_words_aligned": len(spoken),
    }


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def _float_or_none(value):
    if value in (None, "", "nan", "NaN"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load_segmentation_report(path: Path):
    rows = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[row["segment"]] = row
    return rows


def collect_batch(batch: dict, segrep: dict):
    """One dict per segment in this batch, with all raw metrics joined."""
    analysis_csv = Path(batch["analysis_csv"])
    textgrid_dir = Path(batch["textgrid_dir"])
    corpus_dir = Path(batch["corpus_dir"])

    if not analysis_csv.is_file():
        raise FileNotFoundError(f"alignment analysis not found: {analysis_csv}")

    textgrids = {p.stem: p for p in textgrid_dir.rglob("*.TextGrid")}
    wavs = {p.stem: p for p in corpus_dir.rglob("*.wav")}
    texts = {p.stem: p for p in corpus_dir.rglob("*.txt")}

    out = []
    with analysis_csv.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            seg = row["file"]
            begin = _float_or_none(row.get("begin")) or 0.0
            end = _float_or_none(row.get("end")) or 0.0
            duration = round(end - begin, 3)

            rec = {
                "segment": seg,
                "batch": batch["name"],
                "chapter": segrep.get(seg, {}).get("chapter", seg.rsplit("_seg_", 1)[0]),
                "book": seg.split("_")[0],
                "duration_s": duration,
                "overall_log_likelihood": _float_or_none(row.get("overall_log_likelihood")),
                "speech_log_likelihood": _float_or_none(row.get("speech_log_likelihood")),
                "phone_duration_deviation": _float_or_none(row.get("phone_duration_deviation")),
                "snr": _float_or_none(row.get("snr")),
                "segmentation_confidence": segrep.get(seg, {}).get("confidence", ""),
                "segmentation_reason": segrep.get(seg, {}).get("reason", ""),
                "aligned": seg in textgrids,
                "wav_path": str(wavs[seg]) if seg in wavs else "",
                "txt_path": str(texts[seg]) if seg in texts else "",
                "textgrid_path": str(textgrids[seg]) if seg in textgrids else "",
                "speech_begin_s": None,
                "speech_end_s": None,
                "speech_duration_s": None,
                "leading_silence_s": None,
                "trailing_silence_s": None,
                "max_internal_silence_s": None,
                "n_words_aligned": None,
            }

            if rec["aligned"]:
                metrics = silence_metrics(parse_words_tier(textgrids[seg]), duration)
                if metrics is None:
                    rec["aligned"] = False  # TextGrid exists but holds no words
                else:
                    rec.update(metrics)

            out.append(rec)
    return out


# --------------------------------------------------------------------------
# Verdict
# --------------------------------------------------------------------------

def judge(rec: dict, th: dict):
    """Return the list of threshold violations for one segment ([] == pass)."""
    reasons = []

    if not rec["aligned"]:
        reasons.append("alignment_failed")
        return reasons  # no metrics to judge against

    if not rec["wav_path"] or not rec["txt_path"]:
        reasons.append("missing_source_files")

    def under(key, limit, label):
        value = rec[key]
        if limit is not None and value is not None and value < limit:
            reasons.append(label)

    def over(key, limit, label):
        value = rec[key]
        if limit is not None and value is not None and value > limit:
            reasons.append(label)

    # The post-trim speech span is what actually becomes training audio, so
    # the duration bounds are applied to that, not to the raw clip length.
    under("speech_duration_s", th.get("min_duration_s"), "too_short")
    over("speech_duration_s", th.get("max_duration_s"), "too_long")
    under("overall_log_likelihood", th.get("min_overall_log_likelihood"), "low_likelihood")
    over("phone_duration_deviation", th.get("max_phone_duration_deviation"), "phone_duration_outlier")
    under("snr", th.get("min_snr"), "low_snr")
    over("leading_silence_s", th.get("max_leading_silence_s"), "long_leading_silence")
    over("trailing_silence_s", th.get("max_trailing_silence_s"), "long_trailing_silence")
    over("max_internal_silence_s", th.get("max_internal_silence_s"), "long_internal_silence")
    under("n_words_aligned", th.get("min_words"), "too_few_words")

    if th.get("drop_low_confidence_segments") and rec["segmentation_confidence"] == "low":
        reasons.append("segmentation_low_confidence")

    return reasons


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

FIELDS = [
    "segment", "batch", "book", "chapter",
    "qc_pass", "qc_reasons",
    "duration_s", "speech_duration_s", "speech_begin_s", "speech_end_s",
    "leading_silence_s", "trailing_silence_s", "max_internal_silence_s",
    "n_words_aligned",
    "overall_log_likelihood", "speech_log_likelihood",
    "phone_duration_deviation", "snr",
    "aligned", "segmentation_confidence", "segmentation_reason",
    "wav_path", "txt_path", "textgrid_path",
]


def summarise(records):
    total = len(records)
    passed = [r for r in records if r["qc_pass"]]
    log(f"segments considered      : {total}")
    log(f"passed QC                : {len(passed)} ({len(passed)/total:.1%})")
    log(f"dropped                  : {total - len(passed)}")

    reason_counts = Counter()
    for r in records:
        for reason in r["qc_reasons"].split(";"):
            if reason:
                reason_counts[reason] += 1
    log("drop reasons (a segment can fail several):")
    for reason, n in reason_counts.most_common():
        log(f"    {reason:<28} {n:>5}  ({n/total:.1%})")

    kept_hours = sum(r["speech_duration_s"] or 0.0 for r in passed) / 3600
    all_hours = sum(r["duration_s"] for r in records) / 3600
    log(f"audio kept               : {kept_hours:.2f}h of {all_hours:.2f}h "
        f"({kept_hours/all_hours:.1%})")

    per_chapter = defaultdict(lambda: [0, 0])
    for r in records:
        per_chapter[r["chapter"]][0] += 1
        per_chapter[r["chapter"]][1] += bool(r["qc_pass"])
    log(f"chapters                 : {len(per_chapter)}")
    worst = sorted(per_chapter.items(), key=lambda kv: kv[1][1] / kv[1][0])[:5]
    log("lowest-yield chapters:")
    for chapter, (n, ok) in worst:
        log(f"    {chapter:<12} {ok:>3}/{n:<3} ({ok/n:.0%})")


def main(config_path: str) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    thresholds = cfg.get("thresholds", {})

    segrep = load_segmentation_report(Path(cfg["segmentation_report"]))
    log(f"segmentation report      : {len(segrep)} segments")

    records = []
    for batch in cfg["batches"]:
        batch_records = collect_batch(batch, segrep)
        log(f"batch {batch['name']:<20} {len(batch_records):>4} segments, "
            f"{sum(r['aligned'] for r in batch_records):>4} aligned")
        records.extend(batch_records)

    for rec in records:
        reasons = judge(rec, thresholds)
        rec["qc_pass"] = not reasons
        rec["qc_reasons"] = ";".join(reasons)

    out_path = Path(cfg["output_report"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    summarise(records)
    log(f"report written           : {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/qc_filter.py <config.yaml>")
        sys.exit(1)
    main(sys.argv[1])
