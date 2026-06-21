"""
clean_transcripts.py — clean transcript .txt files in pairs/ for MFA input.

Usage: python src/clean_transcripts.py configs/clean_transcripts.yaml
"""

import re
import sys
import unicodedata
from pathlib import Path

import yaml

ZWJ = "‍"
DEVANAGARI = re.compile(r"[ऀ-ॿ‍\s]")

# Cross-reference pattern: Devanagari/space chars followed by digits (Devanagari or ASCII),
# a colon, then more digits.  Matches things like "उत्‍पत्ती २२:१८" or "मत्ती ३:१"
CROSS_REF = re.compile(
    r"[ऀ-ॿ\s]+"          # book name (Devanagari words + spaces)
    r"[०-९\d]+"          # chapter digits (Devanagari or ASCII)
    r"[:：]"                        # colon separator
    r"[०-९\d]+"          # verse digits
    r"(?:[;,\s]*[ऀ-ॿ\s]*[०-९\d]+[:：][०-९\d]+)*"
    # optional additional refs after semicolons
)

# Book-name line: 1–3 Devanagari words then a period (with optional trailing space)
# e.g. "मत्ती." or "मर्कूस."
BOOK_LINE = re.compile(r"^[ऀ-ॿ‍]+(?:\s+[ऀ-ॿ‍]+){0,2}\.\s*$")

# Chapter-number line: ASCII or Devanagari digits followed by a period
# e.g. "1." or "२."
CHAPTER_LINE = re.compile(r"^[०-९\d]+\.\s*$")

MULTI_SPACE = re.compile(r" {2,}")


def is_skip_line(text: str) -> bool:
    t = text.strip()
    return bool(BOOK_LINE.match(t) or CHAPTER_LINE.match(t))


def clean_line(text: str) -> str:
    # NFC normalise
    text = unicodedata.normalize("NFC", text)
    # Remove cross-references
    text = CROSS_REF.sub(" ", text)
    # Keep only Devanagari block, ZWJ, and whitespace
    text = "".join(ch if DEVANAGARI.match(ch) else " " for ch in text)
    # Collapse spaces
    text = MULTI_SPACE.sub(" ", text)
    return text.strip()


def count_words(text: str) -> int:
    return len(text.split())


def process_file(src: Path, dst_dir: Path) -> tuple[int, int]:
    """Return (words_before, words_after)."""
    raw_lines = src.read_text(encoding="utf-8-sig").splitlines()

    before_words = 0
    cleaned_parts = []

    for raw in raw_lines:
        # Files are tab-separated: line_num TAB text
        parts = raw.split("\t", 1)
        text = parts[1] if len(parts) == 2 else parts[0]

        before_words += count_words(text)

        if is_skip_line(text):
            continue

        cleaned = clean_line(text)
        if cleaned:
            cleaned_parts.append(cleaned)

    result = " ".join(cleaned_parts)
    result = MULTI_SPACE.sub(" ", result).strip()

    after_words = count_words(result)

    dst = dst_dir / src.name
    dst.write_text(result, encoding="utf-8")

    return before_words, after_words


def main(config_path: str) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    input_dir = Path(cfg["input_dir"])
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.txt"))
    if not files:
        print(f"No .txt files found in {input_dir}")
        sys.exit(1)

    total_before = 0
    total_after = 0
    show_detail = 3  # print first N files verbosely

    for i, src in enumerate(files):
        before, after = process_file(src, output_dir)
        total_before += before
        total_after += after
        if i < show_detail:
            print(f"{src.name}: {before} words -> {after} words")

    print()
    print(f"Total files : {len(files)}")
    print(f"Words before: {total_before}")
    print(f"Words after : {total_after}")
    print(f"Reduction   : {100*(1 - total_after/total_before):.1f}%" if total_before else "")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/clean_transcripts.py configs/clean_transcripts.yaml")
        sys.exit(1)
    main(sys.argv[1])
