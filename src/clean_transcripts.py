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
#
# The book-name class MUST include ZWJ (U+200D). Sunuwar book names are full of
# conjuncts — "कोरिन्‍थी" has a ZWJ in the middle — and without it the match
# started only *after* the ZWJ, so "१ कोरिन्‍थी १:१७" lost "थी १:१७" and left
# the unspoken fragment "१कोरिन्‍" behind in the transcript. That fragment (and
# bare verse numerals like "१७") is not in the audio and has no entry in the
# G2P dictionary, so MFA saw it as OOV.
#
# The book-name run must also be *bounded*. The original `[ऀ-ॿ\s]+` was greedy
# over whitespace, so it would happily swallow whole preceding sentences up to
# the nearest "N:M"; ZWJ happened to break those runs early and accidentally
# limited the damage. Now that ZWJ is allowed, the length has to be capped
# explicitly: at most three name words, no digits inside them.
# Devanagari letters and marks only: no digits (U+0966-096F) and — critically —
# no danda/double-danda (U+0964-0965). Including the danda would let a book
# name reach back across a sentence boundary and swallow the end of the
# preceding sentence, which would also change the danda count Phase 2 uses to
# decide how many segments a chapter has.
_LETTER = r"[ऀ-ॣ॰-ॿ‍]"
_REF = r"[०-९\d]+[:：][०-९\d]+"

CROSS_REF = re.compile(
    r"(?:[०-९\d]+[ ]+)?"          # optional book ordinal, e.g. the "१" of "१ कोरिन्‍थी"
    rf"(?:{_LETTER}+[ ]+){{0,3}}"  # up to three book-name words
    rf"{_REF}"                     # chapter:verse
    rf"(?:[;,][ ]*(?:{_REF}|[०-९\d]+))*"   # ", १८" / "; ३:४" continuations
)

# Any token still carrying a Devanagari or ASCII digit after cross-reference
# removal. The read-aloud source has no verse numbers in its running text, so
# every such token is a mangled reference remnant, never spoken audio — and
# none of them are in models/mfa_dict.dict, so leaving them in guarantees an
# OOV at alignment time.
DIGIT_TOKEN = re.compile(r"\S*[०-९\d]\S*")

# Book-name line: 1–3 Devanagari words then a period (with optional trailing space)
# e.g. "मत्ती." or "मर्कूस."
BOOK_LINE = re.compile(r"^[ऀ-ॿ‍]+(?:\s+[ऀ-ॿ‍]+){0,2}\.\s*$")

# Chapter-number line: ASCII or Devanagari digits followed by a period
# e.g. "1." or "२."
CHAPTER_LINE = re.compile(r"^[०-९\d]+\.\s*$")

MULTI_SPACE = re.compile(r" {2,}")


def clean_line(text: str) -> tuple[str, int]:
    """Return (cleaned text, number of digit-remnant tokens dropped)."""
    # NFC normalise
    text = unicodedata.normalize("NFC", text)
    # Remove cross-references
    text = CROSS_REF.sub(" ", text)
    # Keep only Devanagari block, ZWJ, and whitespace
    text = "".join(ch if DEVANAGARI.match(ch) else " " for ch in text)
    # Drop leftover numeral fragments the cross-reference pass didn't consume
    dropped = len(DIGIT_TOKEN.findall(text))
    text = DIGIT_TOKEN.sub(" ", text)
    # Collapse spaces
    text = MULTI_SPACE.sub(" ", text)
    return text.strip(), dropped


def count_words(text: str) -> int:
    return len(text.split())


def process_file(src: Path, dst_dir: Path, intro_dir: Path) -> tuple[int, int, int, bool]:
    """Return (words_before, words_after, digit_tokens_dropped, book_name_recovered)."""
    raw_lines = src.read_text(encoding="utf-8-sig").splitlines()

    before_words = 0
    dropped_digits = 0
    cleaned_parts = []
    book_name = None

    for raw in raw_lines:
        # Files are tab-separated: line_num TAB text
        parts = raw.split("\t", 1)
        text = parts[1] if len(parts) == 2 else parts[0]

        before_words += count_words(text)
        t = text.strip()

        if BOOK_LINE.match(t):
            # The narrator reads this aloud before the chapter's verses.
            # Previously discarded along with the chapter-number line below;
            # now kept as a real (not guessed) intro transcript — see
            # segment_from_pauses.py's book-intro handling.
            if book_name is None:
                book_name = t.rstrip("।.॥ ").strip()
            continue

        if CHAPTER_LINE.match(t):
            # Also spoken aloud, but only the bare digit is written here —
            # the actual Sunuwar number word isn't recoverable from this
            # source, so unlike the book name we can't safely reconstruct a
            # transcript for it. Still dropped.
            continue

        cleaned, dropped = clean_line(text)
        dropped_digits += dropped
        if cleaned:
            cleaned_parts.append(cleaned)

    result = " ".join(cleaned_parts)
    result = MULTI_SPACE.sub(" ", result).strip()

    after_words = count_words(result)

    dst = dst_dir / src.name
    dst.write_text(result, encoding="utf-8")

    if book_name:
        (intro_dir / src.name).write_text(book_name, encoding="utf-8")

    return before_words, after_words, dropped_digits, bool(book_name)


def main(config_path: str) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    input_dir = Path(cfg["input_dir"])
    output_dir = Path(cfg["output_dir"])
    intro_dir = Path(cfg["intro_output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    intro_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.txt"))
    if not files:
        print(f"No .txt files found in {input_dir}")
        sys.exit(1)

    total_before = 0
    total_after = 0
    total_digits = 0
    total_intros = 0
    show_detail = 3  # print first N files verbosely

    for i, src in enumerate(files):
        before, after, digits, got_intro = process_file(src, output_dir, intro_dir)
        total_before += before
        total_after += after
        total_digits += digits
        total_intros += int(got_intro)
        if i < show_detail:
            print(f"{src.name}: {before} words -> {after} words "
                  f"({digits} numeral remnants dropped, "
                  f"intro {'recovered' if got_intro else 'MISSING'})")

    print()
    print(f"Total files          : {len(files)}")
    print(f"Words before         : {total_before}")
    print(f"Words after          : {total_after}")
    print(f"Numeral remnants cut : {total_digits}")
    print(f"Book intros recovered: {total_intros} / {len(files)}")
    print(f"Reduction            : {100*(1 - total_after/total_before):.1f}%" if total_before else "")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/clean_transcripts.py configs/clean_transcripts.yaml")
        sys.exit(1)
    main(sys.argv[1])
