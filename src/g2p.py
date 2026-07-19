"""
g2p.py — rule-based grapheme-to-phoneme converter for Devanagari-script languages.

Language-specific mappings live in a YAML table (e.g. configs/g2p_sunuwar.yaml);
this module contains no Sunuwar-specific logic, only the akshara-segmentation
algorithm, so a new language just needs a new table file.

Usage: python src/g2p.py configs/g2p_sunuwar.yaml
"""

import sys
import unicodedata
from pathlib import Path

import yaml

INHERENT_VOWEL = "a"


class G2PTable:
    def __init__(self, cfg: dict):
        self.virama = cfg["virama"]
        self.zwj = cfg["zwj"]
        self.nukta = cfg["nukta"]
        self.nukta_suffix = cfg["nukta_suffix"]
        self.nasal_marks = set(cfg["nasal_marks"])
        self.nasal_suffix = cfg["nasal_suffix"]
        self.visarga = cfg["visarga"]
        self.visarga_phone = cfg["visarga_phone"]
        self.independent_vowels = cfg["independent_vowels"]
        self.matras = cfg["matras"]
        self.consonants = cfg["consonants"]
        self.boundary_strip_chars = cfg["boundary_strip_chars"]

        self.vowel_phones = (
            set(self.independent_vowels.values())
            | set(self.matras.values())
            | {INHERENT_VOWEL}
        )


def word_to_phones(word: str, table: G2PTable) -> tuple:
    """Convert one word to a phone list. Returns (phones, unmapped_chars).

    phones is None if any character in the word had no mapping — the caller
    decides what to do with unmapped tokens, this function never guesses.
    unmapped_chars is a list of (index, char) for every character that
    couldn't be resolved (collected in full, not just the first one).
    """
    phones = []
    unmapped = []
    pending = None  # consonant phone awaiting vowel resolution

    i = 0
    n = len(word)
    while i < n:
        ch = word[i]

        if ch == table.zwj:
            i += 1
            continue

        if ch == table.nukta:
            if pending is not None:
                pending = pending + table.nukta_suffix
            else:
                unmapped.append((i, ch))
            i += 1
            continue

        if ch == table.virama:
            if pending is not None:
                phones.append(pending)
                pending = None
            else:
                unmapped.append((i, ch))
            i += 1
            continue

        if ch in table.consonants:
            if pending is not None:
                phones.append(pending)
                phones.append(INHERENT_VOWEL)
            pending = table.consonants[ch]
            i += 1
            continue

        if ch in table.matras:
            if pending is not None:
                phones.append(pending)
                phones.append(table.matras[ch])
                pending = None
            else:
                phones.append(table.matras[ch])
            i += 1
            continue

        if ch in table.independent_vowels:
            if pending is not None:
                phones.append(pending)
                phones.append(INHERENT_VOWEL)
                pending = None
            phones.append(table.independent_vowels[ch])
            i += 1
            continue

        if ch in table.nasal_marks:
            if pending is not None:
                phones.append(pending)
                phones.append(INHERENT_VOWEL)
                pending = None
            if phones and phones[-1] in table.vowel_phones:
                phones[-1] = phones[-1] + table.nasal_suffix
            else:
                phones.append(table.nasal_suffix)
            i += 1
            continue

        if ch == table.visarga:
            if pending is not None:
                phones.append(pending)
                phones.append(INHERENT_VOWEL)
                pending = None
            phones.append(table.visarga_phone)
            i += 1
            continue

        unmapped.append((i, ch))
        i += 1

    if pending is not None:
        phones.append(pending)
        phones.append(INHERENT_VOWEL)

    if unmapped:
        return None, unmapped
    return phones, []


def extract_vocab(vocab_dir: str, table: G2PTable) -> list:
    """Read every .txt in vocab_dir, tokenise on whitespace, NFC-normalise,
    strip boundary punctuation, return the sorted set of unique tokens."""
    vocab = set()
    for path in sorted(Path(vocab_dir).glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        for raw_token in text.split():
            token = unicodedata.normalize("NFC", raw_token)
            token = token.strip(table.boundary_strip_chars)
            if token:
                vocab.add(token)
    return sorted(vocab)


def build_dictionary(vocab: list, table: G2PTable) -> tuple:
    """Returns (entries, unmapped_report) where entries is a list of
    (word, phones) and unmapped_report is a list of (word, [(idx, char), ...])."""
    entries = []
    unmapped_report = []
    for word in vocab:
        phones, unmapped = word_to_phones(word, table)
        if phones is None:
            unmapped_report.append((word, unmapped))
        else:
            entries.append((word, phones))
    return entries, unmapped_report


def _run_tests(table: G2PTable) -> None:
    cases = [
        # (word, expected_phones, note)
        ("अन्‍काल", ["a", "n", "k", "aa", "l", "a"],
         "virama (cancels न's vowel) + ZWJ (skipped, no phone)"),
        ("अब्राहाम", ["a", "b", "r", "aa", "h", "aa", "m", "a"],
         "consonant cluster ब्र via virama, trailing bare म -> inherent a"),
        ("लोव़", ["l", "o", "vQ", "a"],
         "nukta modifies व -> distinct phone 'vQ', not plain व"),
        ("कं", ["k", "aN"],
         "anusvara nasalises the preceding (inherent) vowel"),
        ("अः", ["a", "H"],
         "visarga is its own trailing phone, not merged into the vowel"),
    ]
    for word, expected, note in cases:
        phones, unmapped = word_to_phones(word, table)
        assert unmapped == [], f"{word!r} had unmapped chars: {unmapped}"
        assert phones == expected, (
            f"{word!r} ({note}): expected {expected}, got {phones}"
        )
    print(f"Assert passed: {len(cases)} hand-traced G2P test cases")


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/g2p_sunuwar.yaml"
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    table = G2PTable(cfg)
    _run_tests(table)

    vocab = extract_vocab(cfg["vocab_source_dir"], table)
    entries, unmapped_report = build_dictionary(vocab, table)

    out_path = Path(cfg["output_dict_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for word, phones in entries:
            f.write(f"{word}\t{' '.join(phones)}\n")

    unmapped_path = Path(cfg["unmapped_report_path"])
    with open(unmapped_path, "w", encoding="utf-8") as f:
        for word, unmapped_chars in unmapped_report:
            chars_desc = ", ".join(f"U+{ord(c):04X}({c})@{i}" for i, c in unmapped_chars)
            f.write(f"{word}\t{chars_desc}\n")

    all_phones = set()
    for _, phones in entries:
        all_phones.update(phones)

    total = len(vocab)
    mapped = len(entries)
    print()
    print("--- G2P dictionary build summary ---")
    print(f"Vocabulary source:     {cfg['vocab_source_dir']}")
    print(f"Total unique tokens:   {total}")
    print(f"Mapped successfully:   {mapped} ({100 * mapped / total:.1f}%)")
    print(f"Unmapped (see report): {len(unmapped_report)} ({100 * len(unmapped_report) / total:.1f}%)")
    print(f"Distinct phones used:  {len(all_phones)}")
    print(f"Dictionary written to: {out_path}")
    print(f"Unmapped report:       {unmapped_path}")

    control_chars = {table.virama, table.zwj, table.nukta}
    leaked = control_chars & all_phones
    assert not leaked, f"control characters leaked into phone inventory: {leaked}"


if __name__ == "__main__":
    main()
