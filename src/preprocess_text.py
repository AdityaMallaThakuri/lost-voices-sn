# Data source: Sunuwar Bible (suzBl), © 2011 Wycliffe Bible Translators, Inc.
# Licence: CC BY-NC-ND 4.0 — non-commercial research use only
# Do not redistribute raw text. Models trained on this data may be released.

import re
import statistics
import sys
import unicodedata
import zipfile
from pathlib import Path

import yaml

_DISCARD_PREFIXES = (
    r'\id', r'\h', r'\toc1', r'\toc2', r'\toc3', r'\mt1',
    r'\c ', r'\s1', r'\r ', r'\ip', r'\io1', r'\io2',
)


def clean_usfm_line(line: str) -> str:
    """Return cleaned Sunuwar text from one USFM line, or '' if the line should be discarded."""
    text = line.strip()

    # 1. Discard full-line markers
    for prefix in _DISCARD_PREFIXES:
        if text.startswith(prefix):
            return ''

    # 2. Strip \em...\em* blocks including content (cross-refs and Nepali footnotes)
    text = re.sub(r'\\em.*?\\em\*', '', text, flags=re.DOTALL)

    # 3. Strip \bd* and \em* closing markers
    text = re.sub(r'\\bd\*', '', text)
    text = re.sub(r'\\em\*', '', text)

    # 4. Strip \bd opening marker (keep enclosed text)
    text = re.sub(r'\\bd\b', '', text)

    # 5. Strip \v N verse marker at line start (keep text after)
    text = re.sub(r'^\\v \d+\s*', '', text)

    # 6. Strip \p and \m markers at line start (keep any text that follows)
    text = re.sub(r'^\\[pm]\s*', '', text)

    # 7. Unicode NFC normalisation
    text = unicodedata.normalize('NFC', text)

    # 8. Strip leading and trailing whitespace
    return text.strip()


def segment_and_filter(text: str, min_tokens: int = 3) -> list[str]:
    """Split on Devanagari danda/double-danda and discard segments shorter than min_tokens."""
    segments = re.split(r'(?<=[।॥])', text)
    result = []
    for seg in segments:
        seg = seg.strip()
        if len(seg.split()) >= min_tokens:
            result.append(seg)
    return result


def passes_script_filter(text: str, max_foreign_ratio: float = 0.20) -> bool:
    """Return True if the line is mostly Devanagari."""
    non_space = [c for c in text if c not in ' \t\n']
    if not non_space:
        return False
    foreign = [c for c in non_space
               if not (0x0900 <= ord(c) <= 0x097F)
               and ord(c) != 0x200D
               and ord(c) not in (0x0964, 0x0965)
               and ord(c) > 127]  # ignore ASCII punctuation
    return len(foreign) / len(non_space) <= max_foreign_ratio


def print_corpus_stats(sentences: list[str]) -> None:
    """Print corpus statistics to stdout."""
    token_counts = [len(s.split()) for s in sentences]
    all_tokens = [tok for s in sentences for tok in s.split()]

    total_sents = len(sentences)
    total_tokens = len(all_tokens)
    unique_types = len(set(all_tokens))
    mean_len = round(sum(token_counts) / len(token_counts), 2) if token_counts else 0.0
    median_len = statistics.median(token_counts) if token_counts else 0

    all_chars = ''.join(sentences)
    non_space = [c for c in all_chars if c not in ' \t\n']
    deva_chars = [c for c in non_space
                  if (0x0900 <= ord(c) <= 0x097F) or ord(c) == 0x200D]
    deva_pct = round(len(deva_chars) / len(non_space) * 100, 2) if non_space else 0.0

    print(f'Total sentences:        {total_sents}')
    print(f'Total tokens:           {total_tokens}')
    print(f'Unique word types:      {unique_types}')
    print(f'Mean sentence length:   {mean_len} tokens')
    print(f'Median sentence length: {median_len} tokens')
    print(f'Devanagari characters:  {deva_pct}%')


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/preprocess.yaml'
    with open(config_path, encoding='utf-8') as fh:
        cfg = yaml.safe_load(fh)

    zip_path        = cfg['zip_path']
    output_path     = cfg['output_path']
    min_tokens      = cfg['min_tokens']
    max_foreign     = cfg['max_foreign_ratio']
    nt_book_codes   = cfg['nt_book_codes']

    all_sentences: list[str] = []

    with zipfile.ZipFile(zip_path) as zf:
        nt_names = [
            n for n in zf.namelist()
            if not n.endswith('/')
            and any(f'-{code}suzBl' in n for code in nt_book_codes)
        ]
        print(f'NT files found in zip: {len(nt_names)}')

        for name in nt_names:
            with zf.open(name) as fh:
                for raw_line in fh:
                    cleaned = clean_usfm_line(raw_line.decode('utf-8'))
                    if not cleaned:
                        continue
                    for seg in segment_and_filter(cleaned, min_tokens=min_tokens):
                        if passes_script_filter(seg, max_foreign_ratio=max_foreign):
                            all_sentences.append(seg)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as fh:
        for sent in all_sentences:
            fh.write(sent + '\n')
    print(f'Output written to: {output_path}')
    print()
    print_corpus_stats(all_sentences)


if __name__ == '__main__':
    main()
