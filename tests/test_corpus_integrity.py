# Data source: Sunuwar Bible (suzBl), © 2011 Wycliffe Bible Translators, Inc.
# Licence: CC BY-NC-ND 4.0 — non-commercial research use only
# Do not redistribute raw text. This file asserts counts only — it must never
# contain, print or write out corpus text.
"""Pin the corpus figures recorded in results/corpus_stats.json.

Two facts are being held still here:

1. The committed corpus is **not** the snapshot the released models were
   trained on. `results/corpus_stats.json` and the committed split both
   describe 15,738 sentences; the file on disk has 15,737.
2. The random sentence-level split leaks. 151 sentences are exact duplicates
   and 25 distinct sentences appear verbatim on both sides of the split.

Neither is repaired — repairing them means regenerating the corpus, which
invalidates every trained model. Recording them means a future edit that
changes either one fails loudly instead of quietly restating a wrong number.

These tests skip when `data/processed/` is absent: the licence work untracks
the corpus, so a fresh clone legitimately will not have it.
"""

import json
import unicodedata
from collections import Counter

import pytest

from conftest import DATA_PROCESSED, DATA_RAW, RESULTS

RAW_CORPUS = DATA_RAW / "sunuwar_nt_raw.txt"
TRAIN = DATA_PROCESSED / "train.txt"
TEST = DATA_PROCESSED / "test.txt"

SKIP_REASON = (
    "corpus not present — data/ is CC BY-NC-ND and is untracked in clones "
    "that exclude it"
)


def _read_sentences(path) -> list[str]:
    """Non-empty lines, NFC-normalised — the definition preprocess_text.py and
    split_corpus.py both use."""
    with open(path, encoding="utf-8") as fh:
        return [
            unicodedata.normalize("NFC", line.rstrip("\n"))
            for line in fh
            if line.strip()
        ]


@pytest.fixture(scope="module")
def stats() -> dict:
    with open(RESULTS / "corpus_stats.json", encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def raw_sentences() -> list[str]:
    if not RAW_CORPUS.exists():
        pytest.skip(SKIP_REASON)
    return _read_sentences(RAW_CORPUS)


@pytest.fixture(scope="module")
def split_sentences() -> tuple[list[str], list[str]]:
    if not (TRAIN.exists() and TEST.exists()):
        pytest.skip(SKIP_REASON)
    return _read_sentences(TRAIN), _read_sentences(TEST)


# ---------------------------------------------------------------------------
# The committed corpus, measured
# ---------------------------------------------------------------------------

def test_committed_corpus_sentence_count(stats, raw_sentences):
    assert len(raw_sentences) == stats["committed_corpus_sentences"] == 15_737


def test_committed_corpus_token_count(stats, raw_sentences):
    tokens = [tok for s in raw_sentences for tok in s.split()]
    assert len(tokens) == stats["committed_corpus_tokens"] == 179_915


def test_committed_corpus_type_count(stats, raw_sentences):
    types = {tok for s in raw_sentences for tok in s.split()}
    assert len(types) == stats["committed_corpus_types"] == 13_123


# ---------------------------------------------------------------------------
# The drift: split + published stats agree with each other, not with the file
# ---------------------------------------------------------------------------

def test_split_sizes_match_the_published_figures(stats, split_sentences):
    train, test = split_sentences
    assert len(train) == stats["train_sentences"] == 14_164
    assert len(test) == stats["test_sentences"] == 1_574


def test_split_total_exceeds_the_committed_corpus_by_one_sentence(
    stats, raw_sentences, split_sentences
):
    """The heart of the drift. If this ever passes trivially (0 difference),
    somebody regenerated the corpus and every number in results/ must be
    restated."""
    train, test = split_sentences
    split_total = len(train) + len(test)

    assert split_total == stats["total_sentences"] == 15_738
    assert split_total - len(raw_sentences) == 1
    assert stats["drift_note"], "the drift must stay documented, not just measured"


# ---------------------------------------------------------------------------
# The leak
# ---------------------------------------------------------------------------

def test_exact_duplicate_sentences_in_the_corpus(stats, raw_sentences):
    counts = Counter(raw_sentences)
    duplicates = sum(count - 1 for count in counts.values() if count > 1)

    assert duplicates == stats["corpus_exact_duplicates"] == 151


def test_train_test_exact_overlap(stats, split_sentences):
    """25 distinct sentences sit on both sides of a supposedly held-out split."""
    train, test = split_sentences
    overlap = set(test) & set(train)

    assert len(overlap) == stats["train_test_exact_overlap"] == 25

    # …and they cover 26 test rows, because one of them is itself duplicated
    # inside test.txt. 26/1574 = 1.65% of the held-out set is not held out.
    contaminated_rows = sum(1 for s in test if s in set(train))
    assert contaminated_rows == 26
    assert contaminated_rows / len(test) == pytest.approx(0.0165, abs=0.0005)
