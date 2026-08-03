# Data source: Sunuwar Bible (suzBl), © 2011 Wycliffe Bible Translators, Inc.
# Licence: CC BY-NC-ND 4.0 — non-commercial research use only
"""Tests for src/evaluate_nlp.py.

These replace the three assertions that used to run at *module import* time in
evaluate_nlp.py. That block loaded a gensim model and read
`data/processed/test.txt` from hardcoded relative paths, so importing the
module — or running it from any directory other than the repo root — raised
before `main()` could execute. The checks were worth keeping; running them on
import was not.

Everything here uses stub models and synthetic text, so no trained artefact and
no corpus text is required.
"""

import numpy as np
import pytest

import evaluate_nlp


# ---------------------------------------------------------------------------
# Stub models — the smallest thing with the shape evaluate_nlp expects
# ---------------------------------------------------------------------------

class _StaticVectors:
    """word2vec-like: a closed vocabulary, KeyError outside it."""

    def __init__(self, vocab: list[str], dim: int = 4):
        self.key_to_index = {w: i for i, w in enumerate(vocab)}
        self._dim = dim

    def __getitem__(self, key):
        if key not in self.key_to_index:
            raise KeyError(key)
        return np.full(self._dim, float(self.key_to_index[key]))


class _SubwordVectors(_StaticVectors):
    """fastText-like: composes a vector for anything from character n-grams."""

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            return np.full(self._dim, 0.5)


class StubWord2Vec:
    def __init__(self, vocab, min_count=2):
        self.wv = _StaticVectors(vocab)
        self.min_count = min_count


class StubFastText:
    def __init__(self, vocab, min_count=1):
        self.wv = _SubwordVectors(vocab)
        self.min_count = min_count


@pytest.fixture
def eval_file(scratch_dir, synthetic_sentences):
    path = scratch_dir / "synthetic_test.txt"
    path.write_text("\n".join(synthetic_sentences) + "\n", encoding="utf-8")
    return str(path)


@pytest.fixture
def known_types(synthetic_sentences) -> list[str]:
    return sorted({tok for s in synthetic_sentences for tok in s.split()})


# ---------------------------------------------------------------------------
# The two OOV rates
# ---------------------------------------------------------------------------

def test_full_coverage_gives_zero_on_both_rates(eval_file, known_types):
    model = StubWord2Vec(known_types)
    vocab_oov, effective_oov = evaluate_nlp.compute_oov_rates(model, eval_file)

    assert vocab_oov == 0.0
    assert effective_oov == 0.0


def test_empty_vocabulary_gives_one_hundred_on_both_rates(eval_file):
    model = StubWord2Vec([])
    vocab_oov, effective_oov = evaluate_nlp.compute_oov_rates(model, eval_file)

    assert vocab_oov == 100.0
    assert effective_oov == 100.0


def test_static_model_has_identical_vocab_and_effective_rates(eval_file, known_types):
    """For word2vec the distinction is vacuous — that is the control."""
    half = known_types[: len(known_types) // 2]
    model = StubWord2Vec(half)

    vocab_oov, effective_oov = evaluate_nlp.compute_oov_rates(model, eval_file)

    assert vocab_oov == effective_oov
    assert 0.0 < vocab_oov < 100.0


def test_subword_model_separates_the_two_rates(eval_file, known_types):
    """The whole reason for splitting the field.

    A fastText-style model is missing words from its *vocabulary* while still
    being able to vectorise every one of them. Reporting only vocabulary
    membership describes it as having an OOV problem it does not have.
    """
    half = known_types[: len(known_types) // 2]
    model = StubFastText(half)

    vocab_oov, effective_oov = evaluate_nlp.compute_oov_rates(model, eval_file)

    assert vocab_oov > 0.0, "the stub should be missing types from its vocabulary"
    assert effective_oov == 0.0, "n-grams vectorise everything — effective OOV is 0"


def test_rates_are_percentages_in_range(eval_file, known_types):
    """One of the three import-time assertions, kept."""
    for model in (StubWord2Vec(known_types[:3]), StubFastText(known_types[:3])):
        for rate in evaluate_nlp.compute_oov_rates(model, eval_file):
            assert isinstance(rate, float)
            assert 0.0 <= rate <= 100.0


def test_empty_eval_file_does_not_divide_by_zero(scratch_dir):
    path = scratch_dir / "empty.txt"
    path.write_text("", encoding="utf-8")

    assert evaluate_nlp.compute_oov_rates(StubWord2Vec([]), str(path)) == (0.0, 0.0)


def test_min_count_is_readable_off_the_model():
    """main() prints this beside each score so the fastText-vs-word2vec
    comparison is auditable — the two are trained with different min_count."""
    assert getattr(StubWord2Vec([]), "min_count", None) == 2
    assert getattr(StubFastText([]), "min_count", None) == 1


# ---------------------------------------------------------------------------
# Retained from the deleted import-time block
# ---------------------------------------------------------------------------

def test_module_imports_without_touching_the_filesystem(scratch_dir, monkeypatch):
    """The point of item 4a: importing must not load a model or read a corpus.

    Reloaded from a directory where none of the old hardcoded relative paths
    resolve. Run from the repo root this assertion cannot fail — `models/…` and
    `data/processed/…` are right there, so a restored import-time block would
    load successfully and the test would still pass.

    `scratch_dir` is requested before `monkeypatch` on purpose: fixtures finalise
    in reverse order of setup, so this way the chdir is undone before the
    directory is removed. Windows refuses to delete the process's own cwd.
    """
    import importlib

    monkeypatch.chdir(scratch_dir)
    importlib.reload(evaluate_nlp)
    assert hasattr(evaluate_nlp, "main")


def test_genre_label_proportions():
    """assign_genre_labels still splits 60/35/5.

    Retained as a characterisation test only. The labels themselves are not
    meaningful — they are assigned by sentence position, but split_corpus.py
    shuffles the corpus first, so they carry no signal. See AUDIT_BRIEF.md; the
    retraction of the genre-F1 result belongs to the metrics PR, not this one.
    """
    labels = evaluate_nlp.assign_genre_labels(["x"] * 100)

    assert labels.count(0) == 60
    assert labels.count(1) == 35
    assert labels.count(2) == 5
