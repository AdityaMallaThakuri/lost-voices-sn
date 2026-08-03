# Data source: Sunuwar Bible (suzBl), © 2011 Wycliffe Bible Translators, Inc.
# Licence: CC BY-NC-ND 4.0 — non-commercial research use only
"""SentencePiece round-trip, with U+200D ZWJ as the thing that must survive.

ZWJ occurs 54,692 times in the NT, inside words, to make Devanagari conjuncts
render correctly. claude.md lists "ZWJ must be preserved in all text
processing — never strip U+200D" as a hard standard, so it has to survive
encode → decode as well as every text-handling step.

All strings here are synthetic Devanagari written by hand, not corpus text.
"""

import pytest

from conftest import SYNTHETIC_ZWJ_WORDS, ZWJ

pytest.importorskip("sentencepiece")


def test_zwj_survives_the_encode_decode_round_trip(tokeniser):
    for word in SYNTHETIC_ZWJ_WORDS:
        assert ZWJ in word, "the fixture itself lost its ZWJ"

        decoded = tokeniser.sp.decode(tokeniser.sp.encode(word))

        assert decoded == word
        assert decoded.count(ZWJ) == word.count(ZWJ)


def test_sentences_round_trip_exactly(tokeniser, synthetic_sentences):
    for sentence in synthetic_sentences:
        assert tokeniser.sp.decode(tokeniser.sp.encode(sentence)) == sentence


def test_synthetic_devanagari_produces_no_unknown_pieces(tokeniser, synthetic_sentences):
    for sentence in synthetic_sentences:
        ids = tokeniser.sp.encode(sentence)
        assert tokeniser.unk_id not in ids, f"[UNK] emitted for {sentence!r}"


def test_encode_wraps_the_sentence_in_cls_and_sep(tokeniser, synthetic_sentences):
    for sentence in synthetic_sentences:
        ids = tokeniser.encode(sentence)

        assert ids[0] == tokeniser.cls_id
        assert ids[-1] == tokeniser.sep_id
        assert ids[1:-1] == tokeniser.sp.encode(sentence)


def test_encode_never_exceeds_max_seq_len(tokeniser):
    long_text = " ".join(SYNTHETIC_ZWJ_WORDS * 200)

    ids = tokeniser.encode(long_text)

    assert len(ids) == tokeniser.max_seq_len


def test_special_ids_are_the_ones_the_checkpoint_was_trained_with(tokeniser):
    """These are load-bearing for models/sunuwar_transformer.pt.

    Note the documentation drift: claude.md specifies [CLS]/[SEP], but ids 2
    and 3 in this model are actually <s> and </s>. The code is self-consistent
    and the checkpoint depends on it — the docs are what is wrong, so the ids
    are pinned here rather than "fixed".
    """
    assert tokeniser.pad_id == 0
    assert tokeniser.unk_id == 1
    assert tokeniser.cls_id == 2
    assert tokeniser.sep_id == 3
    assert tokeniser.sp.id_to_piece(2) == "<s>"
    assert tokeniser.sp.id_to_piece(3) == "</s>"
    assert tokeniser.mask_id == tokeniser.sp.piece_to_id("[MASK]")


def test_piece_size_is_smaller_than_the_configured_vocab_size(tokeniser):
    """6764 real pieces, 8000 embedding rows. Anything that samples an id has
    to respect the smaller number; the embedding table has to keep the larger."""
    assert tokeniser.piece_size == 6764
    assert tokeniser.vocab_size == 8000
    assert tokeniser.piece_size < tokeniser.vocab_size


def test_batch_encode_pads_to_the_longest_row(tokeniser, synthetic_sentences):
    batch = tokeniser.batch_encode(synthetic_sentences)

    assert batch.shape[0] == len(synthetic_sentences)
    longest = max(len(tokeniser.encode(s)) for s in synthetic_sentences)
    assert batch.shape[1] == longest
    for row, sentence in zip(batch, synthetic_sentences):
        ids = tokeniser.encode(sentence)
        assert row[:len(ids)].tolist() == ids
        assert (row[len(ids):] == tokeniser.pad_id).all()
