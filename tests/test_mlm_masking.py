# Data source: Sunuwar Bible (suzBl), © 2011 Wycliffe Bible Translators, Inc.
# Licence: CC BY-NC-ND 4.0 — non-commercial research use only
"""Invariants of the 80/10/10 MLM corruption, and the degenerate-batch guard.

All input here is synthetic token ids, not corpus text.
"""

import pytest

torch = pytest.importorskip("torch")

from model import apply_mlm_mask  # noqa: E402

MLM_PROB = 0.15
MASK_RATIO = 0.8
RANDOM_RATIO = 0.1


def make_batch(tokeniser, rows: int = 256, cols: int = 64, seed: int = 7):
    """A batch shaped like real encoded data: [CLS] … [SEP] then padding."""
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(5, tokeniser.piece_size, (rows, cols), generator=g)
    ids[:, 0] = tokeniser.cls_id
    # Vary where [SEP] lands so padding runs differ per row, as in a real batch.
    lengths = torch.randint(8, cols, (rows,), generator=g)
    for r in range(rows):
        ids[r, lengths[r]] = tokeniser.sep_id
        ids[r, lengths[r] + 1:] = tokeniser.pad_id
    return ids


@pytest.fixture(scope="module")
def batch(tokeniser):
    return make_batch(tokeniser)


@pytest.fixture(scope="module")
def masked(tokeniser, batch):
    g = torch.Generator().manual_seed(42)
    return apply_mlm_mask(
        batch, tokeniser,
        mlm_probability=MLM_PROB, mask_ratio=MASK_RATIO,
        random_ratio=RANDOM_RATIO, generator=g,
    )


# ---------------------------------------------------------------------------
# Which positions may be selected
# ---------------------------------------------------------------------------

def test_special_tokens_are_never_selected(tokeniser, batch, masked):
    _, labels, _ = masked
    selected = labels != -100

    for name, token_id in (
        ("[PAD]", tokeniser.pad_id),
        ("<s> (used as CLS)", tokeniser.cls_id),
        ("</s> (used as SEP)", tokeniser.sep_id),
    ):
        assert not selected[batch == token_id].any(), f"{name} was selected for masking"


def test_padding_is_never_altered(tokeniser, batch, masked):
    masked_ids, _, _ = masked
    is_pad = batch == tokeniser.pad_id

    assert (masked_ids[is_pad] == tokeniser.pad_id).all()


def test_attention_mask_marks_exactly_the_non_pad_positions(tokeniser, batch, masked):
    _, _, attention_mask = masked

    assert torch.equal(attention_mask, (batch != tokeniser.pad_id).long())


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

def test_labels_are_ignore_index_everywhere_except_selected_positions(batch, masked):
    _, labels, _ = masked
    selected = labels != -100

    assert (labels[~selected] == -100).all()
    assert selected.any(), "nothing was selected — the test batch is degenerate"


def test_labels_carry_the_original_id_at_every_selected_position(batch, masked):
    _, labels, _ = masked
    selected = labels != -100

    assert torch.equal(labels[selected], batch[selected])


def test_selection_rate_is_close_to_mlm_probability(tokeniser, batch, masked):
    _, labels, _ = masked
    eligible = (
        (batch != tokeniser.pad_id)
        & (batch != tokeniser.cls_id)
        & (batch != tokeniser.sep_id)
    )
    rate = (labels != -100).sum().item() / eligible.sum().item()

    assert rate == pytest.approx(MLM_PROB, abs=0.02)


# ---------------------------------------------------------------------------
# The 80/10/10 split
# ---------------------------------------------------------------------------

def test_eighty_ten_ten_split_holds_within_tolerance(tokeniser, batch, masked):
    masked_ids, labels, _ = masked
    selected = labels != -100
    n = selected.sum().item()

    was_masked = (masked_ids[selected] == tokeniser.mask_id)
    unchanged = (masked_ids[selected] == batch[selected])
    replaced = ~was_masked & ~unchanged

    # A random draw can land on [MASK] (id 4) or on the original id, so the
    # three buckets are measured with ~1/6760 leakage each. Irrelevant at this
    # tolerance, and the generator is seeded, so this cannot flake.
    assert was_masked.sum().item() / n == pytest.approx(MASK_RATIO, abs=0.02)
    assert replaced.sum().item() / n == pytest.approx(RANDOM_RATIO, abs=0.02)
    assert unchanged.sum().item() / n == pytest.approx(
        1 - MASK_RATIO - RANDOM_RATIO, abs=0.02
    )


def test_every_replacement_id_is_below_piece_size(tokeniser, batch):
    """The fix for ids 6764-7999, which the tokeniser can never emit.

    Run over many independent draws so the assertion is not a single sample:
    the out-of-range band was 15.5% of the random branch before the fix.
    """
    seen_replacements = 0
    for seed in range(20):
        g = torch.Generator().manual_seed(seed)
        masked_ids, labels, _ = apply_mlm_mask(
            batch, tokeniser,
            mlm_probability=MLM_PROB, mask_ratio=MASK_RATIO,
            random_ratio=RANDOM_RATIO, generator=g,
        )
        selected = labels != -100
        replaced = masked_ids[selected][
            (masked_ids[selected] != tokeniser.mask_id)
            & (masked_ids[selected] != batch[selected])
        ]
        seen_replacements += replaced.numel()

        assert (replaced < tokeniser.piece_size).all(), (
            f"seed {seed}: drew an id >= piece_size ({tokeniser.piece_size}); "
            "ids that high exist as embedding rows but no real text contains them"
        )
        assert (replaced >= 4).all(), "drew a special-token id as a replacement"

    assert seen_replacements > 1000, "not enough replacements drawn to be meaningful"
    assert tokeniser.piece_size < tokeniser.vocab_size, (
        "this test is only meaningful while piece_size (6764) < vocab_size (8000)"
    )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_seeded_generator_produces_byte_identical_output_twice(tokeniser, batch):
    def run():
        g = torch.Generator().manual_seed(1234)
        return apply_mlm_mask(batch, tokeniser, generator=g)

    first = run()
    second = run()

    for a, b in zip(first, second):
        assert torch.equal(a, b)


def test_seeded_masking_ignores_the_global_rng(tokeniser, batch):
    """Why evaluate() can be trusted epoch to epoch.

    Training advances the global RNG by a different amount every epoch. If the
    eval mask were drawn from it, the held-out targets would move underneath
    the perplexity curve.
    """
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(99)
    first = apply_mlm_mask(batch, tokeniser, generator=g)

    torch.manual_seed(12345)
    _ = torch.rand(1000)  # advance the global stream, as an epoch of training would
    g = torch.Generator().manual_seed(99)
    second = apply_mlm_mask(batch, tokeniser, generator=g)

    for a, b in zip(first, second):
        assert torch.equal(a, b)


def test_different_seeds_produce_different_masks(tokeniser, batch):
    a = apply_mlm_mask(batch, tokeniser, generator=torch.Generator().manual_seed(1))
    b = apply_mlm_mask(batch, tokeniser, generator=torch.Generator().manual_seed(2))

    assert not torch.equal(a[1], b[1])


def test_training_path_without_a_generator_stays_stochastic(tokeniser, batch):
    """Training must keep drawing fresh corruption every epoch — only the
    held-out evaluation is pinned."""
    a = apply_mlm_mask(batch, tokeniser)
    b = apply_mlm_mask(batch, tokeniser)

    assert not torch.equal(a[1], b[1])
