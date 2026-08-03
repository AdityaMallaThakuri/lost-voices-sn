# Data source: Sunuwar Bible (suzBl), © 2011 Wycliffe Bible Translators, Inc.
# Licence: CC BY-NC-ND 4.0 — non-commercial research use only
"""A batch with nothing to predict must not poison the model.

`CrossEntropyLoss(ignore_index=-100)` over an all-ignored target is a 0/0 mean
and returns NaN. One NaN backward pass turns every weight in the model into
NaN, permanently, and the training loop has no way to notice. At the configured
`batch_size: 32` the probability is ~1e-25, but at the small batch sizes used
for smoke tests it is roughly 1 in 1,250 batches — so it does fire in practice.

These tests drive the real `train_one_epoch` and `evaluate` from
src/train_mlm.py, not a reimplementation.
"""

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

import train_mlm  # noqa: E402
from model import SunuwarBERT  # noqa: E402

TINY_CONFIG = {
    "vocab_size": 16,
    "hidden_dim": 8,
    "num_heads": 2,
    "num_layers": 1,
    "ffn_dim": 16,
    "max_seq_len": 16,
    "dropout": 0.0,
    "activation": "gelu",
    "mlm_probability": 0.15,
    "mask_ratio": 0.8,
    "random_ratio": 0.1,
    "grad_accumulation_steps": 2,
    "fp16": False,
    "seed": 42,
}


@pytest.fixture
def degenerate_loader(tokeniser):
    """Batches containing only [CLS], [SEP] and padding.

    Nothing is eligible for masking, so every label is -100 — the exact shape
    of the failure. [CLS]/[SEP] stay unmasked in the attention mask, so this
    isolates the loss bug rather than tripping an all-padding attention row.
    """
    rows = [
        torch.tensor(
            [tokeniser.cls_id, tokeniser.sep_id] + [tokeniser.pad_id] * 6,
            dtype=torch.long,
        )
        for _ in range(6)
    ]
    return torch.utils.data.DataLoader(rows, batch_size=2, shuffle=False)


@pytest.fixture
def tiny_model():
    torch.manual_seed(0)
    return SunuwarBERT(TINY_CONFIG)


def test_the_unguarded_loss_really_is_nan():
    """The premise. If this ever stops holding, the guard can go."""
    loss = nn.CrossEntropyLoss(ignore_index=-100)(
        torch.randn(8, 16), torch.full((8,), -100)
    )

    assert torch.isnan(loss)


def test_evaluate_returns_a_finite_loss_and_counts_the_skip(
    tiny_model, tokeniser, degenerate_loader
):
    val_loss, perplexity, skipped = train_mlm.evaluate(
        tiny_model, degenerate_loader, tokeniser, TINY_CONFIG, torch.device("cpu")
    )

    assert skipped == 3, "every batch should have been skipped and counted"
    assert not torch.isnan(torch.tensor(val_loss))
    assert not torch.isnan(torch.tensor(perplexity))
    assert torch.isfinite(torch.tensor(perplexity))


def test_train_one_epoch_leaves_every_weight_finite(
    tiny_model, tokeniser, degenerate_loader
):
    """The consequence the guard exists to prevent."""
    optimiser = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lambda _: 1.0)
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    train_loss, skipped = train_mlm.train_one_epoch(
        tiny_model, degenerate_loader, optimiser, scheduler, tokeniser,
        TINY_CONFIG, torch.device("cpu"), scaler,
    )

    assert skipped == 3
    assert not torch.isnan(torch.tensor(train_loss))
    for name, param in tiny_model.named_parameters():
        assert torch.isfinite(param).all(), f"{name} went non-finite"


def test_a_healthy_batch_still_trains_after_a_degenerate_one(tiny_model, tokeniser):
    """The guard must skip the bad batch, not abandon the epoch."""
    good = torch.tensor(
        [[tokeniser.cls_id] + [5, 6, 7, 8, 9, 10, 11] + [tokeniser.sep_id]] * 4,
        dtype=torch.long,
    )
    bad = torch.tensor(
        [[tokeniser.cls_id, tokeniser.sep_id] + [tokeniser.pad_id] * 7] * 4,
        dtype=torch.long,
    )
    loader = torch.utils.data.DataLoader(
        [bad[0], good[0], bad[1], good[1]], batch_size=1, shuffle=False
    )

    config = dict(TINY_CONFIG, mlm_probability=1.0, grad_accumulation_steps=1)
    optimiser = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lambda _: 1.0)
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    before = [p.detach().clone() for p in tiny_model.parameters()]
    train_loss, skipped = train_mlm.train_one_epoch(
        tiny_model, loader, optimiser, scheduler, tokeniser,
        config, torch.device("cpu"), scaler,
    )

    assert skipped == 2, "the two degenerate batches should be skipped"
    assert train_loss > 0.0, "the two healthy batches should still contribute a loss"
    assert any(
        not torch.equal(b, a) for b, a in zip(before, tiny_model.parameters())
    ), "the optimiser never stepped — the healthy batches were lost too"
    for name, param in tiny_model.named_parameters():
        assert torch.isfinite(param).all(), f"{name} went non-finite"
