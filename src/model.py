# Data source: Sunuwar Bible (suzBl), © 2011 Wycliffe Bible Translators, Inc.
# Licence: CC BY-NC-ND 4.0 — non-commercial research use only
# Do not redistribute raw text. Models trained on this data may be released.
"""Canonical SunuwarBERT-small model, tokeniser and MLM masking.

This is the single source of truth. `train_mlm.py`, `app.py` and `demo_mlm.py`
each used to carry their own copy of these three definitions, and they had
already drifted apart. Because `models/sunuwar_transformer.pt` cannot be
regenerated, any divergence in `SunuwarBERT` silently breaks checkpoint
loading in whichever copy was not updated.

`tests/test_model_contract.py` pins the `state_dict` keys and shapes this
module produces against `tests/fixtures/state_dict_contract.json`. If you
rename a layer or change a dimension, that test fails — by design.
"""

import torch
import torch.nn as nn
import sentencepiece as spm


class SunuwarTokeniser:
    """SentencePiece wrapper with the fixed special-token ids used at training time.

    Note on ids 2 and 3: the 8k SentencePiece model defines `<s>` / `</s>` at
    those positions, not `[CLS]` / `[SEP]`. They are *used* as CLS/SEP here.
    The ids are load-bearing for the released checkpoint — do not change them.
    """

    def __init__(self, model_path: str, vocab_size: int, max_seq_len: int):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.pad_id = 0
        self.unk_id = 1
        self.cls_id = 2
        self.sep_id = 3
        self.mask_id = self.sp.piece_to_id("[MASK]")
        # piece_to_id returns the unknown id for a piece the model does not
        # define, so a tokeniser trained without [MASK] in user_defined_symbols
        # yields mask_id == unk_id and apply_mlm_mask then stamps <unk> over 80%
        # of selected positions — silently, with no error and no NaN to notice.
        if self.mask_id == self.sp.unk_id():
            raise ValueError(
                f"{model_path} defines no [MASK] piece: piece_to_id('[MASK]') "
                f"returned the unknown id ({self.mask_id}). Masking would train "
                "on <unk> instead. Retrain the tokeniser with [MASK] in "
                "user_defined_symbols (see configs/spm.yaml)."
            )

    @property
    def piece_size(self) -> int:
        """How many ids the tokeniser can actually emit — 6764, not `vocab_size`.

        `configs/mlm.yaml` asks for 8000 and `train_spm.py` runs with
        `hard_vocab_limit=False`, so SentencePiece settled on the 6764 pieces
        the corpus supports. The embedding table must stay at `vocab_size`
        (the released checkpoint has 8000 rows), but anything that *samples*
        an id has to stay below `piece_size` or it produces tokens that can
        never appear in real text.
        """
        return self.sp.get_piece_size()

    def encode(self, text: str) -> list[int]:
        """Encode to ids wrapped in [CLS] … [SEP], truncated to max_seq_len.

        Truncation is applied *after* [SEP] is appended, so any text longer
        than `max_seq_len - 2` pieces comes back ending on a content piece with
        no terminator. That is deliberate: this is the encode the released
        checkpoint was trained through, and the branch cannot fire on the
        corpus — the longest sentence in train.txt, test.txt or
        sunuwar_nt_raw.txt is 68 pieces (70 with specials) against a
        max_seq_len of 128, i.e. 58 pieces of headroom. Only arbitrary user
        input via app.py or `demo_mlm.py --sentence` can reach it. Pinned by
        tests/test_tokeniser.py rather than changed, since re-appending [SEP]
        would alter tokenisation for a case the trained model never saw.
        """
        ids = self.sp.encode(text)
        ids = [self.cls_id] + ids + [self.sep_id]
        return ids[:self.max_seq_len]

    def batch_encode(self, texts: list[str], pad: bool = True) -> torch.Tensor:
        encoded = [self.encode(t) for t in texts]
        if pad:
            max_len = max(len(e) for e in encoded)
            encoded = [e + [self.pad_id] * (max_len - len(e)) for e in encoded]
        return torch.tensor(encoded, dtype=torch.long)


class SunuwarBERT(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        vocab_size   = config["vocab_size"]
        hidden_dim   = config["hidden_dim"]
        num_heads    = config["num_heads"]
        num_layers   = config["num_layers"]
        ffn_dim      = config["ffn_dim"]
        max_seq_len  = config["max_seq_len"]
        dropout      = config["dropout"]
        activation   = config["activation"]

        self.embedding     = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.encoder  = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.mlm_head = nn.Linear(hidden_dim, vocab_size)

        total = sum(p.numel() for p in self.parameters())
        print(f"SunuwarBERT-small: {total:,} parameters")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.size(1)
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        x = self.embedding(input_ids) + self.pos_embedding(pos_ids)

        # TransformerEncoder expects True where tokens should be *ignored*
        padding_mask = attention_mask == 0

        x = self.encoder(x, src_key_padding_mask=padding_mask)
        return self.mlm_head(x)


def apply_mlm_mask(
    input_ids: torch.Tensor,
    tokeniser: SunuwarTokeniser,
    mlm_probability: float = 0.15,
    mask_ratio: float = 0.8,
    random_ratio: float = 0.1,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the 80/10/10 MLM corruption.

    `generator` — pass a seeded `torch.Generator` (on the same device as
    `input_ids`) to make the masking reproducible. Training should leave this
    None so every epoch sees fresh corruption; *evaluation* must pass one, or
    the held-out perplexity is re-sampled every epoch and the curve, the
    reported best, and the early-stopping trigger are all partly noise.
    """
    masked_ids = input_ids.clone()
    labels     = torch.full_like(input_ids, -100)

    def _rand() -> torch.Tensor:
        return torch.rand(
            input_ids.shape,
            dtype=torch.float,
            device=input_ids.device,
            generator=generator,
        )

    # Eligible positions: not [PAD]=0, [CLS]=2, [SEP]=3
    eligible = (
        (input_ids != tokeniser.pad_id) &
        (input_ids != tokeniser.cls_id) &
        (input_ids != tokeniser.sep_id)
    )

    # Select 15% of eligible tokens
    rand        = _rand()
    selected    = eligible & (rand < mlm_probability)

    # Record original ids as labels at selected positions
    labels[selected] = input_ids[selected]

    # 80/10/10 split over selected tokens
    split = _rand()
    to_mask   = selected & (split < mask_ratio)
    to_random = selected & (split >= mask_ratio) & (split < mask_ratio + random_ratio)
    # remaining selected tokens are left unchanged

    masked_ids[to_mask] = tokeniser.mask_id
    if to_random.any():
        # Upper bound is piece_size (6764), NOT vocab_size (8000). Ids 6764-7999
        # exist as embedding rows but the tokeniser can never emit them, so
        # drawing them injects noise the model will never meet in real data.
        random_tokens = torch.randint(
            4,
            tokeniser.piece_size,
            (to_random.sum().item(),),
            device=input_ids.device,
            generator=generator,
        )
        masked_ids[to_random] = random_tokens

    # Attention mask: 1 for real tokens, 0 for padding
    attention_mask = (input_ids != tokeniser.pad_id).long()

    return masked_ids, labels, attention_mask
