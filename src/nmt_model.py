# Data source: Sunuwar Bible (suzBl), © 2011 Wycliffe Bible Translators, Inc.
# Nepali source: Unlocked Literal Bible (npiulb), Door43 World Missions
# Community, CC BY-SA 4.0 -- see results/suz_npi_parallel_methodology.md.
# Licence: raw parallel text non-commercial research use only, no
# redistribution of raw text. Trained models may be released.
#
# Dependency-light SunuwarNMT-small model + tokenizer + greedy-decode
# inference, factored out of train_nmt.py/demo_nmt.py so inference-only
# callers (the dashboard, translate_interactive.py) don't need to import
# wandb or transformers.get_linear_schedule_with_warmup, which are purely
# training-time dependencies. train_nmt.py, demo_nmt.py, and
# translate_interactive.py all import JointTokeniser/SunuwarNMT/
# greedy_translate/decode_combined_ids from here now -- this module is
# the single source of truth for the model architecture and joint
# vocabulary design (see docstring history in train_nmt.py for the full
# ID-offset-merge rationale).

import csv
import random
import torch
import torch.nn as nn
import sentencepiece as spm


# ---------------------------------------------------------------------------
# Joint tokenizer: ID-offset merge of the two existing SPM models
# ---------------------------------------------------------------------------

class JointTokeniser:
    # Combined-space special tokens -- NOT reusing either SPM model's own
    # special ids (each model's own 0-3 are ignored/skipped entirely).
    PAD_ID = 0
    UNK_ID = 1
    BOS_ID = 2
    EOS_ID = 3
    TAG_2NPI_ID = 4  # "translate what follows into Nepali"
    TAG_2SUZ_ID = 5  # "translate what follows into Sunuwar"
    NUM_SPECIALS = 6

    def __init__(self, suz_model_path: str, npi_model_path: str, max_seq_len: int):
        self.suz_sp = spm.SentencePieceProcessor()
        self.suz_sp.Load(suz_model_path)
        self.npi_sp = spm.SentencePieceProcessor()
        self.npi_sp.Load(npi_model_path)

        self.max_seq_len = max_seq_len

        # Each SPM model's ids 0-3 are its OWN pad/unk/bos/eos -- skip those,
        # content pieces start at id 4 in each model's native space.
        self.suz_content_count = self.suz_sp.vocab_size() - 4
        self.npi_content_count = self.npi_sp.vocab_size() - 4

        self.suz_offset = self.NUM_SPECIALS
        self.npi_offset = self.NUM_SPECIALS + self.suz_content_count

        self.vocab_size = self.NUM_SPECIALS + self.suz_content_count + self.npi_content_count

    def encode_suz(self, text: str) -> list[int]:
        native_ids = self.suz_sp.encode(text)
        return [self._map_suz(i) for i in native_ids]

    def encode_npi(self, text: str) -> list[int]:
        native_ids = self.npi_sp.encode(text)
        return [self._map_npi(i) for i in native_ids]

    def _map_suz(self, native_id: int) -> int:
        if native_id < 4:
            return self.UNK_ID  # one of suz's own specials mid-sequence -> combined UNK
        return self.suz_offset + (native_id - 4)

    def _map_npi(self, native_id: int) -> int:
        if native_id < 4:
            return self.UNK_ID
        return self.npi_offset + (native_id - 4)

    def build_example(self, src_text: str, tgt_text: str, direction: str) -> tuple[list[int], list[int]]:
        """direction is 'suz2npi' or 'npi2suz'. Returns (src_ids, tgt_ids),
        each already truncated to max_seq_len, tgt_ids wrapped in BOS/EOS."""
        if direction == "suz2npi":
            tag = self.TAG_2NPI_ID
            src_ids = [tag] + self.encode_suz(src_text)
            tgt_ids = [self.BOS_ID] + self.encode_npi(tgt_text) + [self.EOS_ID]
        elif direction == "npi2suz":
            tag = self.TAG_2SUZ_ID
            src_ids = [tag] + self.encode_npi(src_text)
            tgt_ids = [self.BOS_ID] + self.encode_suz(tgt_text) + [self.EOS_ID]
        else:
            raise ValueError(f"unknown direction: {direction}")
        return src_ids[: self.max_seq_len], tgt_ids[: self.max_seq_len]


# ---------------------------------------------------------------------------
# Model: standard encoder-decoder transformer (nn.Transformer-based),
# same code style as SunuwarBERT-small / SunuwarCLM-small
# ---------------------------------------------------------------------------

class SunuwarNMT(nn.Module):
    def __init__(self, config: dict, vocab_size: int):
        super().__init__()
        hidden_dim = config["hidden_dim"]
        num_heads = config["num_heads"]
        ffn_dim = config["ffn_dim"]
        num_encoder_layers = config["num_encoder_layers"]
        num_decoder_layers = config["num_decoder_layers"]
        max_seq_len = config["max_seq_len"]
        dropout = config["dropout"]
        activation = config["activation"]

        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=JointTokeniser.PAD_ID)
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_dim)

        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.out_head = nn.Linear(hidden_dim, vocab_size)

        total = sum(p.numel() for p in self.parameters())
        print(f"SunuwarNMT-small: {total:,} parameters")

    def _embed(self, ids: torch.Tensor) -> torch.Tensor:
        seq_len = ids.size(1)
        pos_ids = torch.arange(seq_len, device=ids.device).unsqueeze(0)
        return self.embedding(ids) + self.pos_embedding(pos_ids)

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        src_padding_mask: torch.Tensor,
        tgt_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        src_emb = self._embed(src_ids)
        tgt_emb = self._embed(tgt_ids)

        tgt_len = tgt_ids.size(1)
        causal_mask = torch.triu(
            torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=tgt_ids.device), diagonal=1,
        )

        out = self.transformer(
            src_emb, tgt_emb,
            tgt_mask=causal_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
        )
        return self.out_head(out)


# ---------------------------------------------------------------------------
# Train/val split reconstruction -- moved here (not just greedy_translate/
# decode_combined_ids) because demo_nmt.py's qualitative demo and any
# service that needs to reproduce the exact held-out split both need this,
# and it has no training-only dependency itself (just csv/random) -- keeping
# it in train_nmt.py would have forced demo_nmt.py to import that module
# anyway, reintroducing the wandb/transformers dependency this refactor
# exists to remove.
# ---------------------------------------------------------------------------

def load_and_split(config: dict, tokeniser: JointTokeniser):
    with open(config["aligned_path"], encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)

    rng = random.Random(config["seed"])
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    split_point = int(len(indices) * config["train_split"])
    train_idx = set(indices[:split_point])

    train_examples, val_examples = [], []
    for i, row in enumerate(rows):
        suz_text, npi_text = row["suz_sentence"], row["npi_sentence"]
        ex_s2n = tokeniser.build_example(suz_text, npi_text, "suz2npi")
        ex_n2s = tokeniser.build_example(npi_text, suz_text, "npi2suz")
        target = train_examples if i in train_idx else val_examples
        target.append(ex_s2n)
        target.append(ex_n2s)

    return train_examples, val_examples


# ---------------------------------------------------------------------------
# Greedy-decode inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def greedy_translate(model, tokeniser, src_ids: list, direction: str, device, max_new_tokens: int = 60):
    """direction is 'suz2npi' or 'npi2suz' -- only used to pick which native
    SPM decodes the generated ids back to text."""
    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_padding_mask = torch.zeros_like(src_tensor, dtype=torch.bool)  # no padding, single example

    generated = [JointTokeniser.BOS_ID]
    for _ in range(max_new_tokens):
        tgt_tensor = torch.tensor([generated], dtype=torch.long, device=device)
        tgt_padding_mask = torch.zeros_like(tgt_tensor, dtype=torch.bool)
        logits = model(src_tensor, tgt_tensor, src_padding_mask, tgt_padding_mask)
        next_id = logits[0, -1].argmax().item()
        generated.append(next_id)
        if next_id == JointTokeniser.EOS_ID:
            break

    content_ids = generated[1:]
    if content_ids and content_ids[-1] == JointTokeniser.EOS_ID:
        content_ids = content_ids[:-1]

    return decode_combined_ids(tokeniser, content_ids, direction)


def decode_combined_ids(tokeniser: JointTokeniser, combined_ids: list, direction: str) -> str:
    """Map combined-space ids back to native SPM ids and decode with the
    correct model for the OUTPUT language of this direction.

    A model (especially early in training, or on a mispredicted step) can
    emit a combined-space id that belongs to the WRONG language's range for
    this direction -- e.g. a Sunuwar-range id while generating Nepali output.
    That must not crash decoding (found via a smoke test with a random-init
    model before this was handled): any out-of-range id is dropped rather
    than passed to SentencePiece, which raises IndexError on out-of-range
    piece ids instead of failing gracefully.
    """
    if direction == "suz2npi":
        sp = tokeniser.npi_sp
        offset = tokeniser.npi_offset
        content_count = tokeniser.npi_content_count
    else:
        sp = tokeniser.suz_sp
        offset = tokeniser.suz_offset
        content_count = tokeniser.suz_content_count

    native_ids = []
    dropped = 0
    for cid in combined_ids:
        if cid < JointTokeniser.NUM_SPECIALS:
            continue  # skip stray specials (shouldn't normally appear mid-output)
        native_id = cid - offset + 4  # +4 to undo the "skip native specials 0-3" offset
        if offset <= cid < offset + content_count:
            native_ids.append(native_id)
        else:
            dropped += 1  # id belongs to the other language's range -- drop, don't crash
    text = sp.decode(native_ids) if native_ids else "(empty output)"
    if dropped:
        text += f"  [{dropped} out-of-range token(s) dropped]"
    return text
