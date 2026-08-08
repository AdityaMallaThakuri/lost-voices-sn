# Data source: Sunuwar Bible (suzBl), © 2011 Wycliffe Bible Translators, Inc.
# Nepali source: Unlocked Literal Bible (npiulb), Door43 World Missions
# Community, CC BY-SA 4.0 -- see results/suz_npi_parallel_methodology.md.
# Licence: raw parallel text non-commercial research use only, no
# redistribution of raw text. Trained models may be released.
#
# SunuwarNMT-small -- shared bidirectional Sunuwar<->Nepali encoder-decoder
# transformer (Config A from the Phase 4 feasibility investigation).
#
# JOINT VOCABULARY DESIGN (decision made here, not left implicit):
# Reusing Sunuwar's EXISTING production SPM-8k model (the same tokenizer
# SunuwarBERT-small and SunuwarCLM-small already train with) rather than
# training a fresh, smaller Sunuwar tokenizer just for this model. Two
# options were considered for combining it with the new Nepali SPM-4k:
#   (a) ID-OFFSET MERGE -- keep both SentencePiece models exactly as
#       trained, and map their piece IDs into one combined ID space by
#       offsetting the Nepali IDs past the end of the Sunuwar ID range.
#   (b) ONE JOINT SPM -- retrain a single SentencePiece model from
#       scratch on the concatenation of both languages' text.
# Chose (a). Reasoning: (b) would mean Sunuwar text gets tokenized
# differently here than everywhere else in this project (BERT, CLM),
# which breaks comparability and forfeits the tokenizer already
# validated across two other models for no real benefit -- a joint SPM
# buys shared subword pieces for cognates/loanwords, but Sunuwar and
# Nepali belong to different language families (Kiranti vs Indo-Aryan)
# so genuine cross-lingual subword sharing is expected to be minimal
# anyway (see the Phase 3 investigation: only a handful of induced pairs,
# like Jesus's name, were near-cognates). (a) costs nothing but a larger
# combined vocabulary table, which is straightforward to size for.
#
# CONSEQUENCE FOR CONFIG A'S PARAMETER COUNT, flagged explicitly: the
# investigation's ~7.1M-parameter estimate for Config A assumed a single
# ~6,000-piece joint vocabulary. Reusing the existing Sunuwar SPM-8k
# (6,764 pieces) via ID-offset instead produces a combined vocabulary of
# 6 shared specials + 6,760 Sunuwar content pieces + 3,996 Nepali content
# pieces = 10,762 -- notably larger than 6,000, which pushes the actual
# parameter count above the original ~7.1M estimate (embeddings and the
# output head both scale with vocab size). The real count is printed by
# the dry run below rather than assumed.

import json
import math
import os
import random
import torch
import torch.nn as nn
import torch.utils.data
import sentencepiece as spm
import numpy as np
import wandb
import yaml
from transformers import get_linear_schedule_with_warmup


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
# Data prep: random 90/10 split at the PAIR level (before direction-doubling,
# so a pair's two directions never land on opposite sides of the split),
# then each pair produces two training examples (suz->npi, npi->suz).
# ---------------------------------------------------------------------------

class SunuwarNMTDataset(torch.utils.data.Dataset):
    def __init__(self, examples: list[tuple[list[int], list[int]]]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        src_ids, tgt_ids = self.examples[idx]
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


def load_and_split(config: dict, tokeniser: JointTokeniser):
    import csv
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


def make_dataloader(dataset: SunuwarNMTDataset, batch_size: int, shuffle: bool):
    def collate_fn(batch):
        src_batch, tgt_batch = zip(*batch)
        src_max = max(t.size(0) for t in src_batch)
        tgt_max = max(t.size(0) for t in tgt_batch)
        src_padded = [
            torch.cat([t, torch.full((src_max - t.size(0),), JointTokeniser.PAD_ID, dtype=torch.long)])
            for t in src_batch
        ]
        tgt_padded = [
            torch.cat([t, torch.full((tgt_max - t.size(0),), JointTokeniser.PAD_ID, dtype=torch.long)])
            for t in tgt_batch
        ]
        return torch.stack(src_padded), torch.stack(tgt_padded)

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


def shift_for_teacher_forcing(tgt_ids: torch.Tensor):
    """Standard seq2seq teacher forcing: decoder INPUT is tgt[:-1] (starts
    with BOS), loss LABELS are tgt[1:] (ends with EOS)."""
    decoder_input = tgt_ids[:, :-1]
    labels = tgt_ids[:, 1:].clone()
    labels[labels == JointTokeniser.PAD_ID] = -100
    return decoder_input, labels


# ---------------------------------------------------------------------------
# Training loop -- same AdamW + linear warmup pattern as train_mlm.py /
# train_clm.py
# ---------------------------------------------------------------------------

def train_one_epoch(model, dataloader, optimiser, scheduler, config, device, scaler):
    model.train()
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    fp16 = config.get("fp16", False)
    grad_acc = config["grad_accumulation_steps"]

    total_loss, batch_count = 0.0, 0
    optimiser.zero_grad()

    for step, (src_ids, tgt_ids) in enumerate(dataloader):
        src_ids, tgt_ids = src_ids.to(device), tgt_ids.to(device)
        decoder_input, labels = shift_for_teacher_forcing(tgt_ids)

        src_padding_mask = src_ids == JointTokeniser.PAD_ID
        tgt_padding_mask = decoder_input == JointTokeniser.PAD_ID

        with torch.autocast(device_type=device.type, enabled=fp16):
            logits = model(src_ids, decoder_input, src_padding_mask, tgt_padding_mask)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            loss = loss / grad_acc

        scaler.scale(loss).backward()
        total_loss += loss.item() * grad_acc
        batch_count += 1

        if (step + 1) % grad_acc == 0:
            scaler.unscale_(optimiser)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimiser)
            scaler.update()
            scheduler.step()
            optimiser.zero_grad()

    if batch_count % grad_acc != 0:
        scaler.unscale_(optimiser)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimiser)
        scaler.update()
        scheduler.step()
        optimiser.zero_grad()

    return total_loss / batch_count if batch_count else 0.0


@torch.no_grad()
def evaluate(model, dataloader, config, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    fp16 = config.get("fp16", False)
    total_loss, batch_count = 0.0, 0

    for src_ids, tgt_ids in dataloader:
        src_ids, tgt_ids = src_ids.to(device), tgt_ids.to(device)
        decoder_input, labels = shift_for_teacher_forcing(tgt_ids)
        src_padding_mask = src_ids == JointTokeniser.PAD_ID
        tgt_padding_mask = decoder_input == JointTokeniser.PAD_ID

        with torch.autocast(device_type=device.type, enabled=fp16):
            logits = model(src_ids, decoder_input, src_padding_mask, tgt_padding_mask)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

        total_loss += loss.item()
        batch_count += 1

    mean_loss = total_loss / batch_count if batch_count else 0.0
    return mean_loss, math.exp(mean_loss)


def main():
    with open("configs/nmt.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    wandb.init(project=config["wandb_project"], name=config["wandb_run_name"], config=config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokeniser = JointTokeniser(config["suz_tokeniser_path"], config["npi_tokeniser_path"], config["max_seq_len"])
    print(f"Joint vocab size: {tokeniser.vocab_size:,} "
          f"({tokeniser.suz_content_count:,} Sunuwar + {tokeniser.npi_content_count:,} Nepali + "
          f"{tokeniser.NUM_SPECIALS} specials)")

    model = SunuwarNMT(config, tokeniser.vocab_size).to(device)

    train_examples, val_examples = load_and_split(config, tokeniser)
    print(f"Train examples (both directions): {len(train_examples):,}")
    print(f"Val examples (both directions):   {len(val_examples):,}")

    train_loader = make_dataloader(SunuwarNMTDataset(train_examples), config["batch_size"], shuffle=True)
    val_loader = make_dataloader(SunuwarNMTDataset(val_examples), config["batch_size"], shuffle=False)

    optimiser = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

    grad_acc = config["grad_accumulation_steps"]
    total_steps = config["epochs"] * len(train_loader) // grad_acc
    warmup_steps = int(config["warmup_ratio"] * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimiser, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    scaler = torch.amp.GradScaler("cuda", enabled=config.get("fp16", False))

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    patience = config["early_stopping_patience"]

    for epoch in range(1, config["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimiser, scheduler, config, device, scaler)
        val_loss, val_perplexity = evaluate(model, val_loader, config, device)

        print(f"Epoch {epoch:3d} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | val_perplexity {val_perplexity:.2f}")

        wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_perplexity": val_perplexity})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), config["checkpoint_path"])
            print("Saved best checkpoint")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"\nBest val_loss: {best_val_loss:.4f} at epoch {best_epoch}")

    eval_results = {
        "architecture": "SunuwarNMT-small",
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "joint_vocab_size": tokeniser.vocab_size,
        "best_val_loss": round(best_val_loss, 4),
        "best_epoch": best_epoch,
    }
    os.makedirs(os.path.dirname(config["eval_output_path"]), exist_ok=True)
    with open(config["eval_output_path"], "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)
    print(f"Wrote eval summary to {config['eval_output_path']}")

    wandb.finish()


if __name__ == "__main__":
    main()
