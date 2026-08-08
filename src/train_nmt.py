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
import numpy as np
import wandb
import yaml
from transformers import get_linear_schedule_with_warmup

# JointTokeniser/SunuwarNMT moved to src/nmt_model.py (dependency-light,
# no wandb/transformers) so inference-only callers -- demo_nmt.py,
# translate_interactive.py, the dashboard services -- don't need to import
# this training script (and its wandb/transformers dependencies) at all.
from nmt_model import JointTokeniser, SunuwarNMT, load_and_split


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
    # Label smoothing applied to the TRAINING loss only (standard NMT practice,
    # e.g. the original Transformer paper) -- eval's loss_fn below stays
    # unsmoothed so val_loss/perplexity remain a true, comparable NLL metric
    # for early stopping and cross-run reporting.
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=config.get("label_smoothing", 0.0))
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
