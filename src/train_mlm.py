# Data source: Sunuwar Bible (suzBl), © 2011 Wycliffe Bible Translators, Inc.
# Licence: CC BY-NC-ND 4.0 — non-commercial research use only
# Do not redistribute raw text. Models trained on this data may be released.

import math
import os
import random
import sys
import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
import wandb
import yaml
from transformers import get_linear_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import SunuwarBERT, SunuwarTokeniser, apply_mlm_mask  # noqa: E402


class SunuwarMLMDataset(torch.utils.data.Dataset):
    def __init__(self, file_path: str, tokeniser: SunuwarTokeniser):
        with open(file_path, encoding="utf-8") as f:
            lines = [l.rstrip("\n") for l in f if l.strip()]
        self.encoded = [tokeniser.encode(line) for line in lines]

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(self.encoded[idx], dtype=torch.long)


def make_dataloader(
    dataset: SunuwarMLMDataset,
    tokeniser: SunuwarTokeniser,
    batch_size: int,
    shuffle: bool,
) -> torch.utils.data.DataLoader:
    def collate_fn(batch: list[torch.Tensor]) -> torch.Tensor:
        max_len = max(t.size(0) for t in batch)
        padded  = [
            torch.cat([t, torch.full((max_len - t.size(0),), tokeniser.pad_id, dtype=torch.long)])
            for t in batch
        ]
        return torch.stack(padded)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )


def train_one_epoch(
    model: SunuwarBERT,
    dataloader: torch.utils.data.DataLoader,
    optimiser: torch.optim.Optimizer,
    scheduler,
    tokeniser: SunuwarTokeniser,
    config: dict,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
) -> float:
    model.train()
    loss_fn  = nn.CrossEntropyLoss(ignore_index=-100)
    fp16     = config.get("fp16", False)
    grad_acc = config["grad_accumulation_steps"]

    total_loss   = 0.0
    batch_count  = 0
    optimiser.zero_grad()

    for step, batch in enumerate(dataloader):
        batch = batch.to(device)
        masked_ids, labels, attention_mask = apply_mlm_mask(
            batch,
            tokeniser,
            mlm_probability=config["mlm_probability"],
            mask_ratio=config["mask_ratio"],
            random_ratio=config["random_ratio"],
        )
        masked_ids     = masked_ids.to(device)
        labels         = labels.to(device)
        attention_mask = attention_mask.to(device)

        with torch.autocast(device_type=device.type, enabled=fp16):
            logits = model(masked_ids, attention_mask)
            # logits: (B, T, V) → reshape to (B*T, V); labels: (B, T) → (B*T,)
            loss = loss_fn(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )
            loss = loss / grad_acc

        scaler.scale(loss).backward()
        total_loss  += loss.item() * grad_acc
        batch_count += 1

        if (step + 1) % grad_acc == 0:
            scaler.unscale_(optimiser)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimiser)
            scaler.update()
            scheduler.step()
            optimiser.zero_grad()

    # Flush any remaining accumulated gradients at epoch end
    if batch_count % grad_acc != 0:
        scaler.unscale_(optimiser)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimiser)
        scaler.update()
        scheduler.step()
        optimiser.zero_grad()

    return total_loss / batch_count if batch_count else 0.0


def evaluate(
    model: SunuwarBERT,
    dataloader: torch.utils.data.DataLoader,
    tokeniser: SunuwarTokeniser,
    config: dict,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    loss_fn     = nn.CrossEntropyLoss(ignore_index=-100)
    fp16        = config.get("fp16", False)
    total_loss  = 0.0
    batch_count = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            masked_ids, labels, attention_mask = apply_mlm_mask(
                batch,
                tokeniser,
                mlm_probability=config["mlm_probability"],
                mask_ratio=config["mask_ratio"],
                random_ratio=config["random_ratio"],
            )
            masked_ids     = masked_ids.to(device)
            labels         = labels.to(device)
            attention_mask = attention_mask.to(device)

            with torch.autocast(device_type=device.type, enabled=fp16):
                logits = model(masked_ids, attention_mask)
                loss   = loss_fn(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                )

            total_loss  += loss.item()
            batch_count += 1

    mean_loss   = total_loss / batch_count if batch_count else 0.0
    perplexity  = math.exp(mean_loss)
    return mean_loss, perplexity


def probe_predictions(
    model: SunuwarBERT,
    tokeniser: SunuwarTokeniser,
    probe_sentences: list[str],
    device: torch.device,
) -> None:
    model.eval()
    with torch.no_grad():
        for i, sentence in enumerate(probe_sentences):
            ids = tokeniser.encode(sentence)

            if len(ids) <= 4:
                print(f"Probe {i}: sentence too short to mask position 4, skipping")
                continue

            original_token = tokeniser.sp.id_to_piece(ids[4])
            ids[4]         = tokeniser.mask_id

            input_tensor   = torch.tensor([ids], dtype=torch.long, device=device)
            attention_mask = (input_tensor != tokeniser.pad_id).long()

            logits    = model(input_tensor, attention_mask)        # (1, seq_len, vocab_size)
            top5_ids  = logits[0, 4].topk(5).indices.tolist()
            top5_pieces = [tokeniser.sp.id_to_piece(t) for t in top5_ids]

            print(f"Probe {i}: [{original_token}] → top5: {top5_pieces}")


def main():
    with open("configs/mlm.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    wandb.init(
        project=config["wandb_project"],
        name=config["wandb_run_name"],
        config=config,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokeniser = SunuwarTokeniser(
        model_path=config["tokeniser_path"],
        vocab_size=config["vocab_size"],
        max_seq_len=config["max_seq_len"],
    )

    model = SunuwarBERT(config).to(device)

    train_dataset = SunuwarMLMDataset(config["train_path"], tokeniser)
    val_dataset   = SunuwarMLMDataset(config["test_path"],  tokeniser)
    train_loader  = make_dataloader(train_dataset, tokeniser, batch_size=config["batch_size"], shuffle=True)
    val_loader    = make_dataloader(val_dataset,   tokeniser, batch_size=config["batch_size"], shuffle=False)

    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    grad_acc     = config["grad_accumulation_steps"]
    total_steps  = config["epochs"] * len(train_loader) // grad_acc
    warmup_steps = int(config["warmup_ratio"] * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimiser,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    scaler = torch.amp.GradScaler('cuda', enabled=config.get("fp16", False))

    with open(config["test_path"], encoding="utf-8") as f:
        probe_sentences = [f.readline().strip() for _ in range(3)]

    best_perplexity  = float("inf")
    best_epoch       = 0
    patience_counter = 0
    patience         = config["early_stopping_patience"]

    for epoch in range(1, config["epochs"] + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimiser, scheduler, tokeniser, config, device, scaler,
        )
        val_loss, val_perplexity = evaluate(
            model, val_loader, tokeniser, config, device,
        )

        print(
            f"Epoch {epoch:3d} | "
            f"train_loss {train_loss:.4f} | "
            f"val_loss {val_loss:.4f} | "
            f"val_perplexity {val_perplexity:.2f}"
        )
        probe_predictions(model, tokeniser, probe_sentences, device)

        wandb.log({
            "epoch":          epoch,
            "train_loss":     train_loss,
            "val_loss":       val_loss,
            "val_perplexity": val_perplexity,
        })

        if val_perplexity < best_perplexity:
            best_perplexity  = val_perplexity
            best_epoch       = epoch
            patience_counter = 0
            torch.save(model.state_dict(), config["checkpoint_path"])
            print("Saved best checkpoint")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"\nBest val_perplexity: {best_perplexity:.2f} at epoch {best_epoch}")
    wandb.finish()


if __name__ == "__main__":
    main()
