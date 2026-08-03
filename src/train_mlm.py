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
import yaml
from transformers import get_linear_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import SunuwarBERT, SunuwarTokeniser, apply_mlm_mask  # noqa: E402

DEFAULT_CONFIG_PATH = "configs/mlm.yaml"
WANDB_MODES = ("online", "offline", "disabled")


def amp_enabled(config: dict, device: torch.device) -> bool:
    """Whether mixed precision should actually be on.

    `fp16: true` with no GPU used to hand `torch.autocast` a CPU device type.
    That does not raise — PyTorch warns and silently falls back to bfloat16 —
    so the config claimed fp16 while the run did something else. Gate on the
    device so the two agree.
    """
    return bool(config.get("fp16", False)) and device.type == "cuda"


def init_wandb(config: dict):
    """Honour `wandb_mode` (online / offline / disabled) before touching wandb.

    Returns the wandb module, or None when logging is disabled. wandb is
    imported lazily so `import train_mlm` — which src/demo.py and
    src/smoke_test.py do — never requires the package or a login. Those two
    scripts used to monkey-patch a stub module into sys.modules for exactly
    this reason.
    """
    mode = str(config.get("wandb_mode", "online")).lower()
    if mode not in WANDB_MODES:
        raise ValueError(f"wandb_mode must be one of {WANDB_MODES}, got {mode!r}")

    if mode == "disabled":
        print("wandb: disabled by config")
        return None

    import wandb

    wandb.init(
        project=config["wandb_project"],
        name=config["wandb_run_name"],
        config=config,
        mode=mode,
    )
    return wandb


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
    scaler: torch.amp.GradScaler,
) -> tuple[float, int]:
    model.train()
    loss_fn  = nn.CrossEntropyLoss(ignore_index=-100)
    use_amp  = amp_enabled(config, device)
    grad_acc = config["grad_accumulation_steps"]

    total_loss    = 0.0
    batch_count   = 0
    skipped       = 0
    pending_grads = False
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

        # CrossEntropyLoss(ignore_index=-100) over an all-ignored target is a
        # 0/0 mean: it returns NaN, and one NaN backward poisons every weight.
        # Vanishingly unlikely at batch_size 32, ~1 in 1,250 batches at the
        # small batch sizes used for smoke tests.
        n_masked = int((labels != -100).sum().item())
        if n_masked == 0:
            skipped += 1
            continue

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(masked_ids, attention_mask)
            # logits: (B, T, V) → reshape to (B*T, V); labels: (B, T) → (B*T,)
            loss = loss_fn(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )
            loss = loss / grad_acc

        scaler.scale(loss).backward()
        pending_grads = True
        total_loss  += loss.item() * grad_acc
        batch_count += 1

        if (step + 1) % grad_acc == 0:
            scaler.unscale_(optimiser)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimiser)
            scaler.update()
            scheduler.step()
            optimiser.zero_grad()
            pending_grads = False

    # Flush any remaining accumulated gradients at epoch end. Keyed on whether
    # gradients are actually pending, not on the batch count — a skipped
    # degenerate batch would otherwise desynchronise the two.
    if pending_grads:
        scaler.unscale_(optimiser)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimiser)
        scaler.update()
        scheduler.step()
        optimiser.zero_grad()

    mean_loss = total_loss / batch_count if batch_count else 0.0
    return mean_loss, skipped


def evaluate(
    model: SunuwarBERT,
    dataloader: torch.utils.data.DataLoader,
    tokeniser: SunuwarTokeniser,
    config: dict,
    device: torch.device,
) -> tuple[float, float, int]:
    """Held-out MLM loss and perplexity.

    Two properties this loop needs and previously lacked:

    * **Fixed masking.** The mask used to be re-drawn from the global RNG on
      every call, so consecutive epochs were scored on different held-out
      targets. On ~2,600 masked tokens the sampling spread is comparable to
      the gap between the reported best (14.79 at epoch 30) and the value that
      triggered early stopping (15.60 at epoch 35) — i.e. the stopping
      decision was partly noise. A generator re-seeded identically before
      every pass makes the evaluation set genuinely held *fixed*.
    * **Token weighting.** Averaging per-batch means over-weights batches with
      few masked tokens. Perplexity is a per-token quantity, so weight each
      batch by its masked-token count.
    """
    model.eval()
    loss_fn      = nn.CrossEntropyLoss(ignore_index=-100)
    use_amp      = amp_enabled(config, device)

    generator = torch.Generator(device=device)
    generator.manual_seed(config["seed"])

    total_loss   = 0.0
    total_masked = 0
    skipped      = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            masked_ids, labels, attention_mask = apply_mlm_mask(
                batch,
                tokeniser,
                mlm_probability=config["mlm_probability"],
                mask_ratio=config["mask_ratio"],
                random_ratio=config["random_ratio"],
                generator=generator,
            )
            masked_ids     = masked_ids.to(device)
            labels         = labels.to(device)
            attention_mask = attention_mask.to(device)

            n_masked = int((labels != -100).sum().item())
            if n_masked == 0:
                skipped += 1
                continue

            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(masked_ids, attention_mask)
                loss   = loss_fn(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                )

            total_loss   += loss.item() * n_masked
            total_masked += n_masked

    mean_loss   = total_loss / total_masked if total_masked else 0.0
    perplexity  = math.exp(mean_loss)
    return mean_loss, perplexity, skipped


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
    # claude.md: "All scripts accept a YAML config file as first argument"
    config_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG_PATH
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    print(f"Config: {config_path}")

    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    run = init_wandb(config)

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

    grad_acc = config["grad_accumulation_steps"]
    # train_one_epoch flushes leftover gradients at the end of every epoch, so
    # a partial accumulation group still costs one scheduler step. Floor
    # division dropped it and under-ran the schedule (11,075 vs 11,100 steps
    # over the configured 100 epochs), pinning the LR at 0 before the end.
    steps_per_epoch = math.ceil(len(train_loader) / grad_acc)
    total_steps     = config["epochs"] * steps_per_epoch
    warmup_steps    = int(config["warmup_ratio"] * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimiser,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    print(
        f"Schedule: {steps_per_epoch} optimiser steps/epoch × {config['epochs']} epochs "
        f"= {total_steps} total ({warmup_steps} warmup)"
    )

    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled(config, device))

    with open(config["test_path"], encoding="utf-8") as f:
        probe_sentences = [f.readline().strip() for _ in range(3)]

    best_perplexity  = float("inf")
    best_epoch       = 0
    patience_counter = 0
    patience         = config["early_stopping_patience"]

    for epoch in range(1, config["epochs"] + 1):
        train_loss, train_skipped = train_one_epoch(
            model, train_loader, optimiser, scheduler, tokeniser, config, device, scaler,
        )
        val_loss, val_perplexity, val_skipped = evaluate(
            model, val_loader, tokeniser, config, device,
        )

        print(
            f"Epoch {epoch:3d} | "
            f"train_loss {train_loss:.4f} | "
            f"val_loss {val_loss:.4f} | "
            f"val_perplexity {val_perplexity:.2f}"
        )
        if train_skipped or val_skipped:
            print(
                f"          skipped {train_skipped} train / {val_skipped} val "
                f"batch(es) with no masked token"
            )
        probe_predictions(model, tokeniser, probe_sentences, device)

        if run is not None:
            run.log({
                "epoch":              epoch,
                "train_loss":         train_loss,
                "val_loss":           val_loss,
                "val_perplexity":     val_perplexity,
                "train_batches_skipped": train_skipped,
                "val_batches_skipped":   val_skipped,
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
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
