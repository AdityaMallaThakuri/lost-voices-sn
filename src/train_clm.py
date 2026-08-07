# Data source: Sunuwar Bible (suzBl), © 2011 Wycliffe Bible Translators, Inc.
# Licence: CC BY-NC-ND 4.0 — non-commercial research use only
# Do not redistribute raw text. Models trained on this data may be released.
#
# SunuwarCLM-small — GPT-style causal decoder-only comparison model against
# SunuwarBERT-small (src/train_mlm.py). Two architecture choices were made
# deliberately, not by default, to keep this a controlled comparison where
# the pretraining OBJECTIVE (causal LM vs. MLM) is the only real variable:
#
#   1. Post-LN, not Pre-LN. nn.TransformerEncoderLayer defaults to Post-LN
#      (norm_first=False) and that's what SunuwarBERT-small trains with,
#      even though the reference 210K-parameter Nepali char-level transformer
#      this architecture is adapted from used Pre-LN. We match BERT's Post-LN
#      here on purpose so a difference in results can't be blamed on norm
#      placement instead of the objective.
#   2. Config C from the feasibility investigation: 6 layers, hidden=384,
#      8 heads, ffn=1280 -- identical layers/hidden/heads to SunuwarBERT-small,
#      only FFN width (1280 vs 1024) and vocab table (6,764 actual SPM-8k
#      pieces vs BERT's padded 8000) differ, landing at ~14.72M params
#      against BERT's 14,485,568 (~1.6% over, the closest clean match found;
#      an exact-fit ffn=1216 was rejected as an arbitrary, unmotivated width).

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


class SunuwarTokeniser:
    def __init__(self, model_path: str, vocab_size: int, max_seq_len: int):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.pad_id = 0
        self.unk_id = 1
        # [CLS]/[SEP] are BERT-vocabulary artifacts baked into the shared SPM
        # model; reused here as BOS/EOS so no separate tokenizer is needed.
        self.bos_id = 2
        self.eos_id = 3

    def encode(self, text: str) -> list[int]:
        ids = self.sp.encode(text)
        ids = [self.bos_id] + ids + [self.eos_id]
        return ids[:self.max_seq_len]

    def batch_encode(self, texts: list[str], pad: bool = True) -> torch.Tensor:
        encoded = [self.encode(t) for t in texts]
        if pad:
            max_len = max(len(e) for e in encoded)
            encoded = [e + [self.pad_id] * (max_len - len(e)) for e in encoded]
        return torch.tensor(encoded, dtype=torch.long)


class SunuwarCLM(nn.Module):
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
            norm_first=False,  # Post-LN, matching SunuwarBERT-small -- see module docstring
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

        total = sum(p.numel() for p in self.parameters())
        print(f"SunuwarCLM-small: {total:,} parameters")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.size(1)
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        x = self.embedding(input_ids) + self.pos_embedding(pos_ids)

        # Causal mask: position i may only attend to positions <= i.
        # Built as bool (True = disallowed) to match src_key_padding_mask's
        # dtype -- mixing a float mask with a bool padding mask triggers a
        # PyTorch deprecation warning (and will be a hard error eventually).
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_ids.device),
            diagonal=1,
        )

        # TransformerEncoder expects True where tokens should be *ignored*
        padding_mask = attention_mask == 0

        x = self.encoder(x, mask=causal_mask, src_key_padding_mask=padding_mask, is_causal=True)
        return self.lm_head(x)


class SunuwarCLMDataset(torch.utils.data.Dataset):
    def __init__(self, file_path: str, tokeniser: SunuwarTokeniser):
        with open(file_path, encoding="utf-8") as f:
            lines = [l.rstrip("\n") for l in f if l.strip()]
        self.encoded = [tokeniser.encode(line) for line in lines]

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(self.encoded[idx], dtype=torch.long)


def make_dataloader(
    dataset: SunuwarCLMDataset,
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


def shift_for_clm(
    batch: torch.Tensor,
    tokeniser: SunuwarTokeniser,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Next-token-prediction shift: input = tokens[:-1], target = tokens[1:].

    No random masking (that's the MLM-specific piece this replaces) -- every
    non-pad position is a training signal for causal LM.
    """
    input_ids = batch[:, :-1]
    labels    = batch[:, 1:].clone()

    # Don't train the loss on padding positions.
    labels[labels == tokeniser.pad_id] = -100

    attention_mask = (input_ids != tokeniser.pad_id).long()

    return input_ids, labels, attention_mask


def train_one_epoch(
    model: SunuwarCLM,
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
        input_ids, labels, attention_mask = shift_for_clm(batch, tokeniser)
        labels = labels.to(device)

        with torch.autocast(device_type=device.type, enabled=fp16):
            logits = model(input_ids, attention_mask)
            loss = loss_fn(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
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

    if batch_count % grad_acc != 0:
        scaler.unscale_(optimiser)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimiser)
        scaler.update()
        scheduler.step()
        optimiser.zero_grad()

    return total_loss / batch_count if batch_count else 0.0


def evaluate(
    model: SunuwarCLM,
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
            input_ids, labels, attention_mask = shift_for_clm(batch, tokeniser)
            labels = labels.to(device)

            with torch.autocast(device_type=device.type, enabled=fp16):
                logits = model(input_ids, attention_mask)
                loss   = loss_fn(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                )

            total_loss  += loss.item()
            batch_count += 1

    mean_loss   = total_loss / batch_count if batch_count else 0.0
    perplexity  = math.exp(mean_loss)
    return mean_loss, perplexity


def probe_generation(
    model: SunuwarCLM,
    tokeniser: SunuwarTokeniser,
    probe_sentences: list[str],
    device: torch.device,
    num_new_tokens: int = 12,
    prompt_tokens: int = 4,
) -> None:
    """Qualitative sanity check: greedily continue a short prompt.

    Replaces MLM's probe_predictions() (which masked one position and showed
    top-5 fill-ins) -- a causal LM has no masked position to probe, so the
    natural equivalent is greedy next-token continuation from a short prompt.
    """
    model.eval()
    with torch.no_grad():
        for i, sentence in enumerate(probe_sentences):
            full_ids = tokeniser.encode(sentence)

            if len(full_ids) <= prompt_tokens + 1:
                print(f"Probe {i}: sentence too short for a {prompt_tokens}-token prompt, skipping")
                continue

            # Prompt = BOS + first `prompt_tokens` real tokens (drop EOS).
            ids = full_ids[:prompt_tokens + 1]  # +1 to include the BOS at index 0
            generated = list(ids)

            for _ in range(num_new_tokens):
                input_tensor   = torch.tensor([generated], dtype=torch.long, device=device)
                attention_mask = torch.ones_like(input_tensor)
                logits         = model(input_tensor, attention_mask)
                next_id        = logits[0, -1].argmax().item()
                generated.append(next_id)
                if next_id == tokeniser.eos_id:
                    break

            prompt_pieces    = [tokeniser.sp.id_to_piece(t) for t in ids]
            continued_pieces = [tokeniser.sp.id_to_piece(t) for t in generated[len(ids):]]
            print(f"Probe {i}: prompt={''.join(prompt_pieces).replace(chr(9601), ' ').strip()}")
            print(f"          continuation={''.join(continued_pieces).replace(chr(9601), ' ').strip()}")


def main():
    with open("configs/clm.yaml", encoding="utf-8") as f:
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

    model = SunuwarCLM(config).to(device)

    train_dataset = SunuwarCLMDataset(config["train_path"], tokeniser)
    val_dataset   = SunuwarCLMDataset(config["test_path"],  tokeniser)
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
    val_loss_at_best = None
    train_loss_at_best = None
    patience_counter = 0
    patience         = config["early_stopping_patience"]
    early_stopped    = False
    perplexity_by_epoch = {}
    final_train_loss = None
    epochs_run       = 0

    for epoch in range(1, config["epochs"] + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimiser, scheduler, tokeniser, config, device, scaler,
        )
        val_loss, val_perplexity = evaluate(
            model, val_loader, tokeniser, config, device,
        )
        epochs_run = epoch
        final_train_loss = train_loss
        perplexity_by_epoch[str(epoch)] = round(val_perplexity, 2)

        print(
            f"Epoch {epoch:3d} | "
            f"train_loss {train_loss:.4f} | "
            f"val_loss {val_loss:.4f} | "
            f"val_perplexity {val_perplexity:.2f}"
        )
        probe_generation(model, tokeniser, probe_sentences, device)

        wandb.log({
            "epoch":          epoch,
            "train_loss":     train_loss,
            "val_loss":       val_loss,
            "val_perplexity": val_perplexity,
        })

        if val_perplexity < best_perplexity:
            best_perplexity  = val_perplexity
            best_epoch       = epoch
            val_loss_at_best = val_loss
            train_loss_at_best = train_loss
            patience_counter = 0
            torch.save(model.state_dict(), config["checkpoint_path"])
            print("Saved best checkpoint")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                early_stopped = True
                break

    print(f"\nBest val_perplexity: {best_perplexity:.2f} at epoch {best_epoch}")

    # Same schema as results/bert_eval.json so the two models can be
    # compared side by side -- random_baseline_perplexity for a causal LM
    # is exactly the vocab size (uniform guess over the vocabulary).
    total_params = sum(p.numel() for p in model.parameters())
    eval_results = {
        "architecture":               "SunuwarCLM-small",
        "total_parameters":           total_params,
        "best_val_perplexity":        round(best_perplexity, 2),
        "best_epoch":                 best_epoch,
        "total_epochs_run":           epochs_run,
        "early_stopped":              early_stopped,
        "final_train_loss":           round(final_train_loss, 4) if final_train_loss is not None else None,
        "val_loss_at_best":           round(val_loss_at_best, 4) if val_loss_at_best is not None else None,
        "random_baseline_perplexity": config["vocab_size"],
        "improvement_over_random":    round(config["vocab_size"] / best_perplexity, 1) if best_perplexity else None,
        "perplexity_by_epoch":        perplexity_by_epoch,
    }
    eval_output_path = config.get("eval_output_path", "results/clm_eval.json")
    os.makedirs(os.path.dirname(eval_output_path), exist_ok=True)
    with open(eval_output_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)
    print(f"Wrote eval summary to {eval_output_path}")

    wandb.finish()


if __name__ == "__main__":
    main()
