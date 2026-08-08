# Data source: Sunuwar Bible (suzBl), © 2011 Wycliffe Bible Translators, Inc.
# Licence: CC BY-NC-ND 4.0 — non-commercial research use only

import sys
import math
import argparse
import warnings
warnings.filterwarnings("ignore")

# Force UTF-8 output so Devanagari prints correctly on Windows terminals
# that default to cp1252.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import sentencepiece as spm
import yaml

# ---------------------------------------------------------------------------
# Model classes (self-contained copy, mirroring src/demo.py's pattern for
# SunuwarBERT — no wandb/transformers dependency needed for inference)
# ---------------------------------------------------------------------------

class SunuwarTokeniser:
    def __init__(self, model_path: str, vocab_size: int, max_seq_len: int):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)
        self.vocab_size  = vocab_size
        self.max_seq_len = max_seq_len
        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 2  # reused [CLS] slot -- see train_clm.py's docstring
        self.eos_id = 3  # reused [SEP] slot

    def encode(self, text: str) -> list:
        ids = self.sp.encode(text)
        ids = [self.bos_id] + ids + [self.eos_id]
        return ids[:self.max_seq_len]


class SunuwarCLM(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        vocab_size  = config["vocab_size"]
        hidden_dim  = config["hidden_dim"]
        num_heads   = config["num_heads"]
        num_layers  = config["num_layers"]
        ffn_dim     = config["ffn_dim"]
        max_seq_len = config["max_seq_len"]
        dropout     = config["dropout"]
        activation  = config["activation"]

        self.embedding     = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=ffn_dim,
            dropout=dropout, activation=activation, batch_first=True,
            norm_first=False,  # Post-LN, matching SunuwarBERT-small on purpose
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.size(1)
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.embedding(input_ids) + self.pos_embedding(pos_ids)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_ids.device), diagonal=1,
        )
        padding_mask = attention_mask == 0
        x = self.encoder(x, mask=causal_mask, src_key_padding_mask=padding_mask, is_causal=True)
        return self.lm_head(x)


class SunuwarCLMDataset(torch.utils.data.Dataset):
    def __init__(self, file_path: str, tokeniser: SunuwarTokeniser):
        with open(file_path, encoding="utf-8") as f:
            lines = [l.rstrip("\n") for l in f if l.strip()]
        self.encoded = [tokeniser.encode(line) for line in lines]

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, idx):
        return torch.tensor(self.encoded[idx], dtype=torch.long)


def make_dataloader(dataset, tokeniser, batch_size, shuffle):
    def collate_fn(batch):
        max_len = max(t.size(0) for t in batch)
        padded  = [
            torch.cat([t, torch.full((max_len - t.size(0),), tokeniser.pad_id, dtype=torch.long)])
            for t in batch
        ]
        return torch.stack(padded)

    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn,
    )


def compute_perplexity(model, tokeniser, test_path: str, device, max_sentences: int = 500):
    dataset         = SunuwarCLMDataset(test_path, tokeniser)
    dataset.encoded = dataset.encoded[:max_sentences]
    loader          = make_dataloader(dataset, tokeniser, batch_size=32, shuffle=False)
    loss_fn         = nn.CrossEntropyLoss(ignore_index=-100)

    model.eval()
    total_loss, n_batches = 0.0, 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            input_ids = batch[:, :-1]
            labels    = batch[:, 1:].clone()
            labels[labels == tokeniser.pad_id] = -100
            attn_mask = (input_ids != tokeniser.pad_id).long()

            logits = model(input_ids, attn_mask)
            loss   = loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            total_loss += loss.item()
            n_batches  += 1

    mean_loss = total_loss / n_batches if n_batches else 0.0
    return math.exp(mean_loss)


# ---------------------------------------------------------------------------
# Demo prompts — same three sentences used as training-time probes in
# train_clm.py (first 3 lines of test.txt), so this demo shows what the
# FINAL checkpoint (best_epoch=10) does with prompts you've already seen
# mid-training continuations for.
# ---------------------------------------------------------------------------
DEMO_PROMPTS = [
    {"prompt_words": 4, "note": "Same probe sentence used during training (epoch-by-epoch log)"},
    {"prompt_words": 4, "note": "Same probe sentence used during training (epoch-by-epoch log)"},
    {"prompt_words": 4, "note": "Same probe sentence used during training (epoch-by-epoch log)"},
]


def load_first_n_sentences(path: str, n: int) -> list:
    with open(path, encoding="utf-8") as f:
        return [f.readline().strip() for _ in range(n)]


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

def top_k_next_token(model, tokeniser, ids: list, device, top_k: int = 5):
    input_tensor = torch.tensor([ids], dtype=torch.long, device=device)
    attn_mask    = torch.ones_like(input_tensor)
    with torch.no_grad():
        logits = model(input_tensor, attn_mask)
    probs = F.softmax(logits[0, -1], dim=-1)
    top_probs, top_ids = probs.topk(top_k)
    return [(tokeniser.sp.id_to_piece(tid.item()), prob.item())
            for prob, tid in zip(top_probs, top_ids)]


def greedy_generate(model, tokeniser, prompt_ids: list, device, num_new_tokens: int = 20):
    generated = list(prompt_ids)
    model.eval()
    with torch.no_grad():
        for _ in range(num_new_tokens):
            input_tensor = torch.tensor([generated], dtype=torch.long, device=device)
            attn_mask    = torch.ones_like(input_tensor)
            logits       = model(input_tensor, attn_mask)
            next_id      = logits[0, -1].argmax().item()
            generated.append(next_id)
            if next_id == tokeniser.eos_id:
                break
    return generated


def pieces_to_text(tokeniser, ids: list) -> str:
    pieces = [tokeniser.sp.id_to_piece(t) for t in ids]
    text = "".join(pieces).replace("▁", " ").strip()
    return text


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

SEP  = "─" * 68
DSEP = "═" * 68


def banner():
    print()
    print(DSEP)
    print("   SunuwarCLM-small  ·  Causal (GPT-style) Language Model — Demo")
    print("   Language : Sunuwar (ISO 639-3: suz)  ·  Script : Devanagari")
    print("   Project  : Lost Voices — endangered language NLP & TTS pipeline")
    print(DSEP)


def show_generation(idx, total, note, prompt_text, next_token_preds, continuation_text):
    print(f"\n  Demo {idx}/{total}  ·  {note}")
    print(SEP)
    print(f"  Prompt        :  {prompt_text}")
    print()
    print(f"  Next-token candidates (top 5, greedy picks rank #1 each step):")
    print(f"  {'Rank':<5}  {'Prediction':<24}  {'Confidence':>10}  {'Bar (each block = 5%)'}")
    print(f"  {'────':<5}  {'────────────────────────':<24}  {'──────────':>10}  {'────────────────────'}")
    for rank, (piece, prob) in enumerate(next_token_preds, start=1):
        pct = prob * 100
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"  {rank:<5}  {piece:<24}  {pct:>9.1f}%  {bar}")
    print()
    print(f"  Full greedy continuation ({20} new tokens, stops early at </s>):")
    print(f"  {continuation_text}")
    print(SEP)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SunuwarCLM-small text-continuation demo")
    parser.add_argument("--config",  default="configs/clm.yaml", help="YAML config path")
    parser.add_argument("--no-eval", action="store_true",        help="Skip test-set perplexity")
    parser.add_argument("--prompt",  default=None,
                         help='Custom Sunuwar prompt to continue, e.g. "येसु ख्रीस्‍त आ"')
    parser.add_argument("--new-tokens", type=int, default=20, help="Number of tokens to greedily generate")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokeniser = SunuwarTokeniser(
        model_path  = config["tokeniser_path"],
        vocab_size  = config["vocab_size"],
        max_seq_len = config["max_seq_len"],
    )

    model = SunuwarCLM(config).to(device)
    ckpt  = torch.load(config["checkpoint_path"], map_location=device, weights_only=True)
    model.load_state_dict(ckpt)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())

    banner()
    print(f"\n  Checkpoint  :  {config['checkpoint_path']}")
    print(f"  Parameters  :  {param_count:,}")
    print(f"  Vocabulary  :  {config['vocab_size']:,} SentencePiece pieces (unigram)")
    print(f"  Architecture:  {config['num_layers']} layers · {config['hidden_dim']}d · {config['num_heads']} heads · Post-LN")
    print(f"  Device      :  {device}")

    if not args.no_eval:
        print(f"\n  Computing perplexity on test set (first 500 sentences) ...", end=" ", flush=True)
        ppl = compute_perplexity(model, tokeniser, config["test_path"], device)
        print("done.\n")
        print(f"  {'Metric':<35}  {'Value':>8}")
        print(f"  {'─' * 35}  {'─' * 8}")
        print(f"  {'Test perplexity  (500 sentences)':<35}  {ppl:>8.2f}")
        print(f"  {'Random-baseline perplexity':<35}  {config['vocab_size']:>8,}")
        print(f"  {'Improvement over random':<35}  {config['vocab_size'] / ppl:>7.1f}×")

    print(f"\n\n{DSEP}")
    print("  GREEDY TEXT CONTINUATION")
    print(DSEP)

    if args.prompt:
        entries = [{"prompt_text": args.prompt, "note": "Custom prompt"}]
    else:
        test_sentences = load_first_n_sentences(config["test_path"], len(DEMO_PROMPTS))
        entries = []
        for sent, spec in zip(test_sentences, DEMO_PROMPTS):
            words = sent.split()[: spec["prompt_words"]]
            entries.append({"prompt_text": " ".join(words), "note": spec["note"]})

    for i, entry in enumerate(entries, start=1):
        prompt_ids = tokeniser.encode(entry["prompt_text"])[:-1]  # drop the auto-appended EOS
        next_preds = top_k_next_token(model, tokeniser, prompt_ids, device, top_k=5)
        full_ids   = greedy_generate(model, tokeniser, prompt_ids, device, num_new_tokens=args.new_tokens)
        continuation_text = pieces_to_text(tokeniser, full_ids)
        show_generation(i, len(entries), entry["note"], entry["prompt_text"], next_preds, continuation_text)

    print("\n  End of demo.\n")


if __name__ == "__main__":
    main()
