# Data source: Sunuwar Bible (suzBl), © 2011 Wycliffe Bible Translators, Inc.
# Licence: CC BY-NC-ND 4.0 — non-commercial research use only

import sys
import math
import argparse
import random
import warnings
warnings.filterwarnings("ignore")

# Force UTF-8 output so Devanagari and box-drawing characters print correctly
# on Windows terminals that default to cp1252.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import SunuwarBERT, SunuwarTokeniser, apply_mlm_mask  # noqa: E402


class SunuwarMLMDataset(torch.utils.data.Dataset):
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


# ---------------------------------------------------------------------------
# Demo sentences — real test-set sentences with one word masked.
# CORRECT is the original word so we can score the model.
# ---------------------------------------------------------------------------
DEMO_SENTENCES = [
    {
        "original": "मिनु लिडीया आ खिं लशा बाक्‍तक।",
        "masked":   "मिनु [MASK] आ खिं लशा बाक्‍तक।",
        "correct":  "लिडीया",
        "note":     "Masking a proper name (subject position)",
    },
    {
        "original": "दोपा ग्रानीम देंशा हना, येसु ख्रीस्‍त कली थमा सुइक्‍चा मप्रोंइथु ग्रानीम।",
        "masked":   "दोपा ग्रानीम देंशा हना, येसु ख्रीस्‍त कली थमा [MASK] मप्रोंइथु ग्रानीम।",
        "correct":  "सुइक्‍चा",
        "note":     "Masking a verb in a complex Biblical clause",
    },
    {
        "original": "तन्‍न गो इन कली लोव़ का पचा माल्‍नुङ।",
        "masked":   "तन्‍न गो इन कली लोव़ का [MASK] माल्‍नुङ।",
        "correct":  "पचा",
        "note":     "Masking a verb before sentence-final auxiliary",
    },
    {
        "original": "आफोमी शेंचा मपुंइसीब तौ बाक्‍बा ङा?",
        "masked":   "आफोमी शेंचा [MASK] तौ बाक्‍बा ङा?",
        "correct":  "मपुंइसीब",
        "note":     "Masking a negated verb in a question",
    },
]


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

def encode_with_mask(sentence: str, tokeniser: SunuwarTokeniser):
    """Split sentence on [MASK], encode each side, insert MASK token."""
    if "[MASK]" not in sentence:
        raise ValueError("Sentence must contain [MASK].")

    left, right = sentence.split("[MASK]", 1)
    left_ids  = tokeniser.sp.encode(left.strip())  if left.strip()  else []
    right_ids = tokeniser.sp.encode(right.strip()) if right.strip() else []

    ids      = [tokeniser.cls_id] + left_ids + [tokeniser.mask_id] + right_ids + [tokeniser.sep_id]
    mask_pos = 1 + len(left_ids)
    return ids[:tokeniser.max_seq_len], mask_pos


def predict_masked(model, tokeniser, sentence: str, device, top_k: int = 5):
    ids, mask_pos = encode_with_mask(sentence, tokeniser)
    input_tensor  = torch.tensor([ids], dtype=torch.long, device=device)
    attn_mask     = (input_tensor != tokeniser.pad_id).long()

    model.eval()
    with torch.no_grad():
        logits = model(input_tensor, attn_mask)

    probs             = F.softmax(logits[0, mask_pos], dim=-1)
    top_probs, top_ids = probs.topk(top_k)

    return [(tokeniser.sp.id_to_piece(tid.item()), prob.item())
            for prob, tid in zip(top_probs, top_ids)]


def compute_perplexity(model, tokeniser, test_path: str, device, max_sentences: int = 500):
    dataset         = SunuwarMLMDataset(test_path, tokeniser)
    dataset.encoded = dataset.encoded[:max_sentences]
    loader          = make_dataloader(dataset, tokeniser, batch_size=32, shuffle=False)
    loss_fn         = nn.CrossEntropyLoss(ignore_index=-100)

    model.eval()
    total_loss, n_batches = 0.0, 0

    with torch.no_grad():
        for batch in loader:
            batch                       = batch.to(device)
            masked_ids, labels, attn   = apply_mlm_mask(batch, tokeniser)
            logits                      = model(masked_ids.to(device), attn.to(device))
            loss                        = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1).to(device))
            total_loss += loss.item()
            n_batches  += 1

    mean_loss = total_loss / n_batches if n_batches else 0.0
    return math.exp(mean_loss)


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

SEP  = "─" * 68
DSEP = "═" * 68

def banner():
    print()
    print(DSEP)
    print("   SunuwarBERT-small  ·  Masked Language Model — Inference Demo")
    print("   Language : Sunuwar (ISO 639-3: suz)  ·  Script : Devanagari")
    print("   Project  : Lost Voices — endangered language NLP & TTS pipeline")
    print(DSEP)


def show_prediction(entry, predictions, idx, total):
    print(f"\n  Demo {idx}/{total}  ·  {entry['note']}")
    print(SEP)
    print(f"  Original :  {entry['original']}")
    print(f"  Input    :  {entry['masked']}")
    if entry["correct"] != "—":
        print(f"  Answer   :  {entry['correct']}")
    print()
    print(f"  {'Rank':<5}  {'Prediction':<24}  {'Confidence':>10}  {'Bar (each block = 5%)'}")
    print(f"  {'────':<5}  {'────────────────────────':<24}  {'──────────':>10}  {'────────────────────'}")

    correct_rank = None
    for rank, (piece, prob) in enumerate(predictions, start=1):
        pct       = prob * 100
        # SentencePiece prepends ▁ (U+2581) to word-initial pieces — strip it for comparison
        piece_norm = piece.lstrip("▁")
        is_correct = (piece_norm == entry["correct"] or piece == entry["correct"])
        marker     = "  ✓ correct" if is_correct else ""
        if is_correct:
            correct_rank = rank
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"  {rank:<5}  {piece:<24}  {pct:>9.1f}%  {bar}{marker}")

    print()
    if entry["correct"] == "—":
        pass  # custom input — no correctness check
    elif correct_rank == 1:
        print(f"  ✓  Top-1 hit — correct answer ranked #1")
    elif correct_rank:
        print(f"  ✓  Top-{correct_rank} hit — correct answer at rank #{correct_rank}")
    else:
        print(f"  ✗  Correct answer not in top-{len(predictions)}")
    print(SEP)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SunuwarBERT-small masked prediction demo")
    parser.add_argument("--config",   default="configs/mlm.yaml", help="YAML config path")
    parser.add_argument("--no-eval",  action="store_true",        help="Skip test-set perplexity")
    parser.add_argument("--sentence", default=None,
                        help='Custom Sunuwar sentence with [MASK], e.g. "येसु [MASK] लाई भन्नुभयो ।"')
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokeniser = SunuwarTokeniser(
        model_path  = config["tokeniser_path"],
        vocab_size  = config["vocab_size"],
        max_seq_len = config["max_seq_len"],
    )

    model = SunuwarBERT(config).to(device)
    ckpt  = torch.load(config["checkpoint_path"], map_location=device, weights_only=True)
    model.load_state_dict(ckpt)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())

    # ── Banner ──────────────────────────────────────────────────────────────
    banner()
    print(f"\n  Checkpoint  :  {config['checkpoint_path']}")
    print(f"  Parameters  :  {param_count:,}")
    print(f"  Vocabulary  :  {config['vocab_size']:,} SentencePiece pieces (unigram)")
    print(f"  Architecture:  {config['num_layers']} layers · {config['hidden_dim']}d · {config['num_heads']} heads")
    print(f"  Device      :  {device}")

    # ── Perplexity ───────────────────────────────────────────────────────────
    if not args.no_eval:
        print(f"\n  Computing perplexity on test set (first 500 sentences) ...", end=" ", flush=True)
        ppl = compute_perplexity(model, tokeniser, config["test_path"], device)
        print("done.\n")
        print(f"  {'Metric':<35}  {'Value':>8}")
        print(f"  {'─' * 35}  {'─' * 8}")
        print(f"  {'Test perplexity  (500 sentences)':<35}  {ppl:>8.2f}")
        print(f"  {'Random-baseline perplexity':<35}  {'~8000':>8}")
        print(f"  {'Improvement over random':<35}  {8000 / ppl:>7.0f}×")

    # ── Predictions ──────────────────────────────────────────────────────────
    print(f"\n\n{DSEP}")
    print("  MASKED TOKEN PREDICTIONS")
    print(DSEP)

    sentences = DEMO_SENTENCES
    if args.sentence:
        sentences = [{
            "original": args.sentence,
            "masked":   args.sentence,
            "correct":  "—",
            "note":     "Custom input",
        }]

    for i, entry in enumerate(sentences, start=1):
        preds = predict_masked(model, tokeniser, entry["masked"], device, top_k=5)
        show_prediction(entry, preds, i, len(sentences))

    print("\n  End of demo.\n")


if __name__ == "__main__":
    main()
