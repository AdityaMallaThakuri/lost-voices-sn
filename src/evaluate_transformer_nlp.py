# Data source: Sunuwar Bible (suzBl), © 2011 Wycliffe Bible Translators, Inc.
# Licence: CC BY-NC-ND 4.0 — non-commercial research use only
# Do not redistribute raw text. Models trained on this data may be released.
#
# Downstream genre-classification eval for the transformer models
# (SunuwarBERT-small, SunuwarCLM-small), matching src/evaluate_nlp.py's
# methodology (same genre-label heuristic, same LogisticRegression +
# f1_macro) but pooling each model's own hidden states instead of static
# word2vec/fastText vectors:
#   - BERT:  the [CLS] token's final hidden state (position 0)
#   - CLM:   the last non-padding token's final hidden state (the only
#            position that has seen the whole sequence under a causal mask)
# Neither model exposes hidden states through its public forward() (which
# returns vocab logits after the LM/MLM head), so this script re-runs each
# model's embedding + encoder submodules directly rather than modifying
# train_mlm.py / train_clm.py's forward signature.

import json
import os
import sys

import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_mlm import SunuwarBERT, SunuwarTokeniser as BERTTokeniser
from train_clm import SunuwarCLM, SunuwarTokeniser as CLMTokeniser


def assign_genre_labels(sentences: list) -> list:
    """Identical to evaluate_nlp.py's heuristic, kept in sync deliberately
    so BERT/CLM genre-F1 numbers are comparable to the word2vec/fastText
    ones already in results/embedding_eval.json."""
    n = len(sentences)
    cut1 = int(n * 0.60)
    cut2 = cut1 + int(n * 0.35)
    labels = [0] * cut1 + [1] * (cut2 - cut1) + [2] * (n - cut2)
    print(f"  Genre labels assigned: {labels.count(0)} Narrative, "
          f"{labels.count(1)} Epistles, {labels.count(2)} Apocalyptic")
    return labels


def batch_encode_padded(sentences: list, tokeniser, batch_size: int):
    """Yield (input_ids, attention_mask) tensors, batch_size sentences at a time."""
    for start in range(0, len(sentences), batch_size):
        chunk = sentences[start:start + batch_size]
        encoded = [tokeniser.encode(s) for s in chunk]
        max_len = max(len(e) for e in encoded)
        padded = [e + [tokeniser.pad_id] * (max_len - len(e)) for e in encoded]
        input_ids = torch.tensor(padded, dtype=torch.long)
        attention_mask = (input_ids != tokeniser.pad_id).long()
        yield input_ids, attention_mask


@torch.no_grad()
def pool_bert(model: SunuwarBERT, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    seq_len = input_ids.size(1)
    pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
    x = model.embedding(input_ids) + model.pos_embedding(pos_ids)
    padding_mask = attention_mask == 0
    hidden = model.encoder(x, src_key_padding_mask=padding_mask)
    return hidden[:, 0, :]  # [CLS] is always position 0 -- see SunuwarTokeniser.encode()


@torch.no_grad()
def pool_clm(model: SunuwarCLM, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    seq_len = input_ids.size(1)
    pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
    x = model.embedding(input_ids) + model.pos_embedding(pos_ids)
    causal_mask = torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_ids.device), diagonal=1,
    )
    padding_mask = attention_mask == 0
    hidden = model.encoder(x, mask=causal_mask, src_key_padding_mask=padding_mask, is_causal=True)
    last_real_idx = attention_mask.sum(dim=1) - 1  # 0-indexed position of the last real token
    batch_idx = torch.arange(hidden.size(0), device=hidden.device)
    return hidden[batch_idx, last_real_idx, :]


def vectorise_sentences(sentences: list, labels: list, model, tokeniser, pool_fn, batch_size: int, device):
    vectors, kept_labels = [], []
    label_idx = 0
    for input_ids, attention_mask in batch_encode_padded(sentences, tokeniser, batch_size):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        pooled = pool_fn(model, input_ids, attention_mask).cpu().numpy()
        n = pooled.shape[0]
        vectors.extend(pooled[i] for i in range(n))
        kept_labels.extend(labels[label_idx:label_idx + n])
        label_idx += n
    return vectors, kept_labels


def run_genre_classification(model, tokeniser, pool_fn, train_path, test_path, batch_size, device) -> float:
    with open(train_path, encoding="utf-8") as fh:
        train_sentences = [l.rstrip("\n") for l in fh if l.strip()]
    train_labels = assign_genre_labels(train_sentences)
    X_train, y_train = vectorise_sentences(train_sentences, train_labels, model, tokeniser, pool_fn, batch_size, device)

    with open(test_path, encoding="utf-8") as fh:
        test_sentences = [l.rstrip("\n") for l in fh if l.strip()]
    test_labels = assign_genre_labels(test_sentences)
    X_test, y_test = vectorise_sentences(test_sentences, test_labels, model, tokeniser, pool_fn, batch_size, device)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return float(f1_score(y_test, y_pred, average="macro"))


def load_bert(mlm_config_path: str, device):
    with open(mlm_config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    checkpoint_path = config["checkpoint_path"]
    if not os.path.exists(checkpoint_path):
        print(f"  SKIPPED: checkpoint not found at {checkpoint_path}")
        return None, None
    tokeniser = BERTTokeniser(config["tokeniser_path"], config["vocab_size"], config["max_seq_len"])
    model = SunuwarBERT(config).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model, tokeniser


def load_clm(clm_config_path: str, device):
    with open(clm_config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    checkpoint_path = config["checkpoint_path"]
    if not os.path.exists(checkpoint_path):
        print(f"  SKIPPED: checkpoint not found at {checkpoint_path}")
        print(f"  (train_clm.py saves this on Colab -- download it from Drive, or run this eval on Colab)")
        return None, None
    tokeniser = CLMTokeniser(config["tokeniser_path"], config["vocab_size"], config["max_seq_len"])
    model = SunuwarCLM(config).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model, tokeniser


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/eval_transformer_nlp.yaml"
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    results = {}

    print("\n=== SunuwarBERT-small (pooling: [CLS] token) ===")
    bert_model, bert_tok = load_bert(cfg["mlm_config_path"], device)
    if bert_model is not None:
        f1 = run_genre_classification(
            bert_model, bert_tok, pool_bert,
            cfg["train_path"], cfg["test_path"], cfg["batch_size"], device,
        )
        print(f"  Genre F1 macro: {f1:.4f}")
        results["sunuwarBERT-small"] = {"pooling": "cls_token", "genre_f1_macro": round(f1, 4)}
    else:
        results["sunuwarBERT-small"] = {"pooling": "cls_token", "genre_f1_macro": None, "status": "checkpoint_missing"}

    print("\n=== SunuwarCLM-small (pooling: last non-pad token) ===")
    clm_model, clm_tok = load_clm(cfg["clm_config_path"], device)
    if clm_model is not None:
        f1 = run_genre_classification(
            clm_model, clm_tok, pool_clm,
            cfg["train_path"], cfg["test_path"], cfg["batch_size"], device,
        )
        print(f"  Genre F1 macro: {f1:.4f}")
        results["sunuwarCLM-small"] = {"pooling": "last_token", "genre_f1_macro": round(f1, 4)}
    else:
        results["sunuwarCLM-small"] = {"pooling": "last_token", "genre_f1_macro": None, "status": "checkpoint_missing"}

    os.makedirs(os.path.dirname(cfg["output_path"]), exist_ok=True)
    with open(cfg["output_path"], "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {cfg['output_path']}")

    print()
    print(f"{'model':<20}  {'pooling':<14}  {'genre_f1_macro':>14}")
    print("-" * 54)
    for name, r in results.items():
        f1_str = f"{r['genre_f1_macro']:.4f}" if r["genre_f1_macro"] is not None else "N/A"
        print(f"{name:<20}  {r['pooling']:<14}  {f1_str:>14}")


if __name__ == "__main__":
    main()
