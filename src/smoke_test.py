# Data source: Sunuwar Bible (suzBl), © 2011 Wycliffe Bible Translators, Inc.
# Licence: CC BY-NC-ND 4.0 — non-commercial research use only
# Do not redistribute raw text. Models trained on this data may be released.

import sys

# Force UTF-8 output so Devanagari prints correctly in any terminal
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import torch
import yaml

SAMPLE = "येसु ख्रीस्त आ पर्बम"
NEIGHBOUR_WORD = "येसु"

# ── 1. SentencePiece ─────────────────────────────────────────────────────────
try:
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load("models/sunuwar_spm_8k.model")
    tokens = sp.encode(SAMPLE, out_type=str)
    ids    = sp.encode(SAMPLE)
    print(f"[SPM]  tokens : {tokens}")
    print(f"[SPM]  ids    : {ids}")
    print("PASS  SentencePiece")
except Exception as e:
    print(f"FAIL  SentencePiece — {e}", file=sys.stderr)

# ── 2. word2vec Skip-gram ────────────────────────────────────────────────────
try:
    from gensim.models import Word2Vec
    w2v = Word2Vec.load("models/sunuwar_w2v_sg.model")
    neighbours = w2v.wv.most_similar(NEIGHBOUR_WORD, topn=3)
    print(f"\n[W2V-SG]  top-3 neighbours of '{NEIGHBOUR_WORD}':")
    for word, score in neighbours:
        print(f"          {word}  ({score:.4f})")
    print("PASS  word2vec Skip-gram")
except Exception as e:
    print(f"FAIL  word2vec Skip-gram — {e}", file=sys.stderr)

# ── 3. fastText ──────────────────────────────────────────────────────────────
try:
    from gensim.models import FastText
    ft = FastText.load("models/sunuwar_fasttext.bin")
    neighbours = ft.wv.most_similar(NEIGHBOUR_WORD, topn=3)
    print(f"\n[FT]   top-3 neighbours of '{NEIGHBOUR_WORD}':")
    for word, score in neighbours:
        print(f"          {word}  ({score:.4f})")
    print("PASS  fastText")
except Exception as e:
    print(f"FAIL  fastText — {e}", file=sys.stderr)

# ── 4. SunuwarBERT-small ─────────────────────────────────────────────────────
try:
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from train_mlm import SunuwarBERT

    checkpoint = "models/sunuwar_transformer.pt"
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"{checkpoint} not found — run train_mlm.py first")

    with open("configs/mlm.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = SunuwarBERT(config)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    dummy_ids  = torch.randint(1, config["vocab_size"], (1, 32))
    dummy_mask = torch.ones(1, 32, dtype=torch.long)
    with torch.no_grad():
        logits = model(dummy_ids, dummy_mask)
    print(f"\n[BERT] output shape: {tuple(logits.shape)}")
    print("PASS  SunuwarBERT-small")
except Exception as e:
    print(f"FAIL  SunuwarBERT-small — {e}", file=sys.stderr)
