# Data source: Sunuwar Bible (suzBl), © 2011 Wycliffe Bible Translators, Inc.
# Licence: CC BY-NC-ND 4.0 — non-commercial research use only
# Do not redistribute raw text. Models trained on this data may be released.

import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import os
import torch
import torch.nn.functional as F
import sentencepiece as spm
import yaml
from gensim.models import FastText

sys.path.insert(0, os.path.dirname(__file__))
from train_mlm import SunuwarBERT, SunuwarTokeniser


def load_models():
    sp_proc = spm.SentencePieceProcessor()
    sp_proc.Load("models/sunuwar_spm_8k.model")

    tokeniser = SunuwarTokeniser(
        model_path="models/sunuwar_spm_8k.model",
        vocab_size=8000,
        max_seq_len=128,
    )

    ft = FastText.load("models/sunuwar_fasttext.bin")

    with open("configs/mlm.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert = None
    checkpoint = "models/sunuwar_transformer.pt"
    if os.path.exists(checkpoint):
        bert = SunuwarBERT(config).to(device)
        bert.load_state_dict(torch.load(checkpoint, map_location=device))
        bert.eval()
    else:
        print(f"WARNING: {checkpoint} not found — Demo 3 and Demo 4 will be skipped\n")

    return sp_proc, tokeniser, ft, bert, device


# ── Demo 1 ────────────────────────────────────────────────────────────────────

def demo1_tokenisation(sp_proc):
    print("=" * 62)
    print("Demo 1 — Tokenisation (SentencePiece spm_8k)")
    print("=" * 62)
    sentences = [
        "येसु ख्रीस्त आ पर्बम ब्रेक्शो",
        "परमप्रभु यावे आ लोव़",
        "मुर आन कली पाप रे प्रोंइब",
    ]
    for sent in sentences:
        tokens = sp_proc.encode(sent, out_type=str)
        ids    = sp_proc.encode(sent)
        print(f"  Input  : {sent}")
        print(f"  Tokens : {tokens}")
        print(f"  IDs    : {ids}")
        print()


# ── Demo 2 ────────────────────────────────────────────────────────────────────

def demo2_similarity(ft):
    print("=" * 62)
    print("Demo 2 — Word Similarity (fastText)")
    print("=" * 62)
    for word in ["परमप्रभु", "मुर", "लोव़"]:
        neighbours = ft.wv.most_similar(word, topn=5)
        print(f"  Top 5 neighbours of '{word}':")
        for w, score in neighbours:
            print(f"    {score:.4f}  {w}")
        print()


# ── Demo 3 ────────────────────────────────────────────────────────────────────

def _encode_masked_sentence(tokeniser, sentence):
    """Encode a sentence containing the literal string '[MASK]'.
    Returns (input_ids tensor, index of the [MASK] position in the sequence)."""
    left_text, right_text = sentence.split("[MASK]", 1)
    left_ids  = tokeniser.sp.encode(left_text.strip())  if left_text.strip()  else []
    right_ids = tokeniser.sp.encode(right_text.strip()) if right_text.strip() else []

    ids = [tokeniser.cls_id] + left_ids + [tokeniser.mask_id] + right_ids + [tokeniser.sep_id]
    ids = ids[:tokeniser.max_seq_len]
    mask_pos = 1 + len(left_ids)
    return torch.tensor([ids], dtype=torch.long), mask_pos


def demo3_mlm(tokeniser, bert, device):
    print("=" * 62)
    print("Demo 3 — MLM Prediction (SunuwarBERT-small)")
    print("=" * 62)
    if bert is None:
        print("  SKIPPED — checkpoint not available\n")
        return

    examples = [
        ("येसु ख्रीस्त आ [MASK] ब्रेक्शो", "पर्बम"),
        ("परमप्रभु [MASK] आ लोव़",          "यावे"),
        ("[MASK] आन कली पाप रे प्रोंइब",    "मुर"),
    ]
    for sentence, original in examples:
        input_ids, mask_pos = _encode_masked_sentence(tokeniser, sentence)
        input_ids = input_ids.to(device)
        attn_mask = (input_ids != tokeniser.pad_id).long()

        with torch.no_grad():
            logits = bert(input_ids, attn_mask)

        top5 = [tokeniser.sp.id_to_piece(i) for i in logits[0, mask_pos].topk(5).indices.tolist()]
        print(f"  Sentence : {sentence}")
        print(f"  Original : {original}")
        print(f"  Top 5    : {top5}")
        print()


# ── Demo 4 ────────────────────────────────────────────────────────────────────

def _hidden_state_for_word(bert, tokeniser, sentence, target_word, device):
    """Return the encoder hidden state at the first subword position of target_word."""
    ids = tokeniser.encode(sentence)

    # Find the first subword id that belongs to target_word
    target_piece_ids = tokeniser.sp.encode(target_word)
    pos = None
    for tid in target_piece_ids:
        if tid in ids:
            pos = ids.index(tid)
            break
    if pos is None:
        raise ValueError(f"'{target_word}' not found in encoded '{sentence}'")

    input_tensor = torch.tensor([ids], dtype=torch.long, device=device)
    attn_mask    = (input_tensor != tokeniser.pad_id).long()
    padding_mask = attn_mask == 0
    seq_len      = input_tensor.size(1)
    pos_ids      = torch.arange(seq_len, device=device).unsqueeze(0)

    with torch.no_grad():
        x      = bert.embedding(input_tensor) + bert.pos_embedding(pos_ids)
        hidden = bert.encoder(x, src_key_padding_mask=padding_mask)

    return hidden[0, pos]


def demo4_contextual(tokeniser, bert, device):
    print("=" * 62)
    print("Demo 4 — Contextual Representation Difference")
    print("=" * 62)
    if bert is None:
        print("  SKIPPED — checkpoint not available\n")
        return

    ctx1   = "येसु ख्रीस्त आ पर्बम"
    ctx2   = "परमप्रभु यावे आ लोव़"
    target = "आ"

    vec1    = _hidden_state_for_word(bert, tokeniser, ctx1, target, device)
    vec2    = _hidden_state_for_word(bert, tokeniser, ctx2, target, device)
    cos_sim = F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)).item()

    print(f"  Context 1 : {ctx1}")
    print(f"  Context 2 : {ctx2}")
    print(f"  Target    : '{target}'")
    print(f"  Cosine similarity of hidden states : {cos_sim:.4f}")
    if cos_sim < 0.95:
        print("  PASS — contextual representations differ")
    else:
        print("  NOTE — representations are similar")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sp_proc, tokeniser, ft, bert, device = load_models()
    print()
    demo1_tokenisation(sp_proc)
    demo2_similarity(ft)
    demo3_mlm(tokeniser, bert, device)
    demo4_contextual(tokeniser, bert, device)
