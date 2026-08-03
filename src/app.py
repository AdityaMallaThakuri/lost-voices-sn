# Data source: Sunuwar Bible (suzBl), © 2011 Wycliffe Bible Translators, Inc.
# Licence: CC BY-NC-ND 4.0 — non-commercial research use only

import os
import sys
import math
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import SunuwarBERT, SunuwarTokeniser, apply_mlm_mask  # noqa: E402

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SunuwarBERT Demo",
    page_icon="🏔️",
    layout="wide",
)


class SunuwarMLMDataset(torch.utils.data.Dataset):
    def __init__(self, file_path: str, tokeniser: SunuwarTokeniser):
        with open(file_path, encoding="utf-8") as f:
            lines = [l.rstrip("\n") for l in f if l.strip()]
        self.encoded = [tokeniser.encode(line) for line in lines]

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, idx):
        return torch.tensor(self.encoded[idx], dtype=torch.long)


def make_dataloader(dataset, tokeniser, batch_size=32):
    def collate(batch):
        max_len = max(t.size(0) for t in batch)
        padded  = [torch.cat([t, torch.full((max_len - t.size(0),), tokeniser.pad_id, dtype=torch.long)]) for t in batch]
        return torch.stack(padded)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def encode_with_mask(sentence: str, tokeniser: SunuwarTokeniser):
    left, right = sentence.split("[MASK]", 1)
    left_ids    = tokeniser.sp.encode(left.strip())  if left.strip()  else []
    right_ids   = tokeniser.sp.encode(right.strip()) if right.strip() else []
    ids         = [tokeniser.cls_id] + left_ids + [tokeniser.mask_id] + right_ids + [tokeniser.sep_id]
    return ids[:tokeniser.max_seq_len], 1 + len(left_ids)


def predict_masked(model, tokeniser, sentence: str, top_k: int = 5):
    ids, mask_pos  = encode_with_mask(sentence, tokeniser)
    input_tensor   = torch.tensor([ids], dtype=torch.long)
    attn_mask      = (input_tensor != tokeniser.pad_id).long()
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor, attn_mask)
    probs             = F.softmax(logits[0, mask_pos], dim=-1)
    top_probs, top_ids = probs.topk(top_k)
    return [(tokeniser.sp.id_to_piece(tid.item()), prob.item()) for prob, tid in zip(top_probs, top_ids)]


@st.cache_resource(show_spinner="Loading SunuwarBERT-small checkpoint…")
def load_model():
    import yaml
    with open("configs/mlm.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    tokeniser = SunuwarTokeniser(config["tokeniser_path"], config["vocab_size"], config["max_seq_len"])
    model     = SunuwarBERT(config)
    ckpt      = torch.load(config["checkpoint_path"], map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt)
    model.eval()
    param_count = sum(p.numel() for p in model.parameters())
    return model, tokeniser, config, param_count


@st.cache_data(show_spinner="Computing test-set perplexity…")
def get_perplexity(test_path: str, max_sentences: int = 500):
    model, tokeniser, _, _ = load_model()
    dataset         = SunuwarMLMDataset(test_path, tokeniser)
    dataset.encoded = dataset.encoded[:max_sentences]
    loader          = make_dataloader(dataset, tokeniser)
    loss_fn         = nn.CrossEntropyLoss(ignore_index=-100)
    total, n        = 0.0, 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            masked_ids, labels, attn = apply_mlm_mask(batch, tokeniser)
            logits = model(masked_ids, attn)
            total += loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1)).item()
            n     += 1
    return math.exp(total / n) if n else 0.0


# ---------------------------------------------------------------------------
# Demo sentences
# ---------------------------------------------------------------------------
DEMO_SENTENCES = [
    {
        "label":    "Proper name (subject)",
        "original": "मिनु लिडीया आ खिं लशा बाक्‍तक।",
        "masked":   "मिनु [MASK] आ खिं लशा बाक्‍तक।",
        "correct":  "लिडीया",
    },
    {
        "label":    "Verb in complex clause",
        "original": "दोपा ग्रानीम देंशा हना, येसु ख्रीस्‍त कली थमा सुइक्‍चा मप्रोंइथु ग्रानीम।",
        "masked":   "दोपा ग्रानीम देंशा हना, येसु ख्रीस्‍त कली थमा [MASK] मप्रोंइथु ग्रानीम।",
        "correct":  "सुइक्‍चा",
    },
    {
        "label":    "Verb before auxiliary",
        "original": "तन्‍न गो इन कली लोव़ का पचा माल्‍नुङ।",
        "masked":   "तन्‍न गो इन कली लोव़ का [MASK] माल्‍नुङ।",
        "correct":  "पचा",
    },
    {
        "label":    "Negated verb in question",
        "original": "आफोमी शेंचा मपुंइसीब तौ बाक्‍बा ङा?",
        "masked":   "आफोमी शेंचा [MASK] तौ बाक्‍बा ङा?",
        "correct":  "मपुंइसीब",
    },
]

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

# ── Header ──────────────────────────────────────────────────────────────────
st.title("🏔️ SunuwarBERT-small")
st.markdown(
    "**Masked Language Model · Sunuwar (ISO 639-3: `suz`) · Devanagari script**  \n"
    "First publicly reproducible language model for an endangered Kiranti language of Nepal."
)
st.divider()

# ── Load model ──────────────────────────────────────────────────────────────
model, tokeniser, config, param_count = load_model()

# ── Model info cards ────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Parameters",   f"{param_count:,}")
c2.metric("Layers",       config["num_layers"])
c3.metric("Hidden dim",   config["hidden_dim"])
c4.metric("Vocabulary",   f"{config['vocab_size']:,} pieces")

st.divider()

# ── Perplexity panel ─────────────────────────────────────────────────────────
with st.expander("📊 Model Performance Metrics", expanded=True):
    if st.button("Compute test-set perplexity  (first 500 sentences)"):
        ppl = get_perplexity(config["test_path"])
        st.session_state["ppl"] = ppl

    if "ppl" in st.session_state:
        ppl = st.session_state["ppl"]
        p1, p2, p3 = st.columns(3)
        p1.metric("Test Perplexity",        f"{ppl:.2f}")
        p2.metric("Random Baseline",        "~8,000")
        p3.metric("Improvement Factor",     f"{8000 / ppl:.0f}×")
        st.caption(
            "Perplexity measures how surprised the model is by unseen Sunuwar text. "
            "Lower is better. Random guessing over 8,000 tokens gives ~8,000; "
            f"SunuwarBERT achieves **{ppl:.1f}** — a **{8000/ppl:.0f}× improvement**."
        )

st.divider()

# ── Masked prediction ────────────────────────────────────────────────────────
st.subheader("🔍 Masked Token Prediction")
st.markdown(
    "The model reads a Sunuwar sentence with one word hidden as `[MASK]` "
    "and predicts what that word should be, using bidirectional context."
)

tab_demo, tab_custom = st.tabs(["Demo sentences", "Try your own"])

# ── Demo tab ────────────────────────────────────────────────────────────────
with tab_demo:
    for entry in DEMO_SENTENCES:
        with st.container(border=True):
            col_text, col_chart = st.columns([2, 3])

            with col_text:
                st.markdown(f"**{entry['label']}**")
                st.markdown(f"**Original:** {entry['original']}")
                st.markdown(
                    f"**Input to model:** "
                    + entry["masked"].replace("[MASK]", "**`[MASK]`**")
                )
                st.markdown(f"**Correct answer:** `{entry['correct']}`")

            with col_chart:
                preds = predict_masked(model, tokeniser, entry["masked"])

                labels, values, colors = [], [], []
                correct_rank = None
                for rank, (piece, prob) in enumerate(preds, 1):
                    piece_norm = piece.lstrip("▁")
                    is_correct = piece_norm == entry["correct"] or piece == entry["correct"]
                    if is_correct:
                        correct_rank = rank
                    labels.append(piece)
                    values.append(round(prob * 100, 1))
                    colors.append("#2ecc71" if is_correct else "#3498db")

                # Build a simple HTML bar chart
                chart_html = "<div style='font-family:monospace; font-size:14px;'>"
                for label, value, color in zip(labels, values, colors):
                    bar_w = int(value * 2.5)   # scale: 100% → 250px
                    chart_html += (
                        f"<div style='display:flex; align-items:center; margin:4px 0;'>"
                        f"<span style='width:160px; text-align:right; padding-right:10px; "
                        f"overflow:hidden; white-space:nowrap;'>{label}</span>"
                        f"<div style='background:{color}; width:{bar_w}px; height:22px; "
                        f"border-radius:3px; display:flex; align-items:center; padding-left:6px; "
                        f"color:white; font-size:12px; min-width:40px;'>{value}%</div>"
                        f"</div>"
                    )
                chart_html += "</div>"
                st.markdown(chart_html, unsafe_allow_html=True)

                if correct_rank == 1:
                    st.success(f"✓ Top-1 hit — correct answer ranked #1")
                elif correct_rank:
                    st.warning(f"✓ Top-{correct_rank} hit — correct answer at rank #{correct_rank}")
                else:
                    st.error(f"✗ Correct answer not in top-5")

# ── Custom tab ───────────────────────────────────────────────────────────────
with tab_custom:
    st.markdown(
        "Write any Sunuwar sentence in Devanagari and place `[MASK]` "
        "where you want the model to predict."
    )
    example_hint = "e.g. मिनु [MASK] आ खिं लशा बाक्‍तक।"
    user_input   = st.text_input("Enter a Sunuwar sentence with [MASK]:", placeholder=example_hint)

    if user_input:
        if "[MASK]" not in user_input:
            st.error("Please include `[MASK]` in your sentence.")
        else:
            preds = predict_masked(model, tokeniser, user_input)

            st.markdown("#### Top-5 Predictions")
            cols = st.columns(5)
            for col, (piece, prob) in zip(cols, preds):
                col.metric(label=piece, value=f"{prob*100:.1f}%")

            # Bar chart
            chart_html = "<div style='font-family:monospace; font-size:14px; margin-top:16px;'>"
            for piece, prob in preds:
                bar_w = int(prob * 250)
                pct   = round(prob * 100, 1)
                chart_html += (
                    f"<div style='display:flex; align-items:center; margin:4px 0;'>"
                    f"<span style='width:160px; text-align:right; padding-right:10px;'>{piece}</span>"
                    f"<div style='background:#3498db; width:{bar_w}px; height:22px; "
                    f"border-radius:3px; display:flex; align-items:center; padding-left:6px; "
                    f"color:white; font-size:12px; min-width:40px;'>{pct}%</div>"
                    f"</div>"
                )
            chart_html += "</div>"
            st.markdown(chart_html, unsafe_allow_html=True)

# ── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "**Lost Voices** — Sunuwar Language AI Pipeline · "
    "Data: Sunuwar Bible (suzBl) © 2011 Wycliffe Bible Translators, CC BY-NC-ND 4.0 · "
    "Non-commercial research use only."
)
