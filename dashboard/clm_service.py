"""
clm_service.py — loads SunuwarCLM-small once and exposes a greedy text-
continuation helper for the FastAPI app. Model classes and inference logic
are relocated from src/demo_clm.py's SunuwarTokeniser/SunuwarCLM/
greedy_generate/top_k_next_token/pieces_to_text (no behavior changes),
following the same load-once-in-__init__ pattern already used by
model_service.py's MLMService.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class SunuwarTokeniser:
    def __init__(self, model_path: str, vocab_size: int, max_seq_len: int):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 2  # reused [CLS] slot -- see train_clm.py's docstring
        self.eos_id = 3  # reused [SEP] slot

    def encode(self, text: str) -> list:
        ids = self.sp.encode(text)
        ids = [self.bos_id] + ids + [self.eos_id]
        return ids[: self.max_seq_len]


class SunuwarCLM(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        vocab_size = config["vocab_size"]
        hidden_dim = config["hidden_dim"]
        num_heads = config["num_heads"]
        num_layers = config["num_layers"]
        ffn_dim = config["ffn_dim"]
        max_seq_len = config["max_seq_len"]
        dropout = config["dropout"]
        activation = config["activation"]

        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
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


def top_k_next_token(model, tokeniser, ids: list, device, top_k: int = 5):
    input_tensor = torch.tensor([ids], dtype=torch.long, device=device)
    attn_mask = torch.ones_like(input_tensor)
    with torch.no_grad():
        logits = model(input_tensor, attn_mask)
    probs = F.softmax(logits[0, -1], dim=-1)
    top_probs, top_ids = probs.topk(top_k)
    return [
        {"piece": tokeniser.sp.id_to_piece(tid.item()), "prob": round(prob.item() * 100, 2)}
        for prob, tid in zip(top_probs, top_ids)
    ]


def greedy_generate(model, tokeniser, prompt_ids: list, device, num_new_tokens: int = 20):
    generated = list(prompt_ids)
    model.eval()
    with torch.no_grad():
        for _ in range(num_new_tokens):
            input_tensor = torch.tensor([generated], dtype=torch.long, device=device)
            attn_mask = torch.ones_like(input_tensor)
            logits = model(input_tensor, attn_mask)
            next_id = logits[0, -1].argmax().item()
            generated.append(next_id)
            if next_id == tokeniser.eos_id:
                break
    return generated


def pieces_to_text(tokeniser, ids: list) -> str:
    pieces = [tokeniser.sp.id_to_piece(t) for t in ids]
    text = "".join(pieces).replace("▁", " ").strip()
    return text


class CLMService:
    def __init__(self):
        config_path = PROJECT_ROOT / "configs" / "clm.yaml"
        with open(config_path, encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        tokeniser_path = PROJECT_ROOT / self.config["tokeniser_path"]
        self.tokeniser = SunuwarTokeniser(
            model_path=str(tokeniser_path),
            vocab_size=self.config["vocab_size"],
            max_seq_len=self.config["max_seq_len"],
        )

        self.model = SunuwarCLM(self.config).to(self.device)
        checkpoint_path = PROJECT_ROOT / self.config["checkpoint_path"]
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt)
        self.model.eval()

        self.param_count = sum(p.numel() for p in self.model.parameters())

    def info(self) -> dict:
        return {
            "params": self.param_count,
            "layers": self.config["num_layers"],
            "hidden_dim": self.config["hidden_dim"],
            "vocab_size": self.config["vocab_size"],
            "heads": self.config["num_heads"],
        }

    def generate(self, prompt: str, num_new_tokens: int = 15) -> str:
        if not prompt.strip():
            raise ValueError("Prompt must not be empty.")
        prompt_ids = self.tokeniser.encode(prompt)[:-1]  # drop the auto-appended EOS
        full_ids = greedy_generate(self.model, self.tokeniser, prompt_ids, self.device, num_new_tokens=num_new_tokens)
        return pieces_to_text(self.tokeniser, full_ids)
