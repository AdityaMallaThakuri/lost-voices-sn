"""
model_service.py — loads SunuwarBERT-small once and exposes prediction /
perplexity helpers for the FastAPI app. Model classes are duplicated from
src/train_mlm.py (self-contained, no wandb dependency) rather than imported,
following the same pattern already used by src/app.py and src/demo_mlm.py.
"""

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
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
        self.cls_id = 2
        self.sep_id = 3
        self.mask_id = self.sp.piece_to_id("[MASK]")

    def encode(self, text: str) -> list:
        ids = self.sp.encode(text)
        return ([self.cls_id] + ids + [self.sep_id])[: self.max_seq_len]


class SunuwarBERT(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.embedding = nn.Embedding(config["vocab_size"], config["hidden_dim"], padding_idx=0)
        self.pos_embedding = nn.Embedding(config["max_seq_len"], config["hidden_dim"])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config["hidden_dim"],
            nhead=config["num_heads"],
            dim_feedforward=config["ffn_dim"],
            dropout=config["dropout"],
            activation=config["activation"],
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config["num_layers"])
        self.mlm_head = nn.Linear(config["hidden_dim"], config["vocab_size"])

    def forward(self, input_ids, attention_mask):
        seq_len = input_ids.size(1)
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.embedding(input_ids) + self.pos_embedding(pos_ids)
        x = self.encoder(x, src_key_padding_mask=(attention_mask == 0))
        return self.mlm_head(x)


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
        padded = [
            torch.cat([t, torch.full((max_len - t.size(0),), tokeniser.pad_id, dtype=torch.long)])
            for t in batch
        ]
        return torch.stack(padded)

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)


def apply_mlm_mask(input_ids, tokeniser, mlm_probability=0.15, mask_ratio=0.8, random_ratio=0.1):
    masked_ids = input_ids.clone()
    labels = torch.full_like(input_ids, -100)
    eligible = (input_ids != tokeniser.pad_id) & (input_ids != tokeniser.cls_id) & (input_ids != tokeniser.sep_id)
    rand = torch.rand_like(input_ids, dtype=torch.float)
    selected = eligible & (rand < mlm_probability)
    labels[selected] = input_ids[selected]
    split = torch.rand_like(input_ids, dtype=torch.float)
    masked_ids[selected & (split < mask_ratio)] = tokeniser.mask_id
    to_random = selected & (split >= mask_ratio) & (split < mask_ratio + random_ratio)
    if to_random.any():
        masked_ids[to_random] = torch.randint(4, tokeniser.vocab_size, (to_random.sum().item(),))
    return masked_ids, labels, (input_ids != tokeniser.pad_id).long()


def encode_with_mask(sentence: str, tokeniser: SunuwarTokeniser):
    left, right = sentence.split("[MASK]", 1)
    left_ids = tokeniser.sp.encode(left.strip()) if left.strip() else []
    right_ids = tokeniser.sp.encode(right.strip()) if right.strip() else []
    ids = [tokeniser.cls_id] + left_ids + [tokeniser.mask_id] + right_ids + [tokeniser.sep_id]
    return ids[: tokeniser.max_seq_len], 1 + len(left_ids)


DEMO_SENTENCES = [
    {
        "label": "Proper name (subject)",
        "original": "मिनु लिडीया आ खिं लशा बाक्‍तक।",
        "masked": "मिनु [MASK] आ खिं लशा बाक्‍तक।",
        "correct": "लिडीया",
    },
    {
        "label": "Verb in complex clause",
        "original": "दोपा ग्रानीम देंशा हना, येसु ख्रीस्‍त कली थमा सुइक्‍चा मप्रोंइथु ग्रानीम।",
        "masked": "दोपा ग्रानीम देंशा हना, येसु ख्रीस्‍त कली थमा [MASK] मप्रोंइथु ग्रानीम।",
        "correct": "सुइक्‍चा",
    },
    {
        "label": "Verb before auxiliary",
        "original": "तन्‍न गो इन कली लोव़ का पचा माल्‍नुङ।",
        "masked": "तन्‍न गो इन कली लोव़ का [MASK] माल्‍नुङ।",
        "correct": "पचा",
    },
    {
        "label": "Negated verb in question",
        "original": "आफोमी शेंचा मपुंइसीब तौ बाक्‍बा ङा?",
        "masked": "आफोमी शेंचा [MASK] तौ बाक्‍बा ङा?",
        "correct": "मपुंइसीब",
    },
]


class MLMService:
    def __init__(self):
        config_path = PROJECT_ROOT / "configs" / "mlm.yaml"
        with open(config_path, encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        tokeniser_path = PROJECT_ROOT / self.config["tokeniser_path"]
        self.tokeniser = SunuwarTokeniser(
            model_path=str(tokeniser_path),
            vocab_size=self.config["vocab_size"],
            max_seq_len=self.config["max_seq_len"],
        )

        self.model = SunuwarBERT(self.config)
        checkpoint_path = PROJECT_ROOT / self.config["checkpoint_path"]
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(ckpt)
        self.model.eval()

        self.param_count = sum(p.numel() for p in self.model.parameters())
        self._ppl_cache = None

    def info(self) -> dict:
        return {
            "params": self.param_count,
            "layers": self.config["num_layers"],
            "hidden_dim": self.config["hidden_dim"],
            "vocab_size": self.config["vocab_size"],
            "heads": self.config["num_heads"],
        }

    def predict(self, sentence: str, top_k: int = 5) -> dict:
        if "[MASK]" not in sentence:
            raise ValueError("Sentence must contain [MASK].")
        ids, mask_pos = encode_with_mask(sentence, self.tokeniser)
        input_tensor = torch.tensor([ids], dtype=torch.long)
        attn_mask = (input_tensor != self.tokeniser.pad_id).long()

        with torch.no_grad():
            logits = self.model(input_tensor, attn_mask)

        probs = F.softmax(logits[0, mask_pos], dim=-1)
        top_probs, top_ids = probs.topk(top_k)

        predictions = [
            {
                "piece": self.tokeniser.sp.id_to_piece(tid.item()),
                "piece_clean": self.tokeniser.sp.id_to_piece(tid.item()).lstrip("▁"),
                "prob": round(prob.item() * 100, 2),
            }
            for prob, tid in zip(top_probs, top_ids)
        ]
        return {"predictions": predictions}

    def perplexity(self, max_sentences: int = 500) -> dict:
        if self._ppl_cache is not None:
            return self._ppl_cache

        test_path = PROJECT_ROOT / self.config["test_path"]
        dataset = SunuwarMLMDataset(str(test_path), self.tokeniser)
        dataset.encoded = dataset.encoded[:max_sentences]
        loader = make_dataloader(dataset, self.tokeniser)
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        total, n = 0.0, 0
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                masked_ids, labels, attn = apply_mlm_mask(batch, self.tokeniser)
                logits = self.model(masked_ids, attn)
                total += loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1)).item()
                n += 1

        ppl = math.exp(total / n) if n else 0.0
        result = {
            "perplexity": round(ppl, 2),
            "random_baseline": 8000,
            "improvement_factor": round(8000 / ppl) if ppl else 0,
        }
        self._ppl_cache = result
        return result
