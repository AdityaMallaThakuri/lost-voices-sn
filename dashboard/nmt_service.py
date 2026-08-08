"""
nmt_service.py — loads sunuwarNMT-small once and exposes a translate()
helper for the FastAPI app, following the same load-once-in-__init__
pattern already used by model_service.py's MLMService. JointTokeniser/
SunuwarNMT/greedy_translate come from src/nmt_model.py -- the dependency-
light module factored out of train_nmt.py/demo_nmt.py specifically so this
service doesn't need wandb or transformers installed (see src/nmt_model.py's
docstring for the full reasoning).
"""

import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nmt_model import JointTokeniser, SunuwarNMT, greedy_translate  # noqa: E402


class NMTService:
    def __init__(self):
        config_path = PROJECT_ROOT / "configs" / "nmt.yaml"
        with open(config_path, encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        suz_tokeniser_path = PROJECT_ROOT / self.config["suz_tokeniser_path"]
        npi_tokeniser_path = PROJECT_ROOT / self.config["npi_tokeniser_path"]
        self.tokeniser = JointTokeniser(
            str(suz_tokeniser_path), str(npi_tokeniser_path), self.config["max_seq_len"],
        )

        self.model = SunuwarNMT(self.config, self.tokeniser.vocab_size).to(self.device)
        checkpoint_path = PROJECT_ROOT / self.config["checkpoint_path"]
        state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.param_count = sum(p.numel() for p in self.model.parameters())

    def info(self) -> dict:
        return {
            "params": self.param_count,
            "joint_vocab_size": self.tokeniser.vocab_size,
        }

    def translate(self, text: str, direction: str) -> str:
        if not text.strip():
            raise ValueError("Text must not be empty.")
        if direction not in ("suz2npi", "npi2suz"):
            raise ValueError('direction must be "suz2npi" or "npi2suz".')

        if direction == "suz2npi":
            src_ids = [JointTokeniser.TAG_2NPI_ID] + self.tokeniser.encode_suz(text)
        else:
            src_ids = [JointTokeniser.TAG_2SUZ_ID] + self.tokeniser.encode_npi(text)
        src_ids = src_ids[: self.config["max_seq_len"]]

        return greedy_translate(self.model, self.tokeniser, src_ids, direction, self.device)
