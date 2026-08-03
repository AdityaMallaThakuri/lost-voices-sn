# Data source: Sunuwar Bible (suzBl), © 2011 Wycliffe Bible Translators, Inc.
# Licence: CC BY-NC-ND 4.0 — non-commercial research use only
"""Pin SunuwarBERT's state_dict against the released checkpoint's contract.

`models/sunuwar_transformer.pt` was trained with exactly the keys and shapes
recorded in `tests/fixtures/state_dict_contract.json`, and it cannot be
regenerated. Renaming a layer or changing a dimension makes the checkpoint
unloadable — silently, because `load_state_dict` failures surface only when
someone runs the demo.

This test is the reason `src/model.py` could be extracted safely. It must fail
if the architecture moves.
"""

import json
import math

import pytest
import yaml

from conftest import CONFIGS, FIXTURES

torch = pytest.importorskip("torch")

from model import SunuwarBERT  # noqa: E402

CONTRACT_PATH = FIXTURES / "state_dict_contract.json"


@pytest.fixture(scope="module")
def contract() -> dict:
    with open(CONTRACT_PATH, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def mlm_config() -> dict:
    with open(CONFIGS / "mlm.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="module")
def rebuilt_state_dict(mlm_config) -> dict:
    model = SunuwarBERT(mlm_config)
    return {k: list(v.shape) for k, v in model.state_dict().items()}


def test_config_still_matches_the_architecture_the_checkpoint_was_trained_with(
    contract, mlm_config
):
    """configs/mlm.yaml must not drift from the frozen checkpoint's geometry.

    In particular `vocab_size: 8000` is deliberate. The tokeniser only has 6764
    pieces, which looks like a bug, but the checkpoint's embedding and mlm_head
    tensors have 8000 rows. Lowering it breaks loading permanently.
    """
    for key, expected in contract["architecture"].items():
        assert mlm_config[key] == expected, (
            f"configs/mlm.yaml:{key} is {mlm_config[key]!r}, contract says {expected!r}. "
            "Changing it invalidates models/sunuwar_transformer.pt."
        )


def test_state_dict_keys_match_contract_exactly(contract, rebuilt_state_dict):
    expected = set(contract["state_dict_shapes"])
    actual = set(rebuilt_state_dict)

    assert actual == expected, (
        f"missing from rebuilt model: {sorted(expected - actual)}; "
        f"unexpected in rebuilt model: {sorted(actual - expected)}"
    )


def test_state_dict_shapes_match_contract_exactly(contract, rebuilt_state_dict):
    mismatched = {
        key: (shape, rebuilt_state_dict[key])
        for key, shape in contract["state_dict_shapes"].items()
        if key in rebuilt_state_dict and rebuilt_state_dict[key] != shape
    }
    assert not mismatched, (
        "shape drift (key: contract vs rebuilt): "
        + ", ".join(f"{k}: {c} vs {r}" for k, (c, r) in sorted(mismatched.items()))
    )


def test_parameter_total_matches_the_reported_figure(contract, mlm_config):
    """14,485,568 is the number published in results/bert_eval.json."""
    model = SunuwarBERT(mlm_config)
    total = sum(p.numel() for p in model.parameters())

    assert total == contract["total_parameters"] == 14_485_568

    from_shapes = sum(math.prod(s) for s in contract["state_dict_shapes"].values())
    assert from_shapes == total


def test_forward_pass_produces_vocab_size_logits(mlm_config):
    model = SunuwarBERT(mlm_config)
    model.eval()

    input_ids = torch.randint(4, 100, (2, 16), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        logits = model(input_ids, attention_mask)

    assert tuple(logits.shape) == (2, 16, mlm_config["vocab_size"])


def test_model_py_is_the_only_definition_of_sunuwarbert():
    """The whole point of the extraction: no second copy may reappear."""
    from conftest import SRC

    definers = sorted(
        path.name
        for path in SRC.glob("*.py")
        if "class SunuwarBERT" in path.read_text(encoding="utf-8")
    )
    assert definers == ["model.py"], f"SunuwarBERT is defined in {definers}"
