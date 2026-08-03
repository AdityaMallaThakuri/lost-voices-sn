# Data source: Sunuwar Bible (suzBl), © 2011 Wycliffe Bible Translators, Inc.
# Licence: CC BY-NC-ND 4.0 — non-commercial research use only
"""Shared paths and fixtures.

Every fixture in this suite is **synthetic Devanagari written by hand**. The
Sunuwar NT is CC BY-NC-ND 4.0 and must never be copied into `tests/`; the
strings below are nonsense assembled from Devanagari syllables purely to
exercise the code paths.
"""

import sys
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
CONFIGS = REPO / "configs"
MODELS = REPO / "models"
RESULTS = REPO / "results"
DATA_PROCESSED = REPO / "data" / "processed"
DATA_RAW = REPO / "data" / "raw"
FIXTURES = Path(__file__).resolve().parent / "fixtures"

# Scripts under src/ import each other by bare module name (`from model import ...`),
# relying on the script's own directory being sys.path[0]. Reproduce that for tests.
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# ---------------------------------------------------------------------------
# Synthetic Devanagari — NOT corpus text
# ---------------------------------------------------------------------------

# U+200D ZWJ appears inside two of these, between a virama and the following
# consonant, which is exactly how it occurs in the real corpus (54,692 times).
# claude.md requires it be preserved end to end.
SYNTHETIC_SENTENCES = [
    "काना माना ताना बाना",
    "रिमी सिमी तिमी निमी लिमी",
    "गोपु दोपु मोपु नोपु",
    "सुक्‍ता मुक्‍ता लुक्‍ता",
    "पेंशा देंशा केंशा गेंशा हेंशा",
    "बाक्‍नुङ लाक्‍नुङ चाक्‍नुङ",
    "ङोइ चोइ तोइ नोइ मोइ",
    "आ इ उ ए ओ का",
]


@pytest.fixture(scope="session")
def synthetic_sentences() -> list[str]:
    return list(SYNTHETIC_SENTENCES)


@pytest.fixture
def scratch_dir():
    """A throwaway directory.

    Deliberately not pytest's `tmp_path`: its per-user base directory can be
    left with an ACL that denies access on Windows, which makes every test
    using it error out with PermissionError for reasons that have nothing to
    do with this repository. `tempfile` sidesteps the shared base entirely.
    """
    with tempfile.TemporaryDirectory(prefix="lost-voices-test-") as path:
        yield Path(path)


@pytest.fixture(scope="session")
def spm_model_path() -> str:
    """The committed 8k SentencePiece model, or skip.

    `models/sunuwar_spm_8k.model` is tracked (it carries no redistributable
    text), but a partial clone may not have it.
    """
    path = MODELS / "sunuwar_spm_8k.model"
    if not path.exists():
        pytest.skip(f"{path} not present — tokeniser tests need the committed SPM model")
    return str(path)


@pytest.fixture(scope="session")
def tokeniser(spm_model_path):
    pytest.importorskip("sentencepiece")
    from model import SunuwarTokeniser

    return SunuwarTokeniser(model_path=spm_model_path, vocab_size=8000, max_seq_len=128)
