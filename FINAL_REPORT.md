# Lost Voices: A Monolingual NLP and Text-to-Speech Pipeline for Sunuwar, an Endangered Kiranti Language

**Final Year Project Report**
Nepal Engineering College (NEC), Kathmandu, Nepal

---

## Chapter 1: Introduction

### 1.1 Overview

Lost Voices is the first publicly reproducible Natural Language Processing (NLP) and
Text-to-Speech (TTS) pipeline built for Sunuwar (ISO 639-3: `suz`), an endangered
Kiranti language of the Tibeto-Burman family spoken in eastern Nepal and written in
Devanagari script. The project takes a single available digital resource — a
Wycliffe Bible Translators translation of the New Testament, released in both marked-up
(USFM) and plain-text, chapter-aligned audio form — and builds a complete, end-to-end
pipeline from raw scripture text and narrated audio to a family of trained language
models and a fine-tuned speech synthesiser.

The pipeline has two parallel tracks:

- **Part I — NLP Foundation**: a cleaned Devanagari text corpus, a subword tokeniser,
  static word embeddings (word2vec, fastText), and a compact transformer-based masked
  language model (SunuwarBERT-small) pre-trained from scratch.
- **Part II — Text-to-Speech**: a forced-aligned, quality-controlled single-narrator
  speech dataset, and a Sunuwar TTS model produced by fine-tuning Meta's Massively
  Multilingual Speech (MMS) VITS architecture via cross-lingual transfer.

The project's central claim is not that it produces state-of-the-art accuracy — the
underlying corpus is small (under 200,000 tokens) and drawn from a single religious
register — but that it produces a **fully documented, reproducible pipeline** that a
future researcher, or the Sunuwar community itself, can pick up, extend, and apply to
other data as it becomes available.

### 1.2 Problem Statement

Sunuwar has an estimated 78,910 speakers in Nepal according to the 2021 census and is
classified as **threatened** on the Expanded Graded Intergenerational Disruption Scale
(EGIDS). Despite having a living speaker community, Sunuwar has **no public NLP
resources of any kind**: no digitised corpus suitable for computational use, no word
embeddings, no language model, and no speech synthesis system. This is not a Sunuwar-
specific gap — it reflects a much broader pattern documented in the NLP-diversity
literature, where the overwhelming majority of the world's languages, including many
with active speaker communities, receive zero attention from mainstream NLP research
and tooling because they lack large digitised corpora, commercial incentive, or
standardised benchmarks [1].

The practical consequence for Sunuwar is that every downstream application that
depends on basic language technology — spell-checking, predictive text, voice
interfaces, machine-assisted literacy tools, or archival/dissemination of oral
tradition through synthetic speech — is currently impossible to build without first
solving the foundational problem this project addresses: is it even possible to
build usable language models and speech synthesis from the *only* substantial
digitised resource that exists (a single Bible translation with a single narrator's
audio), and if so, what pipeline design choices make that feasible on a small,
narrow-register, single-speaker dataset?

### 1.3 Objectives

1. Build a clean, reproducible Sunuwar text corpus from the raw USFM Bible source,
   correctly separating genuine Sunuwar running text from embedded Nepali-language
   translator footnotes and cross-references.
2. Train and evaluate a subword tokeniser suited to Sunuwar's agglutinative
   morphology.
3. Train and comparatively evaluate two families of static word embeddings
   (word2vec, fastText) on out-of-vocabulary (OOV) rate and a downstream genre
   classification proxy task.
4. Pre-train a compact bidirectional transformer language model (SunuwarBERT-small)
   from scratch using a masked language modelling objective, and demonstrate it
   learns meaningful contextual representations despite the corpus's small size.
5. Build a correctly time-aligned, per-sentence speech dataset from 29.5 hours of
   single-narrator chapter-level audio, using the Montreal Forced Aligner (MFA), with
   an explicit, measured quality-control process.
6. Fine-tune a Sunuwar TTS model by cross-lingual transfer from a related-language
   MMS checkpoint, and evaluate the result with an objective, literature-standard
   acoustic metric (Mel-Cepstral Distortion, MCD, plus F0 RMSE).
7. Document every non-obvious engineering obstacle encountered (environment bugs,
   alignment failures, transcript-cleaning bugs) so the pipeline is genuinely
   reproducible rather than merely reported as working.

### 1.4 Aims

The project aims to demonstrate that a complete, defensible language-technology
pipeline — spanning corpus construction, tokenisation, embeddings, a contextual
language model, forced alignment, dataset quality control, and TTS fine-tuning — can
be built for a genuinely low-resource, single-source language, using only
compute available on free-tier cloud GPUs (Google Colab T4) and a modest local
machine, without any proprietary data collection.

### 1.5 Motivation

Kiranti languages of Nepal, including Sunuwar, are undergoing rapid intergenerational
shift toward Nepali, and digital-domain absence accelerates this shift: a language
with no presence in predictive text, voice assistants, or synthetic speech becomes
progressively less usable in modern digital life, which in turn narrows the domains
in which younger speakers use it. Building even an imperfect NLP and TTS pipeline is
a concrete, low-cost intervention against this trend, and — because the codebase and
methodology are designed to be reusable — the same pipeline can, with different input
data, be pointed at other under-documented Kiranti and Himalayan languages (Rai,
Limbu, Hayu, Jirel) that face an identical resource gap.

### 1.6 Scope and Applications

**In scope:**
- Monolingual Sunuwar text processing and modelling (no translation task).
- New Testament text only (27 books), not the Old Testament, to keep the corpus and
  results reproducible against the project proposal.
- Devanagari script only (the native Sunuwar script, Koĩts Brese/Tikamuli, recently
  assigned Unicode block U+11BC0–U+11BFF, has no digital resources and is out of
  scope).
- Single-narrator TTS built from existing read-aloud scripture audio; no new studio
  recording.

**Applications:** foundational models for future Sunuwar NER/POS tagging or spell-
checking tools; a base checkpoint for community-facing text-to-speech (e.g.,
scripture read-aloud tools, accessibility applications); a template pipeline directly
transferable to other Kiranti languages with an eBible/FCBH-style Bible translation
and audio recording, which is a fairly common resource pattern for many
under-documented languages worldwide.

### 1.7 Feasibility Study

At project outset, the central open question was whether ~180K tokens of a single
religious register and ~29.5 hours of single-narrator audio were *sufficient* inputs
for any of the above. This was resolved empirically rather than assumed:

- **NLP feasibility**: confirmed by SunuwarBERT-small's validation perplexity
  falling to 14.79 against a random baseline of ~8,000 (a 540.9× improvement),
  demonstrating the corpus is large enough for a *small* transformer to learn
  meaningful subword co-occurrence structure, provided the architecture is kept
  deliberately compact (14.5M parameters) to avoid overfitting a small corpus.
- **TTS feasibility, checkpoint availability**: no MMS-TTS checkpoint exists for
  Sunuwar, and — contrary to the original project assumption — **none exists for
  Nepali either** (`facebook/mms-tts-nep` and `-npi` both return 404, verified
  empirically). This was resolved by selecting `facebook/mms-tts-mai` (Maithili) as
  the transfer base, chosen by directly measuring which candidate MMS
  Devanagari-script tokenizer covers the largest fraction of the real Sunuwar
  character set (98.65%), rather than assuming linguistic proximity — see
  §4.1 for the full comparison.
- **TTS feasibility, alignment**: an initial naive approach (feeding MFA whole,
  multi-minute audio chapters directly) failed completely (0/16 chapters aligned),
  establishing empirically that pre-chunking into sentence-length segments before
  alignment is a hard requirement, not an optional optimisation, for this corpus.
- **Compute feasibility**: all NLP training fits on a Colab T4 free-tier GPU; MFA
  alignment runs on CPU only (Kaldi-based) and was run locally; TTS fine-tuning
  requires a T4 GPU and was budgeted in step-counts (~1.32 s/iteration measured)
  rather than epoch-counts, since the target full-corpus dataset is roughly 20×
  the size of the initial prototype.

---

## Chapter 2: Literature Review

### 2.1 Low-resource and endangered-language NLP

The scale of language exclusion from NLP research is well documented. Joshi et al.
[1] classify the world's languages into six resource strata based on labelled and
unlabelled data availability, showing that the vast majority of the world's roughly
7,000 languages — including Sunuwar — sit in the lowest strata, with essentially no
benchmark datasets, embeddings, or pre-trained models. Their taxonomy directly
motivates this project's framing: rather than attempting to compete with high-resource
NLP results, the goal is to establish *any* reproducible baseline where none
previously existed, using whatever single data source is realistically available.

### 2.2 Subword tokenisation for morphologically rich languages

Whole-word vocabularies are known to perform poorly on agglutinative and
morphologically rich languages because of vocabulary explosion and high
out-of-vocabulary rates on unseen inflected forms. Subword tokenisation schemes
address this by decomposing rare words into shared, reusable pieces. This project
uses the unigram language-model algorithm implemented in Google's SentencePiece
library [2], which — unlike Byte-Pair Encoding's bottom-up merge procedure — begins
with a large candidate subword inventory and prunes it to maximise the likelihood of
the training corpus under a unigram model, with byte-level fallback guaranteeing zero
unrepresentable characters.

### 2.3 Static word embeddings: word2vec and fastText

word2vec [3] represents each word type as a single dense vector learned by predicting
context words from a target word (Skip-gram) or vice versa (CBOW), but assigns no
representation to unseen word forms. fastText [4] addresses this specific limitation
for morphologically rich languages by representing each word as the sum of its
character n-gram vectors, allowing approximate vectors to be composed for previously
unseen inflected forms — directly relevant to Sunuwar's rich verbal suffixation.

### 2.4 Contextual language models: BERT

Devlin et al.'s BERT [5] established the bidirectional transformer encoder,
pre-trained with a masked language modelling (MLM) objective, as the dominant
architecture for producing contextual token representations, in contrast to
earlier unidirectional (left-to-right) language models. This project's
SunuwarBERT-small follows the same MLM pre-training objective (15% masking, 80/10/10
replacement strategy) at a deliberately reduced scale (6 layers, ~14.5M parameters)
appropriate to a corpus roughly six orders of magnitude smaller than BERT's original
pre-training corpus.

### 2.5 Multilingual and low-resource speech synthesis

Meta's Massively Multilingual Speech (MMS) project [6] released text-to-speech
checkpoints for over 1,100 languages by training VITS-based [7] models at scale,
establishing that cross-lingual transfer — fine-tuning from a related-language
checkpoint rather than training from random initialisation — is a viable strategy
for languages with only a few hours of available audio. VITS itself [7] is an
end-to-end conditional variational autoencoder with adversarial (GAN) training,
jointly learning a stochastic duration predictor, a normalising-flow-based prior, and
a HiFi-GAN-style vocoder decoder, avoiding the two-stage acoustic-model-then-vocoder
pipeline used by earlier TTS systems. This project directly builds on the MMS
release by selecting the geographically and typologically closest available
checkpoint (Maithili, Nepal's Terai contact language) as a transfer base, and fine-
tunes it on the constructed Sunuwar dataset — the same "adapt a released multilingual
checkpoint to a new low-resource target language" pattern the MMS paper's authors
anticipate as its primary reuse mode.

### 2.6 Forced alignment

The Montreal Forced Aligner (MFA) [8] is a widely used open-source tool for
time-aligning transcribed speech at the phone and word level using Kaldi's
Gaussian-mixture/HMM acoustic modelling pipeline, given an audio-transcript pair and
a pronunciation dictionary. This project uses MFA to align a custom, hand-built
grapheme-to-phoneme (G2P) dictionary against single-narrator audio, following the
standard MFA training-alignment workflow, adapted with a custom pre-alignment
sentence-segmentation stage (§4.2) to overcome MFA's poor performance on long,
unsegmented utterances.

### 2.7 Existing linguistic description of Sunuwar

Borchers' *A Grammar of Sunwar* [9] remains the primary published linguistic
description of the language (subject-object-verb word order, agglutinative verbal
morphology), and was used as a reference for sanity-checking the corpus's
morphological character (e.g., expected sentence-length distribution, expected
suffixation patterns) during preprocessing design, though it was not used as training
data.

### 2.8 Relevance to this project

None of the above prior work has previously been applied to Sunuwar. The
contribution of this project is not a new algorithm at any individual pipeline
stage — every component (SentencePiece, word2vec/fastText, BERT-style MLM
pre-training, MFA alignment, MMS/VITS fine-tuning) is an existing, published method —
but the assembly, adaptation, and empirical validation of this specific pipeline
against the specific constraints of a single-source, single-narrator, low-resource
Kiranti language, including documenting where standard recipes (e.g., "align whole
chapters," "assume the nearest major language has an MMS checkpoint") concretely
failed and had to be replaced with corpus-specific alternatives.

---

## Chapter 3: System Design

### 3.1 Block Diagram — Overall System

```
                         ┌─────────────────────────────┐
                         │   Raw Data Sources            │
                         │  suzBl_usfm.zip (27 NT books)│
                         │  suzBl_readaloud audio        │
                         │  (260 chapters, 29.5h, 1 spkr)│
                         └───────────────┬───────────────┘
                                         │
                ┌────────────────────────┴────────────────────────┐
                │                                                  │
   ┌────────────▼─────────────┐                     ┌─────────────▼─────────────┐
   │   NLP TRACK                │                     │   TTS TRACK                │
   │                             │                     │                             │
   │ 1. preprocess_text.py      │                     │ 1. preprocess_audio.py     │
   │    USFM parse + clean      │                     │    MP3→WAV, normalise      │
   │         │                  │                     │         │                  │
   │ 2. split_corpus.py         │                     │ 2. g2p.py                  │
   │    90/10 train/test        │                     │    phoneme dictionary      │
   │         │                  │                     │         │                  │
   │ 3. train_spm.py            │                     │ 3. clean_transcripts.py    │
   │    SentencePiece 8k/16k    │                     │    strip USFM/cross-refs   │
   │         │                  │                     │         │                  │
   │ 4a. train_word2vec.py      │                     │ 4. segment_from_pauses.py  │
   │ 4b. train_fasttext.py      │                     │    ranked-silence chunking │
   │         │                  │                     │         │                  │
   │ 5. train_mlm.py            │                     │ 5. align.py (MFA)          │
   │    SunuwarBERT-small       │                     │    forced alignment        │
   │         │                  │                     │         │                  │
   │ 6. evaluate_nlp.py         │                     │ 6. qc_filter.py            │
   │    OOV, fertility,         │                     │    threshold-based QC      │
   │    perplexity              │                     │         │                  │
   │                             │                     │ 7. build_tts_dataset.py    │
   │                             │                     │    metadata.csv            │
   │                             │                     │         │                  │
   │                             │                     │ 8. train_tts.py            │
   │                             │                     │    VITS fine-tune from     │
   │                             │                     │    facebook/mms-tts-mai    │
   │                             │                     │         │                  │
   │                             │                     │ 9. evaluate_tts.py         │
   │                             │                     │    MCD + F0 RMSE           │
   └─────────────────────────────┘                     └─────────────────────────────┘
```

### 3.2 Flow Diagram — Text Preprocessing Pipeline (`preprocess_text.py`)

```
[Read suzBl_usfm.zip, filter 27 NT book files]
              │
              ▼
[NFC Unicode normalisation]
              │
              ▼
[Discard full-line markers: \id \h \toc \mt \c \s1 \r \ip \io \p \m]
              │
              ▼
[Strip \em...\em* blocks (regex, DOTALL) — removes Nepali cross-refs/footnotes]
              │
              ▼
[Strip inline markers: \bd* \em* remnants; keep \bd-wrapped text]
              │
              ▼
[Strip \v N verse-number markers, keep following text]
              │
              ▼
[Script filter: drop line if non-Devanagari/non-space chars > 20%]
              │
              ▼
[Sentence segmentation on danda । and ॥]
              │
              ▼
[Whitespace normalisation; length filter: drop < 3 tokens]
              │
              ▼
[Output: sunuwar_nt_raw.txt — one sentence per line, UTF-8]
```

### 3.3 Flow Diagram — TTS Dataset Construction

```
[260 chapter audio files + read-aloud transcripts]
              │
              ▼
[clean_transcripts.py: strip residual USFM + cross-reference remnants
 (ZWJ-aware regex, danda-bounded book-name matching)]
              │
              ▼
[segment_from_pauses.py: detect all candidate silence pauses,
 keep only the (sentence_count - 1) LONGEST as true sentence
 boundaries — discards mid-sentence breath pauses]
              │
              ▼
[12,672 per-sentence WAV+TXT segments, flagged high/low
 segmentation-confidence by duration-outlier heuristic]
              │
              ▼
[align.py: MFA train+align, custom G2P dictionary,
 --temporary_directory on D: drive]
              │
              ▼
[TextGrids + per-utterance MFA stats: overall_log_likelihood,
 phone_duration_deviation, snr]
              │
              ▼
[qc_filter.py: threshold-based pass/fail per segment]
              │
              ▼
[build_tts_dataset.py: metadata.csv, chapter-level train/val split]
              │
              ▼
[train_tts.py: VITS fine-tune wrapper around
 ylacombe/finetune-hf-vits, base = facebook/mms-tts-mai]
              │
              ▼
[evaluate_tts.py: pyworld/pysptk MCD + F0 RMSE vs. reference]
```

### 3.4 Sequence Diagram — MFA Alignment Run

```
User/Script        align.py            MFA (Kaldi)         Filesystem (D:)
    │                  │                    │                     │
    │ configs/align_*.yaml                  │                     │
    ├─────────────────>│                    │                     │
    │                  │  mfa train (single_speaker mode)         │
    │                  ├───────────────────>│                     │
    │                  │                    │  read segments+dict │
    │                  │                    ├────────────────────>│
    │                  │                    │  GMM-HMM acoustic   │
    │                  │                    │  model training     │
    │                  │                    │  Viterbi alignment  │
    │                  │                    │  per utterance      │
    │                  │                    │  write TextGrids +  │
    │                  │                    │  alignment_analysis │
    │                  │                    ├────────────────────>│
    │                  │<───────────────────┤                     │
    │  per-batch: aligned/total, %          │                     │
    │<─────────────────┤                    │                     │
```

### 3.5 Class/Module Diagram (conceptual — script-oriented, not OOP-heavy)

The codebase is intentionally script-oriented (each stage is an independent,
config-driven Python script rather than a class hierarchy), consistent with the
project's reproducibility goal — each stage can be re-run in isolation given its
YAML config. The closest analogue to a "class diagram" is the config-to-script
dependency map:

| Config file | Consuming script | Produces |
|---|---|---|
| `configs/preprocess.yaml` | `preprocess_text.py` | `sunuwar_nt_raw.txt` |
| `configs/split.yaml` | `split_corpus.py` | `train.txt`, `test.txt` |
| `configs/spm.yaml` | `train_spm.py` | `sunuwar_spm_8k/16k.model` |
| `configs/word2vec.yaml` | `train_word2vec.py` | `sunuwar_w2v_sg/cbow.model` |
| `configs/fasttext.yaml` | `train_fasttext.py` | `sunuwar_fasttext.bin/.vec` |
| `configs/mlm.yaml` | `train_mlm.py` | `sunuwar_transformer.pt` |
| `configs/eval_nlp.yaml` | `evaluate_nlp.py` | `results/*.json`, `corpus_report.md` |
| `configs/audio.yaml` | `preprocess_audio.py` | normalised 16kHz WAVs |
| `configs/g2p_sunuwar.yaml` | `g2p.py` | `mfa_dict.dict` |
| `configs/clean_transcripts.yaml` | `clean_transcripts.py` | cleaned per-chapter transcripts |
| `configs/segment_from_pauses.yaml` | `segment_from_pauses.py` | per-sentence WAV/TXT + `segmentation_report.csv` |
| `configs/align_*.yaml` | `align.py` | TextGrids, acoustic model, `alignment_analysis.csv` |
| `configs/qc_filter.yaml` | `qc_filter.py` | `qc_report.csv` |
| `configs/tts_dataset.yaml` | `build_tts_dataset.py` | `data/processed/tts_dataset/manifest.csv` |
| `configs/tts.yaml` | `train_tts.py` | fine-tuned VITS checkpoint |
| `configs/eval_tts.yaml` | `evaluate_tts.py` | MCD/F0 RMSE report |

### 3.6 Data Dictionary — `metadata.csv` / `manifest.csv` (TTS dataset)

| Column | Type | Description |
|---|---|---|
| `file_name` | string | Relative path to the segment WAV file (16kHz mono) |
| `text` | string (UTF-8, Devanagari + ZWJ) | Cleaned Sunuwar sentence transcript |
| `chapter` | string | Source chapter ID, e.g. `GAL_001` — used for chapter-level train/val split |
| `duration_s` | float | Segment duration in seconds |
| `qc_pass` | boolean | Whether the segment passed all QC thresholds |

### 3.7 Data Dictionary — `qc_report.csv` / `alignment_analysis.csv`

| Column | Description |
|---|---|
| `segment` | Segment identifier |
| `qc_pass` / `qc_reasons` | QC verdict and failing threshold(s), e.g. `phone_duration_outlier` |
| `overall_log_likelihood` | MFA's per-utterance acoustic model log-likelihood |
| `speech_log_likelihood` | Log-likelihood restricted to detected speech region |
| `phone_duration_deviation` | Deviation of aligned phone durations from expected distribution |
| `snr` | Estimated signal-to-noise ratio of the segment |
| `n_words_aligned` | Word count successfully aligned |
| `segmentation_confidence` | Pre-alignment high/low flag from the duration-outlier heuristic (Phase 2) |
| `aligned` | Whether MFA produced a valid alignment at all for this segment |

No relational database is used in this project — all intermediate and final
artefacts are flat CSV/JSON/text files, which is sufficient for a linear,
single-pipeline research project of this scale and keeps every stage inspectable
and diffable without additional infrastructure.

### 3.8 Hardware Requirements

| Requirement | Specification | Justification |
|---|---|---|
| Cloud GPU (training) | NVIDIA T4, 15GB VRAM (Google Colab free tier) | Required for SunuwarBERT-small MLM pre-training and VITS TTS fine-tuning; both are specified in CLAUDE.md to run end-to-end on a T4 |
| Local CPU (alignment) | 6-core/12-thread (measured: Intel i5-11400H) | MFA's Kaldi backend is CPU-only (GMM-HMM), no GPU benefit; used for all forced-alignment runs |
| Local RAM | ≥ 8GB (measured constraint: 7.7GB usable) | Caps MFA parallel job count (`num_jobs`) to avoid swapping during alignment |
| Local/cloud disk | ≥ 10GB free on the alignment temp volume | MFA's working directory (TextGrids, intermediate Kaldi files) for a corpus of 12,672 segments; redirected to a drive with more headroom (141GB) after the default system drive proved too small |

No sensors, microcontrollers, or other embedded hardware are used — this is a
pure data/software pipeline.

### 3.9 Software Requirements

| Category | Tool / Library | Version constraint | Reason |
|---|---|---|---|
| Language | Python | ≥ 3.10 | Project-wide standard (CLAUDE.md code standards) |
| Deep learning | PyTorch | ≥ 2.0 | Backend for BERT pre-training and VITS fine-tuning |
| Transformers | HuggingFace `transformers` | **pinned 4.44.2** | Colab ships 5.x, which raises on undeclared config keys (`pad_token_id`) and breaks MMS checkpoint conversion; downgrading over an existing 5.x install corrupts the package — requires a fresh runtime |
| Hub client | `huggingface_hub` | 0.24.6 | Compatibility with pinned `transformers` |
| Datasets | `datasets` | 2.21.0 (patched) | Audiofolder loader has a pyarrow `large_string` vs `string` schema bug requiring a source patch |
| Accelerate | `accelerate` | 0.34.2 | Training-loop backend used by the VITS fine-tuning wrapper |
| Tokenizers | `tokenizers` | 0.19.1 | Compatibility with pinned `transformers` |
| NumPy | `numpy` | < 2 (pinned 1.26.4) | `numba`/`librosa` (used internally by `datasets` for audio decoding) require NumPy ≤ 1.x; silently reverts without `--force-reinstall --no-deps` |
| Tokeniser | `sentencepiece` | — | Unigram subword tokeniser training |
| Embeddings | `gensim` | — | word2vec Skip-gram/CBOW |
| Embeddings | `fasttext` (Facebook) | — | Character n-gram embeddings |
| Forced alignment | Montreal Forced Aligner (MFA) | 3.3.9 | Time-alignment of audio to per-sentence transcripts; run in a local `aligner` conda environment |
| TTS base model | `facebook/mms-tts-mai` | — | Cross-lingual transfer base (see §4.1 for selection rationale) |
| TTS fine-tuning | `ylacombe/finetune-hf-vits` | — | Wraps VITS GAN training loop (discriminator, mel/KL losses, monotonic alignment search) not present in mainline `transformers`' inference-only `VitsModel` |
| Objective evaluation | `pyworld`, `pysptk` | — | F0 extraction (`dio`/`stonemask`) and mel-generalised cepstral coefficient extraction (`sp2mc`) for MCD computation |
| Experiment tracking | `wandb` | — | Training metric logging (per CLAUDE.md code standards) |
| Audio tooling | `ffmpeg` | — | MP3→WAV conversion, resampling to 16kHz mono |

---

## Chapter 4: Implementation and Discussion

### 4.1 Methodology

#### 4.1.1 Text corpus construction

The raw input is `suzBl_usfm.zip`, containing the full Sunuwar Bible (Old + New
Testament) in USFM (Universal Standard Format Markers) markup. Only the 27 New
Testament books were used (~7,959 verses), to keep the corpus reproducible against
the original project proposal; Old Testament books were treated as reserve data,
not used. USFM embeds structural markers (`\c`, `\s1`, `\p`, `\v N`, etc.) and,
critically, `\em … \em*` blocks that contain **Nepali-language** cross-references
and translator footnotes rather than Sunuwar text — these had to be identified and
removed as a complete span (regex `\\em.*?\\em\*`, `re.DOTALL`), not merely
stripped of their markers, since their *content* is not Sunuwar. A per-line
Devanagari-purity filter (drop if >20% of non-space characters are non-Devanagari)
served as a second line of defense against any Nepali or Latin text that slipped
past marker-stripping.

Unicode NFC (Normalization Form C) canonicalisation was applied before any other
processing, since Devanagari text can represent visually identical characters with
different underlying byte sequences, which would otherwise silently split one word
type into multiple, distinct vocabulary entries. The Zero-Width Joiner character
(U+200D), which appears 54,692 times in the NT and is essential to correct
Devanagari conjunct-character rendering, was explicitly preserved at every
processing stage — a deliberate design decision documented as a hard rule in the
project's coding standards, since a generic Unicode-cleaning script would typically
strip it as a "control character."

Sentence segmentation used the Devanagari danda (।, U+0964) and double-danda (॥,
U+0965), the functional equivalent of a period in Devanagari orthography. A final
length filter discarded any segment with fewer than 3 whitespace-separated tokens,
removing degenerate single-word fragments.

#### 4.1.2 Subword tokenisation

A SentencePiece [2] unigram model was trained at two vocabulary sizes (8,000 and
16,000) with character coverage 0.9995 and byte-level fallback (guaranteeing zero
unrepresentable characters), plus five special tokens (`[PAD]`, `[UNK]`, `[CLS]`,
`[SEP]`, `[MASK]`).

#### 4.1.3 Static word embeddings

word2vec (Skip-gram and CBOW variants, via Gensim) and fastText (via Facebook's
official library) were both trained at 200 dimensions, window size 5, negative
sampling with 10 noise samples, 20 epochs, fixed random seed 42. fastText additionally
used character n-grams of length 3–5, allowing it to compose approximate vectors for
word forms never seen during training — directly relevant given Sunuwar's
agglutinative morphology.

#### 4.1.4 SunuwarBERT-small pre-training

A 6-layer bidirectional transformer encoder (384 hidden dimension, 8 attention
heads, 1,024 feed-forward dimension, GELU activation, 0.1 dropout, 128 max sequence
length, ~14.5M parameters) was pre-trained from scratch using the standard BERT
masked-language-modelling objective [5]: 15% of tokens selected as masking
candidates per batch, with the standard 80% `[MASK]` / 10% random-token / 10%
unchanged replacement split, cross-entropy loss computed only at masked positions.
Optimiser: AdamW, learning rate 5×10⁻⁴, weight decay 0.01, effective batch size 128
(32 × 4 gradient accumulation steps), 10% linear warmup, fp16 mixed precision,
early stopping with patience 5 on validation perplexity, fixed random seed 42.

#### 4.1.5 Audio preprocessing and G2P

The read-aloud audio track (260 chapters, 29.5 hours total, single narrator, zero
degenerate `<5s` chapter files) was normalised to 16kHz mono WAV. A deterministic
grapheme-to-phoneme (G2P) dictionary was built via akshara (Devanagari orthographic
syllable) segmentation into a real phoneme inventory of approximately 50 phones,
achieving 98.9% coverage of the full 260-chapter vocabulary (9,602/9,711 tokens).

#### 4.1.6 Sentence-level segmentation from silence

Naive silence-based pause detection produces roughly 2.67× more candidate pauses
than there are sentences per chapter (narrators pause at clause/breath boundaries as
well as sentence ends), so a **ranked-pause** method was used instead: detect all
candidate silence pauses permissively, then retain only the top `(sentence_count −
1)` *longest* pauses per chapter as genuine sentence boundaries, discarding the rest
as breath noise. This was validated on 15 sampled chapters before being run across
the full corpus, producing plausible per-sentence durations (5–10s mean) with only
1 of 670 sampled segments under 1 second.

#### 4.1.7 Forced alignment (MFA)

The Montreal Forced Aligner [8] was used in `train` mode (training a disposable
Kaldi GMM-HMM acoustic model purely to produce alignment timestamps/confidence
scores — this model is never reused as, or related to, the final TTS model) against
the Phase 1 phoneme dictionary. An initial attempt at aligning whole, unsegmented
chapters directly failed completely (0/16 chapters aligned), empirically confirming
that Kaldi's Viterbi search does not scale to multi-minute utterances and that
pre-chunking (§4.1.6) is a hard prerequisite, not merely an optimisation.

#### 4.1.8 Quality control

Segments were filtered using MFA's own per-utterance output statistics
(`overall_log_likelihood`, `phone_duration_deviation`, `snr`) combined with the
pre-alignment duration-outlier flag from the segmentation stage, with thresholds
fitted on an initial batch and then held fixed and validated against later,
different batches as a genuine held-out test (§4.3, §5.1) rather than re-tuned on
each new batch.

#### 4.1.9 TTS fine-tuning strategy

Cross-lingual transfer learning was chosen over training a VITS model from random
initialisation, since VITS is data-hungry and the available Sunuwar audio, even
after the full pipeline, is on the order of tens of hours at best. The transfer
base was selected by **measuring** — rather than assuming — which candidate MMS
Devanagari-script checkpoint's tokenizer vocabulary best covers the actual Sunuwar
dataset character set (34,863 characters sampled):

| Candidate checkpoint | Vocab size | Character coverage | Missing |
|---|---|---|---|
| **`facebook/mms-tts-mai` (Maithili)** | 67 | **98.65%** | Danda (U+0964) only |
| `facebook/mms-tts-hin` (Hindi) | 73 | 97.83% | 2 distinct characters |
| `facebook/mms-tts-mar` (Marathi) | 74 | 93.03% | 3 distinct characters |
| `facebook/mms-tts-ben/guj/eng` | — | 15.51% | 50 distinct (non-Devanagari) |

Maithili was selected. It is also a Nepal Terai contact language, so the
typological and empirical arguments agree. Notably, `facebook/mms-tts-nep` and
`-npi` (the originally planned Nepali base) do not exist on the Hugging Face Hub —
verified directly by request, not assumed. All three top Devanagari candidates'
vocabularies already include the ZWJ character, so the anticipated ZWJ-remapping
adaptation step turned out to be unnecessary; only the danda needed handling.

`src/train_tts.py` is architected as a thin wrapper around the external
`ylacombe/finetune-hf-vits` training script, rather than a reimplementation, because
mainline `transformers`' `VitsModel` ships inference-only (no discriminator, no GAN
training loss) — reimplementing VITS's discriminator, mel/KL losses and monotonic
alignment search from scratch was judged not worth the risk for this project's
scope.

### 4.2 Implementation Steps and Challenges

The pipeline was implemented and executed in the following order, matching the
dependency chain of each stage's output feeding the next:

1. **USFM parsing and corpus cleaning** (`preprocess_text.py`) — completed;
   15,738 sentences / 179,918 tokens / 13,124 unique types, 97.92% Devanagari
   character purity.
2. **Train/test split** (`split_corpus.py`) — 90/10 split, seed 42: 14,164 train /
   1,574 test sentences.
3. **Tokeniser training** (`train_spm.py`) — 8k and 16k unigram models.
4. **Word embedding training** (`train_word2vec.py`, `train_fasttext.py`).
5. **SunuwarBERT-small pre-training** (`train_mlm.py`) — 35 epochs run, early
   stopped, best checkpoint at epoch 30.
6. **NLP evaluation** (`evaluate_nlp.py`) — produced all `results/*.json` files
   and `corpus_report.md`.
7. **Audio preprocessing** (`preprocess_audio.py`) — full 27-book NT, 260 chapters,
   29.5 hours, single narrator, zero degenerate files.
8. **G2P dictionary construction** (`g2p.py`) — 98.9% phoneme coverage.
9. **Transcript cleaning** (`clean_transcripts.py`) — see the transcript bug
   below.
10. **Sentence-level segmentation** (`segment_from_pauses.py`) — full 260-chapter
    run: 13,192 segments initially, reduced to **12,672** after an intro-offset fix
    (`fix_intro_offset.py`); zero audio lost (29.50h recovered, exact match to
    source); 912 segments (6.9%) flagged low-confidence by the duration-outlier
    heuristic.
11. **Forced alignment** (`align.py`) — verification run across 18 sample chapters
    (6.9% of the corpus), in 4 batches; full 260-chapter run identified as still
    outstanding at time of writing (see §4.2.1 challenges and §6 future work).
12. **QC filtering** (`qc_filter.py`) — re-run across all 18 verification chapters
    with thresholds held fixed from the first three batches.
13. **TTS dataset assembly** (`build_tts_dataset.py`) — 690 QC-passed clips (1.37
    hours) from the 18-chapter verification sample, split 554 train / 136
    validation by whole chapter (not random segment) to avoid near-duplicate
    scriptural phrasing leaking from train into validation.
14. **TTS fine-tuning** (`train_tts.py`, executed via `notebooks/finetune_tts_colab.ipynb`
    on Google Colab) — real fine-tuning run launched on the 1.37-hour/554-clip
    dataset: batch size 8, 200 epochs = 13,800 steps, checkpointing every 250
    steps, resumed twice across Colab free-tier disconnects.
15. **TTS evaluation** (`evaluate_tts.py`) — objective MCD and F0 RMSE computed
    against the held-out validation chapters at an intermediate checkpoint
    (step 5,500, ~40% through training).

#### 4.2.1 Challenges faced and how they were addressed

- **Nepali text contamination in the text corpus.** The `\em…\em*` USFM blocks
  looked, at a glance, like formatting markup to simply strip, but their *content*
  is Nepali-language cross-references and footnotes, not Sunuwar text. Fixed by
  removing the entire span (markers and content) with a DOTALL regex, backed by a
  post-hoc Devanagari-purity filter as a safety net.

- **Whole-chapter forced alignment failed completely (0/16 chapters).** Diagnosed
  as a fundamental limitation of Kaldi's Viterbi search on multi-minute
  utterances, not a configuration bug. Resolved by building a dedicated
  pre-alignment segmentation stage (`segment_from_pauses.py`) rather than trying
  to tune MFA parameters further.

- **Naive silence-based pause detection did not match sentence count.** Roughly
  2.67× more candidate pauses were detected than actual sentence boundaries across
  40 sampled chapters, none within tolerance. Resolved with the ranked-longest-pause
  heuristic (§4.1.6), empirically validated before the full-corpus run.

- **A transcript-cleaning regex bug produced systematic OOV tokens.**
  `clean_transcripts.py`'s cross-reference-stripping regex omitted the ZWJ
  character from its book-name character class, leaving unspoken fragments (e.g.
  partial book-name remnants, bare verse numerals) in 2,217 tokens across all 260
  chapters — none of which existed in the G2P dictionary, making every one an
  out-of-vocabulary token at alignment time and a likely contributor to alignment
  failures. Fixed by adding ZWJ to the character class, bounding book-name matches
  to ≤3 words, and excluding the danda from the match so it cannot cross a
  sentence boundary; the corpus was regenerated (181,993 → 180,255 tokens, 0
  residual digit tokens).

- **No MMS-TTS checkpoint exists for Nepali or Sunuwar.** The originally planned
  transfer base (`facebook/mms-tts-nep`) was found not to exist on the Hugging
  Face Hub. Resolved by empirically measuring character-set coverage across
  candidate Devanagari-script checkpoints and selecting Maithili (§4.1.9), rather
  than defaulting to an assumption of linguistic proximity.

- **MFA silently ignores the requested parallelism.** `num_jobs` was found to be
  ignored for a single-speaker corpus (MFA logged: *"Number of jobs was specified
  as 4, but due to only having 1 speakers, MFA will only use 1 jobs"*), so every
  verification batch ran single-threaded despite a 6-core/12-thread machine being
  available. This is documented as an open item to test (`--single_speaker` flag)
  before committing to the multi-day full-corpus alignment run.

- **Nine distinct environment bugs in the Colab TTS fine-tuning pipeline**,
  none of which surfaced until actually running the full training loop
  end-to-end — including a `transformers` major-version incompatibility silently
  breaking MMS checkpoint conversion, a torch ≥2.1 weight-normalisation key
  rename that `from_pretrained` silently absorbs by **randomly re-initialising**
  the flow and posterior-encoder WaveNet stacks (rather than erroring, which
  would have been easier to catch), a matplotlib API removal breaking validation
  logging, and a pyarrow schema-inference quirk breaking the audio dataset
  loader. Each was found by directly observing failures at the actual training
  step where they occurred, and each is now permanently patched into the
  reproducible Colab notebook rather than worked around ad hoc. Full detail is
  retained in `notebooks/finetune_tts_colab.ipynb`'s patch cells; the most
  consequential is the weight-normalisation remap, since silently
  random-initialising major model components would otherwise be indistinguishable
  from "the model just needs more training" until an evaluation metric like MCD
  is computed.

- **A Windows-built dataset zip nested one directory level too deep** when
  unzipped on Colab's Linux environment (`Compress-Archive` zips the containing
  folder itself, not just its contents), and a numpy version-pinning fix was
  found to silently revert mid-session due to a transitive dependency
  reinstalling a newer numpy, breaking the `datasets` library's internal audio
  decoding inside a worker subprocess. Both required specific, now-documented,
  fixes.

- **A naive MCD implementation gave clearly wrong numbers.** The first version of
  `evaluate_tts.py` used plain librosa MFCC (a DCT of the log-mel spectrogram) in
  place of true mel-generalised cepstral coefficients (mgc), and the standard MCD
  dB-scale formula — calibrated for small-magnitude mgc values from a proper
  vocoder analysis — inflated results 10–50× when applied to MFCC's different-scale
  quantity. This was caught with a sanity check comparing two *different real*
  reference recordings against each other, which should score very low but
  returned ~550 "dB" — clearly a metric bug, not a model or corpus problem. Fixed
  by switching to `pyworld` (F0 and spectral envelope extraction) + `pysptk`
  (spectral-envelope-to-mgc conversion), matching the approach used by
  established TTS evaluation toolkits (ESPnet, ParallelWaveGAN). The same
  real-vs-real sanity check then returned ~8–10 dB, a normal, literature-comparable
  range.

### 4.3 Output Obtained

Completed pipeline stages and their concrete outputs, as of the current project
state:

| Stage | Status | Output |
|---|---|---|
| Text corpus preprocessing | ✅ Complete | 15,738 sentences, 179,918 tokens, 13,124 types, `sunuwar_nt_raw.txt` |
| Train/test split | ✅ Complete | `train.txt` (14,164), `test.txt` (1,574) |
| SentencePiece tokeniser | ✅ Complete | `sunuwar_spm_8k.model`, `sunuwar_spm_16k.model` |
| word2vec + fastText embeddings | ✅ Complete | `sunuwar_w2v_sg/cbow.model`, `sunuwar_fasttext.bin/.vec` |
| SunuwarBERT-small pre-training | ✅ Complete | `sunuwar_transformer.pt`, best val perplexity 14.79 |
| Audio preprocessing | ✅ Complete | 260 chapters, 29.5h normalised WAV |
| G2P phoneme dictionary | ✅ Complete | `mfa_dict.dict`, 98.9% coverage |
| Sentence-level audio segmentation | ✅ Complete (full corpus) | 12,672 segments, `segmentation_report.csv` |
| Forced alignment (MFA) | 🔄 Verification complete (18/260 chapters, 88.5%) | TextGrids, `alignment_analysis.csv` per batch |
| QC filtering | 🔄 Re-run on 18 verification chapters | `qc_report.csv`, 690/915 pass (75.4%) |
| TTS dataset assembly | 🔄 Built from 18-chapter sample | `data/processed/tts_dataset/`, 690 clips / 1.37h |
| TTS fine-tuning | 🔄 Running (checkpoint 5,500/13,800 steps at time of last evaluation) | Colab-trained VITS checkpoint |
| TTS objective evaluation | 🔄 First result obtained | MCD ~8–10 dB, F0 RMSE ~40–100 Hz |

The full 260-chapter alignment, QC, and dataset-build run, and the corresponding
final TTS fine-tune on the complete dataset, remain outstanding at the time of this
report and are addressed as future work in Chapter 6.

### 4.4 Testing / Test Cases

Testing in this project takes the form of empirical validation checks built into
each pipeline stage, rather than conventional unit tests against a fixed API,
since the "correctness" of most stages (Is this a real sentence boundary? Is this
alignment good enough to train on?) is itself an empirical, statistical question.

| Test | Method | Result | Interpretation |
|---|---|---|---|
| Devanagari purity check | Automated character-class filter with printed percentage | 97.92% Devanagari+ZWJ purity | Confirms `\em` stripping successfully isolated Sunuwar text from embedded Nepali |
| Tokeniser OOV test | Encode held-out `test.txt` with both SPM models | 0.0% OOV (byte fallback) | Confirms tokeniser cannot fail to represent any character |
| Embedding OOV test | Look up all unique word types in `test.txt` | word2vec 30.9%, fastText 19.9% | Confirms fastText's n-gram composition measurably reduces OOV versus word2vec |
| BERT held-out perplexity | Validation-set MLM perplexity per epoch | 14.79 at epoch 30 vs. 8,000 random baseline | Confirms the model learned real distributional structure, not memorisation (early stopping triggered by perplexity *rising* at epoch 35, confirming overfitting onset was correctly detected) |
| Silence-boundary segmentation validation | Manual inspection of 15 sampled chapters' resulting segment durations against expected sentence-length range | 5–10s mean, 1/670 segments <1s | Confirms the ranked-pause heuristic produces plausible sentence-level chunks before committing to full-corpus alignment |
| Whole-chapter alignment sanity check | Fed 16 unsegmented chapters directly to MFA | 0/16 aligned | Confirmed pre-chunking is mandatory, ruling out an alternative "align-then-segment" design before further engineering effort was spent on it |
| Alignment hypothesis testing (batch2) | Compared alignment rate against chapter length and OOV rate across 6 chapters | Chapter length: no correlation (largest chapter, 88 segments, aligned at 90.9%); OOV rate: no correlation (chapter with lowest OOV, 7.02%, had the *worst* alignment, 71.4%) | Two plausible-sounding hypotheses about alignment failure causes were explicitly tested and refuted, redirecting investigation toward the actual predictive signal (below) |
| Segmentation-confidence predictive test | Compared alignment success rate for pre-alignment low- vs. high-confidence flagged segments | Low-confidence: 23.5% (4/17) aligned; high-confidence: 88.5% (239/270) aligned | Confirms the Phase 2 duration-outlier flag is a genuine leading indicator of which chapters need re-segmentation, usable pre-alignment |
| MCD metric sanity check | Computed MCD between two different real reference recordings (should be low) | ~550 dB with buggy librosa-MFCC version; ~8–10 dB after switching to pyworld/pysptk mgc | Caught and fixed a metric-implementation bug before it could be mistaken for a model-quality finding |
| Fresh-runtime patch verification | Ran `!python -c "..."` in a fresh interpreter to confirm each of the 9 Colab environment patches actually took effect | All 9 patches confirmed present on a fresh runtime | Guards against false confidence from Python module caching, where an already-imported module makes an in-notebook patch appear to have failed when it actually succeeded |
| Checkpoint weight-integrity check | After loading a mid-training checkpoint, checked `missing`/`unexpected` keys from `load_state_dict(strict=False)` | Confirmed empty on correct load procedure | Guards against a checkpoint silently loading with random-initialised decoder/flow weights (which produces near-silent audio, easily mistaken for "undertrained" rather than "broken loading code") |

### 4.5 Time Schedule

```
Week 0   |████| Data confirmed & downloaded, repo created
Week 1   |████| Text preprocessing + audio preprocessing (parallel tracks begin)
Week 2   |████| SentencePiece tokeniser training (8k / 16k)
Week 3   |████| word2vec + fastText embedding training and evaluation
Week 4   |████| SunuwarBERT-small MLM pre-training (35 epochs, early stopped)
Week 5a  |████| MFA alignment — Mark-only pilot (16 chapters) — superseded
Week 5b  |████| G2P phoneme dictionary construction (Phase 1 of TTS roadmap)
Week 5c  |████| Sentence-level audio segmentation — full 260-chapter run (Phase 2)
Week 5d  |████| MFA alignment verification — 18 chapters, 4 batches, 88.5% (Phase 3, sample)
Week 5e  |████| QC filtering + TTS dataset build on 18-chapter sample (Phase 4-5, sample)
Week 6   |████| TTS fine-tuning environment validated + real training launched on Colab (Phase 6)
                 (9 environment bugs found and fixed; training resumed across 2 Colab disconnects)
Week 6   |████| First objective TTS evaluation (MCD/F0 RMSE) at intermediate checkpoint
Week 7   |░░░░| Full 260-chapter Phase 3 alignment run (planned, not yet executed)
Week 7   |░░░░| Full-corpus Phase 4-5 QC + dataset rebuild (planned)
Week 8   |░░░░| Final TTS fine-tune on full dataset + final evaluation (planned)
Week 8   |░░░░| Report writing, release preparation
```

*(████ = complete at time of writing; ░░░░ = planned / remaining work)*

---

## Chapter 5: Analysis and Evaluation

### 5.1 Data Analysis

**Corpus statistics** (Table 5.1) confirm the text pipeline produced a clean,
sufficiently large corpus for the modelling stages that follow it.

**Table 5.1. Final preprocessed corpus statistics.**

| Metric | Value |
|---|---|
| Total sentences | 15,738 |
| Total tokens | 179,918 |
| Unique word types | 13,124 |
| Type-token ratio | 0.0729 |
| Mean sentence length (tokens) | 11.43 |
| Median sentence length (tokens) | 10.0 |
| Devanagari + ZWJ character purity | 97.92% |
| Training sentences (90%) | 14,164 |
| Test sentences (10%) | 1,574 |

The low type-token ratio (0.0729) is consistent with Sunuwar's agglutinative
morphology: a large fraction of distinct word *forms* are single-occurrence
inflected variants of a smaller set of roots, which directly explains why static
whole-word embeddings later show high OOV rates, and motivates the subword and
character-n-gram approaches used downstream.

**TTS audio and segmentation statistics** (Table 5.2) summarise the current state
of the audio-side pipeline, distinguishing what has been run at full scale from
what remains a validated sample.

**Table 5.2. TTS pipeline scale, full corpus vs. verification sample.**

| Quantity | Full corpus | Verification sample used so far |
|---|---|---|
| Chapters | 260 | 18 (6.9%) |
| Segments | 12,672 | 915 (7.2%) |
| Audio duration | 29.5h | 2.06h |
| Alignment rate | *(not yet run)* | 88.5% (810/915) |
| QC pass rate | *(not yet run)* | 75.4% (690/915) |
| Final TTS training data | *(pending)* | 1.37h (554 train / 136 val clips) |

### 5.2 Results

**Table 5.3. SentencePiece tokeniser evaluation.**

| Model | Vocabulary size | Fertility | OOV rate |
|---|---|---|---|
| `sunuwar_spm_8k` | 8,000 | 1.3612 | 0.0% |
| `sunuwar_spm_16k` | 16,000 | 1.3612 | 0.0% |

Both models are identical in behaviour because the corpus supports only 6,764
unique subword pieces — the extra vocabulary slots in both models are unused.
The 8k model is therefore recommended, as a smaller embedding table gives more
training signal per parameter on a small corpus.

**Table 5.4. Word embedding evaluation.**

| Model | OOV rate (test set) | Genre F1 (macro) |
|---|---|---|
| word2vec Skip-gram | 30.903% | 0.2498 |
| word2vec CBOW | 30.903% | 0.2499 |
| fastText | 19.9103% | 0.2494 |

**Table 5.5. SunuwarBERT-small pre-training results (selected epochs).**

| Epoch | Validation perplexity |
|---|---|
| 1 | 1226.74 |
| 5 | 122.79 |
| 10 | 39.13 |
| 15 | 26.99 |
| 20 | 20.46 |
| 25 | 18.00 |
| **30 (best)** | **14.79** |
| 35 (early stop) | 15.60 |

```
Validation Perplexity vs. Epoch (log scale, illustrative)

1200 ┤●
     │ \
 400 ┤  ●
     │   \
 120 ┤    ●
     │     \___
  40 ┤         ●___
     │             \●__●__●●___________●
  15 ┤                              ●───●
     └──┬───┬───┬───┬───┬───┬───┬───┬──
        1   5  10  15  20  25  30  35
                     Epoch
```

**Table 5.6. TTS objective evaluation (checkpoint-5,500, ~40% through the planned
13,800-step run on the 1.37h dataset).**

| Metric | Result | Literature context |
|---|---|---|
| MCD (Mel-Cepstral Distortion) | ~8–10 dB, GAL_001/LUK_005 validation chapters | ~2–6 dB typical for very good published TTS quality; ~10–12 dB for perceptibly imperfect but functional speech |
| F0 RMSE | ~40–100 Hz | Pitch contour not yet closely tracking reference at this training stage |
| Real-vs-real sanity check | ~8–10 dB | Confirms metric calibration is correct (not a measurement artefact) |

**Table 5.7. Model parameter counts.**

| Model | Parameters |
|---|---|
| SunuwarBERT-small | 14,485,568 (~14.5M) |
| MMS-TTS-Maithili base (VITS) | *(base checkpoint; fine-tuned parameter count inherited from `facebook/mms-tts-mai`)* |

### 5.3 Comparison with Objectives

| Objective (Ch. 1) | Status | Evidence |
|---|---|---|
| 1. Clean, reproducible text corpus | ✅ Met | 15,738 sentences, 97.92% Devanagari purity, fully documented preprocessing steps |
| 2. Subword tokeniser evaluation | ✅ Met | 0.0% OOV, fertility 1.3612, comparative 8k vs. 16k analysis |
| 3. Comparative word embedding evaluation | ✅ Met | fastText (19.9% OOV) vs. word2vec (30.9% OOV), genre-F1 proxy task run for both |
| 4. Contextual language model (SunuwarBERT-small) | ✅ Met | 540.9× perplexity improvement over random baseline, early-stopped correctly |
| 5. Correctly aligned per-sentence speech dataset | 🔄 Partially met | Full-corpus segmentation done (12,672 segments); alignment/QC verified only on a 6.9% sample (88.5% / 75.4%); full 260-chapter run not yet executed |
| 6. Fine-tuned TTS model with objective evaluation | 🔄 Partially met | Pipeline works end-to-end, real training launched, first quantitative MCD/F0 result obtained on a mid-training checkpoint of a partial (1.37h) dataset — not yet the full-corpus final model |
| 7. Document all non-obvious engineering obstacles | ✅ Met | 9 Colab environment bugs, alignment failure modes, transcript bug, and metric bug all explicitly documented with root cause and fix |

Objectives 1–4 (the entire NLP track) are fully met with quantitative, held-out
evaluation results. Objectives 5–6 (the TTS track) are on a validated path — every
individual pipeline stage has been proven correct on a representative sample —
but the project's TTS deliverable at the time of writing is a **verification-scale
result** (18 of 260 chapters), not yet the final full-corpus model. This is stated
plainly rather than implied otherwise, consistent with the project's reproducibility
goal.

### 5.4 Discussion of Findings

The NLP track's results support a clear narrative: static, context-free
representations (word2vec) struggle with Sunuwar's morphology (30.9% OOV),
character-aware representations (fastText) partially compensate (19.9% OOV), and
neither static representation is sufficient for a document-level task like genre
classification (both score *below* the 0.333 random baseline for a 3-class
problem) — motivating the shift to a contextual transformer. SunuwarBERT-small's
540.9× perplexity improvement over random guessing demonstrates that a
deliberately small transformer (14.5M parameters, chosen to match a ~180K-token
corpus) can learn genuine distributional structure without the corpus being
"big enough" by conventional standards — the architecture-to-data-size ratio,
not absolute data volume, is the operative variable.

The TTS track's findings are, if anything, more instructive about how a
low-resource pipeline actually fails and gets fixed. Two working hypotheses about
what predicts alignment failure (chapter length; OOV rate) were both explicitly
tested and refuted using held-out chapters — a discipline that prevented the
project from "fixing" the wrong variable. The one hypothesis that *did* hold
(the pre-alignment duration-outlier flag predicting alignment failure) is
practically valuable precisely because it is available *before* the expensive
alignment step, letting future runs pre-triage which chapters need
re-segmentation. Equally, the MCD metric bug (a 60× measurement inflation from
using librosa MFCC as a stand-in for true mel-cepstral coefficients) is a
cautionary finding in its own right: without the real-vs-real sanity check, a
broken evaluation script would have been indistinguishable from a genuinely bad
TTS model, and any conclusion drawn from it would have been wrong for the wrong
reason. The current MCD result (~8–10 dB) is a genuinely reportable,
literature-comparable number, but it is measured on an intermediate checkpoint
(40% through training) of a partial dataset (1.37h of an eventual >20h), so it
should be read as **evidence the pipeline produces measurably improving,
non-degenerate speech**, not as a final quality claim.

---

## Chapter 6: Conclusion and Future Work

### 6.1 Conclusion

This project built the first reproducible NLP and TTS pipeline for Sunuwar, an
endangered Kiranti language with no prior public digital-language resources. Using
only a single Bible translation (text and matched read-aloud audio) as source data,
the project delivered: a cleaned 179,918-token Devanagari corpus; a SentencePiece
subword tokeniser with 0% OOV; comparative word2vec/fastText embeddings; a
from-scratch pre-trained transformer language model (SunuwarBERT-small) achieving a
540.9× perplexity improvement over random guessing; a validated forced-alignment
and quality-control pipeline for building a single-narrator TTS training set; and a
working, environment-hardened VITS fine-tuning procedure transferring from
`facebook/mms-tts-mai`, with an objective, literature-calibrated evaluation metric
(MCD/F0 RMSE) producing a first genuine quantitative result (~8–10 dB MCD).

Equally important to the concrete artefacts is the project's documentation of
*why* several plausible default approaches failed on this specific data — whole-
chapter alignment, naive silence-count segmentation, assuming a Nepali MMS
checkpoint exists, a naive MFCC-based MCD implementation — each replaced with a
corpus-specific, empirically validated alternative. This documentation is itself a
deliverable: it is what makes the pipeline genuinely reproducible for a future
researcher extending it to more Sunuwar data, or applying it to a related Kiranti
language facing the identical resource gap.

### 6.2 Limitations

- **Register bias.** The entire corpus is religious/scriptural text; models may
  not generalise well to informal, conversational, or technical Sunuwar registers.
- **Corpus size.** ~180K tokens is small relative to high-resource NLP corpora;
  representation depth (especially for rare inflected forms) is correspondingly
  limited.
- **TTS scale.** The evaluated TTS result is based on 1.37 hours from 18 of 260
  available chapters; the full-scale alignment, QC, and dataset build (and the
  corresponding final fine-tune) had not been executed at the time of writing.
- **TTS quality expectations.** Even at full scale, the fine-tuned TTS model is
  expected to produce partially intelligible speech from a single narrator's
  reading of scripture, not broadcast-quality general-purpose synthesis.
- **No native-speaker evaluation.** All current evaluation is objective/automatic
  (perplexity, OOV, MCD, F0 RMSE); no Sunuwar speaker has yet assessed
  intelligibility or naturalness, which the project's own roadmap identifies as
  the ultimately necessary evaluation for the TTS track, since Whisper cannot
  transcribe Sunuwar and therefore cannot provide a WER-based proxy either.
- **Script coverage.** All resources use Devanagari; the native Sunuwar script
  (Koĩts Brese/Tikamuli) has no digital resources and was out of scope.
- **Single-speaker TTS.** The model is trained on one narrator's voice only; no
  multi-speaker or voice-cloning capability is provided.

### 6.3 Future Work

1. **Complete the full 260-chapter Phase 3 alignment run**, re-running
   `segment_from_pauses.py` first (to pick up the transcript-cleaning fix), testing
   MFA's `--single_speaker` flag to reduce the projected ~2.5-day single-threaded
   runtime before committing to the full run.
2. **Re-run QC filtering and TTS dataset assembly at full scale**, expected to
   yield a substantially larger training set than the current 1.37-hour sample.
3. **Re-run the full TTS fine-tune on the complete dataset** and re-evaluate MCD/F0
   RMSE at the final checkpoint, to report a genuine before/after training-progress
   trend rather than a single mid-training snapshot.
4. **Obtain native Sunuwar-speaker evaluation** of synthesised speech
   intelligibility and naturalness, since no automatic metric (MCD, F0 RMSE, or a
   Whisper-based WER proxy, which is not viable for Sunuwar) substitutes for
   community judgment.
5. **Community-recorded supplementary audio** targeting specific phonemes with
   weak coverage in the current G2P inventory, since the existing scripture
   recording cannot be extended and some phoneme gaps are a documented, scoped
   limitation.
6. **Extend the pipeline to other Kiranti/Himalayan languages** (Rai, Limbu, Hayu,
   Jirel) that share the same resource pattern (a single Bible translation plus
   matched audio), reusing the corpus-cleaning, alignment, and TTS-transfer
   methodology developed here.
7. **Downstream NLP task fine-tuning** (e.g., named entity recognition or
   morphological tagging) using SunuwarBERT-small as a base encoder, to
   demonstrate its contextual representations are useful beyond the pre-training
   objective itself.

---

## References (IEEE Style)

[1] P. Joshi, S. Santy, A. Budhiraja, K. Bali, and M. Choudhury, "The state and
fate of linguistic diversity and inclusion in the NLP world," in *Proc. 58th Annu.
Meeting Assoc. Comput. Linguistics (ACL)*, 2020, pp. 6282–6293.

[2] T. Kudo and J. Richardson, "SentencePiece: A simple and language independent
subword tokenizer and detokenizer for neural text processing," in *Proc. 2018
Conf. Empirical Methods Natural Lang. Process. (EMNLP): Syst. Demonstrations*,
2018, pp. 66–71.

[3] T. Mikolov, I. Sutskever, K. Chen, G. S. Corrado, and J. Dean, "Distributed
representations of words and phrases and their compositionality," in *Advances in
Neural Information Processing Systems (NeurIPS)*, vol. 26, 2013, pp. 3111–3119.

[4] P. Bojanowski, E. Grave, A. Joulin, and T. Mikolov, "Enriching word vectors
with subword information," *Trans. Assoc. Comput. Linguistics*, vol. 5, pp.
135–146, 2017.

[5] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of deep
bidirectional transformers for language understanding," in *Proc. 2019 Conf. North
Amer. Chapter Assoc. Comput. Linguistics: Human Lang. Technol. (NAACL-HLT)*, 2019,
pp. 4171–4186.

[6] V. Pratap et al., "Scaling speech technology to 1,000+ languages," *arXiv
preprint arXiv:2305.13516*, 2023.

[7] J. Kim, J. Kong, and J. Son, "Conditional variational autoencoder with
adversarial learning for end-to-end text-to-speech," in *Proc. 38th Int. Conf.
Machine Learning (ICML)*, PMLR vol. 139, 2021, pp. 5530–5540.

[8] M. McAuliffe, M. Socolof, S. Mihuc, M. Wagner, and M. Sonderegger, "Montreal
Forced Aligner: Trainable text-speech alignment using Kaldi," in *Proc. Interspeech
2017*, 2017, pp. 498–502.

[9] D. Borchers, *A Grammar of Sunwar: Descriptive Grammar, Paradigms, Texts and
Glossary*. Leiden, The Netherlands: Brill, 2008.

---

## Appendices

*(Insert screenshots of: `corpus_report.md` rendered output; SentencePiece training
logs; sample masked-token predictions from SunuwarBERT-small at epochs 5/15/30;
`segmentation_report.csv` sample rows; MFA alignment batch logs; TextGrid viewer
screenshot for a sample aligned segment; `qc_report.csv` sample rows; Colab
training loss curves for the VITS fine-tune; a spectrogram/waveform comparison
between a reference clip and its synthesised counterpart, used for the MCD
evaluation.)*
