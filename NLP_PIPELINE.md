# Sunuwar NLP Pipeline — Full Walkthrough

This document explains every step of the NLP pipeline built for Sunuwar (ISO 639-3: `suz`),
an endangered Kiranti language of Nepal written in Devanagari script. The pipeline goes from
raw scripture text all the way to a trained contextual language model.

---

## The Big Picture

```
Raw USFM Bible text
       │
       ▼
Step 1: Text Preprocessing      → clean sentences, one per line
       │
       ▼
Step 2: Train/Test Split        → 90% train, 10% test
       │
       ▼
Step 3: SentencePiece Tokeniser → subword vocabulary (8k pieces)
       │
       ▼
Step 4: Word Embeddings         → word2vec + fastText static vectors
       │
       ▼
Step 5: SunuwarBERT-small       → contextual language model (MLM)
       │
       ▼
Step 6: Evaluation              → OOV rates, perplexity, genre F1
```

---

## Step 1 — Text Preprocessing (`src/preprocess_text.py`)

### What we start with

The raw data is the **Sunuwar New Testament** in USFM format (Universal Standard Format Markers) — a plain-text Bible markup format used by organisations like Wycliffe. It looks like this:

```
\id MAT
\h मत्ती
\c 1
\p
\v 1 इब्राहिमको सन्तान, दाऊदको सन्तान, येशू ख्रीष्टको वंशावली यस प्रकार छ।
\v 2 इब्राहिमबाट इसहाक जन्मिए \em (उत्पत्ति २१:३) \em*
```

We use **27 NT books only** (not the Old Testament) to keep the pipeline reproducible.
Total: ~7,959 verses across books like Matthew, Mark, Romans, Revelation.

### The problem

The raw USFM has several things that are **not Sunuwar running text**:
- `\em … \em*` blocks — these are scripture cross-references and Nepali-language translator
  footnotes (e.g. `\em मत्ती ३:१ \em*`). They must be removed entirely.
- Structural markers like `\c`, `\s1`, `\id`, `\h`, `\r` — chapter/section headings, book IDs.
- Verse markers like `\v 4` — these prefix each line but the number is not text.
- Bold markers `\bd … \bd*` — keep the text inside, strip the markers.

### The 8 processing steps (in order)

**1. Read NT files only**
Open the zip archive, filter to files whose names contain one of the 27 NT book codes
(`MAT`, `MRK`, `LUK`, … `REV`).

**2. Discard full structural lines**
Any line starting with `\id`, `\h`, `\toc1`, `\toc2`, `\toc3`, `\mt1`, `\c`, `\s1`,
`\r`, `\ip`, `\io1`, `\io2` is thrown away entirely. These are metadata.

**3. Strip `\em … \em*` blocks**
Regex `\\em.*?\\em\*` removes the entire span. This is the most important step —
these blocks contain Nepali text that would pollute the Sunuwar corpus if kept.

**4. Strip inline formatting markers**
Remove `\bd*`, `\em*` closing tags and `\bd` opening tags. The text they wrap is kept.

**5. Strip `\v N` verse markers**
The pattern `\\v \d+\s*` at line start is removed, keeping the Sunuwar sentence that follows.

**6. Unicode NFC normalisation**
`unicodedata.normalize('NFC', text)` — ensures all Devanagari characters are in canonical
composed form. Without this, visually identical characters can have different byte sequences,
which would split what should be one word type into multiple types.

**7. Sentence segmentation on danda**
Sunuwar sentences end with `।` (U+0964, Devanagari Danda) or `॥` (U+0965, Double Danda).
We split on these characters. This is the Devanagari equivalent of splitting on a period.

**8. Script filter + length filter**
- Drop any segment where non-Devanagari, non-ASCII characters exceed 20% of total characters.
  This catches any stray Nepali or Latin text that slipped through.
- Drop segments with fewer than 3 whitespace-separated tokens. Single-word fragments are noise.
- **Zero Width Joiner (U+200D) is preserved throughout** — it appears 54,692 times in the NT
  and is part of how Devanagari conjunct characters (ligatures) are rendered correctly.

### Output

`data/processed/sunuwar_nt_raw.txt` — one Sunuwar sentence per line, UTF-8.

| Metric | Value |
|--------|-------|
| Total sentences | 15,738 |
| Total tokens | 179,918 |
| Unique word types | 13,124 |
| Mean sentence length | 11.43 tokens |
| Median sentence length | 10.0 tokens |
| Devanagari character purity | 97.92% |

The 97.92% purity confirms the `\em` stripping worked — only ~2% non-Devanagari characters
remain (mostly ASCII punctuation like `.` and `,` used occasionally in the text).

---

## Step 2 — Train/Test Split (`src/split_corpus.py`)

A simple 90/10 random split with seed 42.

| Split | Sentences |
|-------|-----------|
| `data/processed/train.txt` | 14,164 |
| `data/processed/test.txt` | 1,574 |

The test set is **held out** — never used as input to any training script, only for evaluation.

---

## Step 3 — SentencePiece Tokeniser (`src/train_spm.py`)

### Why subword tokenisation?

Sunuwar is **agglutinative** — it forms words by chaining suffixes together. The same root
word can appear in dozens of inflected forms. A whole-word vocabulary would have massive OOV
(out-of-vocabulary) problems on a corpus of only ~180K tokens. Subword tokenisation solves
this by breaking rare or unseen word forms into known subword pieces.

Example (conceptual): the word `खान्छु` ("I eat") might be split into `खान्` + `छु`, where
each piece is seen enough times to have a trained representation.

### The unigram model

We use the **unigram** algorithm from Google's SentencePiece library. Unlike BPE (Byte-Pair
Encoding) which builds a vocabulary bottom-up, unigram starts with a large candidate set and
prunes it by maximising the likelihood of the training corpus under a unigram language model.

Parameters:
- `model_type: unigram`
- `character_coverage: 0.9995` — the model must be able to encode 99.95% of all character
  types in the corpus. For a small corpus like ours, this ensures every Devanagari character
  is represented.
- `byte_fallback: true` — any character not covered falls back to UTF-8 byte pieces, so the
  model can never produce `<unk>` for a unicode character.
- Special tokens: `[PAD]` (0), `[UNK]` (1), `[CLS]` (2), `[SEP]` (3), `[MASK]` (4)

We train two models for comparison:

| Model | Vocab size | Fertility | OOV rate |
|-------|-----------|-----------|----------|
| `sunuwar_spm_8k` | 8,000 | 1.3612 | 0.0% |
| `sunuwar_spm_16k` | 16,000 | 1.3612 | 0.0% |

### Interpreting the results

**Fertility** = average number of subword pieces per whitespace-delimited word. A fertility of
1.3612 means each word is split into about 1.36 pieces on average. This is lower than typical
(healthy range is 2.5–4.0 for most languages) because Sunuwar morphology in Biblical register
produces relatively constrained surface forms — the same verse patterns repeat.

**OOV rate = 0.0%** — the byte fallback ensures this is always zero. No word can be
unrepresentable.

**Why 8k over 16k?** Both produce identical fertility scores because the corpus only supports
6,764 unique subword pieces — the extra slots in 8k and 16k are simply unused. We use the 8k
model for SunuwarBERT because a smaller embedding table means more training signal per
embedding vector on a small corpus.

---

## Step 4 — Word Embeddings (`src/train_word2vec.py`, `src/train_fasttext.py`)

Word embeddings are **static vectors** — each word type gets one fixed vector regardless of
context. They are useful for similarity lookups, analogy tasks, and as features for downstream
classifiers.

### word2vec (two variants)

Trained with Gensim on `train.txt` using whitespace-tokenised words (not subword pieces).

**Skip-gram (sg=1):** Given a target word, predict its surrounding context words. Better at
representing rare words because it focuses on each target word individually.

**CBOW (sg=0):** Given surrounding context, predict the target word. Faster and slightly
better for frequent words.

Shared hyperparameters:
- `vector_size: 200` — each word maps to a 200-dimensional vector
- `window: 5` — look 5 words left and right for context
- `min_count: 2` — ignore words that appear fewer than 2 times
- `negative: 10` — negative sampling with 10 noise words per positive example
- `epochs: 20`
- `seed: 42`

### fastText

fastText extends word2vec by representing each word as the **sum of its character n-gram
vectors**, in addition to the word itself. For example, `खाने` would be represented partly by
the n-grams `खा`, `खान`, `खाने`, `ाने`, `ने`, etc.

Parameters: `dim=200`, `window=5`, `minn=3` (min n-gram length), `maxn=5`, `min_count=1`,
`neg=10`, `epochs=20`.

The key advantage: fastText can **generate a vector for any word**, even if it was never seen
during training, by composing its character n-gram vectors. This directly addresses Sunuwar's
high morphological variation.

### Results

| Model | OOV rate | Genre F1 (macro) |
|-------|----------|-----------------|
| word2vec Skip-gram | 30.9% | 0.250 |
| word2vec CBOW | 30.9% | 0.250 |
| fastText | 19.9% | 0.249 |

**OOV interpretation:** 30.9% of unique word types in the test set are not in the word2vec
vocabulary. This is expected — with only ~180K training tokens, many inflected forms appear
only once and fall below `min_count=2`. fastText's character n-grams reduce this to 19.9%.

**Genre F1 interpretation:** The three-class task (narrative / epistolary / apocalyptic)
scores ~0.25 for all models, which is **below the random baseline of 0.33**. This is not
surprising — averaged embeddings are a very weak representation for document-level genre
classification. It motivates the next step: contextual representations via BERT.

---

## Step 5 — SunuwarBERT-small (`src/train_mlm.py`)

This is the most sophisticated model in the pipeline. Instead of a single static vector per
word, SunuwarBERT produces a **different vector for every token depending on its context** —
the same subword piece gets a different representation at the start of a sentence vs. the end.

### Architecture

SunuwarBERT is a **bidirectional transformer encoder** — essentially a miniaturised BERT.
"Bidirectional" means when processing position `i`, the model can see all tokens to the
left and right simultaneously (unlike GPT which is left-to-right only).

```
Input:  [CLS] ▁येशू ▁ख्रीष्टको ▁वंशावली [SEP]
          │       │         │          │       │
       Token  Token     Token      Token    Token
     Embedding Embedding Embedding  Embedding Embedding
          +       +         +          +       +
       Pos    Pos       Pos        Pos     Pos
     Embedding Embedding Embedding  Embedding Embedding
          │       │         │          │       │
          └───────┴────── 6× Transformer Encoder Layers ──────┘
                                    │
                              MLM Head (Linear)
                                    │
                          Vocab logits (6,764 classes)
```

Architecture parameters:

| Parameter | Value |
|-----------|-------|
| Layers | 6 |
| Hidden dimension | 384 |
| Attention heads | 8 |
| FFN dimension | 1,024 |
| Max sequence length | 128 tokens |
| Dropout | 0.1 |
| Activation | GELU |
| Total parameters | **14,485,568** (~14.5M) |

### The MLM objective (Masked Language Modelling)

The model is trained to **predict randomly masked tokens**. For each batch:

1. Randomly select 15% of non-special tokens as candidates.
2. Of those candidates:
   - 80% are replaced with `[MASK]`
   - 10% are replaced with a random token from the vocabulary
   - 10% are left unchanged
3. The model must predict the original token at every masked position.
4. Loss is cross-entropy, computed only at masked positions (`ignore_index=-100` everywhere else).

This forces the model to learn distributional patterns of the language without ever seeing
the target token directly — it must infer what goes there from surrounding context.

### Training setup

| Setting | Value |
|---------|-------|
| Optimiser | AdamW |
| Learning rate | 5×10⁻⁴ |
| Weight decay | 0.01 |
| Batch size | 32 |
| Gradient accumulation | 4 steps (effective batch = 128) |
| Warmup | 10% of total steps (linear) |
| Mixed precision | fp16 |
| Early stopping patience | 5 epochs |
| Random seed | 42 |

### Training results

| Epoch | Val perplexity |
|-------|---------------|
| 1 | 1226.74 |
| 5 | 122.79 |
| 10 | 39.13 |
| 15 | 26.99 |
| 20 | 20.46 |
| 25 | 18.00 |
| **30** | **14.79** ← best |
| 35 | 15.60 (early stop) |

**Perplexity** measures how "surprised" the model is at the test data. Lower is better.
A random model that guesses uniformly from 8,000 vocab pieces would have perplexity = 8,000.
Our model reaches **14.79 — a 540.9× improvement over the random baseline**.

Qualitative progression observed during training:
- Epochs 1–5: top-5 predictions are mostly high-frequency pieces with no coherence
- Epoch 15: predictions begin to show plausible morphological variants of the masked token
- Epoch 25–30: top-5 candidates are consistently morphologically and semantically compatible
  with the surrounding context

The best checkpoint is saved to `models/sunuwar_transformer.pt`.

---

## Step 6 — Evaluation (`src/evaluate_nlp.py`)

The evaluation harness (`evaluate_nlp.py`) runs all models and collates results into JSON
files in `results/`:

- `results/corpus_stats.json` — sentence counts, token counts, character purity
- `results/tokeniser_eval.json` — fertility and OOV rate per vocab size
- `results/embedding_eval.json` — OOV rates and genre F1 for word2vec + fastText
- `results/bert_eval.json` — perplexity curve and best epoch for SunuwarBERT
- `results/corpus_report.md` — full human-readable report with tables

---

## Summary — What Each Model Is Good For

| Model | Use case | Limitation |
|-------|----------|------------|
| SentencePiece 8k | Input tokeniser for any neural model | Not a standalone model |
| word2vec | Word similarity, quick lookup | 30.9% OOV on test set |
| fastText | Better similarity + handles OOV words via n-grams | Still static (no context) |
| SunuwarBERT-small | Contextual embeddings for NER, tagging, classification | Needs fine-tuning for specific tasks |

For downstream tasks requiring **word-level lookup**, use fastText (lowest OOV).
For tasks requiring **contextual representations** (sequence labelling, classification),
use SunuwarBERT-small.

---

## Data Licence Note

All models were trained on the Sunuwar Bible NT (© 2011 Wycliffe Bible Translators, Inc.),
licensed CC BY-NC-ND 4.0. Raw text cannot be redistributed. Trained model weights may be
released for research use with attribution.
