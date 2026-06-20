# Lost Voices — Sunuwar Language AI Pipeline
## Claude Code Instruction File

---

## Project in one sentence

Build the first publicly reproducible NLP + TTS pipeline for Sunuwar (ISO 639-3: `suz`),
an endangered Kiranti language of Nepal written in Devanagari script.

---

---

## Repository layout

```
lost-voices-sunuwar/
├── CLAUDE.md                  ← this file
├── README.md
├── data/
│   ├── raw/
│   │   ├── suzBl_usfm.zip     ← full Sunuwar Bible USFM (OT+NT), eBible.org
│   │   ├── suzBl_readaloud.zip← plain Devanagari text, chapter-per-file, no markers
│   │   └── supplementary/     ← web-scraped Sunuwar text (future)
│   ├── processed/
│   │   ├── sunuwar_nt_raw.txt ← output of preprocess_text.py (one sentence per line)
│   │   ├── train.txt          ← 90% split
│   │   ├── test.txt           ← 10% split (held-out, do not train on this)
│   │   └── tts_dataset/       ← MFA-aligned WAV+TXT pairs
│   ├── eval/
│   │   ├── eval_similarity.csv← manually curated Sunuwar word-pair similarity set
│   │   └── eval_analogy.txt   ← word analogy triplets
│   └── provenance.csv         ← data source log (source, url, date, licence)
├── src/
│   ├── preprocess_text.py     ← USFM parser + corpus cleaner 
│   ├── preprocess_audio.py    ← MP3→WAV converter + normaliser 
│   ├── align.py               ← MFA wrapper 
│   ├── train_spm.py           ← SentencePiece tokeniser training 
│   ├── train_word2vec.py      ← word2vec training + eval 
│   ├── train_fasttext.py      ← fastText training + eval 
│   ├── train_mlm.py           ← SunuwarBERT-small MLM pre-training
│   ├── train_tts.py           ← MMS TTS fine-tuning 
│   ├── evaluate_nlp.py        ← NLP evaluation harness 
│   └── evaluate_tts.py        ← TTS WER + MOS evaluation 
├── models/                    ← saved checkpoints (gitignored if large)
├── configs/                   ← YAML config files, one per training script
├── notebooks/                 ← exploratory Colab notebooks
├── results/                   ← evaluation tables, figures, training curves
└── data/provenance.csv
```

---

## Data: what we have

### Primary text source
- **File:** `data/raw/suzBl_usfm.zip`
- **Content:** Full Sunuwar Bible, 66 books (OT + NT), USFM format
- **Licence:** CC BY-NC-ND 4.0, © 2011 Wycliffe Bible Translators, Inc.
- **Use:** Non-commercial academic research. Attribution required. No redistribution of raw text.
- **For NLP:** Use NT books only (27 books, ~7,959 verses, ~200–220K tokens)
- **OT books are bonus data** — use only if NT corpus proves too small for training

### TTS text source ( track)
- **File:** `data/raw/suzBl_readaloud.zip`
- **Content:** Same Bible, plain Devanagari, one `.txt` per chapter, all USFM markers
  and verse numbers already stripped — **no text preprocessing needed**
- **Licence:** Same as above

### NT book files inside suzBl_usfm.zip
Identified by these filename fragments (prefix number + code + `suzBl.usfm`):
```
46-MAT  47-MRK  48-LUK  49-JHN  74-ACT
51-ROM  52-1CO  53-2CO  54-GAL  55-EPH
56-PHP  57-COL  58-1TH  59-2TH  60-1TI
61-2TI  62-TIT  63-PHM  64-HEB  65-JAS
90-1PE  67-2PE  68-1JN  69-2JN  94-3JN
71-JUD  72-REV
```
OT books have prefix numbers 02–45. Do not process these unless explicitly asked.

---

## Language facts Claude Code must know

| Property | Value |
|----------|-------|
| Language | Sunuwar (also Koĩts Lo, Koinch) |
| ISO 639-3 | `suz` |
| Script | Devanagari (working script for all digital resources) |
| Word order | SOV (Subject-Object-Verb) |
| Morphology | Agglutinative, rich verbal suffixation |
| Script Unicode range | U+0900–U+097F |
| Sentence boundary | Devanagari Danda U+0964 (।) and Double-Danda U+0965 (॥) |
| ZWJ usage | U+200D (Zero Width Joiner) appears **54,692 times** in the NT — used inside words for Devanagari conjunct character rendering. **Must be preserved**, not stripped |
| Unique vocab size (NT est.) | ~15,000–30,000 word types |
| Related language for reference | Nepali (also Devanagari, SOV, but different vocabulary) |

---

## USFM marker reference (complete inventory for NT)

These are **all** markers present in the NT. No others exist in this file.

| Marker | Count | Type | Action |
|--------|-------|------|--------|
| `\v N` | 7,959 | Verse number | Strip marker, keep text that follows |
| `\em … \em*` | 7,358 | Italic (cross-refs + rare footnotes) | **Strip entire block including content** — contains scripture cross-references like "मत्ती ३:१" and occasional Nepali explanatory footnotes, neither is Sunuwar running text |
| `\p` | 1,533 | Paragraph | Strip marker, keep nothing (paragraph break only) |
| `\s1` | 903 | Section heading | Strip entire line |
| `\c N` | 260 | Chapter number | Strip entire line |
| `\r` | 238 | Parallel references | Strip entire line |
| `\bd … \bd*` | 162 | Bold | Strip markers, keep text inside |
| `\ip` | 81 | Intro paragraph | Strip entire line |
| `\io1` | 49 | Intro outline level 1 | Strip entire line |
| `\id` | 27 | Book identifier | Strip entire line |
| `\h` | 27 | Running header | Strip entire line |
| `\toc1/2/3` | 27 each | Table of contents | Strip entire line |
| `\mt1` | 27 | Main title | Strip entire line |
| `\io2` | 14 | Intro outline level 2 | Strip entire line |
| `\m` | 13 | Flush-left paragraph | Strip marker, keep text |

**Key rule:** `\em…\em*` content must be fully removed. These are NOT Sunuwar text —
they are parenthetical cross-references and Nepali-language translator footnotes.
Keeping them would pollute the corpus with non-Sunuwar tokens.

---

## Preprocessing pipeline (preprocess_text.py)

### Input
`data/raw/suzBl_usfm.zip` — read NT files only (see list above)

### Processing steps in order

1. **Unicode NFC normalisation** — `unicodedata.normalize('NFC', text)`
2. **Strip full-line markers** — lines starting with `\id`, `\h`, `\toc`, `\mt`, `\c`,
   `\s1`, `\r`, `\ip`, `\io`, `\p`, `\m` → discard entire line
3. **Strip `\em…\em*` blocks** — regex `\\em.*?\\em\*` with re.DOTALL — remove
   entire span including content
4. **Strip inline closing markers** — remove `\bd*`, `\em*` remnants
5. **Strip `\bd` open marker** — `\bd` followed by text (keep the text, strip `\bd`)
6. **Strip `\v N` verse markers** — `\\v \d+\s*` at line start — keep text after
7. **Script filter** — drop any line where non-Devanagari non-space characters
   exceed 20% of total non-space characters. Include ZWJ (U+200D) as valid.
8. **Sentence segmentation** — split on `।` (U+0964) and `॥` (U+0965)
9. **Whitespace normalisation** — strip leading/trailing whitespace, collapse
   internal whitespace runs to single space
10. **Length filter** — discard segments with fewer than 3 whitespace-separated tokens

### Output
`data/raw/sunuwar_nt_raw.txt` — one sentence per line, UTF-8

### Stats to print on completion
- Total sentences written
- Total whitespace tokens
- Unique word types
- Mean and median sentence length (tokens)
- Percentage Devanagari characters (should be >95%)

---

## Model specifications

### SentencePiece tokeniser
- Model type: unigram
- Vocab sizes: 8,000 (primary) and 16,000 (comparison)
- Special tokens: `[PAD]` `[UNK]` `[CLS]` `[SEP]` `[MASK]`
- Character coverage: 0.9995
- Input: `data/processed/train.txt`
- Output: `models/sunuwar_spm_8k.model`, `models/sunuwar_spm_16k.model`

### Word embeddings
- **word2vec:** Gensim, Skip-gram + CBOW, dim=200, window=5, min_count=2,
  negative=10, epochs=20, sg=1
- **fastText:** Facebook fastText, dim=200, window=5, minn=3, maxn=5,
  min_count=1, neg=10, epochs=20
- Output: `models/sunuwar_w2v_sg.model`, `sunuwar_w2v_cbow.model`,
  `sunuwar_fasttext.bin`, `sunuwar_fasttext.vec`

### SunuwarBERT-small
- Architecture: BERT-style bidirectional encoder
- Layers: 6, hidden: 256, heads: 8, FFN: 1024
- Max sequence length: 128 tokens
- Vocabulary: SentencePiece 8k model
- Total parameters: ~15 million
- Dropout: 0.1
- Activation: GELU
- Objective: MLM (15% mask, 80/10/10 strategy)
- Optimiser: AdamW, lr=5e-4, warmup=10%, weight_decay=0.01
- Batch: 32 with 4 gradient accumulation steps (effective 128)
- Early stopping: patience=5 on validation perplexity
- Random seed: **42** (always)

### MMS TTS 
- Base: `facebook/mms-tts` from Hugging Face Hub
- Input: `data/processed/tts_dataset/metadata.csv` (columns: `file_name`, `text`)
- Audio: 16kHz mono WAV
- lr=1e-4, batch=16, max_steps=10000, warmup=1000
- Save checkpoint every 500 steps

---

## Evaluation targets

| Model | Metric | Target |
|-------|--------|--------|
| fastText | OOV rate on test.txt | Near 0% |
| word2vec | OOV rate on test.txt | 20–40% |
| fastText | Downstream genre F1 | Higher than word2vec |
| SunuwarBERT-small | Val perplexity | Below random baseline (~8000) |
| SunuwarBERT-small | Top-5 masked accuracy | Meaningful above chance |
| MMS TTS | WER (Whisper transcription) | Measurably below unintelligible |

---

## Code standards

- **Python 3.10+**
- **All scripts accept a YAML config file** as first argument: `python src/train_spm.py configs/spm.yaml`
- **Random seed 42** fixed at the top of every training script
- **No hardcoded paths** — all paths come from config YAML or argparse
- **UTF-8 everywhere** — always open files with `encoding='utf-8'`
- **ZWJ must be preserved** in all text processing — never strip U+200D
- **NFC normalisation first** in every text-handling script
- Log to stdout with timestamps; use `wandb` for training metrics
- Save intermediate outputs so pipeline can be resumed at any stage
- Every script must run end-to-end in Google Colab with a T4 GPU

---

## Constraints

- **Licence:** Raw text and audio cannot be redistributed publicly. Trained models can be released.
- **No translation task** — this project is monolingual NLP + TTS only. Do not build a translation model unless explicitly asked.
- **No OT data in NLP models** unless explicitly asked — NT only for reproducibility with proposal.
- **No internet calls in scripts** — all downloads happen separately; scripts read from `data/`.
- **Compute:** Google Colab T4 (15GB VRAM) is the baseline. Scripts must run there. A100 is a bonus.

---

## Current project status

| Week | Task | Status |
|------|------|--------|
| Day 0 | eBible data confirmed and downloaded | ✅ Done |
| Day 0 | BibleBrain API key requested | 🔲  pending |
| Week 1 | GitHub repo created | 🔲 Todo |
| Week 1 | preprocess_text.py | 🔲 Todo — next task |
| Week 1 | preprocess_audio.py | 🔲 Todo —  |
| Week 2 | SentencePiece tokeniser | 🔲 Todo |
| Week 3 | word2vec + fastText | 🔲 Todo |
| Week 4 | SunuwarBERT-small | 🔲 Todo |
| Week 5 | MFA alignment | 🔲 Todo —  |
| Week 6 | MMS TTS fine-tuning | 🔲 Todo —  |
| Week 7–8 | Evaluation + report + release | 🔲 Todo |

Update this table as tasks complete.

---

## When starting a new Claude Code session

Always read this file first. Then read the relevant source file if editing existing code.
Never assume state from a previous session — check `data/processed/` for what exists.
