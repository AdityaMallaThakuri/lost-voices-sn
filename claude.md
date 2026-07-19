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
- **For NLP:** Use NT books only (27 books, ~7,959 verses, 15,738 sentences, ~180K tokens after danda segmentation)
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
- Vocabulary: SentencePiece 8k model (effective vocab is 6,764 unique pieces)
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

## TTS roadmap (revised plan, decided 2026-07-19)

Superseding the naive "chunk → align → done" approach used for the Week 5 Mark pilot.
Informed by arXiv:2410.14197 (Unified Framework for Collecting TTS Datasets for 22
Indian Languages) adapted for a **fixed, pre-recorded, single-narrator corpus** — we
cannot record new audio, so corpus-balancing/recording-protocol parts of that paper
don't apply; the coverage/QC/G2P parts do.

### Key facts established
- Full 27-book NT audio: **260 chapters, 29.5 hours, single narrator**, zero degenerate
  (<5s) files. This is enough for solid single-speaker TTS — do not settle for the
  16-chapter Mark pilot as the final training set.
- Week 5 pilot alignment (233/288 chunks, 81%) used a flawed method: chunks were
  split by silence *before* alignment, and each chunk's text was **guessed** by
  proportionally slicing chapter words by duration fraction (assumes constant
  speaking rate — false).
- `mfa_dict.dict` used to map words to **individual Devanagari characters** as
  "phones" (fixed in Phase 1, see below).
- Sunuwar (`suz`) has no existing MMS-TTS checkpoint. Plan: transfer-learn from
  `facebook/mms-tts-nep` (Nepali, same script) rather than random init.
- Whisper cannot transcribe Sunuwar — WER-vs-Whisper (the CLAUDE.md eval target) is
  not a trustworthy intelligibility metric on its own. Treat it as secondary/exploratory
  only; real evaluation needs a Sunuwar speaker's judgment.
- **MFA runs locally, not in Colab.** There's a working conda env `aligner` (MFA
  3.3.9) on this machine already — confirmed via `~/Documents/MFA/command_history.yaml`,
  which also shows the Mark pilot's successful run happened here, not on Colab (an
  earlier session mistakenly believed otherwise). MFA doesn't use GPU (Kaldi/CPU-based),
  so no Colab/GPU is needed for alignment — only for Phase 5's actual TTS fine-tuning.
  Local machine: i5-11400H, 6c/12t, only 7.7GB RAM (the real local constraint — keep
  `num_jobs` low, e.g. 3-4, to avoid swapping).
- **Whole-chapter alignment fails completely, confirmed empirically.** An earlier
  local attempt at feeding MFA whole chapters directly (`mfa train` on unchunked
  `mfa_corpus`) has a log showing `Aligned 0, errors on 16, total 16` — 0% success.
  Kaldi's Viterbi search doesn't scale to multi-minute utterances. This means
  **pre-chunking before alignment is necessary, not the bug** — the bug was always
  just the proportional-word-guessing used to assign text to each chunk. The
  "align-then-segment on whole chapters" idea originally written into Phase 2 below
  is wrong and was replaced (see revised Phase 2).
- **Revised chunking method, validated empirically**: naive pause-count from silence
  detection does NOT match sentence count (~2.67x more pauses than sentences, 0/40
  sampled chapters within even a loose tolerance — narrators pause mid-sentence at
  clause/breath boundaries, not just at the danda). Fix: detect all candidate pauses
  permissively, then keep only the top `(sentence_count - 1)` *longest* pauses as the
  real sentence boundaries, discarding the rest as breath noise. Validated on 15
  sampled chapters: resulting per-sentence segment durations are plausible (5-10s
  mean, only 1/670 segments under 1s). This is now `src/segment_from_pauses.py`.

- **Full-corpus segmentation run completed** (2026-07-19): 260/260 chapters, zero
  fallbacks, **13,192 segments** written to
  `data/processed/audio/mfa_corpus_segments/speaker1/`, total duration recovered
  29.50h (exact match to source — no audio lost). Mean/median duration 8.1s/6.8s.
  912 segments (6.9%) flagged low-confidence (duration outlier). Report at
  `data/processed/audio/segmentation_report.csv`.
- **Disk space blocker found, not yet fixed**: MFA's default temp directory is
  `C:\Users\ASUS\Documents\MFA`, and **C: only has 5.5GB free**. The pilot's temp
  usage was 631MB for 288 segments/1.8h audio; scaling to 13,192 segments/29.5h
  could plausibly need 10-30GB. Must pass `--temporary_directory` pointed at D:
  (152GB free) before running Phase 3 training, or it will likely fail partway
  through a multi-hour run from running out of disk.
- **Phase 3 is not model training** — it's a data-verification step. MFA's
  "acoustic model" here is a disposable Kaldi tool that only produces alignment
  timestamps/confidence scores to validate Phase 2's segmentation guesses; it is
  never used as, or related to, the actual Sunuwar TTS model (that's Phase 6,
  fine-tuning `facebook/mms-tts-nep`). Don't conflate the two when resuming.
- **Phase 3 time/resource expectation**: no GPU needed (Kaldi/CPU). RAM: keep
  `num_jobs` at 3-4 given 7.7GB usable. Time: pilot's 288-segment/1.8h run took
  ~92 min locally; this run is 13,192 segments/29.5h (16-46x more depending on
  which dimension dominates scaling) — realistically many hours, possibly most of
  a day. Plan to run in the background/overnight, not expect a quick turnaround.

### Phases (do in order — each depends on the previous)
1. **G2P / phoneme dictionary** — ✅ done. `src/g2p.py` + `configs/g2p_sunuwar.yaml`,
   deterministic akshara-segmentation, real ~50-phone inventory, 98.9% coverage of
   the full 260-chapter vocab (9,602/9,711 tokens). See "Language facts" section for
   phone table details. Committed.
2. **Segment chapters into per-sentence clips with correct text** — ranked-silence-
   boundary method (above), not align-then-segment (that failed) and not the old
   proportional guess. `src/segment_from_pauses.py` + `configs/segment_from_pauses.yaml`,
   writes to `data/processed/audio/mfa_corpus_segments/speaker1/`, flags
   duration-outlier segments (<1.5s or >20s) as low-confidence rather than dropping
   them silently, writes `data/processed/audio/segmentation_report.csv`.
   `src/split_audio_chunks.py` is now superseded — do not use it going forward.
3. **Align the segments with MFA** — `src/align.py` (train mode) on the full
   corpus using the Phase 1 dictionary, producing the acoustic model + TextGrids.
   Since text per segment is now correct (not guessed), this should perform far
   better than the pilot's 81%.
4. **QC filtering with explicit thresholds** — use MFA's own per-utterance
   `overall_log_likelihood`, `phone_duration_deviation`, `snr` output (confirmed via
   the pilot's `alignment_analysis.csv` — this is MFA's built-in output, no custom
   script needed) plus the pre-alignment duration-outlier flag from step 2, to drop
   bad segments before they become training data.
5. **Build `data/processed/tts_dataset/metadata.csv`** from QC-passed segments.
   Hold out whole chapters (not random segments) for validation to avoid
   near-duplicate scriptural phrasing leaking train→val.
6. **Fine-tune from `facebook/mms-tts-nep`**, not from scratch — cross-lingual
   transfer converges faster and sounds more natural on limited single-speaker data
   than training a VITS model cold. May need Devanagari/ZWJ token remapping.
7. **Evaluate honestly** — MFA self-alignment-confidence of resynthesized audio as
   a cheap automatic signal; real naturalness/intelligibility judgment needs an
   actual Sunuwar speaker (community review), not just Whisper-WER.
8. **Phoneme coverage report** (from phase 1's real phoneme inventory) — diagnostic
   only, since we can't record more to patch gaps; document weak phonemes as a
   known limitation, and as a scoped future-work item (community-recorded
   supplementary set targeting specifically those gaps).

Currently on: **Phase 2 done, Phase 3 not started yet — paused here.** Segmentation
results already confirmed good (see "Full-corpus segmentation run completed" above).
Next concrete steps to resume, in order:
1. Fix `configs/align_train.yaml`: `corpus_dir` still points at the old
   `mfa_corpus_full` (whole chapters — the one that failed 0/16). Repoint it at
   `data/processed/audio/mfa_corpus_segments`.
2. Add `--temporary_directory` pointed at a D: path to the `mfa train` command in
   `src/align.py` (or config) — **do not skip this**, C: only has 5.5GB free and
   will likely run out of space partway through otherwise.
3. Run via the local `aligner` conda env (`conda run -n aligner mfa train ...` or
   activate the env first) — not Colab, MFA already works locally.
4. Expect a long run (many hours) — start it and let it run in the background,
   don't wait on it live.

Do not skip ahead to Phase 6 (TTS fine-tuning) before Phases 3-5 are done.

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
| Day 0 | BibleBrain API key requested | 🔲 pending |
| Week 1 | GitHub repo created | ✅ Done |
| Week 1 | preprocess_text.py | ✅ Done — 15,738 sentences, 179,918 tokens, 13,124 types |
| Week 1 | preprocess_audio.py | ✅ Done — full 27-book NT, 260 chapters, 29.5h, single narrator |
| Week 2 | SentencePiece tokeniser | ✅ Done — 8k + 16k unigram models |
| Week 3 | word2vec + fastText | ✅ Done — fastText OOV 19.9%, word2vec 30.9% |
| Week 4 | SunuwarBERT-small pre-training | ✅ Done — val_perplexity 14.79 epoch 30 |
| Week 5a | MFA alignment — Mark pilot (16 ch) | ✅ Done but flawed — 233/288 (81%), superseded below |
| Week 5b | Phoneme G2P dictionary (roadmap Phase 1) | ✅ Done — 98.9% coverage, ~50-phone inventory |
| Week 5c | Segment chapters into per-sentence clips (roadmap Phase 2) | 🔄 In progress — ran full 260-chapter pass locally, check `segmentation_report.csv` |
| Week 5d | Align segments with MFA (roadmap Phase 3) | 🔲 Todo — next task, run locally via `aligner` conda env, not Colab |
| Week 5e | QC filtering + build tts_dataset/metadata.csv (roadmap Phase 4–5) | 🔲 Todo |
| Week 6 | MMS TTS fine-tuning from mms-tts-nep (roadmap Phase 6) | 🔲 Todo |
| Week 7–8 | Evaluation (roadmap Phase 7–8) + report + release | 🔲 Todo |

Update this table as tasks complete. See "TTS roadmap" section above for the detailed
Week 5–6 plan and why the original Mark-pilot alignment is being redone.

---

## When starting a new Claude Code session

Always read this file first. Then read the relevant source file if editing existing code.
Never assume state from a previous session — check `data/processed/` for what exists.
