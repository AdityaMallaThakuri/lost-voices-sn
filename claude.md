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
- Sunuwar (`suz`) has no existing MMS-TTS checkpoint. **Nor does Nepali** —
  verified empirically 2026-07-27: `facebook/mms-tts-nep` and `-npi` both 404,
  and no Nepali code appears under `facebook/mms-tts` `full_models/` (the
  nearest codes present are `neb`, `new`, `npl`, `npy`). The "transfer-learn
  from mms-tts-nep" plan written into this roadmap was never viable.
- **Base checkpoint is `facebook/mms-tts-mai` (Maithili)**, chosen by measuring
  each candidate tokenizer's character coverage of the actual Sunuwar dataset
  (34,863 chars) rather than by assumption:

  | candidate | vocab | coverage | missing |
  |-----------|-------|----------|---------|
  | **mai**   | 67    | **98.65%** | U+0964 danda only |
  | hin       | 73    | 97.83%   | 2 distinct |
  | mar       | 74    | 93.03%   | 3 distinct |
  | ben/guj/eng | –   | 15.51%   | 50 distinct (non-Devanagari controls) |

  Maithili is also a Nepal contact language (Terai), so the linguistic and
  empirical choices agree. **ZWJ (U+200D) is present in all three Devanagari
  vocabs** — the feared ZWJ remapping is not needed; only the danda is dropped.
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
  fallbacks, 13,192 segments written to
  `data/processed/audio/mfa_corpus_segments/speaker1/`, total duration recovered
  29.50h (exact match to source — no audio lost). Mean/median duration 8.1s/6.8s.
  912 segments (6.9%) flagged low-confidence (duration outlier). Report at
  `data/processed/audio/segmentation_report.csv`. **The intro-offset fix
  (`src/fix_intro_offset.py`, 2026-07-25) then reduced this to 12,672 segments** —
  that is the current count on disk and in `segmentation_report.csv`. Use 12,672,
  not 13,192, for any scaling estimate.
- **Disk space blocker, resolved by config**: MFA's default temp directory is
  `C:\Users\ASUS\Documents\MFA` on a C: drive with only ~10GB free. Measured temp
  usage is modest — batch2's 287 segments used 132MB, and the pilot's 288
  segments/1.8h used 631MB — so 12,672 segments plausibly needs a few GB, not the
  10-30GB feared. Still pass `--temporary_directory` pointed at D:
  (141GB free) before running Phase 3 training, or it may fail partway
  through a multi-hour run from running out of disk.
- **Phase 3 is not model training** — it's a data-verification step. MFA's
  "acoustic model" here is a disposable Kaldi tool that only produces alignment
  timestamps/confidence scores to validate Phase 2's segmentation guesses; it is
  never used as, or related to, the actual Sunuwar TTS model (that's Phase 6,
  fine-tuning `facebook/mms-tts-mai`). Don't conflate the two when resuming.
- **Phase 3 time/resource expectation**: no GPU needed (Kaldi/CPU). Measured on
  the verification batches: 287 segments → 89 min, 350 segments → 108 min, i.e.
  **~0.3 min per segment**. Extrapolated to 12,672 segments that is **~2.5 days**.
- **`num_jobs` is silently ignored on this corpus — MFA runs single-threaded.**
  Confirmed in the batch2 log: `Number of jobs was specified as 4, but due to
  only having 1 speakers, MFA will only use 1 jobs. Use the --single_speaker
  flag if you would like to split utterances across jobs regardless of their
  speaker.` Every verification batch therefore ran on one core of a 6c/12t CPU.
  **Test `--single_speaker` before committing to the full run** — it is the
  difference between ~2.5 days and plausibly well under a day, and it is not
  yet wired into `src/align.py` or the configs. RAM stays the constraint that
  caps parallelism (7.7GB usable), so try 3-4 jobs, not 12.

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
3. **Align the segments with MFA** — 🔄 verification done (18 chapters, 88.5% —
   see "Phase 3 verification complete" below), **full 260-chapter run still todo**.
   `src/align.py` (train mode) using the Phase 1 dictionary, producing the acoustic
   model + TextGrids. Confirmed better than the pilot's 81%.
4. **QC filtering with explicit thresholds** — 🔄 run on all 18 verification
   chapters (see "Phase 4-5 rerun" below), 690/915 = 75.4% pass; **full-corpus run
   still todo**. Uses MFA's own per-utterance `overall_log_likelihood`,
   `phone_duration_deviation`, `snr` output (this is MFA's built-in output, no
   custom script needed) plus the pre-alignment duration-outlier flag from step 2,
   to drop bad segments before they become training data.
5. **Build `data/processed/tts_dataset/metadata.csv`** from QC-passed segments —
   🔄 690 clips/1.37h built on the 18-chapter sample; **full-corpus run still
   todo**. Hold out whole chapters (not random segments) for validation to avoid
   near-duplicate scriptural phrasing leaking train→val.
6. **Fine-tune from `facebook/mms-tts-mai`** (Maithili — `-nep` does not exist,
   see "Key facts established"), not from scratch — cross-lingual transfer
   converges faster and sounds more natural on limited single-speaker data than
   training a VITS model cold. ZWJ remapping turned out NOT to be needed; only the
   danda is missing from the `mai` vocab.
7. **Evaluate honestly** — MFA self-alignment-confidence of resynthesized audio as
   a cheap automatic signal; real naturalness/intelligibility judgment needs an
   actual Sunuwar speaker (community review), not just Whisper-WER.
8. **Phoneme coverage report** (from phase 1's real phoneme inventory) — diagnostic
   only, since we can't record more to patch gaps; document weak phonemes as a
   known limitation, and as a scoped future-work item (community-recorded
   supplementary set targeting specifically those gaps).

**Transcript bug found and fixed 2026-07-27**: `clean_transcripts.py`'s
cross-reference regex omitted ZWJ from its book-name character class, so refs
like "१ कोरिन्‍थी १:१७" were only partly consumed, leaving unspoken fragments
("१कोरिन्‍", bare verse numerals) in the transcripts — 2,217 tokens across all
260 chapters, and **none of them exist in `mfa_dict.dict`**, so every one was
an OOV at alignment time. Likely a contributor to the 9–11% alignment failures.
Fixed (ZWJ added, book-name run bounded to ≤3 words, danda excluded so a match
can't cross a sentence boundary) and `mfa_input/` regenerated: 181,993 →
180,255 tokens, 0 digit tokens. **`mfa_corpus_segments/` still holds the old
dirty text** — re-run `segment_from_pauses.py` before the full Phase 3 run.
Phase 5 strips the remnants as a safety net (`strip_unspoken_numerals`).

### Phase 3 verification complete (2026-07-27) — 18 chapters, 88.5%

All four verification batches are done. **This is the sample, not the corpus:
18 of 260 chapters (6.9%), 915 of 12,672 segments (7.2%), 2.06h of 29.5h.**

| batch | chapters | segments | aligned | rate | wall-clock |
|-------|----------|----------|---------|------|------------|
| batch0_1CO_pilot | 3 (1CO 1-3) | 113 | 101 | 89.4% | – |
| batch1 | 6 (MAT 1, MAT 10, MRK 1, LUK 5, JHN 3, ACT 2) | 350 | 319 | 91.1% | 1h48m |
| batch2 | 6 (ROM 8, GAL 1, EPH 4, PHP 2, HEB 11, JAS 1) | 287 | 243 | 84.7% | 1h29m |
| batch3 | 3 (1PE 2, REV 1, REV 21) | 165 | 147 | 89.1% | 25m |
| **total** | **18** | **915** | **810** | **88.5%** | ~4h |

**88.5% vs the old Mark pilot's 81%**, sustained across 12 different books —
the intro-offset fix generalises and the ranked-silence segmentation is sound.

Outputs per batch: `data/processed/audio/phase3_verification/<batch>/textgrids/`
(the real product — 810 TextGrids, **gitignored**, licensed content),
`models/phase3_verification/<batch>.zip` (disposable Kaldi model), and
`D:/mfa_temp_batch*/mfa_corpus/sat_3_ali/alignment_analysis.csv` (Phase 4's
input — **the only copy, outside the repo, do not clean these temp dirs**).

**Findings from batch2 — three hypotheses tested, two refuted:**
- **Chapter length does NOT degrade alignment.** HEB_011 was the largest chapter
  in the whole sample (88 segments) and hit 90.9%. Refuted.
- **OOV does NOT explain per-chapter variance.** ROM_008 has the *lowest* OOV
  rate of its batch (7.02%) and the *worst* alignment (71.4%); OOV is otherwise
  flat at 7-9.4% everywhere. Refuted.
- **The Phase 2 duration-outlier flag DOES predict failure**: low-confidence
  segments aligned at **23.5% (4/17)** vs **88.5% (239/270)** for high-confidence.
  batch2's two outlier chapters are explained by this — ROM_008 carries 7 flagged
  segments (14% of the chapter, vs 1-2 in healthy chapters) and a 12.0s mean
  duration vs ~8s elsewhere, the signature of the ranked-pause heuristic merging
  sentences. Caveat: this does *not* argue for flipping
  `drop_low_confidence_segments` to true — `alignment_failed` already drops the 13
  that failed, so flipping it would remove only the 4 that succeeded. Its value is
  as a *pre-alignment* predictor of which chapters need re-segmentation.
- **MFA's text normaliser strips the danda itself.** A naive dict lookup makes it
  look like ~77% of OOV tokens are just words with `।` attached, but
  `normalize_oov.log` shows MFA never saw those as OOV. The only real OOVs are the
  transcript-bug remnants below (~37 types). Don't "fix" the danda.

Currently on: **Phase 3 verification done; full 260-chapter Phase 3 NOT started.**
Next concrete steps to resume, in order:
1. Re-run `src/segment_from_pauses.py` — `mfa_corpus_segments/` still holds the
   pre-fix transcripts (see transcript bug above). **This invalidates the 810
   verification TextGrids**, so the full run supersedes rather than extends them.
2. Test `--single_speaker` on one chapter (see throughput note above) before
   committing ~2.5 days of single-threaded compute.
3. Fix `configs/align_train.yaml`: `corpus_dir` still points at the old
   `mfa_corpus_full` (whole chapters — the one that failed 0/16). Repoint it at
   `data/processed/audio/mfa_corpus_segments`.
4. Keep `temporary_directory` on D: (already supported in `src/align.py` and used
   by every `align_batch*.yaml`) — **do not skip this**, C: has ~10GB free.
5. Run via the local `aligner` conda env — not Colab, MFA already works locally.
6. Then re-run Phase 4-5 on the full output.

### Phase 4-5 rerun on all 18 chapters (2026-07-27)

`batch2` added to `configs/qc_filter.yaml` and QC re-run with thresholds held
**fixed** from the batch0/1/3 fit — batch2 was treated as a genuine held-out
test, not re-tuned on. They generalised: batch2 passed at 76.0%, in line with
the other three batches (70.6-84.1%), and ROM_008/EPH_004 (the two chapters
flagged for segmentation damage) were not standout outliers post-QC —
comparable to MAT_010/MAT_001, which have no known segmentation issue. QC is
catching the bad segments as intended; thresholds did not need retuning.

Overall: **690/915 pass (75.4%), 1.34h kept of 2.06h.**

`configs/tts_dataset.yaml`'s `val_chapters` gained `GAL_001` (clean, 83.9% QC
pass) alongside the existing `LUK_005`/`REV_021`, so validation now covers one
clean chapter per register (epistle/Gospel/apocalyptic) rather than lacking
epistles entirely. ROM_008/EPH_004 were deliberately excluded from validation
— their segmentation damage would make the val metric noisy, not
representative; they stay in training where QC filters their bad segments.

Rebuilt `data/processed/tts_dataset/` (gitignored, licensed content): **690
clips / 1.37h**, split 554 train (1.12h) / 136 validation (0.26h) — up from
the previous 472-clip/0.88h prototype. Still well short of what full Phase 3
will produce; **not** yet suitable to replace the running Colab prototype
fine-tune. Its value here was validating that the QC thresholds hold on a
wider, more varied sample before the full 260-chapter run.

Do not skip ahead to Phase 6 (TTS fine-tuning) before Phases 3-5 are done on
the full corpus.

### Phase 6 environment (validated 2026-07-27)

The Colab fine-tune **works**, but only after nine environment fixes, all now
baked into `notebooks/finetune_tts_colab.ipynb` — run that notebook rather than
improvising, and do not "simplify" its patch cells away. The non-obvious ones:

- **Pin `transformers==4.44.2`** (+ `huggingface_hub==0.24.6`,
  `datasets==2.21.0`, `accelerate==0.34.2`, `tokenizers==0.19.1`, `numpy<2`).
  Downgrading *over* an existing 5.x install leaves a mixed package — always
  start from a fresh runtime.
- **Remap `weight_g`/`weight_v` → `parametrizations.weight.original0/1`**
  before conversion. torch ≥2.1 renamed these, and `from_pretrained` silently
  random-initialises the flow + posterior-encoder WaveNet stacks instead of
  erroring. This is the fix most likely to be dropped by accident and the one
  that most damages quality if it is.
- `save_file` needs `metadata={"format": "pt"}`.
- Pass discriminator/generator paths separately to the conversion script;
  `--language_code` re-derives the generator and undoes the remap.
- `data_dir` is not a trainer field — the local path goes in `dataset_name`.
- Two upstream bugs in `run_vits_finetuning.py` need patching: the
  unconditional `batch["speaker_id"]` read (single-speaker path) and
  `utils/plot.py`'s use of matplotlib's removed `tostring_rgb()`.

Verify patches in a **fresh interpreter** (`!python -c ...`) — modules already
imported in the notebook are cached and will report a patch as absent.

Throughput: **1.32 s/it on a Colab T4**. Plan the full-corpus run in step
counts, not epochs (~20x the clips).

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

## Downstream genre-classification results (measured, 2026-08-08)

Genre-classification F1 (macro), all 5 NLP models, on the same corpus-position
genre-label heuristic (`assign_genre_labels`: first 60% of a file = Narrative,
next 35% = Epistles, last 5% = Apocalyptic — a proxy from corpus structure,
**not real annotated genre metadata**). Static embeddings mean-pool word
vectors (`src/evaluate_nlp.py`); BERT pools its `[CLS]` token, the causal LM
(SunuwarCLM-small, built as a matched-parameter comparison model against
SunuwarBERT-small — see `src/train_clm.py`) pools its last non-padding token
(`src/evaluate_transformer_nlp.py`). Both scripts' `LogisticRegression` use
`class_weight="balanced"` — **required**, not optional: with an unweighted
classifier, the Apocalyptic class (~5% of every split) scored a flat 0.0 F1
across *all five* models, indistinguishable from "no signal for this class."
Reweighting revealed real, non-zero signal for Apocalyptic everywhere, so
this setting isn't cosmetic — an unweighted eval on this label distribution
will silently hide a third of the classification problem.

| model | genre_f1_macro | Narrative | Epistles | Apocalyptic |
|---|---|---|---|---|
| fasttext | 0.2520 | 0.3760 | 0.2902 | 0.0899 |
| word2vec_cbow | 0.2621 | 0.3763 | 0.3343 | 0.0758 |
| word2vec_sg | 0.2685 | 0.4121 | 0.3070 | 0.0864 |
| SunuwarBERT-small | 0.2869 | 0.4134 | 0.3473 | 0.1002 |
| SunuwarCLM-small | 0.2884 | 0.4302 | 0.3390 | 0.0959 |

**Reading this honestly:**
- Both transformers beat every static embedding on **every individual genre
  class**, not just in the aggregate macro number — a modest but consistent
  edge (~0.018–0.02 macro F1 over the best embedding, word2vec_sg).
- The word2vec-vs-fastText target above (fastText expected higher) does
  **not** hold here — fastText is the *weakest* of the three embeddings on
  this task (0.2520) despite having much better OOV coverage (19.9% vs
  30.9%). OOV coverage and downstream genre signal are apparently not the
  same thing on this task.
- SunuwarBERT-small vs. SunuwarCLM-small is a **wash, not a finding**: BERT
  edges CLM on Epistles and Apocalyptic, CLM edges BERT on Narrative, and the
  macro gap (0.2869 vs 0.2884) is well within what a single train/test split
  would produce from noise alone. Don't read a pretraining-objective
  conclusion into this without cross-validation.
- Perplexity is **not** a fair comparison between the two: SunuwarBERT-small
  scores far lower val perplexity (14.79) than SunuwarCLM-small (36.88), but
  MLM perplexity (bidirectional context, only 15% of tokens scored) and
  causal perplexity (left-context-only, every token scored) are on different
  scales by construction — this gap is expected and doesn't mean BERT is the
  better language model. The genre-classification table above is the fair,
  apples-to-apples comparison between the two.

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
| Week 5d | Align segments with MFA (roadmap Phase 3) | 🔄 Verification complete — all 4 batches, 18 chapters, 810/915 = 88.5%; **full 260-chapter run still todo** (~2.5 days single-threaded, test `--single_speaker` first), run locally via `aligner` conda env, not Colab |
| Week 5e | QC filtering + build tts_dataset/metadata.csv (roadmap Phase 4–5) | 🔄 Re-run on all 18 verification chapters (batch2 added) — 690/915 pass QC (75.4%), 1.37h dataset, thresholds held fixed and generalised without retuning. Must be re-run again after the full Phase 3 |
| Week 6 | MMS TTS fine-tuning from mms-tts-**mai** (roadmap Phase 6) | 🔄 Running — prototype fine-tune launched 2026-07-27 on the 0.88h dataset, 9,000 steps / ~3h18m on a Colab T4. Pipeline validated end to end |
| Week 6 | SunuwarCLM-small — causal LM comparison model (`src/train_clm.py`) | ✅ Done — matched-parameter (14.72M vs. BERT's 14.49M) decoder-only model, Post-LN to match BERT deliberately (controlled comparison). Trained 2026-08-07, early-stopped epoch 15, best val_perplexity 36.88 at epoch 10 (`results/clm_eval.json`) — not directly comparable to BERT's 14.79, see note below |
| Week 6–7 | Downstream genre-classification eval, all 5 NLP models (`src/evaluate_nlp.py`, `src/evaluate_transformer_nlp.py`) | ✅ Done (2026-08-08) — see "Downstream genre-classification results" section below for the full table and honest caveats. Both transformers beat all embeddings on every class; BERT-vs-CLM is a wash |
| Week 7–8 | Evaluation (roadmap Phase 7–8) + report + release | 🔄 Eval data built (2026-08-07) — `results/eval_similarity.csv` (90 pairs) and `results/eval_analogy.txt` (30 quadruples) both finalized, corpus-mined + IDF-weighted, with `*_methodology.md` docs for each. Actually scoring word2vec/fastText/SunuwarBERT-small against these files is still 🔲 Todo |

Update this table as tasks complete. See "TTS roadmap" section above for the detailed
Week 5–6 plan and why the original Mark-pilot alignment is being redone.

---

## When starting a new Claude Code session

Always read this file first. Then read the relevant source file if editing existing code.
Never assume state from a previous session — check `data/processed/` for what exists.
