# Lost Voices — Sunuwar Language AI Pipeline

The first publicly reproducible NLP and Text-to-Speech pipeline for **Sunuwar** (ISO 639-3: `suz`), an endangered Kiranti language of eastern Nepal written in Devanagari script.

---

## What this project builds

**Part I — NLP Foundation**
- Cleaned Devanagari Sunuwar text corpus (~220K tokens, New Testament)
- SentencePiece subword tokeniser (8k and 16k vocabulary)
- word2vec and fastText word embeddings
- SunuwarBERT-small — a compact transformer (~15M parameters) pre-trained from scratch using Masked Language Modelling

**Part II — Text-to-Speech**
- MFA-aligned Sunuwar speech dataset (~4–8 hours, ~5,000–8,000 segments)
- Fine-tuned Sunuwar TTS model based on Meta's MMS (Massively Multilingual Speech)

The primary contribution is a **fully documented, reproducible pipeline** — not peak accuracy. Every component is designed to be extended by future researchers and the Sunuwar community.

---

## Why Sunuwar

Sunuwar has approximately 78,910 speakers in Nepal (2021 census) and is classified as **threatened** on the EGIDS scale. Despite a living speaker community, no public NLP resources exist for the language — no corpus, no embeddings, no language model, no speech synthesis system. This project directly addresses that gap.

---

## Repository structure

```
lost-voices-sunuwar/
├── data/
│   ├── raw/              — source text and audio (not redistributed, see licence)
│   ├── processed/        — cleaned corpus, TTS dataset, train/test splits
│   └── eval/             — word similarity and analogy evaluation sets
├── src/
│   ├── preprocess_text.py
│   ├── preprocess_audio.py
│   ├── align.py
│   ├── train_spm.py
│   ├── train_word2vec.py
│   ├── train_fasttext.py
│   ├── train_mlm.py
│   ├── train_tts.py
│   ├── evaluate_nlp.py
│   └── evaluate_tts.py
├── configs/              — YAML config files for each training script
├── models/               — saved checkpoints (see Releases)
├── notebooks/            — exploratory Colab notebooks
├── results/              — evaluation tables and figures
├── CLAUDE.md             — Claude Code instruction file
├── SKILLS.md             — Sunuwar-specific technical reference
└── data/provenance.csv   — data source and licence log
```

---

## Data sources

| Source | Content | Licence |
|--------|---------|---------|
| [eBible.org — suzBl](https://ebible.org/Scriptures/suzBl_usfm.zip) | Sunuwar Bible, USFM text | CC BY-NC-ND 4.0, © 2011 Wycliffe Bible Translators |
| [Faith Comes By Hearing](https://www.bible.is/SUZWBT) | Sunuwar NT audio recordings | FCBH licence, non-commercial |
| Supplementary web text | Community pages, archived documents | Various, documented in provenance.csv |

**Note:** Raw text and audio cannot be redistributed under their licences. Trained models derived from this data are released openly.

---

## Reproducing the pipeline

### Requirements

```
Python >= 3.10
torch >= 2.0
transformers
sentencepiece
gensim
fasttext
montreal-forced-aligner >= 3.0
wandb
ffmpeg
```

Install all Python dependencies:

```bash
pip install torch transformers sentencepiece gensim wandb
pip install fasttext-wheel
```

### Step 1 — Download data

```bash
wget https://ebible.org/Scriptures/suzBl_usfm.zip -P data/raw/
wget https://ebible.org/Scriptures/suzBl_readaloud.zip -P data/raw/
```

Audio must be downloaded separately from [bible.is/SUZWBT](https://www.bible.is/SUZWBT) or via the [BibleBrain API](https://4.dbt.io/api_key/request) (free key required).

### Step 2 — Preprocess text

```bash
python src/preprocess_text.py configs/preprocess.yaml
```

Output: `data/processed/sunuwar_nt_raw.txt` — one sentence per line.

### Step 3 — Train tokeniser and embeddings

```bash
python src/train_spm.py configs/spm.yaml
python src/train_word2vec.py configs/word2vec.yaml
python src/train_fasttext.py configs/fasttext.yaml
```

### Step 4 — Pre-train SunuwarBERT-small

```bash
python src/train_mlm.py configs/mlm.yaml
```

Requires a GPU. Runs on Google Colab T4 (~8–12 hours).

### Step 5 — Audio preprocessing and alignment (TTS track)

```bash
python src/preprocess_audio.py configs/audio.yaml
python src/align.py configs/align.yaml
```

### Step 6 — Fine-tune MMS TTS

```bash
python src/train_tts.py configs/tts.yaml
```

Requires a GPU. Runs on Google Colab T4/A100 (~10,000 steps).

### Step 7 — Evaluate

```bash
python src/evaluate_nlp.py configs/eval_nlp.yaml
python src/evaluate_tts.py configs/eval_tts.yaml
```

---

## Released artefacts

| Artefact | Description |
|----------|-------------|
| `sunuwar_corpus_clean.txt` | Preprocessed Sunuwar NT corpus |
| `sunuwar_spm_8k.model` | SentencePiece tokeniser, 8k vocab |
| `sunuwar_spm_16k.model` | SentencePiece tokeniser, 16k vocab |
| `sunuwar_w2v_sg.model` | word2vec Skip-gram embeddings |
| `sunuwar_w2v_cbow.model` | word2vec CBOW embeddings |
| `sunuwar_fasttext.bin` | fastText model with subword vectors |
| `sunuwar_transformer.pt` | SunuwarBERT-small checkpoint |
| `sunuwar_tts.pt` | Fine-tuned MMS TTS checkpoint |
| `eval_similarity.csv` | Sunuwar word-pair similarity evaluation set |
| TTS dataset | Aligned Sunuwar speech segments (where licence permits) |

Models are hosted on [Hugging Face Hub](#) *(link added on release)*.

---

## Acknowledged limitations

- **Register bias** — the entire corpus is religious text. Models may not generalise to informal or conversational Sunuwar.
- **Corpus size** — ~220K tokens is significantly smaller than high-resource NLP corpora. Representation depth is limited accordingly.
- **TTS quality** — the fine-tuned TTS model is expected to produce partially intelligible speech, not broadcast-quality synthesis. The pipeline's value is its reproducibility.
- **Script** — all resources use Devanagari. The native Sunuwar script (Koĩts Brese / Tikamuli) has only recently received a Unicode block (U+11BC0–U+11BFF) and no digital resources currently use it.
- **No native-speaker evaluation** — word similarity annotation relies on bilingual dictionaries rather than native-speaker judgements.

---

## Extending this work

This pipeline is designed to be extended. Likely next steps:

- Collect studio-recorded Sunuwar speech to improve TTS quality
- Build a Sunuwar ASR system using the aligned dataset as a starting point
- Develop a Nepali–Sunuwar machine translation model using SunuwarBERT-small as the target-side encoder
- Apply the same pipeline to other Kiranti and Himalayan languages: Rai, Limbu, Hayu, Jirel

---

## Academic context

This project is submitted as a final-year Bachelor of Engineering project at **Nepal Engineering College (NEC)**, Kathmandu, Nepal, and as an entry for **Hult Prize Nepal 2026**.

CSR support provided by **SAS Industries, Bhaktapur** (franchise of BIAAREE Industries Limited).

---

## Licence

All code in this repository is released under the **MIT Licence**.

Trained models are released under **CC BY 4.0**.

Raw text and audio data are subject to their original source licences (CC BY-NC-ND 4.0 for eBible text; FCBH terms for audio) and are not included in this repository. See `data/provenance.csv` for full source documentation.

---

## Citation

If you use this work, please cite:

```bibtex
@misc{lostvoices2026,
  title     = {Lost Voices: Monolingual NLP Pipeline and Text-to-Speech
               Fine-Tuning for an Endangered Himalayan Language},
  year      = {2026},
  note      = {Final Year Project, Nepal Engineering College.
               \url{https://github.com/AdityaMallaThakuri/lost-voices-sunuwar}}
}
```

---

## References

Key references from the academic proposal:

- Joshi et al. (2020). *The state and fate of linguistic diversity and inclusion in the NLP world.* ACL.
- Pratap et al. (2023). *Scaling speech technology to 1,000+ languages.* arXiv:2305.13516.
- Kim et al. (2021). *VITS: Conditional variational autoencoder with adversarial learning for end-to-end TTS.* ICML.
- Devlin et al. (2019). *BERT: Pre-training of deep bidirectional transformers.* NAACL-HLT.
- McAuliffe et al. (2017). *Montreal Forced Aligner.* Interspeech.
- Borchers (2008). *A Grammar of Sunwar.* Brill.
