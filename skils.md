# Lost Voices — Sunuwar NLP Skills Reference

This file documents hard-won knowledge about the Sunuwar data and language.
Claude Code should read this alongside CLAUDE.md when working on any text
processing, tokenisation, or model training task.

---

## Skill 1: Parsing Sunuwar USFM files correctly

### The one tricky rule
`\em…\em*` blocks appear **inside verse lines** and must be fully removed.
They contain two things — neither is Sunuwar running text:

1. **Cross-references** (most common): `मत्ती ३:१` `यसैया ४०:३` — Nepali book names
   + Devanagari numerals + colon/semicolon separators. These look like Sunuwar but
   they're citation strings, not sentences.
2. **Translator footnotes** (rare, ~57 in NT): short Nepali-language explanations
   the translator added. Example: `सास्त्री यहूदी आन कली परमप्रभु यावे आ अरेशो लोव़ शेंब बाक्शो बाक्तेक्म।`

**Both must be stripped.** Keeping them introduces Nepali tokens and citation
noise into a Sunuwar corpus.

### Correct stripping regex
```python
import re
# Strip em blocks first, before processing verse text
line = re.sub(r'\\em.*?\\em\*', '', line, flags=re.DOTALL)
```

### The `\bd` marker keeps its text
`\bd bold text\bd*` — strip the markers but keep the enclosed text.
```python
line = re.sub(r'\\bd\*?', '', line)
```

### Lines to discard entirely (no text to keep)
```python
DISCARD_LINE_PREFIXES = (
    r'\id', r'\h', r'\toc1', r'\toc2', r'\toc3', r'\mt1',
    r'\c ', r'\s1', r'\r ', r'\ip', r'\io1', r'\io2'
)
```
Any line whose stripped content starts with one of these → skip the entire line.

### Lines to keep text from
- `\v N text` → strip `\v N `, keep the rest as the verse text
- `\p`, `\m` → strip marker, if any text follows keep it (rare but possible)

---

## Skill 2: ZWJ handling in Sunuwar Devanagari

The NT corpus contains **54,692** Zero Width Joiner characters (U+200D).
They appear *inside words* as part of Devanagari conjunct character sequences.

Example (ZWJ shown as `|`):
```
ब्|ZWJ|वाक्|ZWJ|कुम  →  ब्‍वाक्‍कुम  (one word, renders as conjunct)
```

### Rules
- **Never strip ZWJ** from text at any stage of the pipeline
- **Treat ZWJ as part of the word** for tokenisation purposes
- When computing "percentage Devanagari characters", count ZWJ as valid
  (i.e. include it with U+0900–U+097F, not as foreign character)
- SentencePiece will handle ZWJ correctly if trained on the raw text — do not
  pre-process it away before tokeniser training

### Character classes for Sunuwar text
```python
def is_sunuwar_char(c):
    cp = ord(c)
    return (
        0x0900 <= cp <= 0x097F  # Devanagari block
        or cp == 0x200D          # ZWJ (conjunct marker)
        or cp == 0x0964          # Danda (sentence boundary)
        or cp == 0x0965          # Double danda
        or c in ' \t\n'          # Whitespace
        or c in '""\'\'(),।॥'   # Common punctuation in text
    )
```

---

## Skill 3: Sentence segmentation for Sunuwar

### Primary boundary markers
- `।` (U+0964, Devanagari Danda) — primary sentence-final punctuation
- `॥` (U+0965, Double Danda) — used at end of chapters/sections

### What NOT to use as sentence boundaries
- `,` — comma is common within Sunuwar sentences, not a boundary
- `.` — rarely appears; when it does it's usually inside numbers (e.g. `ए. डी.` = A.D.)
- `?` — question mark appears in some verses but splitting on it loses context

### Correct segmentation
```python
import re

def segment_sentences(text):
    # Split on danda/double-danda, keep the boundary marker attached to previous segment
    segments = re.split(r'(?<=[।॥])', text)
    result = []
    for seg in segments:
        seg = seg.strip()
        if seg and len(seg.split()) >= 3:  # minimum 3 tokens
            result.append(seg)
    return result
```

### Expected output characteristics
- Mean sentence length: ~20–25 tokens
- Some verses produce 1 sentence, others produce 2–3 (when internal dandas present)
- Target total: ~8,000–12,000 sentences from NT alone

---

## Skill 4: Script filtering threshold

After stripping all USFM markers, clean Sunuwar verse text is typically >98%
Devanagari+ZWJ by character count (excluding spaces). The 20% foreign-character
threshold is deliberately lenient to allow:
- Quoted speech with `"` `"` `'` `'`
- Names transliterated from Greek/Hebrew using Devanagari
- Rare Devanagari numerals mixed with verse flow

```python
def passes_script_filter(text, threshold=0.20):
    """Return True if the line is mostly Devanagari."""
    non_space = [c for c in text if c not in ' \t\n']
    if not non_space:
        return False
    foreign = [c for c in non_space
               if not (0x0900 <= ord(c) <= 0x097F)
               and ord(c) != 0x200D
               and ord(c) not in (0x0964, 0x0965)
               and ord(c) > 127]  # ignore ASCII punctuation
    return len(foreign) / len(non_space) <= threshold
```

---

## Skill 5: SentencePiece training for Sunuwar

### Why Sunuwar needs its own tokeniser
Standard multilingual tokenisers (mBERT, XLM-R) allocate very few vocab slots
to Sunuwar. Every Sunuwar word gets fragmented into byte-level or character-level
pieces, degrading representation quality. Training from scratch on the corpus
gives morphologically aware subword units.

### Configuration that works
```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input='data/processed/train.txt',
    model_prefix='models/sunuwar_spm_8k',
    vocab_size=8000,
    model_type='unigram',
    character_coverage=0.9995,  # high coverage for Devanagari
    pad_id=0,
    unk_id=1,
    bos_id=2,   # [CLS]
    eos_id=3,   # [SEP]
    user_defined_symbols=['[MASK]', '[PAD]'],
    byte_fallback=True,         # handles any OOV character via byte encoding
    split_by_whitespace=True,
)
```

### Fertility check (run after training)
Fertility = average number of subword tokens per whitespace-separated word.
Good range for Sunuwar: **2.5–4.0**.
Below 2.5: vocab too large, memorising whole words.
Above 4.0: vocab too small, over-fragmenting.

---

## Skill 6: Evaluating embeddings without standard benchmarks

No Sunuwar word similarity or analogy benchmarks exist. We build our own.

### Word-pair similarity dataset (eval_similarity.csv)
Format: `word1,word2,score`
Score range: 1–5 (1 = unrelated, 5 = near-synonyms)
Source: Use the SIL lexicon glosses — find Sunuwar words with similar English glosses
and pairs known to be unrelated.
Minimum viable: **50 pairs**.

### Genre classification proxy task
The NT corpus has a natural genre split:
- **Gospels + Acts** (narrative): Matthew, Mark, Luke, John, Acts
- **Epistles** (instructional): Romans through Jude
- **Apocalyptic**: Revelation

Train a logistic regression on averaged embeddings. Random baseline = 33%.
Any result above ~50% confirms the embeddings capture register differences.

---

## Skill 7: BERT training on small Sunuwar corpus

### The overfitting risk
At ~220K tokens, the corpus is ~50× smaller than typical BERT pre-training.
Signs of overfitting:
- Train loss continues falling while val perplexity rises or plateaus
- Top-1 predictions on probe sentences stop diversifying after epoch 20

### Mitigation in order of importance
1. **Dynamic masking** — re-sample which tokens to mask each epoch (not fixed)
2. **Dropout 0.1** on all attention + FFN layers
3. **Weight decay 0.01** in AdamW
4. **Early stopping** patience=5 on val perplexity
5. **Small architecture** — 6 layers × 256 hidden is deliberate, not a limitation

### Probe sentences to track training
Pick 3 sentences from test.txt at the start and log top-5 predictions for a
masked token in each sentence at every checkpoint. This gives a qualitative
"is the model learning Sunuwar?" signal alongside the perplexity number.

### Random baseline perplexity
With vocab size 8000, random-guess perplexity = 8000.
Any trained model should reach below 1000 within 20 epochs on this corpus size.
Below 200 is a good result. Below 100 is excellent.

---

## Skill 8: MFA alignment without a pre-trained Sunuwar model

Montreal Forced Aligner has no Sunuwar acoustic model. Use training mode:

```bash
# Step 1: train acoustic model on available Sunuwar audio
mfa train \
  data/processed/audio/mfa_input/ \
  data/processed/audio/lexicon.txt \
  models/sunuwar_acoustic_model

# Step 2: align using trained model  
mfa align \
  data/processed/audio/mfa_input/ \
  data/processed/audio/lexicon.txt \
  models/sunuwar_acoustic_model \
  data/processed/audio/textgrids/
```

### Pilot on Mark first
Mark has 16 chapters, ~678 verses, ~1.5 hours of audio.
If alignment success rate on Mark > 80%, scale to all NT.
If < 80%, check: audio sample rate (must be 16kHz), text encoding (must be UTF-8
Devanagari), lexicon coverage.

### Segment filtering after alignment
Keep segments where:
- Duration: 0.5s ≤ duration ≤ 15s
- Alignment confidence: above MFA's default threshold
- Text: at least 3 tokens

Discard everything else. Report how many segments were discarded and why.

---

## Data provenance (required in every data-touching script)

The Sunuwar Bible text is **CC BY-NC-ND 4.0**.
This means:
- ✅ Use for non-commercial research
- ✅ Train models on it
- ✅ Release trained models publicly
- ❌ Do not redistribute the raw text files publicly
- ❌ Do not create derivative text works from it
- ❌ Do not use commercially

Every script that reads from `data/raw/` must include this comment at the top:
```python
# Data source: Sunuwar Bible (suzBl), © 2011 Wycliffe Bible Translators, Inc.
# Licence: CC BY-NC-ND 4.0 — non-commercial research use only
# Do not redistribute raw text. Models trained on this data may be released.
```
