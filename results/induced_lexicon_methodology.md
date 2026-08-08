# induced_suz_npi_lexicon.tsv — Methodology (Phase 3)

## Pipeline

- **eflomal** (Cython/C, MCMC-based word aligner), run on the
  10,412-pair sentence-aligned corpus (`results/suz_npi_sentence_aligned.tsv`)
  from Phase 2.
- **Punctuation-separated tokenization**: danda (`।॥`), comma, quotes
  (`" " ' '` and ASCII `"`), `?`/`!`/etc. spaced out as their own
  tokens rather than glued to the adjacent word or stripped entirely —
  consistent with this project's existing tokenization convention.
  ZWJ (U+200D) is left untouched, staying glued inside words.
- **Raw output**: 38,964 unique word pairs, 125,919 total aligned
  token-pair occurrences, extracted by mapping eflomal's alignment
  indices back to the tokenized source/target lines and counting
  co-occurrence frequency.
- **Runtime**: 31.2 seconds on Colab's default environment — no
  custom build toolchain needed, `gcc`/`make` were already present
  (this had failed to install entirely on the local Windows machine,
  which has neither).

## Filtering

- **Punctuation-to-punctuation pairs removed**: 738 (38,226 remain).
- **Frequency distribution**: a sharp cliff between `freq=1` (71.69%
  of pairs) and `freq=2` (13.02%) — consistent with chance
  co-occurrence noise concentrating almost entirely at `freq=1`.
- **Threshold selected: ≥3** (5,846 pairs, 15.29% of the
  punctuation-filtered set), chosen as a middle ground between
  removing chance noise (≥2 is the minimum defensible cutoff, given
  where the cliff sits) and retaining a workable candidate set.
  Reported for transparency even though ≥3 was the one selected:

  | Threshold | Pairs survive | % of punctuation-filtered set |
  |---|---|---|
  | ≥2 | 10,822 | 28.31% |
  | ≥3 | 5,846 | 15.29% |
  | ≥5 | 2,991 | 7.82% |

## Validation against SIL lexicon (measured)

- At threshold ≥3: **2,376 of 5,846 pairs** have their Sunuwar word
  present in `suz_nep_lexicon.tsv`; of those, **415 agree** with SIL's
  listed gloss(es) — **measured precision 17.47%**.
- **Coverage**: 256 of 8,606 SIL headwords (**2.97%**) appear in the
  induced set at all.
- **Coverage is primarily a domain-mismatch artifact, not evidence of
  induction failure.** SIL's lexicon is a general-vocabulary
  dictionary; the induction source is exclusively NT scripture text.
  Most of SIL's everyday vocabulary (household items, nature terms,
  etc.) simply never occurs in this corpus, regardless of alignment
  quality.

## Manual precision spot-check (the key corrective finding)

- **40 pairs sampled** (20 `disagree`, 20 `not_in_SIL`), each read in
  its real corpus sentence context by the project lead. **Important
  limitation, stated explicitly**: the project lead is
  English/Nepali-literate, not a Sunuwar speaker — judgment was made
  from Nepali translation plausibility and sentence context, not
  independent Sunuwar fluency.
- **Result: approximately 26/40 (65%) judged genuinely correct**,
  with several more borderline/related, versus the **17.47% measured
  precision** at the same threshold — **roughly a 4x gap** between
  measured and human-judged precision.
- **Primary cause of the gap**: SIL's lexicon lists only one or two
  glosses per headword, not exhaustive synonym sets. Many induced
  translations that are actually correct get scored as "disagree"
  simply because the correct synonym wasn't SIL's chosen gloss (e.g.
  `मिनु`→`र` scored as disagreement against SIL's `अनि` — both
  genuinely mean "and").
- **Secondary, distinct failure mode, named explicitly**:
  high-frequency generic Sunuwar auxiliaries/copulas produce
  reproducible collocation artifacts — the aligner locks onto a
  frequently-adjacent content word rather than the auxiliary's true,
  near-meaningless grammatical function. Confirmed in two independent
  sampled instances of `बाक्‍त` aligning to two unrelated Nepali verbs
  (`बताइदिए` "told," `बनाए` "made"), and previously in the
  high-frequency review (`येसुमी`→`भन्‍नुभयो`, "Jesus"→"said," driven
  by the fixed Biblical-narrative collocation "Jesus said..." rather
  than true lexical correspondence). This is a **real error category,
  not a measurement artifact**, and is the most likely source of
  genuine remaining error in the induced lexicon.

## Limitations

- **Corpus size** (10,412 sentence pairs) is small by typical
  statistical word-alignment standards (usually 100K+ sentence
  pairs). No established minimum-viable-size guidance was found in
  available eflomal/fast_align documentation, so results should be
  read as "workable but noisy" rather than benchmarked against a
  known-good scale.
- **The precision estimate is based on a 40-pair human-judged
  sample, not full manual annotation** — a rough corrective factor
  (~4x), not an exact figure.
- **Judgment was made without independent Sunuwar-side verification.**
  A genuine Sunuwar mistranslation that happens to produce
  Nepali-plausible output in its sentence context would not be caught
  by this method — the spot-check can confirm plausibility, not
  ground-truth correctness, without a Sunuwar-fluent reviewer.

## Final files

- **`results/induced_suz_npi_lexicon.tsv`** — 5,846 rows (threshold
  ≥3). Columns: `suz_word`, `npi_word`, `frequency`,
  `sil_match_status` (`agree` / `disagree` / `not_in_SIL`).
- **`results/induced_lexicon_methodology.md`** — this file.

Scratch files (`scratchpad/induced_lexicon_filtered_DRAFT.tsv`,
`scratchpad/precision_spotcheck_sample.tsv`) remain in the scratch
directory as working artifacts, superseded by the files above as the
Phase 3 deliverable.
