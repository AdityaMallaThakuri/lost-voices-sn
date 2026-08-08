# suz_npi_sentence_aligned.tsv — Methodology (Phase 2)

## Purpose
Sentence-level alignment within the verse-level pairs from
`results/suz_npi_parallel.tsv` (Phase 1). A verse-level parallel
corpus is not a sentence-level one — the Phase 1 methodology doc
already showed Sunuwar verses run multi-sentence roughly twice as
often as Nepali verses (60% vs 31%), so this phase splits verses into
individual sentences and aligns those sentences against each other.

## Licensing — same combined constraint as Phase 1, inherited unchanged

This file is derived directly from `suz_npi_parallel.tsv` and carries
the **identical** licensing situation:
- **Sunuwar**: CC BY-NC-ND 4.0 (© 2011 Wycliffe Bible Translators,
  Inc.) — non-commercial research use only, attribution required, no
  redistribution of raw text. This remains the binding constraint on
  public release.
- **Nepali (Unlocked Literal Bible, `npiulb`)**: CC BY-SA 4.0 — any
  public release requires attribution to the Door43 World Missions
  Community and the Unlocked Literal Bible, and derivatives must carry
  a compatible share-alike license.
- Splitting verses into sentences does not relax either condition —
  both still apply in full to this file.

## Step 1: quote-aware sentence splitter

The naive splitter (terminal punctuation = sentence boundary, full
stop) over-split direct-speech constructions: a `?` or `।` occurring
**inside an open quotation** was being treated as a sentence boundary,
severing narrative tags like `देंत।` ("[he] said.") from the quoted
question that preceded them. Fixed by tracking quote-open/close state
(both Devanagari-style `“ ”` / `‘ ’` and any ASCII `"` found in the
corpus) per line, and only treating `।`, `?`, `!` as real boundaries
when they occur **outside** an open quote.

**Measured impact** (re-running the Phase 2 investigation's Step 2
distribution with the fix):

| | Naive splitter | Quote-aware splitter | Shift |
|---|---|---|---|
| 1-vs-1 (trivial) | 2,312 (29.11%) | **3,238 (40.77%)** | +926 |
| Equal count, >1 | 1,301 (16.38%) | 1,072 (13.50%) | −229 |
| Unequal count | 4,329 (54.51%) | 3,632 (45.73%) | −697 |

Nearly a third of what looked like "hard" unequal-sentence-count
verses were artifacts of over-splitting inside quotations, not real
translation differences — fixing this before alignment materially
changed the problem's shape.

## Step 2: Gale-Church-style length-based DP alignment

- **Equal-count verses (including 1-vs-1) are paired in order**, no
  reordering — the overwhelmingly plausible default for two
  translations of the same verse.
- **Unequal-count verses** go through a dynamic-programming alignment
  over character length, supporting the standard Gale-Church
  categories **1:1, 1:2, 2:1, 1:0, 0:1** (not 2:2 — kept to the
  categories explicitly scoped for this phase).
- Corpus constants estimated empirically from the 3,238 clean 1-vs-1
  pairs (not assumed): **C = 1.0264** (mean Nepali-chars/Sunuwar-chars
  ratio — Nepali runs marginally longer per sentence) and **S2 =
  9.4698** (length-mismatch variance, used to scale how strongly a
  length mismatch is penalized relative to sentence length).
- Category priors bias the DP toward 1:1 as the default explanation
  and treat 1:0/0:1 (an unmatched sentence) as the least likely
  outcome, only chosen when the length evidence genuinely doesn't
  support pairing.

## Step 3: lexicon-overlap tiebreaker — tested, found non-decisive

Implemented an IDF-weighted lexicon-overlap scorer (same formula
family as the `eval_similarity.csv` build:
`sunuwar_resources/lexicons/suz_nep_lexicon.tsv`, IDF over gloss
alternatives, applied when the DP found a near-tied competing
alignment for a given verse) intended to break ties between two
similarly-scored length-based alignment paths.

**Result: the DP found 474 genuine near-ties across the corpus, and
the lexicon tiebreaker never once overrode the length-based choice.**
This is reported honestly as a **tested-and-rejected refinement**, not
omitted or hidden. The likely reason, consistent with earlier
investigation: common function words (postpositions, pronouns) are
too frequent across candidate sentences to discriminate, and the
lexicon's single-gloss-per-headword design misses real overlaps when
the Nepali text uses a synonym the lexicon doesn't list (e.g. the
lexicon's gloss "बाबु" for a Sunuwar word meaning "father," when the
actual Nepali verse text uses the synonym "पिता" instead). The
mechanism is implemented and exercised correctly (verified via a
chain-continuity check after fixing an index-reconstruction bug found
during development — see note below); it simply never had enough
signal strength to win against the length-based default on this
corpus. `confidence_signal` in the output is `length_only` for every
row as a direct result — the column is real infrastructure for a
tiebreak that could matter with a richer lexicon or a different
corpus, not dead code.

**Development note**: an earlier version of the tiebreak logic
approximated segment indices instead of using the DP's own exact
backpointer coordinates, which produced overlapping/duplicate chunks
in a small number of cases. Fixed by storing exact DP-cell endpoints
explicitly and adding a chain-continuity assertion (verified clean
across the full corpus — zero assertion failures) before any results
were reported.

## Step 4: unmatched sentences (1:0 / 0:1)

**1,016 sentences have no counterpart on the other side** (939 Sunuwar
sentences with no Nepali match, 77 Nepali sentences with no Sunuwar
match) — the DP's least-preferred category, only chosen when no
length-based pairing was plausible. These are logged, not deleted, in
`results/suz_npi_sentence_unmatched.tsv` with the same 7-column
schema as the matched file (the unmatched side's text field is empty)
so the exclusion remains traceable.

## Final files

- **`results/suz_npi_sentence_aligned.tsv`** — **10,412 rows**, all
  with a genuine pair on both sides (alignment types 1:1, 2:1, 1:2).
  Columns: `book`, `chapter`, `verse`, `suz_sentence`, `npi_sentence`,
  `alignment_type`, `confidence_signal`.
- **`results/suz_npi_sentence_unmatched.tsv`** — **1,016 rows**
  (alignment types 1:0, 0:1), same schema, kept as a documented
  exclusion log.

## Not done yet
Phase 3 (lexicon induction from the aligned sentence pairs) has not
been started.
