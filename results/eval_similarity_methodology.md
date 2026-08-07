# eval_similarity.csv — Methodology

## Purpose
Word-pair similarity scores for evaluating Sunuwar word embeddings
(word2vec, fastText) and SunuwarBERT-small, by comparing the model's
cosine similarity for each pair against the scores in this file
(Spearman/Pearson correlation).

## Source data (internal use only — not redistributed)
Candidate word pairs and their gloss-overlap scores were derived from
`sunuwar_resources/lexicons/nepali-sunwar.sil-apk.dict.jsonl`, a
Sunuwar↔Nepali/English dictionary extracted from the SIL International
Dictionary App. This lexicon is © SIL International and marked private
research use only — **not for public redistribution**. It was used
only to compute scores and select word pairs; no lexicon content
(Nepali or English glosses) appears in the final public file. The
lexicon itself lives outside version control (gitignored) and stays on
the local machine.

## Pipeline

1. **Noise filtering.** Dictionary entries were dropped if their
   English gloss list was empty, if the headword (after stripping the
   leading tone/glottal marker used in this lexicon) was a single
   character, or if the headword contained digits.

2. **Corpus attestation.** A headword only becomes eligible for a
   similarity pair if it's actually attested in our own cleaned
   Sunuwar corpus (`data/raw/sunuwar_nt_raw.txt`), not just present in
   the dictionary — otherwise we'd be scoring words the embeddings
   never saw. A headword counts as attested if it is at least 3
   characters long and is a prefix of some whitespace-tokenized corpus
   word (this catches inflected corpus forms built on a dictionary
   citation form, not just exact matches). This is a deliberately loose
   criterion: even so, only 272 of 3,804 filtered dictionary entries
   (~7%) turned out to be corpus-attested, reflecting how little
   direct overlap exists between this general-purpose dictionary and
   the New Testament corpus's vocabulary and phrasing.

3. **Similarity scoring — IDF-weighted gloss overlap.** For every pair
   of corpus-attested headwords, each entry's English gloss list is
   treated as a bag of individual gloss words. Each gloss word's IDF
   (inverse document frequency) is computed across the full filtered
   dictionary, so generic words that show up in many unrelated entries
   (e.g. "is," "of," "made") count for less than gloss words that are
   distinctive to a small number of entries. The pair score is the
   ratio of (IDF-weighted intersection of the two gloss-word sets) to
   (IDF-weighted union) — an IDF-weighted Jaccard measure, producing a
   continuous score rather than discrete tiers.

4. **Same-stem/same-lemma exclusion.** Pairs were dropped if word1 and
   word2 share a common character prefix that is at least 60% of the
   shorter word's length. This removes pairs that are really just
   suffix variants of the same lemma (e.g. two inflected forms of one
   verb root), which would otherwise inflate the high-similarity end
   of the scale without testing genuine semantic similarity between
   distinct words. 3 pairs were removed by this automatic rule out of
   95 candidates.

   The prefix rule only catches suffix-appended variants; it misses
   cases where a grammatical marker is *infixed* rather than
   appended, breaking the shared prefix. A manual linguistic read of
   the surviving high-scoring pairs caught two such misses and pulled
   them by hand:
   - `लचा` / `लतीके` — glosses "go / to / mind / with / goal" vs
     "going / of / mind / with / because / goal": the same verb root
     (~"go") in a bare form vs. a nominalized/purposive form, not two
     distinct words.
   - `पुंइतीके` / `पुंइसीतीके` — glosses "for / of / sth / because /
     asking / begging" vs "for / of / os / sth / because / asking":
     same root (~"ask/request"), with an infixed reflexive/detail
     marker (`सी`) breaking the prefix match the automatic rule relies
     on.
   Neither removed pair was the min or max of the filtered set, so the
   min-max rescaling below did not need to be recomputed after this
   manual pass. Final count after both the automatic rule and the
   manual pass: 90 pairs (95 candidates − 3 automatic − 2 manual).

5. **Min-max rescaling.** After the same-stem exclusion, the remaining
   pair scores were rescaled with min-max normalization
   (`(score - min) / (max - min)`) to span the full 0–1 range. This
   was done after exclusion, not before, since dropping the same-stem
   pairs could shift which pairs define the min and max.

## Known limitations
- The raw gloss-overlap score distribution is heavily skewed toward
  zero — most dictionary entries have unique, largely non-overlapping
  multi-word glosses, so genuinely high-overlap pairs are rare. Before
  rescaling, only a handful of pairs (out of a 20,000-pair random
  sample used to characterize the distribution) scored above 0.6.
  Min-max rescaling stretches this thin high end across the full
  range, so scores near 1.0 in the final file represent the *most
  similar pairs found in this dataset*, not an absolute ceiling of
  semantic similarity.
- Because corpus attestation is the binding constraint (only 272
  usable entries), the final set (90 pairs) draws from a fairly small
  pool. Some scores are based on gloss lists with only 2–4 words,
  making individual pair scores noisier than they would be with richer
  glosses.
- Gloss quality itself is heuristic, not gold-standard — the source
  lexicon's own notice describes it as reconstructed from a search
  index rather than canonical dictionary entries, so multi-sense gloss
  lists may mix distinct word senses.
- 15 independently sampled random word pairs were scored as a sanity
  check before finalizing and landed at the low end of the
  distribution (raw scores 0.000–0.063 out of a 0–0.63 raw range),
  supporting that the formula assigns low scores to genuinely
  unrelated words.

## Final file
`results/eval_similarity.csv` — columns `word1,word2,score` only.
90 pairs, scores in [0, 1] after min-max rescaling. No Nepali or
English text, and no other SIL lexicon content, appears in this file.
