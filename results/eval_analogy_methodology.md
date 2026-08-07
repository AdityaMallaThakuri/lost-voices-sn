# eval_analogy.txt — Methodology

## Purpose
Sunuwar word-analogy quadruples (a:b :: c:d) for evaluating word2vec,
fastText, and SunuwarBERT-small embeddings via the standard vector
arithmetic test (does `vec(b) - vec(a) + vec(c) ≈ vec(d)`?).

## Source — no licensing concern
Unlike `eval_similarity.csv`, this file is mined entirely from our own
cleaned corpus (`data/raw/sunuwar_nt_raw.txt`) and its word-frequency
statistics. No SIL lexicon content, no Nepali/English glosses, and no
other third-party material is used anywhere in this pipeline.

## Pipeline

1. **Candidate verb paradigms.** Starting from a feasibility
   investigation that clustered corpus word forms by shared stem
   (stripping 1-5 trailing characters, keeping stems with 3+ surface
   forms and length ≥3 characters), 8 stem groups were identified as
   plausible verb paradigms rather than coincidental substring matches
   between unrelated words: `बाक्`, `दुम्`, `दें`, `शें`, `पुंइ`, `प्र`,
   `सेल्`, `कोव़`.

2. **Paradigm cleanup.**
   - `बाक्`/`बाक्‍`/`बाक`, `दुम्`/`दुम्‍`/`दुम`, and `सेल्`/`सेल्‍` are
     each the same root counted multiple times — the variants differ
     only by whether a Zero-Width Joiner (U+200D) is attached to the
     stem or included/excluded by the strip-length window used during
     candidate discovery. These were merged into one paradigm each.
   - From the `प्र` group, `प्रभु`, `प्रभुमी`, and `प्रभुम` were
     manually excluded — these are inflected/case-marked forms of
     "Lord/master" (a noun/title), not verb inflections, and would
     have contaminated the paradigm's suffix statistics.
   - Final paradigm count: 8, ranging from 55 to 155 surface forms
     each.

3. **Suffix normalization (ZWJ).** Three of the eight paradigms
   (`बाक्`, `दुम्`, `सेल्`) have stems ending in a consonant conjunct
   that requires a ZWJ before any suffix; the other five don't. Raw
   suffix strings for the "same" grammatical suffix therefore differed
   only by a leading ZWJ (e.g. `‍चा` vs `चा`) and would not match
   as equal strings. All suffix comparisons across paradigms are done
   on the ZWJ-stripped ("normalized") suffix; each paradigm's own
   correctly-ZWJ'd surface form is still used when reconstructing the
   actual quadruple words, so every word written to the final file is
   an exact, real corpus-attested token.

4. **Cross-paradigm consistency filter.** A normalized suffix pair
   only qualifies as a candidate analogy dimension if it appears among
   both paradigms' top-8-most-frequent suffixes in **at least 3** of
   the 8 paradigms — this is meant to filter out suffix pairs that are
   common by coincidence in a single paradigm rather than a productive,
   general pattern. 11 suffix pairs passed this bar, appearing in 3-5
   paradigms each (e.g. `(ब, शो)` in 5 paradigms, `(त, माक्‍त)` in 3).
   Two paradigms — `पुंइ` and `प्र` — ended up contributing **zero**
   quadruples: their top-8 suffix sets don't overlap enough with the
   suffix cluster shared by the other six paradigms.

5. **Quadruple construction + frequency floor.** For every qualifying
   suffix pair and every pair of paradigms sharing it, a quadruple
   `root1+sufA, root1+sufB, root2+sufA, root2+sufB` was built and kept
   only if **all four** forms are attested in the corpus with
   frequency ≥5 (avoids rare/noisy forms). This produced 53 candidates.

6. **Diversity-capped trim to ~30.** To avoid the final set being
   dominated by one or two paradigm/suffix combinations, each unique
   (paradigm-pair, suffix-pair) combination was capped at 2 quadruples,
   keeping the 2 with the highest combined corpus frequency (sum of
   all 4 forms' frequencies) when more than 2 existed. This alone
   didn't reduce the set (no combination had more than 2 candidates),
   so the lowest-combined-frequency quadruples were dropped until
   reaching the ~30 target.
   - **Final count: 30 quadruples.**
   - **Distinct paradigm pairs represented: 8 of 8** possible pairs
     among the 6 paradigms that produced any quadruples.
   - **Distinct suffix pairs represented: 10 of the 11** that passed
     the consistency filter.

## Corpus-context plausibility check (honest result: weak/inconclusive)

For the two suffix pairs with the most quadruples in the trimmed set —
**(ब, शो)** with 7 quadruples and **(चा, ब)** with 4 — corpus sentences
containing any word ending in each suffix (not limited to the 8
paradigms) were pulled and scanned for nearby time-adverb-like words
("then," "before," "now," "until," etc.).

- **Suffix `शो` and suffix `चा` samples were clean** — every matched
  word in both samples was a genuine verb form (participle/converb-
  looking forms like माइश्‍शो, कुरशो, दुम्‍शो for `शो`; जरमेचा, लाइक्‍चा,
  प्रोंइचा, साइक्‍चा for `चा`).
- **The suffix `ब` sample was contaminated by proper nouns** — several
  of the top matches (याकूब, अम्‍मीनादाब, राहाब — "Jacob," "Amminadab,"
  "Rahab," from the Matthew 1 genealogy) are Biblical names that
  coincidentally end in `ब` and have nothing to do with the verbal
  suffix `-ब` used in the paradigms. This is the same kind of
  false-positive risk flagged during paradigm cleanup (step 2 above)
  and in the earlier `eval_similarity.csv` work — a short suffix/stem
  string will always risk catching unrelated words by coincidence.
- **Time-word co-occurrence rates:** suffix `चा` sentences had
  time-words in 5/10 samples, `शो` in 4/10, and `ब` in 3/10 (though
  that count is unreliable given the contamination above).
- **Honest verdict:** this is not a clean split. The differences
  between 5/10, 4/10, and 3/10 are small and could easily be sampling
  noise rather than a real tense/aspect distinction between suffixes,
  and the `ब` count specifically can't be trusted due to the
  proper-noun contamination. This check should be read as "nothing
  ruled the suffix pairs out as implausible," not as "confirmed these
  suffixes mark different grammatical categories."

## Known limitations

- **`पुंइ` and `प्र` are entirely unrepresented** in the final file —
  their most productive suffixes don't overlap enough with the other
  six paradigms' shared suffix cluster to pass the cross-paradigm
  consistency filter. A different filter design (e.g. a lower
  frequency-rank cutoff than top-8) might recover them, at the cost of
  including noisier, less-productive suffixes.
- **Infix-marked morphology is not represented at all.** This is a
  suffix-only v1: any grammatical contrast marked by an infix rather
  than a suffix (the confirmed real example from the similarity work,
  `पुंइतीके` vs `पुंइसीतीके`, where `सी` is inserted before the final
  `तीके` rather than appended) is invisible to this pipeline by
  construction. A feasibility check specifically for infix patterns
  found 0 qualifying pairs in a sample of the 50 most frequent word
  forms, which is weak evidence the gap is small — but that check was
  limited to a 50-word sample, not the full corpus, so this shouldn't
  be read as a confident "infixes are rare here."
- **Grammatical category labels are unverified.** Suffix pairs here
  were validated only for cross-paradigm statistical consistency and a
  weak/inconclusive corpus-context plausibility check — not against a
  reference grammar. We do not have verified access to Borchers 2008
  (the primary descriptive grammar of Sunuwar/Kiranti-Kõits), so we
  cannot state with confidence what tense/aspect/person category any
  given suffix (e.g. `त` vs `ब` vs `शो`) actually marks. The quadruples
  are defensible as *distributionally consistent morphological
  contrasts*, not as confirmed instances of a named grammatical
  category.

## Final file
`results/eval_analogy.txt` — 30 lines, `a b c d` space-separated,
Sunuwar text only, one quadruple per line.
