# suz_npi_parallel.tsv — Methodology

## Purpose
A verse-aligned Sunuwar↔Nepali parallel corpus, for future translation-
model work. This is Phase 1 (extraction + alignment) only — no sentence-
level re-splitting or word alignment has been done yet (see "Not done
yet" below).

## Sources and licensing — read before any public release

- **Sunuwar**: `data/raw/suzBl_usfm.zip`, © 2011 Wycliffe Bible
  Translators, Inc., **CC BY-NC-ND 4.0** — non-commercial research use
  only, attribution required, no redistribution of raw text. Same
  restriction that already applies to every other Sunuwar text file in
  this project (per the top-level project licence notes).
- **Nepali**: Unlocked Literal Bible (`npiulb`), Door43 World Missions
  Community, 2019, downloaded from `https://ebible.org/Scriptures/npiulb_usfm.zip`,
  **CC BY-SA 4.0** — share-alike. Any public release of this file (or
  anything derived from it) must carry attribution to the Door43 World
  Missions Community and the Unlocked Literal Bible, and any derivative
  work must be released under CC BY-SA 4.0 or a compatible license.
- **Combined effect**: this file inherits the *stricter* of the two
  constraints on the Sunuwar side (no public redistribution of the raw
  text at all, per the existing project-wide restriction) **and** the
  share-alike/attribution obligation from the Nepali side. Both
  conditions apply simultaneously — this is not a file to publish
  as-is without resolving the Sunuwar licensing question first, and
  even once that's resolved, the Nepali CC BY-SA attribution/share-alike
  terms still need to be honored.

## Extraction method

1. **Book identity comes from each file's own `\id` line, not the
   filename.** eBible.org's numeric filename prefixes are project-
   specific ordering numbers, not a shared standard across sources —
   confirmed during investigation that Sunuwar and Nepali use
   completely different prefixes for the same book (e.g. Sunuwar's
   Matthew is `46-MATsuzBl.usfm`, Nepali's is `70-MATnpiulb.usfm`). The
   3-letter code inside `\id MAT ...` matches on both sides and is the
   only reliable join-key component.

2. **Verse boundaries**: `\c N` sets the current chapter; each `\v N`
   (or `\v N-M` bridge marker) starts a new verse unit whose text
   accumulates across any wrapped lines until the next `\v`, `\c`, or
   end of file.

3. **Bridge handling**: when a `\v N-M` marker appears (only on the
   Nepali side, never observed on the Sunuwar side), its single text
   blob is recorded against **every** verse number from N through M,
   so it can still join against however many separate verse IDs exist
   on the other side for that range. **Exactly 2 bridges found across
   the full NT, both in Acts**: `1:21-22` and `8:37-38`. No others
   anywhere in the other 26 books.

4. **Noise stripped, Sunuwar side**: `\em...\em*` blocks removed
   entirely (cross-references/footnotes, confirmed not to be Sunuwar
   running text — same rule `preprocess_text.py` already uses),
   `\bd`/`\bd*` markers stripped but inner text kept, and full-line
   markers (`\r`, `\s1`, `\ip`, `\io1`, `\io2`, `\id`, `\h`, `\toc1-3`,
   `\mt1`, `\m`, `\p`, `\c`) dropped entirely — none of these carry
   verse-attributable text.

5. **Noise stripped, Nepali side**: only the generic structural markers
   (`\id`, `\ide`, `\h`, `\toc1-3`, `\mt1`, `\c`, `\p`) — confirmed
   across all 27 books (not just the 7-book investigation sample) that
   npiulb carries **no** footnote (`\f`), cross-reference (`\x`),
   poetry (`\q`), or other markers requiring special handling anywhere
   in the NT.

## Join and coverage

Joined on `(book_code, chapter, verse)`. **7,959 of 7,959 verses
matched on both sides — zero verses present on only one side**,
across all 27 NT books.

## Quality filter: length-ratio sanity check

For every joined pair, computed `len(suz_text) / len(npi_text)`.
Distribution: min 0.158, p5 0.664, p25 0.891, **median 1.087**, p75
1.333, p95 1.812, max 4.341 — centered close to 1.0, as expected for
genuinely parallel text.

**Threshold: pairs with ratio < 0.3 or > 3.0 are excluded** as likely
verse-boundary content mismatches rather than extraction errors. 17 of
7,959 pairs (0.21%) were excluded on this basis.

**Why these 17 are excluded, not fixed**: manually inspected several
of them (e.g. 1 John 5:7, Mark 3:14, Mark 5:24, Luke 24:39) and
confirmed each is a genuine **translation-tradition difference in
where a sentence's content is assigned to a verse number**, not a
parsing bug — e.g. Sunuwar's 1 John 5:7 contains the complete "the
Spirit testifies... the Spirit is truth" thought while Nepali's 5:7 is
only the introductory clause "there are three that testify," with the
list of three deferred to verse 8. Both verses exist and are correctly
extracted on both sides; the two translations simply split the
underlying sentence across the verse boundary differently. This isn't
something a smarter joiner could fix — the verse-ID granularity itself
doesn't line up with the sentence granularity at these 17 spots, so
they're logged and excluded rather than force-realigned.

## Final files

- **`results/suz_npi_parallel.tsv`** — 7,942 pairs that passed the
  ratio check. Columns: `book`, `chapter`, `verse`, `suz_text`,
  `npi_text`.
- **`results/suz_npi_parallel_excluded.tsv`** — the 17 excluded pairs,
  same 5 columns, kept as a documented exclusion log (not deleted) so
  the exclusion is traceable rather than silent.

## Known scale issue: verse-level ≠ sentence-level

A rough scale check (counting Devanagari danda `।`/`॥` per verse,
since that's the sentence-boundary convention in both languages per
CLAUDE.md) shows the two sides pack sentences into verses at
noticeably different rates:

| | Sunuwar | Nepali |
|---|---|---|
| 0 danda (no terminal punctuation) | 3.82% | 9.78% |
| exactly 1 danda (single sentence) | 35.76% | 59.49% |
| 2+ danda (likely multi-sentence) | **60.43%** | 30.72% |

**Sunuwar verses are multi-sentence roughly twice as often as Nepali
verses are** (60% vs 31%). Only 27.37% of verses look multi-sentence
on *both* sides simultaneously; 63.77% look multi-sentence on *at
least one* side. This means the current file is a genuine **verse-
level**, not sentence-level, parallel corpus — a meaningful fraction of
"pairs" actually align a multi-sentence Sunuwar span against a
shorter Nepali span or vice versa. This is a real limitation to keep
in mind for any downstream use (e.g. sentence-level MT training would
want these split first), and is exactly why sentence-level re-splitting
was deliberately **not** attempted in this phase — the risk of naive
per-danda splitting introducing new misalignments (given the count
mismatch above) is high enough to warrant its own dedicated,
separately-reviewed phase.

## Not done yet
- Sentence-level re-splitting within verses (Phase 2, not started).
- Word-level alignment (Phase 3, not started).
- Public-release licensing decision (the Sunuwar CC BY-NC-ND
  restriction still applies in full).
