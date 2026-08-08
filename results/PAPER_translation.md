# A Reproducible Sunuwar-Nepali Parallel Corpus, Induced Bilingual Lexicon, and Baseline Neural Machine Translation System

## Abstract

We present the first Sunuwar-Nepali parallel-text resource pipeline for
Sunuwar (ISO 639-3: `suz`), an endangered Kiranti language of Nepal with no
prior parallel-corpus resources of any kind. Starting from the Sunuwar and
Nepali New Testaments, we build a verse-aligned corpus (7,942 pairs), refine
it to a sentence-aligned corpus (10,412 pairs) via a quote-aware sentence
splitter and a Gale-Church-style dynamic-programming aligner with
empirically-fit corpus constants, induce a bilingual lexicon via MCMC-based
word alignment (eflomal) and validate it against an existing human-curated
SIL lexicon, and train a baseline direction-tagged bidirectional
encoder-decoder neural machine translation (NMT) system. We report every
stage's results honestly rather than optimistically: the induced lexicon's
measured precision against the SIL gold resource (17.47%) undershoots its
human-judged precision (~65%) by roughly 4x, a gap traced to the gold
lexicon's single-gloss-per-headword sparsity rather than induction failure;
and the baseline NMT system, while producing fluent, grammatically
well-formed output in both directions with zero pipeline-level defects,
achieves low translation adequacy (BLEU 2.23-3.97, chrF 21.40-24.92) at the
present ~10K-pair data scale. Consistent with the companion NLP resource
report's framing, this paper prioritizes reproducibility and honest
reporting of what a from-scratch pipeline can and cannot achieve at this
scale over any claim of state-of-the-art accuracy.

## 1. Introduction

Sunuwar has, prior to this work, no parallel-text resources of any kind —
no verse-aligned, sentence-aligned, or word-aligned bitext exists in the
published literature or in any public repository. This is a strictly
harder starting point than most low-resource NLP work, which typically
begins from at least some parallel data (even if small) between the target
language and English.

Nepali, not English, is the practically relevant pairing language for this
work: Sunuwar speakers are overwhelmingly Nepali-bilingual (Nepali is
Nepal's lingua franca and the language of schooling, administration, and
regional commerce in Sunuwar-speaking areas), whereas English bilingualism
among Sunuwar speakers is comparatively rare. A Sunuwar-Nepali resource is
therefore of direct practical use to the community this project serves —
for translation aids, bilingual dictionaries, and language-documentation
tools — in a way a Sunuwar-English resource would not be.

This paper's contributions are:

1. A verse-aligned and, refined further, sentence-aligned Sunuwar-Nepali
   parallel corpus, built from the two languages' independently-translated
   New Testaments and joined on a language-agnostic key (each source
   file's own `\id` book-code line, not filename), with all quality
   filtering and exclusions logged and traceable rather than silent.
2. An induced bilingual lexicon (via MCMC word alignment), validated
   against an existing human-curated SIL lexicon and further corrected via
   manual spot-check — a validation methodology whose findings (see
   Section 7) generalize beyond this specific lexicon.
3. A baseline bidirectional NMT model (shared encoder-decoder, direction-
   tagged) and its honest quantitative (chrF/BLEU) and qualitative
   evaluation, including an evaluation-design finding (per-direction
   scoring, never blended) and a hypothesis-reversing directional
   asymmetry reported as unconfirmed rather than over-interpreted.

## 2. Related Work

Bible text has long served as a practical starting point for massively
multilingual and low-resource parallel-corpus work, since it is one of the
few texts translated into thousands of languages by design
[Christodoulopoulos & Steedman, 2015]. More recent work has built
standardized, machine-learning-ready releases of this data specifically
for low-resource machine translation benchmarking [Daspit et al., 2023,
eBible Corpus]. *(Bibliographic detail flag: the exact venue/identifier
for both of these citations could not be pulled from an existing file in
this repository — they were supplied by name in the task instructions
referencing a "companion NEC report" not present in `results/` at the time
of writing. The entries below are standard-form citations from general
bibliographic knowledge, not copied from a verified source file, and
should be checked against the original companion report before any
external release of this paper.)*

Two large multilingual model families were considered as an alternative to
training a translation model from scratch: Meta's No Language Left Behind
(NLLB) machine translation models, and Meta's Massively Multilingual
Speech (MMS) project, whose text-to-speech checkpoints are already used
elsewhere in this project's TTS track (`facebook/mms-tts-suz`, per project
memory) [Pratap et al., 2023]. Neither was adopted for the NMT work
reported here: the Phase 4 feasibility investigation found no Sunuwar
(`suz`) code and no viable Nepali proxy in the checkpoint families
practically available for fine-tuning at the time, and a from-scratch
joint-vocabulary model (Section 6) was judged the more tractable path
given the project's existing SentencePiece tokenizers. This is recorded
here as a design decision with a documented reason, not an oversight.

## 3. Data Sources and Licensing

- **Sunuwar**: `data/raw/suzBl_usfm.zip`, © 2011 Wycliffe Bible
  Translators, Inc., **CC BY-NC-ND 4.0** — non-commercial research use
  only, attribution required, no redistribution of raw text. This is the
  same restriction that applies to every other Sunuwar text file used
  elsewhere in this project.
- **Nepali**: Unlocked Literal Bible (`npiulb`), Door43 World Missions
  Community, 2019, `https://ebible.org/Scriptures/npiulb_usfm.zip`,
  **CC BY-SA 4.0** — share-alike. Any public release of this file, or
  anything derived from it, must carry attribution to the Door43 World
  Missions Community and the Unlocked Literal Bible, and any derivative
  work must be released under CC BY-SA 4.0 or a compatible license.
- **Combined effect**: every file produced in this pipeline (verse-aligned,
  sentence-aligned, and the induced lexicon) inherits the *stricter* of
  the two constraints on the Sunuwar side (no public redistribution of the
  raw text at all) **and** the share-alike/attribution obligation from the
  Nepali side simultaneously. None of these files are cleared for public
  release as-is; the Sunuwar licensing question would need to be resolved
  first, and the Nepali CC BY-SA terms would still need to be honored even
  then. Trained *models* (e.g. the NMT checkpoint) may be released per this
  project's general licensing stance, distinct from the raw/derived text.

## 4. Corpus Construction

### 4.1 Verse-Level Alignment (Phase 1)

Book identity is derived from each source file's own `\id` line, not its
filename — eBible.org's numeric filename prefixes are project-specific
ordering numbers, not a shared standard across sources (confirmed: Sunuwar
Matthew is `46-MATsuzBl.usfm`, Nepali Matthew is `70-MATnpiulb.usfm`). The
3-letter code inside each `\id` line is the only reliable join-key
component across the two sources.

Verse boundaries follow `\c N` (chapter) and `\v N` (or `\v N-M` bridge)
markers, with wrapped lines accumulated until the next marker. **Exactly 2
bridge markers were found across the full New Testament, both in
Acts: `1:21-22` and `8:37-38`**, both on the Nepali side only (never
observed on the Sunuwar side); each bridge's text is recorded against
every verse number in its range so it can join against however many
separate verse IDs exist on the Sunuwar side.

Noise stripping followed the same rules as the project's existing
`preprocess_text.py` on the Sunuwar side (`\em...\em*` blocks removed
entirely as cross-references/footnotes, not running text; `\bd`/`\bd*`
stripped but inner text kept; full-line structural markers dropped); the
Nepali side required only the generic structural markers (`\id`, `\ide`,
`\h`, `\toc1-3`, `\mt1`, `\c`, `\p`), since npiulb was confirmed (across
all 27 books) to carry no footnote, cross-reference, or poetry markers
anywhere in the NT.

Joining on `(book_code, chapter, verse)` yielded **7,959 of 7,959 verses
matched on both sides — zero verses present on only one side**, across all
27 NT books.

**Quality filter**: for every joined pair, the character-length ratio
`len(suz_text) / len(npi_text)` was computed. The distribution (min 0.158,
p5 0.664, p25 0.891, **median 1.087**, p75 1.333, p95 1.812, max 4.341) is
centered close to 1.0, as expected for genuinely parallel text. Pairs with
ratio < 0.3 or > 3.0 were excluded as likely verse-boundary content
mismatches rather than extraction errors — **17 of 7,959 pairs (0.21%)**
were excluded on this basis. Manual inspection of several (e.g. 1 John
5:7, Mark 3:14, Mark 5:24, Luke 24:39) confirmed each is a genuine
translation-tradition difference in where a sentence's content is assigned
to a verse number, not a parsing bug — both languages' text exists and is
correctly extracted, but the two translation traditions split the
underlying sentence across the verse boundary differently, a mismatch no
smarter joiner could fix. These 17 are logged in
`results/suz_npi_parallel_excluded.tsv`, not deleted.

**Final Phase 1 output**: **7,942 pairs** in `results/suz_npi_parallel.tsv`
(columns: `book`, `chapter`, `verse`, `suz_text`, `npi_text`).

A rough per-verse danda (`।`/`॥`) count showed the two languages pack
sentences into verses at markedly different rates: Sunuwar verses contain
2+ dandas (likely multi-sentence) **60.43%** of the time versus Nepali's
**30.72%** — Sunuwar verses run multi-sentence roughly twice as often.
Only 27.37% of verses look multi-sentence on both sides simultaneously.
This established that the Phase 1 output is a genuine verse-level, not
sentence-level, parallel corpus, and motivated Phase 2.

### 4.2 Sentence-Level Alignment (Phase 2)

**Quote-aware sentence splitter.** An initial naive splitter (treating any
terminal `।`/`?`/`!` as a sentence boundary) over-split direct-speech
constructions, severing narrative tags (e.g. `देंत।`, "[he] said.") from
the quoted question preceding them, whenever that punctuation occurred
inside an open quotation. This was fixed by tracking quote-open/close
state (Devanagari `“ ” ‘ ’` and any ASCII `"` found in the corpus) per
line, treating `।`, `?`, `!` as real sentence boundaries only when they
occur **outside** an open quote. The measured impact of this fix:

| | Naive splitter | Quote-aware splitter | Shift |
|---|---|---|---|
| 1-vs-1 (trivial) | 2,312 (29.11%) | **3,238 (40.77%)** | +926 |
| Equal count, >1 | 1,301 (16.38%) | 1,072 (13.50%) | −229 |
| Unequal count | 4,329 (54.51%) | 3,632 (45.73%) | −697 |

Nearly a third of what appeared to be "hard" unequal-sentence-count verses
were artifacts of over-splitting inside quotations, not real translation
differences — fixing the splitter materially changed the shape of the
alignment problem before any alignment algorithm was applied.

**Gale-Church-style DP alignment.** Equal-count verses (including 1-vs-1)
are paired in order with no reordering. Unequal-count verses go through a
dynamic-programming alignment over character length, supporting the
standard Gale-Church categories **1:1, 1:2, 2:1, 1:0, 0:1** (2:2 was
explicitly out of scope for this phase). Corpus constants were estimated
empirically from the 3,238 clean 1-vs-1 pairs, not assumed: **C = 1.0264**
(mean Nepali-chars/Sunuwar-chars ratio) and **S2 = 9.4698** (length-
mismatch variance). Category priors bias the DP toward 1:1 as the default
explanation and treat 1:0/0:1 as the least-likely outcome.

**Lexicon-overlap tiebreaker — tested and rejected.** An IDF-weighted
lexicon-overlap scorer (same formula family used to build
`eval_similarity.csv`) was implemented to break ties between similarly-
scored competing DP alignment paths. **The DP found 474 genuine near-ties
across the corpus, and the lexicon tiebreaker never once overrode the
length-based choice.** This is reported as a tested-and-rejected
refinement, not omitted or hidden: the likely reason is that common
function words are too frequent across candidate sentences to
discriminate, and the lexicon's single-gloss-per-headword design misses
real overlaps when the Nepali text uses a synonym the lexicon doesn't
list. The mechanism was verified to be exercised correctly (a chain-
continuity check, added after fixing an index-reconstruction bug found
during development, passed with zero failures across the full corpus); it
simply never had enough signal strength to win against the length-based
default on this corpus. `confidence_signal` is `length_only` for every row
as a direct consequence — real infrastructure for a tiebreak that could
matter with a richer lexicon or different corpus, not dead code.

**Unmatched sentences.** **1,016 sentences have no counterpart on the
other side** (939 Sunuwar sentences with no Nepali match, 77 Nepali
sentences with no Sunuwar match) — the DP's least-preferred category,
chosen only when no length-based pairing was plausible. These are logged
in `results/suz_npi_sentence_unmatched.tsv`, not deleted.

**Final Phase 2 output**: **`results/suz_npi_sentence_aligned.tsv`**,
**10,412 rows**, all with a genuine pair on both sides (alignment types
1:1, 2:1, 1:2). Columns: `book`, `chapter`, `verse`, `suz_sentence`,
`npi_sentence`, `alignment_type`, `confidence_signal`.

## 5. Induced Bilingual Lexicon (Phase 3)

**Pipeline**: eflomal (Cython/C, MCMC-based word aligner) run on the
10,412-pair sentence-aligned corpus, using punctuation-separated
tokenization (danda, comma, quotes, `?`/`!`, etc. spaced out as their own
tokens rather than glued to adjacent words or stripped; ZWJ U+200D left
untouched inside words per this project's convention). Runtime: 31.2
seconds on Colab's default environment (eflomal failed to install
entirely on the local Windows machine, which lacks `gcc`/`make`). Raw
output: **38,964 unique word pairs, 125,919 total aligned token-pair
occurrences**.

**Filtering**: 738 punctuation-to-punctuation pairs removed (38,226
remain). The frequency distribution showed a sharp cliff between `freq=1`
(71.69% of pairs) and `freq=2` (13.02%), consistent with chance
co-occurrence noise concentrating at `freq=1`. A threshold of **≥3** was
selected as a middle ground between removing chance noise and retaining a
workable candidate set:

| Threshold | Pairs survive | % of punctuation-filtered set |
|---|---|---|
| ≥2 | 10,822 | 28.31% |
| ≥3 | 5,846 | 15.29% |
| ≥5 | 2,991 | 7.82% |

**Validation against the SIL lexicon** (`sunuwar_resources/lexicons/
suz_nep_lexicon.tsv`): at threshold ≥3, **2,376 of 5,846 pairs** have their
Sunuwar word present in the SIL lexicon; of those, **415 agree** with
SIL's listed gloss(es) — **measured precision 17.47%**. Coverage: **256 of
8,606 SIL headwords (2.97%)** appear in the induced set at all — a
domain-mismatch artifact (SIL is a general-vocabulary dictionary; the
induction source is exclusively NT scripture text), not evidence of
induction failure.

**Manual precision spot-check — the key corrective finding.** 40 pairs
were sampled (20 `disagree`, 20 `not_in_SIL`) and read in their real
corpus sentence context by the project lead (English/Nepali-literate, not
a Sunuwar speaker — judgment was made from Nepali translation
plausibility and sentence context, an explicit limitation, not independent
Sunuwar fluency). **Result: approximately 26/40 (65%) judged genuinely
correct**, versus the **17.47% measured precision** at the same threshold
— **roughly a 4x gap** between measured and human-judged precision. The
primary cause: SIL's lexicon lists only one or two glosses per headword,
not exhaustive synonym sets, so many correct induced translations are
scored "disagree" simply because the correct synonym wasn't SIL's chosen
gloss (e.g. `मिनु`→`र` scored as disagreement against SIL's `अनि`, though
both genuinely mean "and").

A **secondary, distinct failure mode** is named explicitly: high-frequency
generic Sunuwar auxiliaries/copulas produce reproducible collocation
artifacts, where the aligner locks onto a frequently-adjacent content word
rather than the auxiliary's true, near-meaningless grammatical function.
Confirmed in two independent sampled instances of `बाक्‍त` aligning to two
unrelated Nepali verbs (`बताइदिए` "told," `बनाए` "made"), and previously in
`येसुमी`→`भन्‍नुभयो` ("Jesus"→"said"), driven by the fixed Biblical-
narrative collocation "Jesus said..." rather than true lexical
correspondence. This is a real error category, not a measurement artifact,
and the most likely source of genuine remaining error in the induced
lexicon.

**Final Phase 3 output**: **`results/induced_suz_npi_lexicon.tsv`**, 5,846
rows (threshold ≥3), columns `suz_word`, `npi_word`, `frequency`,
`sil_match_status`.

## 6. Baseline Neural Machine Translation (Phase 4)

**Architecture (Config A)**: shared bidirectional encoder-decoder
transformer. hidden_dim=256, num_heads=4, ffn_dim=512, 3 encoder layers, 3
decoder layers, max_seq_len=128, dropout=0.2, GELU activation. **Joint
vocabulary: 10,762 tokens** — Sunuwar SPM-8k's 6,760 content pieces +
Nepali SPM-4k's 3,996 content pieces + 6 shared specials (`PAD`, `UNK`,
`BOS`, `EOS`, `<2npi>`, `<2suz>`), merged via **ID-offset**, not a
freshly-retrained joint SentencePiece model — chosen because a single
joint SPM would tokenize Sunuwar differently than the project's existing
SunuwarBERT-small/SunuwarCLM-small tokenizer, and because Phase 3's
induced lexicon found only a handful of near-cognates between the two
(unrelated) language families, suggesting minimal benefit from shared
cross-lingual subwords. **Actual parameter count: 9,508,362** — larger
than the ~7.1M pre-training estimate, a discrepancy traced entirely to the
joint vocabulary coming out at 10,762 tokens rather than the ~6,000 the
original estimate assumed.

**Direction-tagged shared model**: a single encoder-decoder handles both
directions via `<2npi>`/`<2suz>` tags prepended to the source sequence.
Each of the 10,412 sentence pairs is doubled into both directions for
training. The 90/10 train/val split is performed **at the sentence-pair
level, before doubling into directions**, preventing within-pair direction
leakage (guaranteeing a pair's suz→npi and npi→suz examples land on the
same side of the split). This is separate from, and does not address, the
~0.66% cross-pair synoptic-parallel leakage (near-duplicate parallel
Gospel passages on both sides of a random split) identified and accepted
as negligible during the Phase 4 feasibility investigation.

**Training runs**:

| Run | Regularization | best_val_loss | val_perplexity | best_epoch |
|---|---|---|---|---|
| Run 1 (baseline) | dropout=0.1, weight_decay=0.01, no label smoothing | 4.0815 | 59.23 | 12 |
| Run 2 (regularized) | label_smoothing=0.1, dropout=0.2, weight_decay=0.05 | 3.9696 | 52.96 | 19 |

Run 2 represents a real **~10.6% val_perplexity improvement**, verified
apples-to-apples since the evaluation loss function was kept unsmoothed in
both runs (only training loss used label smoothing). Train-loss values
between the two runs are *not* directly comparable, since label smoothing
structurally raises the floor a smoothed training loss can reach. All
results below are from Run 2.

**Quantitative evaluation** (full greedy-decode inference over the entire
held-out validation split, 1,042 sentence pairs × 2 directions = 2,084
examples, scored with `sacrebleu` corpus BLEU and chrF, **separately per
direction**):

| Direction | BLEU | chrF |
|-----------|------|------|
| suz→npi   | 2.23 | 21.40 |
| npi→suz   | 3.97 | 24.92 |
| corpus avg| 3.10 | 23.16 |

Both directions score well below typical usable low-resource NMT
baselines (BLEU ~15-20+, chrF ~40-50+). **npi→suz outscores suz→npi on
both metrics, consistently across every divergence example inspected** —
this **reverses the pre-training hypothesis** from the Phase 4 feasibility
investigation, which expected Sunuwar-as-target to be the harder direction
due to its richer agglutinative verbal morphology. A plausible but
**unconfirmed** explanation offered: source-side comprehension may matter
more than target-side generation complexity at this corpus size, meaning
Nepali source sentences may be easier for the encoder to represent
usefully than Sunuwar source sentences — offered as a hypothesis for
future investigation, not a conclusion this evaluation can support on its
own.

**Qualitative findings**: outputs are fluent and grammatically well-formed
in the target language in both directions, and the joint vocabulary/
direction-tagging mechanism never misfired in the full validation run (no
cross-language token generation observed). Content adequacy is poor: most
outputs diverge substantially from the reference meaning, often defaulting
to frequent generic narrative/thematic templates from the corpus's
Biblical register (e.g. "Paul said...", "the gospel of Jesus Christ...")
rather than reflecting the source sentence's specific content. chrF-high/
BLEU-low cases are the clearest signature of this failure mode (correct
topic and register, wrong specific proposition); BLEU-high/chrF-low cases
are short-sentence scoring artifacts, not genuine quality signals.

## 7. Discussion

Across all four phases, the pipeline components themselves — USFM
parsing and `\id`-based joining, quote-aware sentence splitting,
Gale-Church DP alignment, eflomal word alignment, joint-vocabulary
construction via ID-offset merge, direction-tagged shared training, and
greedy decoding — worked correctly end-to-end. No architectural or
implementation defect was found at any stage. Every honest limitation
encountered instead traces to the same root cause: **the underlying
corpus is small** (10,412 sentence pairs, or 7,942 at the coarser
verse-level granularity), sufficient for surface-level correctness
(complete verse coverage, fluent target-language generation) but not for
fine-grained tasks that require broad lexical or distributional coverage
(exhaustive lexicon induction, translation adequacy).

The most broadly citable methodological point in this paper, not specific
to Sunuwar, comes from Section 5's SIL-lexicon validation: **measuring an
induced lexicon's precision against a sparse, single-gloss-per-headword
gold resource systematically understates true precision**, because a
correct induced translation is scored as an error whenever it happens to
use a synonym the gold resource didn't happen to list. The ~4x gap
observed here (17.47% measured vs. ~65% human-judged) is large enough that
it should not be treated as a Sunuwar-specific quirk. Any project
evaluating an induced or automatically-derived lexicon against a similarly
sparse gold dictionary — a common situation for genuinely low-resource
languages, where "gold" resources are themselves small and
non-exhaustive — should expect a comparable undercount and budget for a
manual spot-check rather than trusting the raw precision number at face
value. This generalizes the specific collocation-artifact finding (Section
5) as well: a real remaining error category exists independent of the
gold-sparsity measurement gap, and the two should not be conflated when
interpreting a similarly-structured evaluation elsewhere.

## 8. Limitations

- **Corpus size ceiling**: 7,942 pairs at verse-level granularity, 10,412
  pairs at sentence-level granularity — small by the standards of both
  statistical word alignment (typically 100K+ pairs) and neural machine
  translation, and the root cause of every downstream adequacy/coverage
  limitation reported above.
- **No independent Sunuwar-side verification of the lexicon spot-check**:
  the 40-pair manual check (Section 5) was judged by an English/
  Nepali-literate, non-Sunuwar-fluent reviewer from translation
  plausibility and sentence context. A genuine Sunuwar mistranslation that
  happens to produce Nepali-plausible output in context would not be
  caught by this method.
- **Synoptic-parallel cross-book leakage is not solvable by any split
  strategy used here**: the ~0.66% cross-pair leakage from near-duplicate
  parallel Gospel passages (e.g. the same teaching appearing in Matthew,
  Mark, and Luke) landing on both sides of a random split is a property of
  the underlying text, not the split algorithm — the Phase 4 split fix
  (pre-doubling, sentence-pair-level) addresses a different, narrower
  leakage source (within-pair direction leakage) and does not touch this
  one.
- **Domain mismatch between the SIL lexicon and the NT-only induction
  source**: SIL's lexicon is general-vocabulary; the induction source is
  exclusively New Testament scripture text, so the low raw coverage
  (2.97% of SIL headwords) reflects domain non-overlap, not induction
  failure, but does mean the induced lexicon cannot be assumed
  representative of everyday Sunuwar vocabulary.
- **NMT translation adequacy is low at this data scale** (Section 6);
  the model is not fit for practical translation use as released, only as
  a documented baseline for future scaling comparisons.

## 9. Future Work

- **English WEB pairing**: already licensing-cleared per the Phase 1
  investigation, but not yet built — would add a third parallel language,
  potentially enabling pivot-based triangulation for additional training
  signal.
- **Fine-tuning from pretrained multilingual representations** (e.g. an
  NLLB or similar multilingual encoder-decoder) as an alternative to
  training a translation transformer from scratch, to bring prior
  cross-lingual and target-fluency knowledge that the current model had to
  learn entirely from 10,412 pairs.
- **Extension to Tamang and Puma**, per the project's broader Kiranti-
  language roadmap, applying the same verse-alignment → sentence-alignment
  → lexicon-induction → baseline-NMT pipeline documented in this paper.

## References

[1] C. Christodoulopoulos and M. Steedman, "A massively parallel corpus:
the Bible in 100 languages," *Language Resources and Evaluation*, vol. 49,
no. 2, pp. 375-395, 2015. *(Bibliographic detail not independently
verified against a repo source file — see Section 2 flag.)*

[2] Daspit et al., "The eBible Corpus: Data and Model Benchmarks for Bible
Translation for Low-Resource Languages," 2023. *(Bibliographic detail not
independently verified against a repo source file — see Section 2 flag.)*

[3] T. Kudo and J. Richardson, "SentencePiece: A simple and language
independent subword tokenizer and detokenizer for neural text processing,"
in *Proc. 2018 Conf. Empirical Methods Natural Lang. Process. (EMNLP):
Syst. Demonstrations*, 2018, pp. 66-71.

[4] T. Mikolov, I. Sutskever, K. Chen, G. S. Corrado, and J. Dean,
"Distributed representations of words and phrases and their
compositionality," in *Advances in Neural Information Processing Systems
(NeurIPS)*, vol. 26, 2013, pp. 3111-3119.

[5] P. Bojanowski, E. Grave, A. Joulin, and T. Mikolov, "Enriching word
vectors with subword information," *Trans. Assoc. Comput. Linguistics*,
vol. 5, pp. 135-146, 2017.

[6] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training
of deep bidirectional transformers for language understanding," in *Proc.
2019 Conf. North Amer. Chapter Assoc. Comput. Linguistics: Human Lang.
Technol. (NAACL-HLT)*, 2019, pp. 4171-4186.

[7] V. Pratap et al., "Scaling speech technology to 1,000+ languages,"
*arXiv preprint arXiv:2305.13516*, 2023.

[8] R. Östling and J. Tiedemann, "Efficient word alignment with Markov
Chain Monte Carlo," *Prague Bulletin of Mathematical Linguistics*, vol.
106, pp. 125-146, 2016.

[9] W. A. Gale and K. W. Church, "A program for aligning sentences in
bilingual corpora," *Computational Linguistics*, vol. 19, no. 1, pp.
75-102, 1993.

---

*All quantitative figures in this paper were pulled directly from
`results/suz_npi_parallel_methodology.md` (Phase 1),
`results/suz_npi_sentence_alignment_methodology.md` (Phase 2),
`results/induced_lexicon_methodology.md` (Phase 3), and
`results/nmt_methodology.md` (Phase 4). No figure was paraphrased from
memory or approximated; the two reference entries flagged above are the
only content in this paper not directly sourced from an existing repo
file.*
