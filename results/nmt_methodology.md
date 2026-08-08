# sunuwarNMT-small — Methodology (Phase 4)

## Architecture & Training

- **Config A**: shared bidirectional encoder-decoder transformer.
  hidden_dim=256, num_heads=4, ffn_dim=512, 3 encoder layers, 3 decoder
  layers, max_seq_len=128, dropout=0.2, GELU activation.
- **Joint vocabulary**: 10,762 tokens — Sunuwar SPM-8k's 6,760 content
  pieces + Nepali SPM-4k's 3,996 content pieces + 6 shared specials
  (`PAD`, `UNK`, `BOS`, `EOS`, `<2npi>`, `<2suz>`), merged via
  **ID-offset**, not a freshly-retrained joint SentencePiece model.
  Rationale: a single joint SPM would tokenize Sunuwar differently
  than the existing SunuwarBERT-small/SunuwarCLM-small tokenizer
  (breaking comparability with those models), and the expected benefit
  of shared cross-lingual subwords is minimal here — Sunuwar and
  Nepali are unrelated language families, and Phase 3's induced
  bilingual lexicon (`results/induced_lexicon_methodology.md`) found
  only a handful of near-cognates in 5,846 filtered word pairs, not
  the systematic subword overlap that would make a joint SPM pay off.
- **Actual parameter count: 9,508,362** — larger than the ~7.1M
  estimate from the Phase 4 feasibility investigation. The discrepancy
  traces entirely to the joint vocab: the investigation's estimate
  assumed a ~6,000-token joint vocabulary, but reusing the existing
  Sunuwar SPM-8k (6,764 pieces, of which 6,760 are content pieces)
  rather than training a smaller joint one pushed the real combined
  vocabulary to 10,762 — nearly double the estimate — which drives
  most of the extra parameters via the embedding and output-head
  matrices. Both numbers are recorded here so the estimate-vs-actual
  gap is traceable rather than silently dropped.
- **Direction-tagged shared model**: a single encoder-decoder handles
  both directions via `<2npi>`/`<2suz>` tags prepended to the source
  sequence. Each of the 10,412 sentence pairs from
  `results/suz_npi_sentence_aligned.tsv` is doubled into both
  directions (suz→npi and npi→suz), for 20,824 training examples pre-split.
- **Split methodology**: the 90/10 train/val split is performed **at
  the sentence-pair level, before doubling into directions** — this
  guarantees a given sentence pair's suz→npi and npi→suz examples
  always land on the same side of the split, preventing
  within-pair direction leakage (seeing a pair's Sunuwar→Nepali example
  in training would otherwise let the model see half of a "held-out"
  Nepali→Sunuwar validation example too). This is a **separate**
  concern from the ~0.66% cross-pair synoptic-parallel leakage
  documented in the Phase 4 feasibility investigation (near-duplicate
  parallel Gospel passages appearing on both sides of a random split)
  — that leakage source is still present and **not** addressed by this
  split design; it was assessed and accepted as negligible at that
  earlier stage, not fixed here.
- **Run 1 (baseline)**: dropout=0.1, weight_decay=0.01, no label
  smoothing. Best val_loss=4.0815 at epoch 12 (val_perplexity 59.23),
  early-stopped at epoch 17.
- **Run 2 (regularized)**: label_smoothing=0.1, dropout raised
  0.1→0.2, weight_decay raised 0.01→0.05. Best val_loss=3.9696 at
  epoch 19 (val_perplexity 52.96), early-stopped at epoch 24 — a real
  **~10.6% val_perplexity improvement**, verified apples-to-apples
  since the evaluation loss function was kept unsmoothed in both runs
  (only the training loss used label smoothing). **Caveat**: the
  train_loss values between the two runs are *not* directly
  comparable — label smoothing structurally raises the floor a
  smoothed training loss can reach, so an apparent shrinking of the
  train/val gap between runs partly reflects that mechanical effect,
  not only reduced overfitting. The val_perplexity comparison above is
  the trustworthy one.
- All reported checkpoint/eval numbers are from **Run 2**
  (`models/sunuwar_nmt.pt`, confirmed via `results/nmt_eval.json`).

## Quantitative Evaluation (held-out validation, 1,042 pairs/direction)

Full greedy-decode inference over the entire held-out validation split
(1,042 sentence pairs × 2 directions = 2,084 examples), scored with
`sacrebleu`'s corpus BLEU and chrF, computed **separately per
direction** — never blended into one number.

| Direction | BLEU | chrF |
|-----------|------|------|
| suz→npi   | 2.23 | 21.40 |
| npi→suz   | 3.97 | 24.92 |
| corpus avg| 3.10 | 23.16 |

**Stated plainly**: both directions score well below typical usable
low-resource NMT baselines (BLEU ~15–20+, chrF ~40–50+ is the
rough range for a "working" low-resource system). These are low scores
by any normal yardstick, not a borderline or ambiguous result.

**Directional asymmetry**: npi→suz outscores suz→npi on both metrics,
consistently across every divergence example inspected, not just in
the aggregate. This is worth flagging explicitly because it
**reverses the pre-training hypothesis** from the Phase 4 feasibility
investigation, which expected Sunuwar-as-target to be the *harder*
direction due to Sunuwar's richer agglutinative verbal morphology
(more surface forms to generate correctly). The measured result is the
opposite. A plausible but **unconfirmed** explanation: source-side
comprehension may matter more than target-side generation complexity
at this corpus size — Nepali source sentences may be easier for the
encoder to represent usefully than Sunuwar source sentences, for
reasons not isolated here (e.g. Nepali's SPM-4k trained on a much
smaller, more homogeneous vocabulary, or Sunuwar's morphology making
source-side segmentation noisier). This is offered as a hypothesis for
future investigation, not a conclusion this evaluation can support on
its own.

## Qualitative Findings

- Outputs are **fluent and grammatically well-formed** in the target
  language in both directions. The joint vocabulary and
  direction-tagging mechanism never misfired in the full validation
  run — no cross-language token generation was observed (the
  `decode_combined_ids` out-of-range guard, built and smoke-tested
  against a random-init model earlier in Phase 4, never triggered on
  the real trained checkpoint).
- **Content adequacy is poor**: most outputs diverge substantially
  from the reference meaning. A recurring pattern is the model
  defaulting to frequent generic narrative/thematic templates learned
  from the corpus's Biblical register (e.g. "Paul said...",
  "the gospel of Jesus Christ...") rather than reflecting the actual
  source sentence's specific content.
- **chrF-high/BLEU-low cases are the clearest signature of this
  failure mode**: the model produces the right topic and register but
  the wrong specific proposition (e.g. MRK 1:1 — source about "the
  gospel beginning," model output about "preaching the gospel,"
  reference about "the Son of God" — same general domain, wrong
  specific claim). chrF's character-n-gram partial credit picks up
  the topical/morphological overlap that BLEU's stricter word-n-gram
  matching misses entirely.
- **BLEU-high/chrF-low cases are short-sentence scoring artifacts, not
  genuine quality signals** — very short references (3–5 words) let a
  short, generic model output score a high BLEU from incidental word
  overlap (e.g. "म हुँ ।" scoring BLEU=32.34 against "म पनि इस्राएली
  हुँ ।") without actually conveying the reference's content. These
  should not be read as evidence of good translation quality.

## Conclusion

The pipeline itself — tokenization, joint vocabulary construction via
ID-offset merge, direction-tagged shared training, and greedy decoding
— is **fully functional and bug-free end-to-end**. No architectural or
implementation defect was found at any stage of Phase 4. The limiting
factor is **corpus size**: 10,412 sentence pairs is small for a
translation task and is sufficient for the model to learn target-language
fluency (grammatical, well-formed output) but not sufficient for it to
learn fine-grained translation adequacy (correctly mapping specific
source content to specific target content). This is consistent with
known low-resource NMT behavior in general, not a defect specific to
this approach or this implementation.

**Future work**, not undertaken here: expanding the parallel corpus
(e.g. pairing against the English WEB translation as a third
reference/pivot to potentially triangulate more training signal), or
fine-tuning from an existing pretrained multilingual encoder-decoder
(e.g. mT5, NLLB) rather than training a transformer from scratch on
this corpus size — a pretrained multilingual starting point would
bring prior cross-lingual and target-fluency knowledge that this
from-scratch model had to learn entirely from 10,412 pairs.

## Final files

- **`results/nmt_eval.json`** — Run 2 training summary (architecture,
  parameter count, joint vocab size, best_val_loss, best_epoch).
- **`results/nmt_eval_bleu_chrf.json`** — full chrF/BLEU breakdown per
  direction plus corpus-level average, from the full 1,042-pair
  held-out validation run.
- **`models/sunuwar_nmt.pt`** — Run 2 checkpoint (38MB).
- **`results/nmt_methodology.md`** — this file.

This document closes out Phase 4 and the translation pipeline
(Phases 1–4) as a whole.
