# Data source: Sunuwar Bible (suzBl), © 2011 Wycliffe Bible Translators, Inc.
# Nepali source: Unlocked Literal Bible (npiulb), Door43 World Missions
# Community, CC BY-SA 4.0 -- see results/suz_npi_parallel_methodology.md.
# Licence: raw parallel text non-commercial research use only, no
# redistribution of raw text. Trained models may be released.
#
# Full chrF/BLEU evaluation of SunuwarNMT-small over the held-out
# validation split, computed SEPARATELY per direction (never blended --
# a single mixed number would hide a direction-specific weakness, exactly
# the gap flagged after the first training run). Requires sacrebleu
# (installed by the notebook cell before this one).

import sys
import os
import csv
import json
import random
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import yaml
import sacrebleu

from train_nmt import JointTokeniser, SunuwarNMT
from demo_nmt import greedy_translate

random.seed(42)


def main():
    with open("configs/nmt.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokeniser = JointTokeniser(config["suz_tokeniser_path"], config["npi_tokeniser_path"], config["max_seq_len"])
    model = SunuwarNMT(config, tokeniser.vocab_size).to(device)
    state_dict = torch.load(config["checkpoint_path"], map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    with open(config["eval_output_path"], encoding="utf-8") as f:
        eval_meta = json.load(f)
    print(f"Checkpoint confirmed: best_val_loss={eval_meta['best_val_loss']} (epoch {eval_meta['best_epoch']})")

    # Rebuild the EXACT same val split as training (same seed, same ratio)
    with open(config["aligned_path"], encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)

    rng = random.Random(config["seed"])
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    split_point = int(len(indices) * config["train_split"])
    val_row_indices = indices[split_point:]
    print(f"Val pool: {len(val_row_indices)} sentence pairs -> {len(val_row_indices) * 2} examples (both directions)")

    # ------------------------------------------------------------------
    # Inference over the full val split, both directions
    # ------------------------------------------------------------------
    start = time.time()
    results = {"suz2npi": [], "npi2suz": []}

    for count, idx in enumerate(val_row_indices, 1):
        row = rows[idx]
        suz_text, npi_text = row["suz_sentence"], row["npi_sentence"]

        src_ids_s2n = ([JointTokeniser.TAG_2NPI_ID] + tokeniser.encode_suz(suz_text))[: config["max_seq_len"]]
        out_s2n = greedy_translate(model, tokeniser, src_ids_s2n, "suz2npi", device)
        results["suz2npi"].append((row["book"], row["chapter"], row["verse"], suz_text, out_s2n, npi_text))

        src_ids_n2s = ([JointTokeniser.TAG_2SUZ_ID] + tokeniser.encode_npi(npi_text))[: config["max_seq_len"]]
        out_n2s = greedy_translate(model, tokeniser, src_ids_n2s, "npi2suz", device)
        results["npi2suz"].append((row["book"], row["chapter"], row["verse"], npi_text, out_n2s, suz_text))

        if count % 100 == 0:
            elapsed = time.time() - start
            print(f"  {count}/{len(val_row_indices)} pairs done ({elapsed:.1f}s elapsed, "
                  f"~{elapsed / count * len(val_row_indices):.0f}s estimated total)")

    elapsed = time.time() - start
    print(f"\nInference complete: {len(val_row_indices) * 2} examples in {elapsed:.1f}s ({elapsed / 60:.2f} min)")

    # ------------------------------------------------------------------
    # chrF / BLEU per direction, never blended
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("chrF / BLEU per direction")
    print("=" * 78)

    scores = {}
    for direction in ["suz2npi", "npi2suz"]:
        hyps = [r[4] for r in results[direction]]
        refs = [r[5] for r in results[direction]]
        clean_hyps = [h.split("  [")[0] if "  [" in h else h for h in hyps]

        bleu = sacrebleu.corpus_bleu(clean_hyps, [refs])
        chrf = sacrebleu.corpus_chrf(clean_hyps, [refs])

        scores[direction] = {"bleu": round(bleu.score, 4), "chrf": round(chrf.score, 4), "n_examples": len(clean_hyps)}
        print(f"\n{direction}: n={len(clean_hyps)}")
        print(f"  BLEU: {bleu.score:.4f}")
        print(f"  chrF: {chrf.score:.4f}")

    corpus_avg_bleu = (scores["suz2npi"]["bleu"] + scores["npi2suz"]["bleu"]) / 2
    corpus_avg_chrf = (scores["suz2npi"]["chrf"] + scores["npi2suz"]["chrf"]) / 2
    print(f"\nCorpus-level average (simple mean of the two directions):")
    print(f"  BLEU: {corpus_avg_bleu:.4f}")
    print(f"  chrF: {corpus_avg_chrf:.4f}")

    output_path = "results/nmt_eval_bleu_chrf.json"
    final_json = {
        "architecture": "SunuwarNMT-small",
        "checkpoint_best_val_loss": eval_meta["best_val_loss"],
        "checkpoint_best_epoch": eval_meta["best_epoch"],
        "num_val_pairs": len(val_row_indices),
        "suz2npi": scores["suz2npi"],
        "npi2suz": scores["npi2suz"],
        "corpus_avg_bleu": round(corpus_avg_bleu, 4),
        "corpus_avg_chrf": round(corpus_avg_chrf, 4),
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_json, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {output_path}")

    # ------------------------------------------------------------------
    # Per-sentence chrF vs BLEU divergence check
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("Per-sentence chrF vs BLEU divergence check")
    print("=" * 78)

    for direction in ["suz2npi", "npi2suz"]:
        print(f"\n--- {direction} ---")
        per_sentence = []
        for (book, chap, verse, src, hyp, ref) in results[direction]:
            clean_hyp = hyp.split("  [")[0] if "  [" in hyp else hyp
            sent_bleu = sacrebleu.sentence_bleu(clean_hyp, [ref], smooth_method="exp").score
            sent_chrf = sacrebleu.sentence_chrf(clean_hyp, [ref]).score
            per_sentence.append((book, chap, verse, src, clean_hyp, ref, sent_bleu, sent_chrf))

        high_chrf_low_bleu = sorted(per_sentence, key=lambda x: -(x[7] - x[6]))[:3]
        high_bleu_low_chrf = sorted(per_sentence, key=lambda x: -(x[6] - x[7]))[:3]

        print(f"\n  Top 3: chrF HIGH, BLEU LOW (partial/morphological credit chrF catches, BLEU misses):")
        for (book, chap, verse, src, hyp, ref, sb, sc) in high_chrf_low_bleu:
            print(f"\n  [{book} {chap}:{verse}]  sentence_BLEU={sb:.2f}  sentence_chrF={sc:.2f}")
            print(f"    SRC: {src}")
            print(f"    HYP: {hyp}")
            print(f"    REF: {ref}")

        print(f"\n  Top 3: BLEU HIGH, chrF LOW (rare -- exact word overlap without much character overlap):")
        for (book, chap, verse, src, hyp, ref, sb, sc) in high_bleu_low_chrf:
            print(f"\n  [{book} {chap}:{verse}]  sentence_BLEU={sb:.2f}  sentence_chrF={sc:.2f}")
            print(f"    SRC: {src}")
            print(f"    HYP: {hyp}")
            print(f"    REF: {ref}")


if __name__ == "__main__":
    main()
