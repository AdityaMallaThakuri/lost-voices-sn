# Data source: Sunuwar Bible (suzBl), © 2011 Wycliffe Bible Translators, Inc.
# Nepali source: Unlocked Literal Bible (npiulb), Door43 World Missions
# Community, CC BY-SA 4.0 -- see results/suz_npi_parallel_methodology.md.
# Licence: raw parallel text non-commercial research use only, no
# redistribution of raw text. Trained models may be released.

import sys
import random
import warnings
warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import os
import json
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nmt_model import JointTokeniser, SunuwarNMT, load_and_split, greedy_translate, decode_combined_ids

random.seed(42)

# Known best_val_loss from the two training runs discussed, used only to
# report which run's weights are actually loaded -- not a hard check.
KNOWN_RUNS = {
    4.0815: "Run 1 (baseline, no label smoothing, dropout=0.1, weight_decay=0.01)",
    3.9696: "Run 2 (label_smoothing=0.1, dropout=0.2, weight_decay=0.05)",
}


def identify_checkpoint_run(config: dict):
    """Best-effort identification of which training run produced the
    currently-saved checkpoint, using results/nmt_eval.json (written at the
    END of whichever run last completed). Prints a clear confirmation or a
    clear warning -- never silently assumes."""
    eval_path = config.get("eval_output_path", "results/nmt_eval.json")
    if not os.path.exists(eval_path):
        print(f"  WARNING: {eval_path} not found -- cannot confirm which run's "
              f"weights are in {config['checkpoint_path']}. Proceeding anyway, "
              f"but treat the examples below as unverified-provenance until "
              f"this is checked.")
        return
    with open(eval_path, encoding="utf-8") as f:
        eval_data = json.load(f)
    best_val_loss = eval_data.get("best_val_loss")
    best_epoch = eval_data.get("best_epoch")
    match = KNOWN_RUNS.get(best_val_loss)
    if match:
        print(f"  Confirmed via {eval_path}: best_val_loss={best_val_loss} (epoch {best_epoch}) "
              f"matches {match}")
    else:
        print(f"  {eval_path} reports best_val_loss={best_val_loss} (epoch {best_epoch}) -- "
              f"does not match either previously-discussed run exactly. "
              f"This may be a newer/different run than the two compared so far.")


def main():
    with open("configs/nmt.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokeniser = JointTokeniser(config["suz_tokeniser_path"], config["npi_tokeniser_path"], config["max_seq_len"])
    model = SunuwarNMT(config, tokeniser.vocab_size).to(device)

    checkpoint_path = config["checkpoint_path"]
    print(f"\nLoading checkpoint: {checkpoint_path}")
    identify_checkpoint_run(config)

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    # Rebuild the EXACT same train/val split used during training (same
    # seed, same train_split ratio) so these are genuine held-out examples,
    # not new sentences the model may have already seen.
    with open(config["aligned_path"], encoding="utf-8") as f:
        import csv
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)

    rng = random.Random(config["seed"])
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    split_point = int(len(indices) * config["train_split"])
    val_row_indices = indices[split_point:]

    print(f"\nHeld-out validation pool: {len(val_row_indices)} sentence pairs "
          f"(same seed={config['seed']}, train_split={config['train_split']} as training)")

    sample_rng = random.Random(123)  # different seed from the split itself, just for which 10 to show
    sampled_val_indices = sample_rng.sample(val_row_indices, 10)

    print("\n" + "=" * 90)
    print("QUALITATIVE INFERENCE -- 5 examples per direction, greedy decoding")
    print("=" * 90)

    for direction, label in [("suz2npi", "Sunuwar -> Nepali"), ("npi2suz", "Nepali -> Sunuwar")]:
        print(f"\n{'-'*90}\n{label}\n{'-'*90}")
        shown = 0
        for idx in sampled_val_indices:
            if shown >= 5:
                break
            row = rows[idx]
            suz_text, npi_text = row["suz_sentence"], row["npi_sentence"]
            if direction == "suz2npi":
                src_text, ref_text = suz_text, npi_text
                src_ids = [JointTokeniser.TAG_2NPI_ID] + tokeniser.encode_suz(src_text)
            else:
                src_text, ref_text = npi_text, suz_text
                src_ids = [JointTokeniser.TAG_2SUZ_ID] + tokeniser.encode_npi(src_text)
            src_ids = src_ids[: config["max_seq_len"]]

            output_text = greedy_translate(model, tokeniser, src_ids, direction, device)

            print(f"\n[{row['book']} {row['chapter']}:{row['verse']}]")
            print(f"  SOURCE:    {src_text}")
            print(f"  MODEL:     {output_text}")
            print(f"  REFERENCE: {ref_text}")
            shown += 1


if __name__ == "__main__":
    main()
