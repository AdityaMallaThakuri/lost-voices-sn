# Data source: Sunuwar Bible (suzBl), © 2011 Wycliffe Bible Translators, Inc.
# Nepali source: Unlocked Literal Bible (npiulb), Door43 World Missions
# Community, CC BY-SA 4.0 -- see results/suz_npi_parallel_methodology.md.
# Licence: raw parallel text non-commercial research use only, no
# redistribution of raw text. Trained models may be released.
#
# Interactive REPL for sunuwarNMT-small (Run 2 checkpoint). Lets you type
# arbitrary Sunuwar or Nepali sentences and see the model's greedy-decoded
# translation, reusing the exact tokenizer/model/decoding code already
# used by demo_nmt.py and evaluate_nmt.py -- no new translation logic here.
#
# Known ceiling (see results/nmt_methodology.md): this model scores
# BLEU 2.23-3.97 / chrF 21.40-24.92 on held-out validation data drawn from
# the SAME Bible-register training corpus. Arbitrary free-typed sentences
# are out-of-domain relative to that corpus and should be expected to
# translate worse, not better, than the held-out numbers -- fluent output,
# low content adequacy, is the expected outcome, not a bug.

import sys
import os
import warnings
warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8")

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nmt_model import JointTokeniser, SunuwarNMT, greedy_translate
from demo_nmt import identify_checkpoint_run


def main():
    with open("configs/nmt.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading tokenizers and model...")
    tokeniser = JointTokeniser(config["suz_tokeniser_path"], config["npi_tokeniser_path"], config["max_seq_len"])
    model = SunuwarNMT(config, tokeniser.vocab_size).to(device)

    identify_checkpoint_run(config)
    state_dict = torch.load(config["checkpoint_path"], map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    print("\n" + "=" * 78)
    print("sunuwarNMT-small -- interactive translation")
    print("=" * 78)
    print("Known ceiling: BLEU 2.23-3.97 / chrF 21.40-24.92 on in-corpus")
    print("held-out data (results/nmt_methodology.md). Free-typed sentences")
    print("are out-of-domain relative to the NT training corpus -- expect")
    print("fluent but often loosely-related output, not accurate translation.")
    print("=" * 78)
    print("\nCommands: 's' = Sunuwar->Nepali, 'n' = Nepali->Sunuwar, 'q' = quit")

    direction = None
    while True:
        if direction is None:
            choice = input("\nDirection ([s]uz->npi / [n]pi->suz / [q]uit): ").strip().lower()
            if choice in ("q", "quit", "exit"):
                break
            if choice not in ("s", "n"):
                print("  Please type 's', 'n', or 'q'.")
                continue
            direction = "suz2npi" if choice == "s" else "npi2suz"
            label = "Sunuwar -> Nepali" if direction == "suz2npi" else "Nepali -> Sunuwar"
            print(f"  Direction set: {label}. (type 'switch' to change, 'q' to quit)")

        prompt = "  Sunuwar> " if direction == "suz2npi" else "  Nepali> "
        text = input(prompt).strip()
        if not text:
            continue
        if text.lower() in ("q", "quit", "exit"):
            break
        if text.lower() == "switch":
            direction = None
            continue

        if direction == "suz2npi":
            src_ids = [JointTokeniser.TAG_2NPI_ID] + tokeniser.encode_suz(text)
        else:
            src_ids = [JointTokeniser.TAG_2SUZ_ID] + tokeniser.encode_npi(text)
        src_ids = src_ids[: config["max_seq_len"]]

        output_text = greedy_translate(model, tokeniser, src_ids, direction, device)
        print(f"  -> {output_text}")

    print("\nDone.")


if __name__ == "__main__":
    main()
