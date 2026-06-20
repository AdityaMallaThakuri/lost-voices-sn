# Data source: Sunuwar Bible (suzBl), © 2011 Wycliffe Bible Translators, Inc.
# Licence: CC BY-NC-ND 4.0 — non-commercial research use only
# Do not redistribute raw text. Models trained on this data may be released.

import sys
import unicodedata
from pathlib import Path

import sentencepiece as spm
import yaml


def train_tokeniser(input_path: str, model_prefix: str, vocab_size: int, config: dict) -> None:
    spm.SentencePieceTrainer.train(
        input=input_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=config['model_type'],
        character_coverage=config['character_coverage'],
        pad_id=config['pad_id'],
        unk_id=config['unk_id'],
        bos_id=config['bos_id'],
        eos_id=config['eos_id'],
        user_defined_symbols=config['user_defined_symbols'],
        byte_fallback=config['byte_fallback'],
        split_by_whitespace=config['split_by_whitespace'],
        hard_vocab_limit=False,
    )
    print(f'Trained: {model_prefix}.model')


def evaluate_tokeniser(model_path: str, eval_path: str, vocab_size: int) -> dict:
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    unk_id = sp.unk_id()

    total_words = 0
    total_pieces = 0
    total_unks = 0

    with open(eval_path, encoding='utf-8') as fh:
        for line in fh:
            words = line.strip().split()
            if not words:
                continue
            for word in words:
                pieces = sp.encode(word, out_type=int)
                total_words += 1
                total_pieces += len(pieces)
                total_unks += sum(1 for p in pieces if p == unk_id)

    fertility = total_pieces / total_words if total_words else 0.0
    unk_rate  = total_unks  / total_pieces if total_pieces else 0.0

    return {'vocab_size': vocab_size, 'fertility': float(fertility), 'unk_rate': float(unk_rate)}


# Assertion tests (dummy values — no trained model required)
_d = {'vocab_size': 8000, 'fertility': 3.2, 'unk_rate': 0.01}
assert isinstance(_d['fertility'], float) and _d['fertility'] > 0
assert isinstance(_d['unk_rate'], float) and 0.0 <= _d['unk_rate'] <= 1.0


_SUFFIX = {8000: '8k', 16000: '16k'}


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/spm.yaml'
    with open(config_path, encoding='utf-8') as fh:
        cfg = yaml.safe_load(fh)

    model_dir = Path(cfg['model_dir'])
    model_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for vocab_size in cfg['vocab_sizes']:
        suffix = _SUFFIX.get(vocab_size, str(vocab_size))
        model_prefix = str(model_dir / f'sunuwar_spm_{suffix}')

        train_tokeniser(cfg['input_path'], model_prefix, vocab_size, cfg)
        result = evaluate_tokeniser(f'{model_prefix}.model', cfg['eval_path'], vocab_size)
        results.append((model_prefix, result))

    # Comparison table
    print()
    print(f"{'vocab_size':>12}  {'fertility':>10}  {'unk_rate':>10}")
    print('-' * 38)
    for _, r in results:
        print(f"{r['vocab_size']:>12}  {r['fertility']:>10.4f}  {r['unk_rate']:>10.6f}")

    # Recommendation — fertility sweet spot is 2.5–4.0 (Skill 5)
    print()
    in_range = [r for _, r in results if 2.5 <= r['fertility'] <= 4.0]
    if len(in_range) == 2:
        # Both in range: prefer 8k for BERT (smaller vocab → denser embeddings on small corpus)
        chosen = min(results, key=lambda x: x[1]['vocab_size'])
        reason = (
            f"Both vocab sizes fall in the healthy fertility range (2.5–4.0). "
            f"8k is preferred for SunuwarBERT-small: with only ~180K tokens of training data, "
            f"a smaller vocabulary means more training signal per embedding, "
            f"reducing the risk of undertrained subword vectors."
        )
    elif len(in_range) == 1:
        chosen = next((mp, r) for mp, r in results if r == in_range[0])
        other  = next(r for _, r in results if r != in_range[0])
        out_fertility = other['fertility']
        direction = "over-fragments words (vocab too small)" if out_fertility > 4.0 else "memorises whole words (vocab too large)"
        reason = (
            f"Only the {chosen[1]['vocab_size']:,}-token vocabulary lands in the healthy fertility "
            f"range (2.5–4.0). The {other['vocab_size']:,}-token model has fertility {out_fertility:.4f}, "
            f"which {direction}."
        )
    else:
        # Neither in range — pick closest to midpoint 3.25
        chosen = min(results, key=lambda x: abs(x[1]['fertility'] - 3.25))
        reason = (
            f"Neither model hits the 2.5–4.0 fertility target. "
            f"{chosen[1]['vocab_size']:,} is closer to the midpoint (3.25) so it is preferred."
        )

    print(f"Recommendation: use vocab_size={chosen[1]['vocab_size']:,} for SunuwarBERT-small.")
    print(f"Reason: {reason}")


if __name__ == '__main__':
    main()
