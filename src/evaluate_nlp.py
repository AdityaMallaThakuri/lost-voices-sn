# Data source: Sunuwar Bible (suzBl), © 2011 Wycliffe Bible Translators, Inc.
# Licence: CC BY-NC-ND 4.0 — non-commercial research use only
# Do not redistribute raw text. Models trained on this data may be released.

import json
import sys
from pathlib import Path

import numpy as np
import yaml
from gensim.models import FastText, Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


def load_embedding_model(model_path: str):
    try:
        model = Word2Vec.load(model_path)
        print(f'Loaded Word2Vec model: {model_path}')
        return model
    except Exception:
        pass

    try:
        model = FastText.load(model_path)
        print(f'Loaded FastText model: {model_path}')
        return model
    except Exception:
        pass

    raise ValueError(f'Could not load {model_path} as Word2Vec or FastText.')


def read_types(eval_path: str) -> set:
    with open(eval_path, encoding='utf-8') as fh:
        return {
            token
            for line in fh
            for token in line.strip().split()
            if token
        }


def compute_oov_rates(model, eval_path: str) -> tuple:
    """Return (vocab_oov_rate, effective_oov_rate) as percentages.

    These are two different quantities and the distinction is the whole point
    of comparing fastText against word2vec:

    * `vocab_oov_rate` — share of test word types absent from the model's
      vocabulary. This is what the old single `oov_rate` measured.
    * `effective_oov_rate` — share of test word types the model cannot produce
      a vector for at all. For word2vec the two are identical. For fastText
      they are not: character n-grams synthesise a vector for unseen words, so
      its effective rate is near zero, which is the property claude.md's target
      table actually asks for ("fastText OOV rate on test.txt: near 0%").

    Reporting only the vocabulary figure understates fastText's advantage and
    describes it as the wrong kind of win.
    """
    unique_types = read_types(eval_path)
    if not unique_types:
        return 0.0, 0.0

    vocab_oov = sum(1 for t in unique_types if t not in model.wv.key_to_index)

    effective_oov = 0
    for token in unique_types:
        try:
            model.wv[token]
        except KeyError:
            effective_oov += 1

    n = len(unique_types)
    return vocab_oov / n * 100, effective_oov / n * 100


def vectorise_sentences(model, sentences: list, labels: list) -> tuple:
    is_fasttext = type(model).__name__ == 'FastText'
    vectors = []
    kept_labels = []

    for sentence, label in zip(sentences, labels):
        tokens = sentence.strip().split()
        if is_fasttext:
            token_vecs = [model.wv[tok] for tok in tokens if tok]
        else:
            token_vecs = [model.wv[tok] for tok in tokens
                          if tok in model.wv.key_to_index]

        if not token_vecs:
            continue

        vectors.append(np.mean(token_vecs, axis=0))
        kept_labels.append(label)

    return vectors, kept_labels


def assign_genre_labels(sentences: list) -> list:
    n = len(sentences)
    cut1 = int(n * 0.60)
    cut2 = cut1 + int(n * 0.35)

    labels = [0] * cut1 + [1] * (cut2 - cut1) + [2] * (n - cut2)

    print(f'Genre labels assigned: {labels.count(0)} Narrative, '
          f'{labels.count(1)} Epistles, {labels.count(2)} Apocalyptic')
    return labels


def run_genre_classification(model, train_path: str, test_path: str) -> float:
    with open(train_path, encoding='utf-8') as fh:
        train_sentences = [l.rstrip('\n') for l in fh if l.strip()]
    train_labels = assign_genre_labels(train_sentences)
    X_train, y_train = vectorise_sentences(model, train_sentences, train_labels)

    with open(test_path, encoding='utf-8') as fh:
        test_sentences = [l.rstrip('\n') for l in fh if l.strip()]
    test_labels = assign_genre_labels(test_sentences)
    X_test, y_test = vectorise_sentences(model, test_sentences, test_labels)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return float(f1_score(y_test, y_pred, average='macro'))


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/eval_nlp.yaml'
    with open(config_path, encoding='utf-8') as fh:
        cfg = yaml.safe_load(fh)

    models_to_eval = [
        ('word2vec_sg',   cfg['word2vec_sg_path']),
        ('word2vec_cbow', cfg['word2vec_cbow_path']),
        ('fasttext',      cfg['fasttext_path']),
    ]

    results = {}
    rows = []
    skipped = []

    for name, path in models_to_eval:
        print(f'\n=== {name} ===')

        # models/*.bin and *.vec are gitignored, so sunuwar_fasttext.bin is
        # absent from every clone. Aborting the whole run on it means nobody
        # who clones this repo can produce any of these numbers.
        if not Path(path).exists():
            print(f'  WARNING: {path} not found — skipping {name}.')
            print(f'           (Paths in the config are relative, so run from '
                  f'the repository root. models/*.bin and *.vec are gitignored, '
                  f'so sunuwar_fasttext.bin in particular is absent from every '
                  f'clone and has to be retrained with src/train_fasttext.py.)')
            results[name] = {'status': 'skipped', 'reason': f'{path} not found'}
            skipped.append(name)
            continue

        model = load_embedding_model(path)

        # min_count is read off the trained model, not off a config file, so
        # the printed comparison reflects what was actually trained.
        min_count = getattr(model, 'min_count', None)

        vocab_oov, effective_oov = compute_oov_rates(model, cfg['test_path'])
        f1 = run_genre_classification(model, cfg['train_path'], cfg['test_path'])

        print(f'  class:              {type(model).__name__}')
        print(f'  min_count:          {min_count}')
        print(f'  vocab OOV rate:     {vocab_oov:.2f}%   '
              '(test types absent from the vocabulary)')
        print(f'  effective OOV rate: {effective_oov:.2f}%   '
              '(test types the model cannot vectorise at all)')
        print(f'  Genre F1 macro:     {f1:.4f}')

        results[name] = {
            'model_class':        type(model).__name__,
            'min_count':          min_count,
            'vocab_oov_rate':     round(vocab_oov, 4),
            'effective_oov_rate': round(effective_oov, 4),
            'genre_f1_macro':     round(f1, 4),
        }
        rows.append((name, min_count, vocab_oov, effective_oov, f1))

    Path(cfg['output_path']).parent.mkdir(parents=True, exist_ok=True)
    with open(cfg['output_path'], 'w', encoding='utf-8') as fh:
        json.dump(results, fh, indent=2)
    print(f'\nResults saved to {cfg["output_path"]}')

    print()
    print(f'{"model":<16}  {"min_count":>9}  {"vocab_oov":>10}  '
          f'{"effective_oov":>13}  {"genre_f1_macro":>14}')
    print('-' * 72)
    for name, min_count, vocab_oov, effective_oov, f1 in rows:
        print(f'{name:<16}  {str(min_count):>9}  {vocab_oov:>9.2f}%  '
              f'{effective_oov:>12.2f}%  {f1:>14.4f}')

    if skipped:
        print(f'\nSkipped (model file absent): {", ".join(skipped)}')

    print(
        '\nNote: min_count is printed because the models do not share it. Any '
        '\nvocab_oov gap between fastText and word2vec is confounded by that '
        '\ndifference and cannot be attributed to subword n-grams until it is '
        '\ncontrolled for. See NLP_PIPELINE.md Step 4.'
    )


if __name__ == '__main__':
    main()
