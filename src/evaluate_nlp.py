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


def compute_oov_rate(model, eval_path: str) -> float:
    with open(eval_path, encoding='utf-8') as fh:
        unique_types = {
            token
            for line in fh
            for token in line.strip().split()
            if token
        }
    if not unique_types:
        return 0.0
    oov_count = sum(1 for t in unique_types if t not in model.wv.key_to_index)
    return oov_count / len(unique_types) * 100


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

    for name, path in models_to_eval:
        print(f'\n=== {name} ===')
        model   = load_embedding_model(path)
        oov     = compute_oov_rate(model, cfg['test_path'])
        f1      = run_genre_classification(model, cfg['train_path'], cfg['test_path'])
        print(f'  OOV rate:       {oov:.2f}%')
        print(f'  Genre F1 macro: {f1:.4f}')
        results[name] = {'oov_rate': round(oov, 4), 'genre_f1_macro': round(f1, 4)}
        rows.append((name, oov, f1))

    Path(cfg['output_path']).parent.mkdir(parents=True, exist_ok=True)
    with open(cfg['output_path'], 'w', encoding='utf-8') as fh:
        json.dump(results, fh, indent=2)
    print(f'\nResults saved to {cfg["output_path"]}')

    print()
    print(f'{"model":<18}  {"oov_rate":>10}  {"genre_f1_macro":>14}')
    print('-' * 48)
    for name, oov, f1 in rows:
        print(f'{name:<18}  {oov:>9.2f}%  {f1:>14.4f}')


# Assertion tests
_m = load_embedding_model('models/sunuwar_w2v_sg.model')
assert hasattr(_m, 'wv'), 'loaded model must have a .wv attribute'

_oov = compute_oov_rate(_m, 'data/processed/test.txt')
assert isinstance(_oov, float) and 0.0 <= _oov <= 100.0, f'OOV rate out of range: {_oov}'

_dummy_sents = ['x'] * 100
_labels = assign_genre_labels(_dummy_sents)
assert _labels.count(0) == 60 and _labels.count(1) == 35 and _labels.count(2) == 5, \
    f'Label counts wrong: {_labels.count(0)}/{_labels.count(1)}/{_labels.count(2)}'


if __name__ == '__main__':
    main()
