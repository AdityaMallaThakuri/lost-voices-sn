# Data source: Sunuwar Bible (suzBl), © 2011 Wycliffe Bible Translators, Inc.
# Licence: CC BY-NC-ND 4.0 — non-commercial research use only
# Do not redistribute raw text. Models trained on this data may be released.

import time
import unicodedata
from pathlib import Path

from gensim.models import FastText


def train_fasttext(input_path: str, model_dir: str, config: dict) -> FastText:
    with open(input_path, encoding='utf-8') as fh:
        sentences = [
            unicodedata.normalize('NFC', line).split()
            for line in fh
            if line.strip()
        ]

    model_dir_path = Path(model_dir)
    model_dir_path.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    model = FastText(
        sentences=sentences,
        vector_size=config['vector_size'],
        window=config['window'],
        min_count=config['min_count'],
        min_n=config['minn'],
        max_n=config['maxn'],
        negative=config['negative'],
        epochs=config['epochs'],
        seed=config['seed'],
        workers=config['workers'],
    )

    bin_path = model_dir_path / 'sunuwar_fasttext.bin'
    vec_path = model_dir_path / 'sunuwar_fasttext.vec'
    model.save(str(bin_path))
    model.wv.save_word2vec_format(str(vec_path))
    print(f'fastText trained in {time.time() - t0:.1f}s  ->  {bin_path}')
    print(f'Word vectors saved              ->  {vec_path}')

    return model


def print_nearest_neighbours(model, probe_words: list, model_name: str) -> None:
    print(f'\n--- Nearest neighbours ({model_name}) ---')
    for word in probe_words:
        neighbours = model.wv.most_similar(word, topn=10)
        print(f'  {word}:')
        for neighbour, score in neighbours:
            print(f'    {score:.4f}  {neighbour}')


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


def main() -> None:
    import sys
    import yaml

    config_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/fasttext.yaml'
    with open(config_path, encoding='utf-8') as fh:
        cfg = yaml.safe_load(fh)

    model = train_fasttext(cfg['input_path'], cfg['model_dir'], cfg)

    print_nearest_neighbours(model, cfg['probe_words'], 'fastText')

    ft_oov = compute_oov_rate(model, cfg['eval_path'])

    W2V_OOV = 30.90

    print()
    print(f'{"model":<18}  {"vocab_oov_rate":>14}  {"can_encode_oov":<18}')
    print('-' * 56)
    print(f'{"word2vec SG":<18}  {W2V_OOV:>13.2f}%  {"no":<18}')
    print(f'{"word2vec CBOW":<18}  {W2V_OOV:>13.2f}%  {"no":<18}')
    print(f'{"fastText":<18}  {ft_oov:>13.2f}%  {"yes (subword)":<18}')


if __name__ == '__main__':
    main()
