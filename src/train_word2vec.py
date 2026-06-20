# Data source: Sunuwar Bible (suzBl), © 2011 Wycliffe Bible Translators, Inc.
# Licence: CC BY-NC-ND 4.0 — non-commercial research use only
# Do not redistribute raw text. Models trained on this data may be released.

import time
import unicodedata
from pathlib import Path

from gensim.models import Word2Vec


def train_word2vec(input_path: str, model_dir: str, config: dict) -> tuple:
    with open(input_path, encoding='utf-8') as fh:
        sentences = [
            unicodedata.normalize('NFC', line).split()
            for line in fh
            if line.strip()
        ]

    model_dir_path = Path(model_dir)
    model_dir_path.mkdir(parents=True, exist_ok=True)

    shared_kwargs = dict(
        sentences=sentences,
        vector_size=config['vector_size'],
        window=config['window'],
        min_count=config['min_count'],
        negative=config['negative'],
        sample=config['sample'],
        epochs=config['epochs'],
        seed=config['seed'],
        workers=config['workers'],
    )

    t0 = time.time()
    sg_model = Word2Vec(**shared_kwargs, sg=1)
    sg_path = model_dir_path / 'sunuwar_w2v_sg.model'
    sg_model.save(str(sg_path))
    print(f'Skip-gram trained in {time.time() - t0:.1f}s  ->  {sg_path}')

    t0 = time.time()
    cbow_model = Word2Vec(**shared_kwargs, sg=0)
    cbow_path = model_dir_path / 'sunuwar_w2v_cbow.model'
    cbow_model.save(str(cbow_path))
    print(f'CBOW      trained in {time.time() - t0:.1f}s  ->  {cbow_path}')

    return sg_model, cbow_model


def print_nearest_neighbours(model: Word2Vec, probe_words: list, model_name: str) -> None:
    print(f'\n--- Nearest neighbours ({model_name}) ---')
    for word in probe_words:
        if word in model.wv:
            neighbours = model.wv.most_similar(word, topn=10)
            print(f'  {word}:')
            for neighbour, score in neighbours:
                print(f'    {score:.4f}  {neighbour}')
        else:
            print(f'  [OOV] {word}')


def compute_oov_rate(model: Word2Vec, eval_path: str) -> float:
    with open(eval_path, encoding='utf-8') as fh:
        unique_types = {
            token
            for line in fh
            for token in line.strip().split()
            if token
        }
    if not unique_types:
        return 0.0
    oov_count = sum(1 for t in unique_types if t not in model.wv)
    return oov_count / len(unique_types) * 100


def main() -> None:
    import sys
    import yaml

    config_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/word2vec.yaml'
    with open(config_path, encoding='utf-8') as fh:
        cfg = yaml.safe_load(fh)

    sg_model, cbow_model = train_word2vec(cfg['input_path'], cfg['model_dir'], cfg)

    print_nearest_neighbours(sg_model,   cfg['probe_words'], 'Skip-gram')
    print_nearest_neighbours(cbow_model, cfg['probe_words'], 'CBOW')

    sg_oov   = compute_oov_rate(sg_model,   cfg['eval_path'])
    cbow_oov = compute_oov_rate(cbow_model, cfg['eval_path'])

    print('\n--- OOV rate summary ---')
    print(f'{"model":<12}  {"oov_rate":>10}')
    print('-' * 26)
    print(f'{"skip-gram":<12}  {sg_oov:>9.2f}%')
    print(f'{"cbow":<12}  {cbow_oov:>9.2f}%')


if __name__ == '__main__':
    main()
