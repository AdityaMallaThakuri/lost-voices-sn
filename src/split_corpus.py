import random
import sys
import unicodedata
from pathlib import Path

import yaml

SEED = 42


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/split.yaml'
    with open(config_path, encoding='utf-8') as fh:
        cfg = yaml.safe_load(fh)

    input_path  = cfg['input_path']
    train_path  = cfg['train_path']
    test_path   = cfg['test_path']
    test_ratio  = cfg['test_ratio']

    with open(input_path, encoding='utf-8') as fh:
        sentences = [unicodedata.normalize('NFC', line.rstrip('\n')) for line in fh if line.strip()]

    random.seed(SEED)
    random.shuffle(sentences)

    split = int(len(sentences) * (1 - test_ratio))
    train = sentences[:split]
    test  = sentences[split:]

    Path(train_path).parent.mkdir(parents=True, exist_ok=True)
    Path(test_path).parent.mkdir(parents=True, exist_ok=True)

    with open(train_path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(train) + '\n')

    with open(test_path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(test) + '\n')

    print(f'Total sentences:  {len(sentences)}')
    print(f'Train sentences:  {len(train)}  ({len(train)/len(sentences)*100:.1f}%)')
    print(f'Test  sentences:  {len(test)}   ({len(test)/len(sentences)*100:.1f}%)')
    print(f'Train -> {train_path}')
    print(f'Test  -> {test_path}')


if __name__ == '__main__':
    main()
