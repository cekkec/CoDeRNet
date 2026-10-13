"""Prepare the LEVIR-MCI caption files used by train.py / test.py.

Reads LevirCCcaptions.json (shipped with the LEVIR-MCI dataset) and writes the
files expected by --list_path / --token_folder:

    <out_dir>/
    ├── train.txt, val.txt, test.txt   # image file names per split
    ├── vocab.json                     # {token: index}
    └── tokens/<image>.txt             # the 5 tokenized captions of each image (JSON list)

Tokenization and vocabulary construction follow Change-Agent
(https://github.com/Chen-Yang-Liu/Change-Agent, preprocess_data.py), so the
generated files match the ones used to train the released checkpoints.

Usage:
    python preprocess_levir_mci.py \
        --captions_json <LEVIR_MCI_ROOT>/LevirCCcaptions.json \
        --out_dir ./data/LEVIR_MCI
"""
import argparse
import json
import os

SPECIAL_TOKENS = {'<NULL>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3}
PUNCT_TO_KEEP = [';', ',']      # kept as separate tokens
PUNCT_TO_REMOVE = ['?', '.']    # dropped


def tokenize(sentence):
    """Split a raw caption on spaces, keep ';' and ',' as tokens, drop '?' and '.',
    and wrap the result with <START> / <END>."""
    s = sentence
    for p in PUNCT_TO_KEEP:
        s = s.replace(p, ' ' + p)
    for p in PUNCT_TO_REMOVE:
        s = s.replace(p, '')
    tokens = [t for t in s.split(' ') if t != '']
    return ['<START>'] + tokens + ['<END>']


def build_vocab(all_tokens, min_count):
    """Special tokens first, then every word occurring more than `min_count`
    times, in alphabetical order."""
    counts = {}
    for tokens_list in all_tokens:
        for tokens in tokens_list:
            for t in tokens:
                counts[t] = counts.get(t, 0) + 1
    vocab = dict(SPECIAL_TOKENS)
    for t, c in sorted(counts.items()):
        if t not in vocab and c > min_count:
            vocab[t] = len(vocab)
    return vocab


def main(args):
    with open(args.captions_json, 'r') as f:
        data = json.load(f)

    token_dir = os.path.join(args.out_dir, 'tokens')
    os.makedirs(token_dir, exist_ok=True)

    split_lists = {'train': [], 'val': [], 'test': []}
    all_tokens = []
    max_len = 0
    for img in data['images']:
        name = img['filename']
        tokens_list = [tokenize(s['raw']) for s in img['sentences']]
        all_tokens.append(tokens_list)
        max_len = max(max_len, max(len(t) for t in tokens_list))

        with open(os.path.join(token_dir, name.split('.')[0] + '.txt'), 'w') as f:
            f.write(json.dumps(tokens_list))
        split_lists[img['split']].append(name)

    for split, names in split_lists.items():
        with open(os.path.join(args.out_dir, split + '.txt'), 'w') as f:
            f.write(''.join(n + '\n' for n in names))

    vocab = build_vocab(all_tokens, args.word_count_threshold)
    with open(os.path.join(args.out_dir, 'vocab.json'), 'w') as f:
        json.dump(vocab, f)

    print(f'images: ' + ', '.join(f'{k}={len(v)}' for k, v in split_lists.items()))
    print(f'vocab size: {len(vocab)}  (words with count > {args.word_count_threshold} + 4 special tokens)')
    print(f'longest caption: {max_len} tokens incl. <START>/<END>  (--max_length must be >= this; default 41)')
    print(f'written to {args.out_dir}/  ->  pass it as --list_path')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LEVIR-MCI caption preprocessing for CoDeRNet')
    parser.add_argument('--captions_json', required=True,
                        help='path to LevirCCcaptions.json from the LEVIR-MCI dataset')
    parser.add_argument('--out_dir', default='./data/LEVIR_MCI',
                        help='output folder (becomes --list_path for train.py / test.py)')
    parser.add_argument('--word_count_threshold', type=int, default=5,
                        help='a word enters the vocabulary if it occurs more than this many times')
    main(parser.parse_args())
