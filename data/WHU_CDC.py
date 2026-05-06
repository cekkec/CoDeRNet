import torch
from torch.utils.data import Dataset
import json
import os
import numpy as np
import random as pyrandom
from random import randint, random, uniform
from imageio import imread

from .LEVIR_MCI import (
    encode, augment_geometric, remap_caption_tokens,
)


def build_whu_vocab(caption_json_path, split='train'):
    """Build vocab from WHU-CDC caption JSON (training split only). Returns {token: idx} dict."""
    with open(caption_json_path, 'r') as f:
        data = json.load(f)

    word_set = set()
    for img_entry in data['images']:
        if img_entry['filepath'] != split:
            continue
        for sent in img_entry['sentences']:
            for tok in sent['tokens']:
                word_set.add(tok.lower())

    # Same special token convention as LEVIR-MCI
    vocab = {'<NULL>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3}
    for i, w in enumerate(sorted(word_set), start=4):
        vocab[w] = i

    return vocab


class WHUCDCDataset(Dataset):
    """WHU-CDC dataset for change detection + change captioning.

    Key differences from LEVIR-MCI:
      - Captions stored in a single JSON (whuCCcaptions.json), not per-image .txt
      - Labels are bool (2-class: no-change / change), not 3-class
      - Tokens in JSON don't include <START>/<END> — we add them
      - File list comes from JSON split field, not .txt files
    """
    def __init__(self, data_folder, caption_json_path, split,
                 word_vocab=None, max_length=26, allow_unk=0,
                 base_seed=None, deterministic_data=False):
        self.split = split
        self.max_length = max_length
        # Geometric augmentation (flip + 90deg rotation) is applied only at
        # train time; caption direction tokens are remapped accordingly.
        self.augment = (split == 'train')
        self.base_seed = base_seed
        self.deterministic_data = deterministic_data
        self._epoch = 0
        self.word_vocab = word_vocab
        self.allow_unk = allow_unk

        assert self.split in {'train', 'val', 'test'}

        # Load caption JSON
        with open(caption_json_path, 'r') as f:
            all_data = json.load(f)

        # Filter by split
        self.files = []
        for entry in all_data['images']:
            if entry['filepath'] != split:
                continue

            fname = entry['filename']
            img_fileA = os.path.join(data_folder, split, 'A', fname)
            img_fileB = os.path.join(data_folder, split, 'B', fname)
            seg_label_file = os.path.join(data_folder, split, 'label', fname)

            imgA = imread(img_fileA)
            imgB = imread(img_fileB)
            seg_label = imread(seg_label_file)

            # Build token lists: add <START> and <END>
            captions = []
            for sent in entry['sentences']:
                tokens = ['<START>'] + [t.lower() for t in sent['tokens']] + ['<END>']
                captions.append(tokens)

            self.files.append({
                "imgA": imgA, "imgB": imgB, "seg_label": seg_label,
                "captions": captions, "name": fname,
            })

        # Compute mean/std (WHU-CDC specific — placeholder, will compute below)
        self._compute_stats()

    def _compute_stats(self):
        """Use precomputed WHU-CDC mean/std (computed over training set)."""
        # Computed from WHU-CDC training set (A+B images)
        self.mean = [0.48598 * 255, 0.46535 * 255, 0.42587 * 255]
        self.std = [0.19541 * 255, 0.18924 * 255, 0.20196 * 255]

    def set_epoch(self, epoch):
        self._epoch = epoch

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        if self.deterministic_data and self.base_seed is not None:
            seed = int(self.base_seed) + self._epoch * len(self.files) + int(index)
            pyrandom.seed(seed)
            np.random.seed(seed % (2**32))

        datafiles = self.files[index]
        name = datafiles["name"]

        imgA = np.asarray(datafiles["imgA"], np.float32)
        imgB = np.asarray(datafiles["imgB"], np.float32)
        seg_label = datafiles["seg_label"].copy()

        # bool -> uint8 (False=0, True=1)
        seg_label = seg_label.astype(np.uint8)

        if seg_label.ndim == 3:
            seg_label = seg_label[:, :, 0]

        # Geometric augmentation (train only). aug_info drives caption remap.
        aug_info = {"hflip": False, "vflip": False, "rot_k": 0}
        if self.augment:
            imgA, imgB, seg_label, aug_info = augment_geometric(imgA, imgB, seg_label)

        # HWC -> CHW + normalize
        imgA = imgA.transpose(2, 0, 1)
        imgB = imgB.transpose(2, 0, 1)
        for i in range(len(self.mean)):
            imgA[i, :, :] -= self.mean[i]
            imgA[i, :, :] /= self.std[i]
            imgB[i, :, :] -= self.mean[i]
            imgB[i, :, :] /= self.std[i]

        # Captions
        if self.word_vocab is not None:
            caption_list = datafiles["captions"]

            token_all = np.zeros((len(caption_list), self.max_length), dtype=int)
            token_all_len = np.zeros((len(caption_list), 1), dtype=int)

            for j, tokens in enumerate(caption_list):
                if self.augment:
                    tokens = remap_caption_tokens(tokens, aug_info)

                tokens_encode = encode(tokens, self.word_vocab,
                                       allow_unk=self.allow_unk == 1)
                clip_len = min(len(tokens_encode), self.max_length)
                token_all[j, :clip_len] = tokens_encode[:clip_len]
                token_all_len[j] = clip_len

            j = randint(0, len(caption_list) - 1)
            token = token_all[j]
            token_len = token_all_len[j].item()
        else:
            token_all = np.zeros(1, dtype=int)
            token = np.zeros(1, dtype=int)
            token_len = np.zeros(1, dtype=int)
            token_all_len = np.zeros(1, dtype=int)

        return (imgA.copy(), imgB.copy(), seg_label.copy(),
                token_all.copy(), token_all_len.copy(),
                token.copy(), np.array(token_len), name)
