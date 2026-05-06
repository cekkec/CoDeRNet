import torch
from torch.utils.data import Dataset
import json
import os
import numpy as np
import random as pyrandom
from random import randint, random, uniform
from imageio import imread


def encode(seq_tokens, token_to_idx, allow_unk=False):
    seq_idx = []
    for token in seq_tokens:
        if token not in token_to_idx:
            if allow_unk:
                token = '<UNK>'
            else:
                raise KeyError('Token "%s" not in vocab' % token)
        seq_idx.append(token_to_idx[token])
    return seq_idx


# ═══════════════════════════════════════════════════════════
#  Geometric Augmentation (returns aug_info for caption remap)
# ═══════════════════════════════════════════════════════════

def augment_geometric(imgA, imgB, seg_label):
    """Geometric augmentation with transform record for caption remap."""
    aug = {"hflip": False, "vflip": False, "rot_k": 0}

    if random() > 0.5:
        imgA = np.flip(imgA, axis=1).copy()
        imgB = np.flip(imgB, axis=1).copy()
        seg_label = np.flip(seg_label, axis=1).copy()
        aug["hflip"] = True

    if random() > 0.5:
        imgA = np.flip(imgA, axis=0).copy()
        imgB = np.flip(imgB, axis=0).copy()
        seg_label = np.flip(seg_label, axis=0).copy()
        aug["vflip"] = True

    if random() > 0.5:
        k = randint(1, 3)
        imgA = np.rot90(imgA, k, axes=(0, 1)).copy()
        imgB = np.rot90(imgB, k, axes=(0, 1)).copy()
        seg_label = np.rot90(seg_label, k, axes=(0, 1)).copy()
        aug["rot_k"] = k

    return imgA, imgB, seg_label, aug


# ═══════════════════════════════════════════════════════════
#  Caption Token Remap (direction words adjusted for geo aug)
# ═══════════════════════════════════════════════════════════

def _swap_tokens(tokens, mapping_2tok, mapping_1tok):
    out = list(tokens)
    n = len(out)

    used = [False] * n
    for i in range(n - 1):
        pair = (out[i], out[i + 1])
        if pair in mapping_2tok and not used[i] and not used[i + 1]:
            new_pair = mapping_2tok[pair]
            out[i], out[i + 1] = new_pair
            used[i] = used[i + 1] = True

    for i in range(n):
        if not used[i] and out[i] in mapping_1tok:
            out[i] = mapping_1tok[out[i]]

    return out


def apply_hflip(tokens):
    map2 = {
        ("upper", "left"): ("upper", "right"), ("upper", "right"): ("upper", "left"),
        ("lower", "left"): ("lower", "right"), ("lower", "right"): ("lower", "left"),
        ("top", "left"): ("top", "right"), ("top", "right"): ("top", "left"),
        ("bottom", "left"): ("bottom", "right"), ("bottom", "right"): ("bottom", "left"),
    }
    map1 = {"left": "right", "right": "left"}
    return _swap_tokens(tokens, map2, map1)


def apply_vflip(tokens):
    map2 = {
        ("upper", "left"): ("lower", "left"), ("lower", "left"): ("upper", "left"),
        ("upper", "right"): ("lower", "right"), ("lower", "right"): ("upper", "right"),
        ("top", "left"): ("bottom", "left"), ("bottom", "left"): ("top", "left"),
        ("top", "right"): ("bottom", "right"), ("bottom", "right"): ("top", "right"),
    }
    map1 = {"top": "bottom", "bottom": "top", "upper": "lower", "lower": "upper"}
    return _swap_tokens(tokens, map2, map1)


def apply_rot90_ccw(tokens):
    map2 = {
        ("upper", "left"): ("lower", "left"), ("lower", "left"): ("lower", "right"),
        ("lower", "right"): ("upper", "right"), ("upper", "right"): ("upper", "left"),
        ("top", "left"): ("bottom", "left"), ("bottom", "left"): ("bottom", "right"),
        ("bottom", "right"): ("top", "right"), ("top", "right"): ("top", "left"),
    }
    map1 = {
        "top": "left", "left": "bottom", "bottom": "right", "right": "top",
        "upper": "left", "lower": "right",
    }
    return _swap_tokens(tokens, map2, map1)


def remap_caption_tokens(tokens, aug):
    out = list(tokens)
    if aug["hflip"]:
        out = apply_hflip(out)
    if aug["vflip"]:
        out = apply_vflip(out)
    for _ in range(aug["rot_k"] % 4):
        out = apply_rot90_ccw(out)
    return out


# ═══════════════════════════════════════════════════════════
#  Dataset
# ═══════════════════════════════════════════════════════════

class LEVIRCCDataset(Dataset):
    def __init__(self, data_folder, list_path, split, token_folder=None,
                 vocab_file=None, max_length=41, allow_unk=0, max_iters=None,
                 base_seed=None, deterministic_data=False):
        self.mean = [0.39073 * 255, 0.38623 * 255, 0.32989 * 255]
        self.std = [0.15329 * 255, 0.14628 * 255, 0.13648 * 255]
        self.list_path = list_path
        self.split = split
        self.max_length = max_length
        # Geometric augmentation (flip + 90deg rotation) is applied only at
        # train time; caption direction tokens are remapped accordingly.
        self.augment = (split == 'train')
        self.base_seed = base_seed
        self.deterministic_data = deterministic_data
        self._epoch = 0

        assert self.split in {'train', 'val', 'test'}
        self.img_ids = [i_id.strip() for i_id in open(os.path.join(list_path + split + '.txt'))]

        if vocab_file is not None:
            with open(os.path.join(list_path + vocab_file + '.json'), 'r') as f:
                self.word_vocab = json.load(f)
            self.allow_unk = allow_unk

        if max_iters is not None:
            n_repeat = int(np.ceil(max_iters / len(self.img_ids)))
            self.img_ids = self.img_ids * n_repeat + self.img_ids[:max_iters - n_repeat * len(self.img_ids)]

        self.files = []
        for name in self.img_ids:
            if split == 'train':
                img_fileA = os.path.join(data_folder, split, 'A', name.split('-')[0])
            else:
                img_fileA = os.path.join(data_folder, split, 'A', name)
            img_fileB = img_fileA.replace('/A/', '/B/')
            seg_label_file = img_fileA.replace('/A/', '/label/')

            imgA = imread(img_fileA)
            imgB = imread(img_fileB)
            seg_label = imread(seg_label_file)

            token_id = name.split('-')[-1] if '-' in name else None
            token_file = None
            if token_folder is not None:
                token_file = os.path.join(token_folder, name.split('.')[0] + '.txt')

            self.files.append({
                "imgA": imgA, "imgB": imgB, "seg_label": seg_label,
                "token": token_file, "token_id": token_id,
                "name": name.split('-')[0] if split == 'train' else name
            })

    def set_epoch(self, epoch):
        self._epoch = epoch

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        if self.deterministic_data and self.base_seed is not None:
            # Per-sample seeding: deterministic + epoch-varying augmentation
            seed = int(self.base_seed) + self._epoch * len(self.files) + int(index)
            pyrandom.seed(seed)
            np.random.seed(seed % (2**32))

        datafiles = self.files[index]
        name = datafiles["name"]

        imgA = np.asarray(datafiles["imgA"], np.float32)
        imgB = np.asarray(datafiles["imgB"], np.float32)
        seg_label = datafiles["seg_label"].copy()

        if seg_label.ndim == 3:
            seg_label = seg_label[:, :, 0]

        # Geometric augmentation (train only). aug_info drives caption remap.
        aug_info = {"hflip": False, "vflip": False, "rot_k": 0}
        if self.augment:
            imgA, imgB, seg_label, aug_info = augment_geometric(imgA, imgB, seg_label)

        # Label mapping
        seg_label = seg_label.copy()
        seg_label[seg_label == 255] = 2
        seg_label[seg_label == 128] = 1

        # HWC -> CHW + normalize
        imgA = imgA.transpose(2, 0, 1)
        imgB = imgB.transpose(2, 0, 1)
        for i in range(len(self.mean)):
            imgA[i, :, :] -= self.mean[i]
            imgA[i, :, :] /= self.std[i]
            imgB[i, :, :] -= self.mean[i]
            imgB[i, :, :] /= self.std[i]

        if datafiles["token"] is not None:
            caption = open(datafiles["token"]).read()
            caption_list = json.loads(caption)

            token_all = np.zeros((len(caption_list), self.max_length), dtype=int)
            token_all_len = np.zeros((len(caption_list), 1), dtype=int)
            for j, tokens in enumerate(caption_list):
                nochange_cap = ['<START>', 'the', 'scene', 'is', 'the', 'same', 'as', 'before', '<END>']
                if self.split == 'train' and nochange_cap in caption_list:
                    tokens = nochange_cap

                # Caption geo remap (direction words adjusted for augmentation)
                if self.augment:
                    tokens = remap_caption_tokens(tokens, aug_info)

                tokens_encode = encode(tokens, self.word_vocab, allow_unk=self.allow_unk == 1)
                token_all[j, :len(tokens_encode)] = tokens_encode
                token_all_len[j] = len(tokens_encode)

            if datafiles["token_id"] is not None:
                id = int(datafiles["token_id"])
                token = token_all[id]
                token_len = token_all_len[id].item()
            else:
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
