"""
CoDeRNet test/inference entry point — supports both LEVIR-MCI and WHU-CDC.

Usage:
  python test.py --dataset levir --checkpoint .../best_*.pth [--save_pred]
  python test.py --dataset whu   --checkpoint .../best_*.pth [--save_pred]

When --save_pred is set, per-image predicted masks (PNG) and predicted/reference
captions (TXT) are written under --result_path. Aggregate metrics (mIoU + the
captioning suite) are always printed at the end.
"""
import cv2
import torch
from torch.utils import data
import argparse
import json
import numpy as np
import os
import time
from tqdm import tqdm

from data.LEVIR_MCI import LEVIRCCDataset
from data.WHU_CDC import WHUCDCDataset, build_whu_vocab
from model.change_model import ChangeModel
from utils_tool.utils import *
from utils_tool.metrics import Evaluator


def seg_to_color(seg, num_classes=3):
    """Returns BGR-ordered array (for cv2.imwrite).
    LEVIR-MCI (3-class): class 1 (Road) -> Yellow, class 2 (Building) -> Red.
    WHU-CDC  (2-class):   class 1 (Change) -> White.
    """
    h, w = seg.shape
    bgr = np.zeros((h, w, 3), dtype=np.uint8)
    if num_classes == 3:
        bgr[seg == 1] = [0, 255, 255]
        bgr[seg == 2] = [0, 0, 255]
    else:
        bgr[seg == 1] = [255, 255, 255]
    return bgr


def main(args):
    # Vocab
    if args.dataset == 'levir':
        with open(os.path.join(args.list_path, args.vocab_file + '.json'), 'r') as f:
            word_vocab = json.load(f)
    else:  # whu
        word_vocab = build_whu_vocab(args.caption_json)
    idx_to_word = list(word_vocab.keys())

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model = ChangeModel(args, word_vocab)
    model.load_state_dict(checkpoint['model_dict'])
    model = model.cuda()
    model.eval()

    result_dir = None
    if args.save_pred:
        result_dir = os.path.join(args.result_path,
                                  os.path.basename(args.checkpoint).replace('.pth', ''))
        for sub in ['masks', 'captions']:
            os.makedirs(os.path.join(result_dir, sub), exist_ok=True)

    nochange_list = [
        "the scene is the same as before ",
        "there is no difference ",
        "the two scenes seem identical ",
        "no change has occurred ",
        "almost nothing has changed "
    ]

    # Dataset
    if args.dataset == 'levir':
        test_ds = LEVIRCCDataset(args.data_folder, args.list_path, 'test',
                                  args.token_folder, args.vocab_file,
                                  args.max_length, args.allow_unk)
    else:  # whu
        test_ds = WHUCDCDataset(args.data_folder, args.caption_json, 'test',
                                 word_vocab=word_vocab, max_length=args.max_length,
                                 allow_unk=args.allow_unk)

    test_loader = data.DataLoader(test_ds, batch_size=1, shuffle=False,
                                   num_workers=args.workers, pin_memory=True)

    references, hypotheses = [], []
    change_refs, change_hyps = [], []
    nochange_refs, nochange_hyps = [], []
    change_acc = nochange_acc = 0
    evaluator = Evaluator(num_class=args.num_classes)
    t0 = time.time()

    with torch.no_grad():
        for imgA, imgB, seg_label, token_all, _, _, _, name in tqdm(test_loader, desc='Testing'):
            imgA_cuda, imgB_cuda = imgA.cuda(), imgB.cuda()
            token_all = token_all.squeeze(0).cuda()
            sample_name = name[0]
            key = sample_name.split('.')[0]

            seg_out, _ = model(imgA_cuda, imgB_cuda, train_goal=0,
                               target_size=seg_label.shape[-2:])
            seq = model.sample_caption(imgA_cuda, imgB_cuda)

            pred_seg = np.argmax(seg_out.cpu().numpy(), axis=1)
            gt_seg = seg_label.numpy()
            evaluator.add_batch(gt_seg, pred_seg)

            img_tokens = [
                [w for w in c if w not in {word_vocab['<START>'], word_vocab['<END>'], word_vocab['<NULL>']}]
                for c in token_all.tolist()]
            references.append(img_tokens)
            pred_seq = [w for w in seq if w not in {
                word_vocab['<START>'], word_vocab['<END>'], word_vocab['<NULL>']}]
            hypotheses.append(pred_seq)

            pred_caption = " ".join([idx_to_word[i] for i in pred_seq])
            ref_caption = " ".join([idx_to_word[i] for i in img_tokens[0]])

            if ref_caption in nochange_list:
                nochange_refs.append(img_tokens)
                nochange_hyps.append(pred_seq)
                if pred_caption in nochange_list: nochange_acc += 1
            else:
                change_refs.append(img_tokens)
                change_hyps.append(pred_seq)
                if pred_caption not in nochange_list: change_acc += 1

            # Save per-image predictions (mask PNG + caption TXT)
            if args.save_pred and result_dir:
                cv2.imwrite(os.path.join(result_dir, 'masks', key + '.png'),
                            seg_to_color(pred_seg[0], args.num_classes))
                ref_all = ""
                for tl in img_tokens:
                    ref_all += " ".join([idx_to_word[i] for i in tl]) + " .  "
                with open(os.path.join(result_dir, 'captions', key + '_cap.txt'), 'w') as f:
                    f.write(f'pred: {pred_caption}\nref:  {ref_all}\n')

    test_time = time.time() - t0

    if args.dataset == 'whu':
        mIoU = evaluator.Change_IoU()
    else:
        mIoU, _ = evaluator.Mean_Intersection_over_Union()
    sc = get_eval_score(references, hypotheses)

    iou_label = "cIoU" if args.dataset == 'whu' else "mIoU"
    print(f'\n{iou_label:<8} {"B1":<8} {"B2":<8} {"B3":<8} {"B4":<8} {"M":<8} {"R":<8} {"C":<8}')
    print(f'{mIoU*100:<8.2f} {sc["Bleu_1"]*100:<8.2f} {sc["Bleu_2"]*100:<8.2f} '
          f'{sc["Bleu_3"]*100:<8.2f} {sc["Bleu_4"]*100:<8.2f} {sc["METEOR"]*100:<8.2f} '
          f'{sc["ROUGE_L"]*100:<8.2f} {sc["CIDEr"]*100:<8.2f}')
    print(f'\nTime: {test_time:.1f}s')

    if args.save_pred and result_dir:
        print(f'Predictions saved to: {result_dir}/')


def run_test(args):
    torch.cuda.set_device(args.gpu_id)
    main(args)


# ────────────────────────────────────────────────────────────────────
# CLI entry point (LEVIR-MCI / WHU-CDC unified)
# ────────────────────────────────────────────────────────────────────

_LEVIR_DEFAULTS = {
    'data_folder':  '../images',
    'list_path':    '../LEVIR_MCI/',
    'token_folder': '../LEVIR_MCI/tokens',
    'caption_json': None,
    'max_length':   41,
    'num_classes':  3,
}

_WHU_DEFAULTS = {
    'data_folder':  '../whu_CDC_dataset/images',
    'list_path':    None,
    'token_folder': None,
    'caption_json': '../whu_CDC_dataset/whuCCcaptions.json',
    'max_length':   26,
    'num_classes':  2,
}


def _parse_test_args():
    parser = argparse.ArgumentParser(description='CoDeRNet test (LEVIR-MCI / WHU-CDC)')

    # Dataset switch
    parser.add_argument('--dataset', choices=['levir', 'whu'], required=True)

    # Data (defaults filled per --dataset)
    parser.add_argument('--data_folder',  default=None)
    parser.add_argument('--list_path',    default=None)
    parser.add_argument('--token_folder', default=None)
    parser.add_argument('--vocab_file',   default='vocab')
    parser.add_argument('--caption_json', default=None)
    parser.add_argument('--max_length',   type=int, default=None)
    parser.add_argument('--num_classes',  type=int, default=None)
    parser.add_argument('--allow_unk',    type=int, default=1)
    parser.add_argument('--gpu_id',       type=int, default=0)

    # Model (must match training config)
    parser.add_argument('--backbone',         default='tu-convnext_base')
    parser.add_argument('--d_model',          type=int, default=256)
    parser.add_argument('--n_heads',          type=int, default=8)
    parser.add_argument('--decoder_n_layers', type=int, default=1)
    parser.add_argument('--dropout',          type=float, default=0.1)
    parser.add_argument('--workers',          type=int, default=4)

    # Test-specific
    parser.add_argument('--checkpoint',  required=True)
    parser.add_argument('--result_path', default='./predict_result/')
    parser.add_argument('--save_pred',   action='store_true', default=False,
                        help='Save per-image predicted masks (PNG) and captions (TXT).')

    args = parser.parse_args()
    _apply_dataset_defaults(args)
    return args


def _apply_dataset_defaults(args):
    defaults = _LEVIR_DEFAULTS if args.dataset == 'levir' else _WHU_DEFAULTS
    for key, value in defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)


if __name__ == '__main__':
    args = _parse_test_args()
    run_test(args)
