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
from utils_tool.utils import get_eval_score
from utils_tool.metrics import Evaluator
from utils_tool.options import add_data_args, add_model_args, apply_dataset_defaults

# Released checkpoints, looked up under ./checkpoints/ when --checkpoint is not given
_RELEASED_CKPT = {'levir': 'levir_mci.pth', 'whu': 'whu_cdc.pth'}


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
    state_dict = checkpoint['model_dict'] if 'model_dict' in checkpoint else checkpoint
    model = ChangeModel(args, word_vocab)
    model.load_state_dict(state_dict)
    model = model.cuda()
    model.eval()

    result_dir = None
    if args.save_pred:
        result_dir = os.path.join(args.result_path,
                                  os.path.basename(args.checkpoint).replace('.pth', ''))
        for sub in ['masks', 'captions']:
            os.makedirs(os.path.join(result_dir, sub), exist_ok=True)

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

            # Save per-image predictions (mask PNG + caption TXT)
            if args.save_pred and result_dir:
                cv2.imwrite(os.path.join(result_dir, 'masks', key + '.png'),
                            seg_to_color(pred_seg[0], args.num_classes))
                pred_caption = " ".join([idx_to_word[i] for i in pred_seq])
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
    cIoU_bin = evaluator.Binary_Change_IoU() * 100
    sc = get_eval_score(references, hypotheses)

    iou_label = "cIoU" if args.dataset == 'whu' else "mIoU"
    print(f'\n{iou_label:<8} {"binIoU":<8} {"B1":<8} {"B2":<8} {"B3":<8} {"B4":<8} {"M":<8} {"R":<8} {"C":<8}')
    print(f'{mIoU*100:<8.2f} {cIoU_bin:<8.2f} {sc["Bleu_1"]*100:<8.2f} {sc["Bleu_2"]*100:<8.2f} '
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

def _parse_test_args():
    parser = argparse.ArgumentParser(description='CoDeRNet test (LEVIR-MCI / WHU-CDC)')
    add_data_args(parser)
    add_model_args(parser)       # must match the training configuration

    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--gpu_id',  type=int, default=0)

    # Test-specific
    parser.add_argument('--checkpoint',  default=None,
                        help='model weights (.pth); default: ./checkpoints/levir_mci.pth '
                             'or ./checkpoints/whu_cdc.pth depending on --dataset')
    parser.add_argument('--result_path', default='./predict_result/')
    parser.add_argument('--save_pred',   action='store_true', default=False,
                        help='Save per-image predicted masks (PNG) and captions (TXT).')

    args = parser.parse_args()
    apply_dataset_defaults(args, parser)
    if args.checkpoint is None:
        args.checkpoint = os.path.join('./checkpoints', _RELEASED_CKPT[args.dataset])
    if not os.path.isfile(args.checkpoint):
        parser.error(f'checkpoint not found: {args.checkpoint}')
    return args


if __name__ == '__main__':
    args = _parse_test_args()
    run_test(args)
