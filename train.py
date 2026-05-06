"""
CoDeRNet training entry point — supports both LEVIR-MCI and WHU-CDC.

Usage:
  python train.py --dataset levir --savepath ./Results/exp01/
  python train.py --dataset whu   --savepath ./Results/exp01/

All hyperparameters can be overridden via CLI; dataset-specific defaults
(data paths, max_length, num_classes, ...) are applied automatically based
on --dataset.
"""
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils import data
import argparse
import json
import time
import numpy as np
import os
import random
from tqdm import tqdm

from data.LEVIR_MCI import LEVIRCCDataset
from data.WHU_CDC import WHUCDCDataset, build_whu_vocab
from model.change_model import ChangeModel
from utils_tool.utils import *
from utils_tool.metrics import Evaluator

# Checkpoint-selection criterion used in the paper:
# mIoU (detection) + BLEU-4 (lexical precision) + CIDEr (consensus semantic).
# METEOR and ROUGE-L are still computed and reported, but not used for selection.
EARLY_STOP_SELECTOR = "SUM_mIoU_B4_CIDEr"
SELECTORS = {
    EARLY_STOP_SELECTOR: lambda m: m["mIoU"] + m["B4"] + m["C"],
}


def _worker_init_fn(worker_id):
    """Per-worker seed for full reproducibility with num_workers > 0."""
    seed = torch.initial_seed() % (2**32)
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        # Force math-only SDP (flash/mem_efficient backward is non-deterministic)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)


def dice_loss(pred, target, num_classes, smooth=1.0):
    pred_soft = F.softmax(pred, dim=1)
    target_oh = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    inter = (pred_soft * target_oh).sum(dim=dims)
    card = pred_soft.sum(dim=dims) + target_oh.sum(dim=dims)
    return 1.0 - ((2.0 * inter + smooth) / (card + smooth)).mean()


def cross_entropy_2d_deterministic(logits, target):
    """Deterministic CE for 2D segmentation: log_softmax + gather."""
    # logits: (B, C, H, W), target: (B, H, W)
    logp = F.log_softmax(logits, dim=1)
    loss = -logp.gather(1, target.unsqueeze(1)).squeeze(1)
    return loss.mean()


def cross_entropy_1d_deterministic(logits, target):
    """Deterministic CE for 1D (captioning): log_softmax + gather."""
    # logits: (N, C), target: (N,)
    logp = F.log_softmax(logits, dim=1)
    loss = -logp.gather(1, target.unsqueeze(1)).squeeze(1)
    return loss.mean()


def _make_empty_record():
    return {"score": -1, "Acc": 0, "mIoU": 0, "F1": 0, "FWIoU": 0,
            "B1": 0, "B2": 0, "B3": 0, "B4": 0, "M": 0, "R": 0, "C": 0, "epoch": 0}


class Trainer:
    def __init__(self, args):
        self.args = args

        # Run name = backbone + timestamp (paper-fixed architecture choices omitted)
        name = f"{args.backbone}_{time_file_str()}"
        if not os.path.exists(args.savepath):
            os.makedirs(args.savepath)
        self.args.savepath = os.path.join(args.savepath, name)
        os.makedirs(self.args.savepath, exist_ok=True)
        self.log = open(os.path.join(self.args.savepath, f'{name}.log'), 'w')
        self.print_config(args)

        # Multi-selector best tracking
        self.best_tracker = {sel: _make_empty_record() for sel in SELECTORS}
        self.best_paths = {sel: None for sel in SELECTORS}  # checkpoint file per selector

        # Vocab
        if args.dataset == 'levir':
            with open(os.path.join(args.list_path, args.vocab_file + '.json'), 'r') as f:
                self.word_vocab = json.load(f)
        else:  # whu
            self.word_vocab = build_whu_vocab(args.caption_json)

        # Model
        self.model = ChangeModel(args, self.word_vocab).cuda()

        # Log module-level parameter counts
        total_params = sum(p.numel() for p in self.model.parameters())
        print_log(f'[Params] total: {total_params:,}', self.log)
        for name in ('encoder', 'fusion', 'projection', 'comp_inter',
                     'seg_decoder', 'cap_decoder'):
            module = getattr(self.model, name, None)
            if module is not None:
                p = sum(x.numel() for x in module.parameters())
                print_log(f'[Params] {name}: {p:,}', self.log)

        # Loss
        self.criterion_det = None  # use cross_entropy_2d_deterministic
        self.criterion_cap = None  # use cross_entropy_1d_deterministic

        # Optimizer
        param_groups = self.model.get_param_groups(args.encoder_lr, args.decoder_lr)
        self.optimizer = torch.optim.Adam(param_groups)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.5)

        # Data (seeded generator + worker_init_fn for reproducibility)
        g = torch.Generator()
        g.manual_seed(args.seed)
        if args.dataset == 'levir':
            train_ds = LEVIRCCDataset(
                args.data_folder, args.list_path, 'train',
                args.token_folder, args.vocab_file, args.max_length,
                args.allow_unk,
                base_seed=args.seed, deterministic_data=args.deterministic)
            val_ds = LEVIRCCDataset(
                args.data_folder, args.list_path, 'val',
                args.token_folder, args.vocab_file, args.max_length,
                args.allow_unk,
                base_seed=args.seed, deterministic_data=args.deterministic)
        else:  # whu
            train_ds = WHUCDCDataset(
                args.data_folder, args.caption_json, 'train',
                word_vocab=self.word_vocab, max_length=args.max_length,
                allow_unk=args.allow_unk,
                base_seed=args.seed, deterministic_data=args.deterministic)
            val_ds = WHUCDCDataset(
                args.data_folder, args.caption_json, 'val',
                word_vocab=self.word_vocab, max_length=args.max_length,
                allow_unk=args.allow_unk,
                base_seed=args.seed, deterministic_data=args.deterministic)
        train_kwargs = dict(
            batch_size=args.train_batchsize, shuffle=True,
            num_workers=args.workers, pin_memory=True, drop_last=True,
            worker_init_fn=_worker_init_fn, generator=g,
        )
        val_kwargs = dict(
            batch_size=1, shuffle=False,
            num_workers=args.workers, pin_memory=True,
        )
        self.train_loader = data.DataLoader(train_ds, **train_kwargs)
        self.val_loader = data.DataLoader(val_ds, **val_kwargs)

        self.evaluator = Evaluator(num_class=args.num_classes)
        self.hist = np.zeros((args.num_epochs * 2 * len(self.train_loader), 5))
        self.index_i = 0

        # Checkpoint resume
        if args.checkpoint is not None:
            ckpt = torch.load(args.checkpoint, map_location='cpu')
            self.model.load_state_dict(ckpt['model_dict'])
            print(f'Loaded checkpoint from {args.checkpoint}')

    def print_config(self, args):
        d = vars(args)
        w = max(len(k) for k in d) + 3
        print_log("\n" + "=" * 80, self.log)
        for k, v in sorted(d.items()):
            print_log(f"  {k:<{w}} {v}", self.log)
        print_log("=" * 80, self.log)
        print_log(f"  Selector: {EARLY_STOP_SELECTOR}", self.log)
        print_log("=" * 80 + "\n", self.log)

    def training(self, epoch):
        args = self.args
        self.train_loader.dataset.set_epoch(epoch)
        self.model.train()
        self.optimizer.zero_grad()
        accum_steps = max(1, 64 // args.train_batchsize)

        for it, (imgA, imgB, seg_label, _, _, token, token_len, _) in enumerate(self.train_loader):
            t0 = time.time()
            imgA = imgA.cuda(non_blocking=True)
            imgB = imgB.cuda(non_blocking=True)
            seg_label = seg_label.cuda(non_blocking=True)
            token = token.squeeze(1).cuda(non_blocking=True)
            token_len = token_len.cuda(non_blocking=True)

            seg_out, cap_out = self.model(
                imgA, imgB, token, token_len,
                target_size=seg_label.shape[-2:],
                train_goal=2)  # joint training

            # Seg loss = CE + Dice
            det_loss = cross_entropy_2d_deterministic(seg_out, seg_label.long())
            det_loss = det_loss + dice_loss(seg_out, seg_label.long(), args.num_classes)

            # Cap loss (teacher-forced cross-entropy)
            scores, caps_sorted, dec_lens, sort_ind, cap_hidden = cap_out
            targets = caps_sorted[:, 1:]
            scores_pk = pack_padded_sequence(scores, dec_lens, batch_first=True).data
            targets_pk = pack_padded_sequence(targets, dec_lens, batch_first=True).data
            cap_loss = cross_entropy_1d_deterministic(scores_pk, targets_pk.long())

            # Joint loss: stop-gradient-normalized sum
            loss = (det_loss / det_loss.detach().clamp(min=1e-8) +
                    cap_loss / cap_loss.detach().clamp(min=1e-8))

            (loss / accum_steps).backward()

            if args.grad_clip is not None:
                torch.nn.utils.clip_grad_value_(self.model.parameters(), args.grad_clip)

            if (it + 1) % accum_steps == 0 or (it + 1) == len(self.train_loader):
                self.optimizer.step()
                self.optimizer.zero_grad()

            # Logging
            self.hist[self.index_i, 0] = time.time() - t0
            if seg_out is not None:
                self.hist[self.index_i, 1] = det_loss.item()
                self.hist[self.index_i, 2] = accuracy(
                    seg_out.permute(0, 2, 3, 1).reshape(-1, seg_out.size(1)),
                    seg_label.reshape(-1), 1)
            if cap_out is not None:
                self.hist[self.index_i, 3] = cap_loss.item()
                self.hist[self.index_i, 4] = accuracy(scores_pk, targets_pk, 5)

            self.index_i += 1
            if self.index_i % args.print_freq == 0:
                sl = slice(self.index_i - args.print_freq, self.index_i)
                print_log(
                    f'Epoch [{epoch}][{it}/{len(self.train_loader)}] '
                    f'Time: {np.mean(self.hist[sl, 0]) * args.print_freq:.1f} '
                    f'Det: {np.mean(self.hist[sl, 1]):.4f} '
                    f'Acc: {np.mean(self.hist[sl, 2]):.1f} '
                    f'Cap: {np.mean(self.hist[sl, 3]):.5f} '
                    f'Top5: {np.mean(self.hist[sl, 4]):.1f}',
                    self.log)

        self.lr_scheduler.step()

    def validation(self, epoch):
        self.model.eval()
        wv = self.word_vocab
        references, hypotheses = [], []
        self.evaluator.reset()

        with torch.inference_mode():
            for imgA, imgB, seg_label, token_all, _, _, _, _ in tqdm(
                    self.val_loader, desc=f'val_ep{epoch}'):
                imgA = imgA.cuda(non_blocking=True)
                imgB = imgB.cuda(non_blocking=True)
                token_all = token_all.squeeze(0).cuda(non_blocking=True)

                seg_out, _ = self.model(imgA, imgB, train_goal=0,
                                        target_size=seg_label.shape[-2:])
                seq = self.model.sample_caption(imgA, imgB)

                # Seg metrics
                if seg_out is not None:
                    pred = seg_out.data.cpu().numpy()
                    gt = seg_label.cpu().numpy()
                    self.evaluator.add_batch(gt, np.argmax(pred, axis=1))

                # Cap metrics
                if seq is not None:
                    img_tokens = [
                        [w for w in c if w not in {wv['<START>'], wv['<END>'], wv['<NULL>']}]
                        for c in token_all.tolist()]
                    references.append(img_tokens)
                    hypotheses.append(
                        [w for w in seq if w not in {wv['<START>'], wv['<END>'], wv['<NULL>']}])

        # Feature similarity analysis (every 10 epochs)
        if epoch % 10 == 0:
            sample_A, sample_B = next(iter(self.val_loader))[:2]
            sim = self.model.compute_feature_similarity(
                sample_A.cuda(non_blocking=True),
                sample_B.cuda(non_blocking=True),
            )
            print_log(f"\n[Feature Similarity @ epoch {epoch}]", self.log)
            for s in ['1/4', '1/8', '1/16', '1/32']:
                pre = sim.get(f'seg_vs_cap_{s}_pre', 0)
                post = sim.get(f'seg_vs_cap_{s}_post', None)
                line = f"  {s:>4s}  seg-cap={pre:.4f}"
                if post is not None:
                    line += f"  -> {post:.4f} (post-inter)"
                print_log(line, self.log)

        # Interaction stats (every 10 epochs, averaged over 50 val samples).
        if epoch % 10 == 0:
            stats_accum = {}
            n_samples = 0
            for idx, batch in enumerate(self.val_loader):
                if idx >= 50:
                    break
                sA = batch[0].cuda(non_blocking=True)
                sB = batch[1].cuda(non_blocking=True)
                stats = self.model.compute_interaction_stats(sA, sB)
                if stats is not None:
                    for k, v in stats.items():
                        stats_accum[k] = stats_accum.get(k, 0) + v
                    n_samples += 1
            if n_samples > 0:
                print_log(f"\n[Interaction Stats @ epoch {epoch} (avg over {n_samples} samples)]", self.log)
                for k, v in stats_accum.items():
                    print_log(f"  {k}={v/n_samples:.4f}", self.log)

        # Compute metrics
        curr_IoU = "0.0000"
        curr_Acc = self.evaluator.Pixel_Accuracy()
        if self.args.dataset == 'whu':
            curr_mIoU = self.evaluator.Change_IoU()
        else:
            curr_mIoU, curr_IoU = self.evaluator.Mean_Intersection_over_Union()
        curr_F1, _ = self.evaluator.F1_Score()
        curr_FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        iou_label = "cIoU" if self.args.dataset == 'whu' else "mIoU"
        print_log(
            f"\nDetection: Acc={curr_Acc * 100:.1f}  {iou_label}={curr_mIoU * 100:.1f}  "
            f"F1={curr_F1 * 100:.1f}  FWIoU={curr_FWIoU * 100:.1f}",
            self.log,
        )

        curr_B1 = curr_B2 = curr_B3 = curr_B4 = curr_M = curr_R = curr_C = 0.0
        if len(references) > 0:
            score_dict = get_eval_score(references, hypotheses)
            curr_B1 = score_dict["Bleu_1"]
            curr_B2 = score_dict["Bleu_2"]
            curr_B3 = score_dict["Bleu_3"]
            curr_B4 = score_dict["Bleu_4"]
            curr_M = score_dict["METEOR"]
            curr_R = score_dict["ROUGE_L"]
            curr_C = score_dict["CIDEr"]
            print_log(
                f"Captioning: B1={curr_B1 * 100:.1f} B2={curr_B2 * 100:.1f} "
                f"B3={curr_B3 * 100:.1f} B4={curr_B4 * 100:.1f} "
                f"M={curr_M * 100:.1f} R={curr_R * 100:.1f} C={curr_C * 100:.1f}",
                self.log,
            )

        Acc_p = curr_Acc * 100
        mIoU_p = curr_mIoU * 100
        F1_p = curr_F1 * 100
        FWIoU_p = curr_FWIoU * 100
        B1_p = curr_B1 * 100
        B2_p = curr_B2 * 100
        B3_p = curr_B3 * 100
        B4_p = curr_B4 * 100
        M_p = curr_M * 100
        R_p = curr_R * 100
        C_p = curr_C * 100

        # Current metrics dict (used by selectors)
        metrics = {
            "Acc": Acc_p, "mIoU": mIoU_p, "F1": F1_p, "FWIoU": FWIoU_p,
            "B1": B1_p, "B2": B2_p, "B3": B3_p, "B4": B4_p,
            "M": M_p, "R": R_p, "C": C_p, "epoch": epoch,
        }

        # Update best tracker per selector + save checkpoints
        state = None  # lazy: only serialize once if needed
        saved_any = False
        for sel_name, sel_fn in SELECTORS.items():
            sel_score = sel_fn(metrics)
            if sel_score > self.best_tracker[sel_name]["score"] and epoch > 3:
                self.best_tracker[sel_name] = {"score": sel_score, **metrics}
                # Save checkpoint for this selector
                if state is None:
                    state = {"model_dict": self.model.state_dict()}
                ckpt_path = os.path.join(self.args.savepath, f"best_{sel_name}.pth")
                torch.save(state, ckpt_path)
                self.best_paths[sel_name] = ckpt_path
                saved_any = True
                print_log(f"  [BEST {sel_name}] score={sel_score:.2f} -> saved {os.path.basename(ckpt_path)}", self.log)
            elif sel_score > self.best_tracker[sel_name]["score"]:
                # Update tracker but don't save (epoch <= 3)
                self.best_tracker[sel_name] = {"score": sel_score, **metrics}

        # Print best tracker table
        iou_col = "cIoU" if self.args.dataset == 'whu' else "mIoU"
        print_log("-" * 110, self.log)
        print_log(
            f"{'Selector':<20} | {'Score':<8} | {iou_col:<6} | {'B4':<6} | "
            f"{'M':<6} | {'R':<6} | {'C':<6} | {'Ep':<4} | {'Ckpt':<5}",
            self.log)
        print_log("-" * 110, self.log)
        for sel_name in SELECTORS:
            d = self.best_tracker[sel_name]
            score = d["score"] if d["score"] != -1 else 0.0
            has_ckpt = "Yes" if self.best_paths[sel_name] else "-"
            print_log(
                f"{sel_name:<20} | {score:<8.2f} | {d['mIoU']:<6.1f} | {d['B4']:<6.1f} | "
                f"{d['M']:<6.1f} | {d['R']:<6.1f} | {d['C']:<6.1f} | {d['epoch']:<4} | {has_ckpt:<5}",
                self.log)
        print_log("-" * 110, self.log)

        # Save val summary (overwritten each epoch, final state = best per selector)
        val_summary_path = os.path.join(self.args.savepath, "val_summary.txt")
        with open(val_summary_path, 'w') as f:
            f.write(f"{'Selector':<20} {'Score':<10} {iou_col:<8} {'B4':<8} {'M':<8} {'R':<8} {'C':<8} {'Ep':<5} {'Ckpt':<5}\n")
            f.write("-" * 95 + "\n")
            for sel_name in SELECTORS:
                d = self.best_tracker[sel_name]
                score = d["score"] if d["score"] != -1 else 0.0
                has_ckpt = "Yes" if self.best_paths[sel_name] else "-"
                f.write(f"{sel_name:<20} {score:<10.2f} {d['mIoU']:<8.1f} {d['B4']:<8.1f} "
                        f"{d['M']:<8.1f} {d['R']:<8.1f} {d['C']:<8.1f} {d['epoch']:<5} {has_ckpt:<5}\n")

    @torch.inference_mode()
    def _collect_gate_stats(self, loader, max_batches=50):
        """Aggregate per-batch interaction/gate stats for P2 data-adaptivity check.

        Each per-direction operator (ours/confcross/consensus/diffgate/pairgate/...)
        returns a dict of scalar stats from its forward pass (mean over spatial/channel).
        We collect those scalars across up to `max_batches` val/test samples and
        summarize the dataset-level distribution (mean/std/min/max over batches).

        The summary is the primary evidence for the disagreement principle's
        falsifiable prediction: gate mean should differ between datasets
        (e.g. WHU low gate vs LEVIR higher gate).
        """
        if not getattr(self.args, 'interaction', False):
            return None
        if not hasattr(self.model, 'comp_inter'):
            return None

        self.model.eval()
        accum = {}
        n = 0
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            imgA = batch[0].cuda(non_blocking=True)
            imgB = batch[1].cuda(non_blocking=True)
            try:
                stats = self.model.compute_interaction_stats(imgA, imgB)
            except Exception:
                return None
            if stats is None:
                return None
            for k, v in stats.items():
                accum.setdefault(k, []).append(float(v))
            n += 1

        if n == 0:
            return None
        summary = {}
        for k, values in accum.items():
            arr = np.array(values)
            summary[k] = {
                'mean':      float(arr.mean()),
                'std':       float(arr.std()),
                'min':       float(arr.min()),
                'max':       float(arr.max()),
                'n_batches': n,
            }
        return summary

    def auto_test(self):
        """Run test inference on all saved best checkpoints after training."""
        saved = {sel: path for sel, path in self.best_paths.items() if path and os.path.exists(path)}
        if not saved:
            print_log("\n[Auto-Test] No checkpoints saved, skipping.", self.log)
            return

        # Deduplicate by epoch: selectors that peaked at the same epoch share one eval
        epoch_groups = {}  # epoch -> (ckpt_path, [selector_names])
        for sel, path in saved.items():
            ep = self.best_tracker[sel]["epoch"]
            if ep not in epoch_groups:
                epoch_groups[ep] = (path, [])
            epoch_groups[ep][1].append(sel)

        print_log(f"\n{'='*110}", self.log)
        print_log(f"[Auto-Test] {len(saved)} selectors -> {len(epoch_groups)} unique epoch(s) to evaluate on TEST split", self.log)
        print_log(f"{'='*110}", self.log)

        args = self.args
        word_vocab = self.word_vocab

        if args.dataset == 'levir':
            test_ds = LEVIRCCDataset(args.data_folder, args.list_path, 'test',
                                     args.token_folder, args.vocab_file, args.max_length,
                                     args.allow_unk)
        else:  # whu
            test_ds = WHUCDCDataset(args.data_folder, args.caption_json, 'test',
                                    word_vocab=word_vocab, max_length=args.max_length,
                                    allow_unk=args.allow_unk)
        test_kwargs = dict(
            batch_size=1, shuffle=False,
            num_workers=args.workers, pin_memory=True,
        )
        test_loader = data.DataLoader(test_ds, **test_kwargs)

        test_results = {}  # selector -> metrics

        for ep, (ckpt_path, selectors) in sorted(epoch_groups.items()):
            sel_label = "+".join(selectors)
            print_log(f"\n--- Testing ep{ep}: {os.path.basename(ckpt_path)} (selectors: {sel_label}) ---", self.log)

            ckpt = torch.load(ckpt_path, map_location='cpu')
            self.model.load_state_dict(ckpt['model_dict'])
            self.model.eval()

            references, hypotheses = [], []
            evaluator = Evaluator(num_class=args.num_classes)

            with torch.inference_mode():
                for imgA, imgB, seg_label, token_all, _, _, _, _ in tqdm(test_loader, desc=f'Test(ep{ep})'):
                    imgA_c = imgA.cuda(non_blocking=True)
                    imgB_c = imgB.cuda(non_blocking=True)
                    token_all = token_all.squeeze(0).cuda(non_blocking=True)

                    seg_out, _ = self.model(imgA_c, imgB_c, train_goal=0,
                                            target_size=seg_label.shape[-2:])
                    seq = self.model.sample_caption(imgA_c, imgB_c)

                    pred_seg = np.argmax(seg_out.cpu().numpy(), axis=1)
                    evaluator.add_batch(seg_label.numpy(), pred_seg)

                    img_tokens = [
                        [w for w in c if w not in {word_vocab['<START>'], word_vocab['<END>'], word_vocab['<NULL>']}]
                        for c in token_all.tolist()]
                    references.append(img_tokens)
                    hypotheses.append(
                        [w for w in seq if w not in {word_vocab['<START>'], word_vocab['<END>'], word_vocab['<NULL>']}])

            if args.dataset == 'whu':
                mIoU_t = evaluator.Change_IoU()
            else:
                mIoU_t, _ = evaluator.Mean_Intersection_over_Union()
            sc = get_eval_score(references, hypotheses)
            result = {
                "mIoU": mIoU_t * 100,
                "B1": sc["Bleu_1"] * 100, "B2": sc["Bleu_2"] * 100,
                "B3": sc["Bleu_3"] * 100, "B4": sc["Bleu_4"] * 100,
                "M": sc["METEOR"] * 100, "R": sc["ROUGE_L"] * 100, "C": sc["CIDEr"] * 100,
            }

            iou_l = "cIoU" if args.dataset == 'whu' else "mIoU"
            print_log(
                f"  {iou_l}={result['mIoU']:.1f}  B4={result['B4']:.1f}  "
                f"M={result['M']:.1f}  R={result['R']:.1f}  C={result['C']:.1f}",
                self.log)

            for sel in selectors:
                test_results[sel] = result

        # Final summary table
        iou_col = "cIoU" if args.dataset == 'whu' else "mIoU"
        print_log(f"\n{'='*110}", self.log)
        print_log("[Auto-Test] Final Summary (TEST split)", self.log)
        print_log(f"{'='*110}", self.log)
        print_log(
            f"{'Selector':<20} | {'ValScore':<10} | {iou_col:<8} | {'B4':<8} | "
            f"{'M':<8} | {'R':<8} | {'C':<8} | {'ValEp':<5}",
            self.log)
        print_log("-" * 90, self.log)
        for sel_name in SELECTORS:
            if sel_name not in test_results:
                continue
            r = test_results[sel_name]
            val_ep = self.best_tracker[sel_name]["epoch"]
            val_score = self.best_tracker[sel_name]["score"]
            print_log(
                f"{sel_name:<20} | {val_score:<10.2f} | {r['mIoU']:<8.1f} | {r['B4']:<8.1f} | "
                f"{r['M']:<8.1f} | {r['R']:<8.1f} | {r['C']:<8.1f} | {val_ep:<5}",
                self.log)
        print_log("=" * 110, self.log)

        # Gate statistics collection (P2 data-adaptivity check) --
        # load the primary selector's best checkpoint as a consistent reference.
        gate_summary = None
        primary_sel = EARLY_STOP_SELECTOR
        best_primary_path = self.best_paths.get(primary_sel)
        if best_primary_path and os.path.exists(best_primary_path):
            ckpt = torch.load(best_primary_path, map_location='cpu')
            self.model.load_state_dict(ckpt['model_dict'])
            self.model.eval()
            gate_summary = self._collect_gate_stats(test_loader, max_batches=50)

        if gate_summary is not None:
            print_log(f"\n{'='*110}", self.log)
            print_log(f"[Gate Statistics]  (best {primary_sel} checkpoint, 50 test batches)", self.log)
            print_log(f"{'='*110}", self.log)
            for k in sorted(gate_summary):
                s = gate_summary[k]
                print_log(
                    f"  {k:<28} mean={s['mean']:.4f}  std={s['std']:.4f}  "
                    f"min={s['min']:.4f}  max={s['max']:.4f}  n={s['n_batches']}",
                    self.log)

        # Save summary to file
        summary_path = os.path.join(self.args.savepath, "test_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("\n")
            f.write(f"{'Selector':<20} {'ValScore':<10} {iou_col:<8} {'B1':<8} {'B2':<8} {'B3':<8} {'B4':<8} {'M':<8} {'R':<8} {'C':<8} {'ValEp':<5}\n")
            f.write("-" * 105 + "\n")
            for sel_name in SELECTORS:
                if sel_name not in test_results:
                    continue
                r = test_results[sel_name]
                val_ep = self.best_tracker[sel_name]["epoch"]
                val_score = self.best_tracker[sel_name]["score"]
                f.write(f"{sel_name:<20} {val_score:<10.2f} {r['mIoU']:<8.2f} {r['B1']:<8.2f} {r['B2']:<8.2f} "
                        f"{r['B3']:<8.2f} {r['B4']:<8.2f} {r['M']:<8.2f} {r['R']:<8.2f} {r['C']:<8.2f} {val_ep:<5}\n")
            if gate_summary is not None:
                f.write(f"\n# Gate Statistics (best {primary_sel} checkpoint, 50 test batches)\n")
                for k in sorted(gate_summary):
                    s = gate_summary[k]
                    f.write(
                        f"# {k:<28} mean={s['mean']:.4f}  std={s['std']:.4f}  "
                        f"min={s['min']:.4f}  max={s['max']:.4f}  n={s['n_batches']}\n")
        print_log(f"\nTest summary saved: {summary_path}", self.log)


def run_training(args):
    """Main training entry point."""
    set_random_seed(args.seed, deterministic=args.deterministic)
    torch.cuda.set_device(args.gpu_id)

    trainer = Trainer(args)
    train_start = time.time()
    for epoch in range(args.num_epochs):
        trainer.training(epoch)
        trainer.validation(epoch)
    elapsed = time.time() - train_start
    h, m = int(elapsed // 3600), int((elapsed % 3600) // 60)
    print_log(f"\nTotal training time: {h}h {m}m ({elapsed:.0f}s)", trainer.log)

    trainer.auto_test()
    trainer.log.close()


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


def _parse_train_args():
    parser = argparse.ArgumentParser(description='CoDeRNet training (LEVIR-MCI / WHU-CDC)')

    # Dataset switch (drives data-path / class-count defaults)
    parser.add_argument('--dataset', choices=['levir', 'whu'], required=True)

    # Data (defaults filled per --dataset; override here if needed)
    parser.add_argument('--data_folder',  default=None)
    parser.add_argument('--list_path',    default=None)
    parser.add_argument('--token_folder', default=None)
    parser.add_argument('--vocab_file',   default='vocab')
    parser.add_argument('--caption_json', default=None)
    parser.add_argument('--max_length',   type=int, default=None)
    parser.add_argument('--num_classes',  type=int, default=None)
    parser.add_argument('--allow_unk',    type=int, default=1)

    # Encoder / model
    parser.add_argument('--backbone',  default='tu-convnext_base')
    parser.add_argument('--d_model',   type=int, default=256)
    parser.add_argument('--n_heads',          type=int, default=8)
    parser.add_argument('--decoder_n_layers', type=int, default=1)
    parser.add_argument('--dropout',          type=float, default=0.1)

    # Training
    parser.add_argument('--train_batchsize', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=150)
    parser.add_argument('--encoder_lr', type=float, default=1e-4)
    parser.add_argument('--decoder_lr', type=float, default=1e-4)
    parser.add_argument('--grad_clip',  type=float, default=None)
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--workers',    type=int, default=4)
    parser.add_argument('--seed',       type=int, default=42)
    parser.add_argument('--deterministic', action='store_true', default=False)
    parser.add_argument('--gpu_id',     type=int, default=0)

    # Checkpoint / output
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--savepath',   default='./Results/')

    args = parser.parse_args()
    _apply_dataset_defaults(args)
    return args


def _apply_dataset_defaults(args):
    """Fill data-path / class-count defaults based on --dataset."""
    defaults = _LEVIR_DEFAULTS if args.dataset == 'levir' else _WHU_DEFAULTS
    for key, value in defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)


if __name__ == '__main__':
    args = _parse_train_args()
    run_training(args)
