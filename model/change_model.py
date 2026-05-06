"""
CoDeRNet: Change Detection + Change Captioning Model.
Encoder -> Fusion -> BranchProjection -> CompInteraction (CoDeR) -> Seg + Cap Decoders.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Encoder
from .fusion import ChangeFusion
from .projection import BranchProjection
from .comp_interaction import CompInteraction
from .seg_decoder import SegDecoder
from .cap_decoder import CapDecoder


class ChangeModel(nn.Module):
    def __init__(self, args, word_vocab):
        super().__init__()
        d_model = args.d_model

        self.encoder = Encoder(backbone_name=args.backbone, d_model=d_model, pretrained=True)
        self.fusion = ChangeFusion(d_model=d_model)

        self.projection = BranchProjection(d_model=d_model)

        self.comp_inter = CompInteraction(
            d_model=d_model,
            seg_recv_scale=1,                # 1/8  (boundary-aware)
            cap_recv_scale=2,                # 1/16 (semantic-abstract)
            variant='light',                 # adapter = Conv1×1
        )

        self.seg_decoder = SegDecoder(
            d_model=d_model, num_classes=args.num_classes,
            use_refine=True, refine_n_blocks=2)
        self.cap_decoder = CapDecoder(
            d_model=d_model, vocab_size=len(word_vocab),
            max_length=args.max_length, word_vocab=word_vocab,
            n_head=args.n_heads, n_layers=args.decoder_n_layers,
            dropout=args.dropout,
            use_meso=True, meso_pool_size=2,    # 2x2 -> 4 meso tokens at 1/8 scale
            num_classes=args.num_classes)

    def _extract_feats(self, imgA, imgB, return_stats=False):
        feat_A = self.encoder(imgA)
        feat_B = self.encoder(imgB)
        chg_feats = self.fusion(feat_A, feat_B)

        seg_feats, cap_feats = self.projection(chg_feats)
        seg_feats, cap_feats, inter_stats = self.comp_inter(seg_feats, cap_feats)

        if return_stats:
            return seg_feats, cap_feats, inter_stats
        return seg_feats, cap_feats

    def forward(self, imgA, imgB, encoded_captions=None, caption_lengths=None,
                target_size=None, train_goal=2):
        seg_feats, cap_feats = self._extract_feats(imgA, imgB)

        seg_out = None
        cap_out = None

        if train_goal in [0, 2]:
            seg_out = self.seg_decoder(seg_feats, target_size=target_size)

        if train_goal in [1, 2] and encoded_captions is not None:
            cap_out = self.cap_decoder(cap_feats, encoded_captions, caption_lengths)

        return seg_out, cap_out

    @torch.inference_mode()
    def sample_caption(self, imgA, imgB):
        _, cap_feats = self._extract_feats(imgA, imgB)
        return self.cap_decoder.sample(cap_feats)

    @torch.inference_mode()
    def compute_feature_similarity(self, imgA, imgB):
        """Cosine similarity between seg and cap branches at each scale,
        before and after CoDeR routing — used for diagnostic logging."""
        feat_A = self.encoder(imgA)
        feat_B = self.encoder(imgB)
        chg_feats = self.fusion(feat_A, feat_B)

        seg_feats, cap_feats = self.projection(chg_feats)

        # Before routing
        scale_names = ['1/4', '1/8', '1/16', '1/32']
        result = {}
        for i, s in enumerate(scale_names):
            sg = seg_feats[i].flatten(1)
            cp = cap_feats[i].flatten(1)
            result[f'seg_vs_cap_{s}_pre'] = F.cosine_similarity(sg, cp, dim=1).mean().item()

        # After routing
        seg_feats_post, cap_feats_post, _ = self.comp_inter(seg_feats, cap_feats)
        for i, s in enumerate(scale_names):
            sg = seg_feats_post[i].flatten(1)
            cp = cap_feats_post[i].flatten(1)
            result[f'seg_vs_cap_{s}_post'] = F.cosine_similarity(sg, cp, dim=1).mean().item()

        return result

    @torch.inference_mode()
    def compute_interaction_stats(self, imgA, imgB):
        """Returns interaction confidence/mask stats."""
        _, _, stats = self._extract_feats(imgA, imgB, return_stats=True)
        return stats

    def get_param_groups(self, encoder_lr, decoder_lr):
        return [
            {'params': list(self.encoder.parameters()),     'lr': encoder_lr},
            {'params': list(self.fusion.parameters()),      'lr': decoder_lr},
            {'params': list(self.projection.parameters()),  'lr': decoder_lr},
            {'params': list(self.comp_inter.parameters()),  'lr': decoder_lr},
            {'params': list(self.seg_decoder.parameters()), 'lr': decoder_lr},
            {'params': list(self.cap_decoder.parameters()), 'lr': decoder_lr},
        ]
