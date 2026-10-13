"""
CoDeRNet: Change Detection + Change Captioning Model.
Encoder -> Fusion -> BranchProjection -> CompInteraction (CoDeR) -> Seg + Cap Decoders.
"""
import torch
import torch.nn as nn

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

        if str(args.backbone).startswith('tips'):
            # Optional ViT backbone, imported only when requested so that the
            # default ConvNeXt path does not depend on it.
            from .encoder_tips import TIPSEncoder
            self.encoder = TIPSEncoder(
                backbone_name=args.backbone, d_model=d_model,
                npz_path=getattr(args, 'tips_npz', None),
                img_size=getattr(args, 'tips_img_size', 224),
                pretrained=True)
        else:
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
        """train_goal selects the heads to run: 0 = detection only,
        1 = captioning only, 2 = both (joint training)."""
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
    def compute_interaction_stats(self, imgA, imgB):
        """Returns interaction confidence/mask stats."""
        _, _, stats = self._extract_feats(imgA, imgB, return_stats=True)
        return stats

    def get_param_groups(self, encoder_lr, decoder_lr, vit_lr=None):
        # A TIPS encoder is split into the ViT backbone (lower LR) and its
        # multi-scale adapter (decoder LR).
        if hasattr(self.encoder, 'vit_parameters'):
            enc_groups = [
                {'params': list(self.encoder.vit_parameters()),
                 'lr': vit_lr if vit_lr is not None else encoder_lr},
                {'params': list(self.encoder.adapter_parameters()), 'lr': decoder_lr},
            ]
        else:
            enc_groups = [{'params': list(self.encoder.parameters()), 'lr': encoder_lr}]
        return enc_groups + [
            {'params': list(self.fusion.parameters()),      'lr': decoder_lr},
            {'params': list(self.projection.parameters()),  'lr': decoder_lr},
            {'params': list(self.comp_inter.parameters()),  'lr': decoder_lr},
            {'params': list(self.seg_decoder.parameters()), 'lr': decoder_lr},
            {'params': list(self.cap_decoder.parameters()), 'lr': decoder_lr},
        ]
