"""Bidirectional Confidence-Decomposed Routing (CoDeR) dispatcher.

CoDeR is applied bidirectionally with independent estimators and adapters:
    seg ← cap at scale s_g (default 1/8, fine for boundary refinement)
    cap ← seg at scale s_c (default 1/16, coarser for semantic abstraction)

Within each direction, source and target features are taken from the same
spatial scale, so the routing mask is applied directly without resizing.
The asymmetry across directions reflects the different spatial granularity
required by detection and captioning.
"""
from torch import nn

from .interactions import CoDeR


class CompInteraction(nn.Module):
    def __init__(self, d_model=256, seg_recv_scale=1, cap_recv_scale=2,
                 variant='light'):
        super().__init__()
        self.seg_recv_scale = seg_recv_scale
        self.cap_recv_scale = cap_recv_scale

        scale_names = ['1/4', '1/8', '1/16', '1/32']

        # Two independent CoDeR blocks, one per direction.
        self.seg_recv = CoDeR(d_model, variant=variant)   # cap -> seg
        self.cap_recv = CoDeR(d_model, variant=variant)   # seg -> cap

        print(f'[CompInteraction] cap -> seg @ {scale_names[seg_recv_scale]}')
        print(f'[CompInteraction] seg -> cap @ {scale_names[cap_recv_scale]}')

    def forward(self, seg_feats, cap_feats):
        """Apply CoDeR bidirectionally at the configured scales.

        Args:
            seg_feats: list of 4 seg-branch features  [F^(1/4), F^(1/8), F^(1/16), F^(1/32)]
            cap_feats: list of 4 cap-branch features  (same layout)
        Returns:
            seg_feats_out, cap_feats_out: copies with one entry replaced by the
                CoDeR output at the corresponding receive scale.
            stats: dict of per-direction routing statistics (means).
        """
        seg_feats_out = list(seg_feats)
        cap_feats_out = list(cap_feats)
        stats = {}

        # cap -> seg: seg branch receives at seg_recv_scale
        si = self.seg_recv_scale
        seg_new, s = self.seg_recv(seg_feats[si], cap_feats[si])
        seg_feats_out[si] = seg_new
        stats.update({f'seg_recv_{k}': v for k, v in s.items()})

        # seg -> cap: cap branch receives at cap_recv_scale
        ci = self.cap_recv_scale
        cap_new, s = self.cap_recv(cap_feats[ci], seg_feats[ci])
        cap_feats_out[ci] = cap_new
        stats.update({f'cap_recv_{k}': v for k, v in s.items()})

        return seg_feats_out, cap_feats_out, stats
