"""
Change Fusion.

For per-scale features (F_t0, F_t1) extracted from a bi-temporal pair, build the
shared change representation
    F^(s) = Conv1x1( [ F_t0,  F_t1,  |F_t0 - F_t1|,  F_t0 - F_t1 ] )
which combines the two temporal features with a magnitude difference cue and a
signed difference cue. The 1x1 convolution then projects the concatenation back
to d_model.
"""
import torch
import torch.nn as nn


class ChangeFusion(nn.Module):
    IN_MULT = 4   # [F_t0, F_t1, |Δ|, Δ]

    def __init__(self, d_model=256, num_scales=4):
        super().__init__()
        self.fuse = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d_model * self.IN_MULT, d_model, 1, bias=False),
                nn.BatchNorm2d(d_model),
                nn.ReLU(inplace=True),
            ) for _ in range(num_scales)
        ])
        print(f'[ChangeFusion] in_channels={self.IN_MULT}*d_model')

    def forward(self, feat_A, feat_B):
        """
        Args:
            feat_A, feat_B: lists of `num_scales` feature maps from the two
                temporal images, each (B, d_model, H_i, W_i).
        Returns:
            chg_feats: list of fused feature maps at the same scales.
        """
        chg_feats = []
        for i in range(len(feat_A)):
            a, b = feat_A[i], feat_B[i]
            cat = torch.cat([a, b, torch.abs(a - b), a - b], dim=1)
            chg_feats.append(self.fuse[i](cat))
        return chg_feats
