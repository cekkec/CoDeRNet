"""Branch-Specific Feature Projection.

From shared change features R^(s), produce per-branch (seg, cap) feature spaces:
    F_seg^(s) = ψ_seg^(s)(R^(s))
    F_cap^(s) = ψ_cap^(s)(R^(s))

Each ψ is a lightweight 1×1 conv + BN + ReLU + 3×3 conv + BN + ReLU block,
applied independently per scale and per branch. These provide separate
feature spaces on which CoDeR estimates branch-dependent routing cues.
"""
import torch.nn as nn


def _make_proj(d_model):
    return nn.Sequential(
        nn.Conv2d(d_model, d_model, 1, bias=False),
        nn.BatchNorm2d(d_model),
        nn.ReLU(inplace=True),
        nn.Conv2d(d_model, d_model, 3, padding=1, bias=False),
        nn.BatchNorm2d(d_model),
        nn.ReLU(inplace=True),
    )


class BranchProjection(nn.Module):
    def __init__(self, d_model=256, num_scales=4):
        super().__init__()
        self.seg_proj = nn.ModuleList([_make_proj(d_model) for _ in range(num_scales)])
        self.cap_proj = nn.ModuleList([_make_proj(d_model) for _ in range(num_scales)])
        print(f'[BranchProjection] d_model={d_model}, num_scales={num_scales}')

    def forward(self, chg_feats):
        """
        Args:
            chg_feats: [R^(1/4), R^(1/8), R^(1/16), R^(1/32)] each (B, d_model, H, W)
        Returns:
            seg_feats, cap_feats: branch-specific feature lists at the same scales.
        """
        seg_feats = [self.seg_proj[i](chg_feats[i]) for i in range(len(chg_feats))]
        cap_feats = [self.cap_proj[i](chg_feats[i]) for i in range(len(chg_feats))]
        return seg_feats, cap_feats
