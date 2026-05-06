"""
Segmentation Decoder.
FPN top-down pathway + ResBlock refinement, then a 2-stage transposed
convolution head produces per-class logits at the input resolution.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(dim)
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + x)


class SegDecoder(nn.Module):
    def __init__(self, d_model=256, num_classes=3, use_refine=False, refine_n_blocks=2):
        super().__init__()
        self.use_refine = use_refine

        self.lateral = nn.ModuleList([
            nn.Sequential(nn.Conv2d(d_model, d_model, 1, bias=False),
                          nn.BatchNorm2d(d_model), nn.ReLU(inplace=True))
            for _ in range(4)
        ])
        self.fpn = nn.ModuleList([ResBlock(d_model) for _ in range(3)])

        # Optional refinement blocks after FPN (before seg_head)
        if use_refine:
            self.refine_blocks = nn.Sequential(
                *[ResBlock(d_model) for _ in range(refine_n_blocks)])
            print(f'[SegDecoder] Refinement enabled ({refine_n_blocks} ResBlocks)')

        self.seg_head = nn.Sequential(
            nn.ConvTranspose2d(d_model, d_model // 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(d_model // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(d_model // 2, d_model // 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(d_model // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model // 4, num_classes, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _fpn_merge(self, lat_fine, lat_coarse, idx):
        up = F.interpolate(lat_coarse, size=lat_fine.shape[2:],
                           mode='bilinear', align_corners=False)
        return self.fpn[idx](lat_fine + up)

    def forward(self, chg_feats, target_size=None):
        lat = [self.lateral[i](chg_feats[i]) for i in range(4)]
        lat[2] = self._fpn_merge(lat[2], lat[3], 0)
        lat[1] = self._fpn_merge(lat[1], lat[2], 1)
        lat[0] = self._fpn_merge(lat[0], lat[1], 2)

        feat = lat[0]
        if self.use_refine:
            feat = self.refine_blocks(feat)

        out = self.seg_head(feat)
        if target_size is not None:
            out = F.interpolate(out, size=target_size, mode='bilinear', align_corners=False)
        return out
