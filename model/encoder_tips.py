"""TIPSv2 (vision-language pretrained ViT) encoder for CoDeRNet.

Same interface as the default smp `Encoder`:
    forward(x: (B, 3, H, W)) -> [F^(1/4), F^(1/8), F^(1/16), F^(1/32)], each (B, d_model, ., .)

A ViT is single-scale (one patch grid), so four intermediate layers are taken
and each is resampled to one of the four pyramid scales, then projected to
d_model.

Notes:
- The released TIPS checkpoints store a 32x32 position embedding (448 px input),
  so the ViT is built with img_size=448 and loaded as is. The forward pass
  resizes the image to `img_size` (default 224) and the position embedding is
  interpolated accordingly.
- TIPS takes un-normalized [0, 1] images, whereas the data loaders standardize
  their inputs. The encoder therefore rescales its input with the ImageNet
  mean/std before the ViT; this is the setting used for our TIPS results.
- xformers is disabled by default (XFORMERS_DISABLED=1) to use standard
  attention, which keeps runs reproducible.
"""
import os
os.environ.setdefault("XFORMERS_DISABLED", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import tips_vit


_TIPS_BUILDERS = {
    'tips_vit_b14':  (tips_vit.vit_base,   'mlp',    768),
    'tips_vit_l14':  (tips_vit.vit_large,  'mlp',    1024),
    'tips_vit_so14': (tips_vit.vit_so400m, 'mlp',    1152),
}

# Default intermediate layers per ViT depth (four evenly spaced blocks)
_DEFAULT_LAYERS = {12: [2, 5, 8, 11], 24: [5, 11, 17, 23]}


class TIPSEncoder(nn.Module):
    def __init__(self, backbone_name='tips_vit_b14', d_model=256,
                 npz_path=None, img_size=224, layers=None, pretrained=True):
        super().__init__()
        assert backbone_name in _TIPS_BUILDERS, f'unknown TIPS backbone {backbone_name}'
        builder, ffn, embed_dim = _TIPS_BUILDERS[backbone_name]
        self.img_size = img_size

        # Built at 448 to match the checkpoint's 32x32 position embedding
        self.vit = builder(
            img_size=448, patch_size=14, ffn_layer=ffn, block_chunks=0,
            init_values=1.0, interpolate_antialias=True, interpolate_offset=0.0,
        )
        depth = len(self.vit.blocks)
        self.layers = layers if layers is not None else _DEFAULT_LAYERS.get(depth, list(range(depth - 4, depth)))

        if pretrained:
            if npz_path is None:
                raise ValueError(
                    'A TIPS backbone needs --tips_npz pointing to the vision '
                    'checkpoint (*_vision.npz) of the chosen variant.')
            npz = npz_path
            ckpt = {k: torch.tensor(v) for k, v in dict(np.load(npz, allow_pickle=False)).items()}
            missing, unexpected = self.vit.load_state_dict(ckpt, strict=False)
            print(f'[TIPSEncoder] loaded {os.path.basename(npz)} | missing={len(missing)} unexpected={len(unexpected)}')

        # ImageNet statistics used to rescale the standardized input (see forward)
        self.register_buffer('inorm_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('inorm_std',  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # Per-scale projection: embed_dim -> d_model (resampling happens in forward)
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, d_model, 1, bias=False),
                nn.BatchNorm2d(d_model),
                nn.ReLU(inplace=True),
            ) for _ in range(4)
        ])

    def vit_parameters(self):
        """ViT backbone parameters (lower learning-rate group)."""
        return self.vit.parameters()

    def adapter_parameters(self):
        """Projection adapter parameters (decoder learning-rate group)."""
        return self.proj.parameters()

    def forward(self, x):
        B, C, H, W = x.shape
        # Rescale the standardized input back to an image-like range for TIPS
        x01 = x * self.inorm_std + self.inorm_mean
        # Resize to the ViT input resolution
        if self.img_size != H or self.img_size != W:
            xv = F.interpolate(x01, size=(self.img_size, self.img_size),
                               mode='bilinear', align_corners=False)
        else:
            xv = x01
        # Four intermediate layers, all at the patch-grid resolution
        feats = self.vit.get_intermediate_layers(xv, n=self.layers, reshape=True)
        # Resample to 1/4, 1/8, 1/16, 1/32 of the input size, then project
        targets = [(H // 4, W // 4), (H // 8, W // 8), (H // 16, W // 16), (H // 32, W // 32)]
        out = []
        for i, (f, tgt) in enumerate(zip(feats, targets)):
            f = F.interpolate(f, size=tgt, mode='bilinear', align_corners=False)
            out.append(self.proj[i](f))
        return out
