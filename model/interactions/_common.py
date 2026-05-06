"""Shared utilities for the CoDeR direction block."""
import torch.nn as nn


def _make_adapter(d_model, variant='light'):
    """Source-to-target channel projection. Keeps dim: d_model -> d_model."""
    if variant == 'light':
        return nn.Conv2d(d_model, d_model, 1, bias=False)
    elif variant == 'deep':
        return nn.Sequential(
            nn.Conv2d(d_model, d_model, 1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model, d_model, 3, padding=1, bias=False),
        )
    else:
        raise ValueError(f'Unknown variant: {variant}')


def _init_residual_near_zero(module, std=1e-3):
    """Near-zero init for conv/linear layers in the residual branch so the
    initial routed output is approximately zero (i.e., F_target' ~ F_target
    at initialization)."""
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.normal_(m.weight, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
