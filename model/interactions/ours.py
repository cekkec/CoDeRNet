"""Confidence-Decomposed Routing (CoDeR).

For a target feature F_t (receiving branch) and source feature F_s
(sending branch) at the same spatial scale:

    C_t = sigmoid(h_t(F_t))      # target-side sufficiency cue
    C_s = sigmoid(h_s(F_s))      # source-side reliability cue
    M   = (1 - C_t) * C_s        # routing mask
    F_t' = F_t + M * g(F_s)      # selective residual update

Routing is active where the target branch is uncertain AND the source branch
is reliable; the lightweight adapter g re-projects the source feature to the
target's representation space before the residual addition.
"""
import torch
import torch.nn as nn

from ._common import _make_adapter, _init_residual_near_zero


class CoDeR(nn.Module):
    def __init__(self, d_model, variant='light'):
        super().__init__()
        # Routing estimators: zero-init -> initial C ~ 0.5
        # M = (1-0.5)*0.5 = 0.25 with near-zero adapter -> initial residual ~ 0.
        self.h_target = nn.Conv2d(d_model, 1, 1)
        self.h_source = nn.Conv2d(d_model, 1, 1)
        nn.init.zeros_(self.h_target.weight)
        nn.init.zeros_(self.h_target.bias)
        nn.init.zeros_(self.h_source.weight)
        nn.init.zeros_(self.h_source.bias)

        # Source -> target adapter; near-zero init for residual stability.
        self.adapter = _make_adapter(d_model, variant)
        _init_residual_near_zero(self.adapter, std=1e-3)

    def forward(self, F_target, F_source):
        C_target = torch.sigmoid(self.h_target(F_target))
        C_source = torch.sigmoid(self.h_source(F_source))
        mask = (1 - C_target) * C_source
        F_target_new = F_target + mask * self.adapter(F_source)

        stats = {
            'C_target': C_target.mean().item(),
            'C_source': C_source.mean().item(),
            'mask': mask.mean().item(),
        }
        return F_target_new, stats
