"""Confidence-Decomposed Routing module.

Single direction-block operator with the interface:
    forward(F_target, F_source) -> (F_target_new, stats_dict)

Spatial scale (where each direction operates) is controlled by the outer
CompInteraction via seg_recv_scale / cap_recv_scale.
"""
from .ours import CoDeR

__all__ = ['CoDeR']
