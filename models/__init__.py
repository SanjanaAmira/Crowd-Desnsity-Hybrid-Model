"""
Model architectures for Crowd Density Estimation.

This package contains:
- lcdnet: Lightweight Crowd Density Network (for sparse scenes)
- csrnet: CSRNet for dense crowd counting (for congested scenes)
"""

from .lcdnet import LCDNet, DepthwiseSeparableConv
from .csrnet import CSRNet, create_csrnet
