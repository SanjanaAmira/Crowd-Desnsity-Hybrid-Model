"""
Utility modules for LCDNet Crowd Density Estimation.

This package contains:
- density_generator: Functions for creating density maps from annotations
- metrics: Evaluation metrics (MAE, MSE) for crowd counting
"""

from .density_generator import generate_density_map, create_gaussian_kernel
from .metrics import compute_mae, compute_mse, evaluate_model
