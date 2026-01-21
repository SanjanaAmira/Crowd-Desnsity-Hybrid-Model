"""
Evaluation Metrics for Crowd Counting.

This module provides evaluation metrics for crowd density estimation:
- MAE (Mean Absolute Error): Average absolute difference between predicted
  and ground truth counts. Lower is better.
- MSE (Mean Squared Error): Average squared difference. Penalizes large
  errors more than MAE. Lower is better.

These are the standard metrics used in crowd counting literature.

Author: Thesis Implementation
"""

import numpy as np
import torch
from typing import Union, Tuple, List


def compute_mae(
    pred_counts: Union[np.ndarray, List[float]],
    gt_counts: Union[np.ndarray, List[float]]
) -> float:
    """
    Compute Mean Absolute Error between predicted and ground truth counts.
    
    MAE = (1/N) * Σ|pred_i - gt_i|
    
    This metric represents the average error in person count across all
    test images. For example, MAE of 2.5 means on average, the model's
    count is off by 2.5 people per image.
    
    Args:
        pred_counts: Array of predicted counts (one per image).
        gt_counts: Array of ground truth counts (one per image).
    
    Returns:
        Mean Absolute Error as a float.
    
    Example:
        >>> compute_mae([10, 20, 30], [12, 18, 32])
        2.0
    """
    pred_counts = np.array(pred_counts)
    gt_counts = np.array(gt_counts)
    
    assert len(pred_counts) == len(gt_counts), \
        f"Length mismatch: {len(pred_counts)} vs {len(gt_counts)}"
    
    mae = np.mean(np.abs(pred_counts - gt_counts))
    return float(mae)


def compute_mse(
    pred_counts: Union[np.ndarray, List[float]],
    gt_counts: Union[np.ndarray, List[float]]
) -> float:
    """
    Compute Mean Squared Error between predicted and ground truth counts.
    
    MSE = (1/N) * Σ(pred_i - gt_i)²
    
    MSE penalizes large errors more heavily than MAE due to the squaring.
    This makes it useful for identifying models that occasionally make
    very large errors.
    
    Args:
        pred_counts: Array of predicted counts (one per image).
        gt_counts: Array of ground truth counts (one per image).
    
    Returns:
        Mean Squared Error as a float.
    
    Note:
        Often, papers report RMSE (Root MSE) = sqrt(MSE) for interpretability.
    
    Example:
        >>> compute_mse([10, 20, 30], [12, 18, 32])
        4.0
    """
    pred_counts = np.array(pred_counts)
    gt_counts = np.array(gt_counts)
    
    assert len(pred_counts) == len(gt_counts), \
        f"Length mismatch: {len(pred_counts)} vs {len(gt_counts)}"
    
    mse = np.mean((pred_counts - gt_counts) ** 2)
    return float(mse)


def compute_rmse(
    pred_counts: Union[np.ndarray, List[float]],
    gt_counts: Union[np.ndarray, List[float]]
) -> float:
    """
    Compute Root Mean Squared Error.
    
    RMSE = sqrt(MSE)
    
    RMSE is in the same units as the counts, making it more interpretable.
    
    Args:
        pred_counts: Array of predicted counts.
        gt_counts: Array of ground truth counts.
    
    Returns:
        Root Mean Squared Error as a float.
    """
    return np.sqrt(compute_mse(pred_counts, gt_counts))


def density_to_count(density_map: Union[np.ndarray, torch.Tensor]) -> float:
    """
    Convert a density map to a person count by summing all pixel values.
    
    The density map is constructed such that its integral (sum) equals
    the number of people in the image.
    
    Args:
        density_map: 2D array or tensor containing the density map.
                    Can be single map or batch.
    
    Returns:
        Total count (float). For batches, returns sum for each map.
    """
    if isinstance(density_map, torch.Tensor):
        # Handle batch dimension if present
        if density_map.dim() == 4:  # (B, C, H, W)
            return density_map.sum(dim=(1, 2, 3)).cpu().numpy()
        elif density_map.dim() == 3:  # (C, H, W)
            return float(density_map.sum().cpu().numpy())
        else:  # (H, W)
            return float(density_map.sum().cpu().numpy())
    else:
        return float(np.sum(density_map))


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cuda"
) -> Tuple[float, float, float]:
    """
    Evaluate a crowd counting model on a dataset.
    
    Runs inference on all images in the dataloader and computes
    MAE, MSE, and RMSE metrics.
    
    Args:
        model: PyTorch model that takes images and returns density maps.
        dataloader: DataLoader yielding (images, density_maps, counts).
        device: Device to run inference on ('cuda' or 'cpu').
    
    Returns:
        Tuple of (MAE, MSE, RMSE).
    """
    model.eval()
    
    pred_counts = []
    gt_counts = []
    
    with torch.no_grad():
        for images, density_maps, counts in dataloader:
            images = images.to(device)
            
            # Forward pass
            pred_density = model(images)
            
            # Convert density maps to counts
            batch_pred_counts = density_to_count(pred_density)
            batch_gt_counts = counts.numpy()
            
            # Handle single-sample case
            if isinstance(batch_pred_counts, float):
                pred_counts.append(batch_pred_counts)
            else:
                pred_counts.extend(batch_pred_counts.tolist())
            
            if isinstance(batch_gt_counts, float):
                gt_counts.append(batch_gt_counts)
            else:
                gt_counts.extend(batch_gt_counts.tolist())
    
    mae = compute_mae(pred_counts, gt_counts)
    mse = compute_mse(pred_counts, gt_counts)
    rmse = compute_rmse(pred_counts, gt_counts)
    
    return mae, mse, rmse


if __name__ == "__main__":
    # Test metrics
    print("Testing evaluation metrics...")
    
    # Test data
    pred = [10.5, 20.3, 30.1, 45.2, 12.8]
    gt = [10, 22, 28, 45, 15]
    
    mae = compute_mae(pred, gt)
    mse = compute_mse(pred, gt)
    rmse = compute_rmse(pred, gt)
    
    print(f"Predictions: {pred}")
    print(f"Ground Truth: {gt}")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    # Test density to count
    density_map = np.random.rand(256, 256) * 0.001
    density_map[128, 128] = 1.0  # Simulate one person
    count = density_to_count(density_map)
    print(f"\nDensity map sum: {count:.4f}")
    
    print("\nMetrics test passed!")
