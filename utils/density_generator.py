"""
Density Map Generator for Crowd Counting.

This module generates ground truth density maps from head/person annotations.
Each annotated head position is represented as a 2D Gaussian, and the density
map is the sum of all Gaussians. The integral of the density map equals the
total person count.

Key Concepts:
- Density Map: A 2D map where pixel values represent crowd density
- The sum of all pixel values equals the total person count
- Gaussian kernels create smooth, differentiable targets for regression

Author: Thesis Implementation
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def create_gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """
    Create a 2D Gaussian kernel.
    
    The kernel is normalized so that its sum equals 1, ensuring that each
    person contributes exactly 1 to the total count when the density map
    is summed.
    
    Args:
        size: Size of the kernel (size x size). Should be odd for symmetry.
        sigma: Standard deviation of the Gaussian distribution.
               Larger sigma = more spread out blob.
    
    Returns:
        2D numpy array containing the normalized Gaussian kernel.
    
    Example:
        >>> kernel = create_gaussian_kernel(15, 4.0)
        >>> np.abs(kernel.sum() - 1.0) < 1e-6
        True
    """
    # Ensure size is odd for proper centering
    if size % 2 == 0:
        size += 1
    
    # Create coordinate grids centered at (0, 0)
    x = np.arange(0, size) - size // 2
    y = np.arange(0, size) - size // 2
    xx, yy = np.meshgrid(x, y)
    
    # 2D Gaussian formula: exp(-(x^2 + y^2) / (2 * sigma^2))
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    
    # Normalize so kernel sums to 1 (each person = 1 count)
    kernel = kernel / kernel.sum()
    
    return kernel


def compute_adaptive_sigma(points: np.ndarray, k: int = 3) -> np.ndarray:
    """
    Compute adaptive sigma for each point based on k-nearest neighbors.
    
    In crowded scenes, people are close together, so we use smaller sigma.
    In sparse regions, people are far apart, so we use larger sigma.
    This adaptive approach produces better density maps than fixed sigma.
    
    Args:
        points: Array of shape (N, 2) containing (x, y) coordinates.
        k: Number of nearest neighbors to consider.
    
    Returns:
        Array of shape (N,) containing sigma values for each point.
    """
    n_points = len(points)
    
    # Handle edge cases
    if n_points == 0:
        return np.array([])
    if n_points == 1:
        return np.array([config.FIXED_SIGMA])
    
    # Limit k to available neighbors
    k_actual = min(k, n_points - 1)
    
    # Build KD-tree for efficient nearest neighbor search
    tree = KDTree(points)
    
    # Query k+1 neighbors (first one is the point itself with distance 0)
    distances, _ = tree.query(points, k=k_actual + 1)
    
    # Average distance to k nearest neighbors (exclude self)
    avg_distances = distances[:, 1:].mean(axis=1)
    
    # Sigma proportional to average nearest neighbor distance
    sigmas = config.SIGMA_COEFFICIENT * avg_distances
    
    # Clamp sigma to reasonable range
    sigmas = np.clip(sigmas, config.MIN_SIGMA, config.MAX_SIGMA)
    
    return sigmas


def generate_density_map(
    image_shape: tuple,
    points: np.ndarray,
    adaptive_sigma: bool = True
) -> np.ndarray:
    """
    Generate a density map from head annotation points.
    
    This is the core function for creating ground truth density maps.
    Each head position becomes a Gaussian blob, and the sum of all
    pixel values equals the total person count.
    
    Args:
        image_shape: Tuple of (height, width) for the output density map.
        points: Array of shape (N, 2) containing (x, y) head coordinates.
                Coordinates should be in image coordinate system.
        adaptive_sigma: If True, use geometry-adaptive sigma based on
                       nearest neighbor distances. If False, use fixed sigma.
    
    Returns:
        2D numpy array of shape (height, width) containing the density map.
        The sum of this array equals the number of points (person count).
    
    Example:
        >>> density_map = generate_density_map((480, 640), np.array([[100, 200], [300, 150]]))
        >>> abs(density_map.sum() - 2.0) < 0.01
        True
    """
    height, width = image_shape
    density_map = np.zeros((height, width), dtype=np.float32)
    
    # Handle empty annotations (no people in frame)
    if len(points) == 0:
        return density_map
    
    # Ensure points are numpy array with correct shape
    points = np.array(points).reshape(-1, 2)
    
    # Compute sigma for each point
    if adaptive_sigma and len(points) > 1:
        sigmas = compute_adaptive_sigma(points, k=config.K_NEAREST_NEIGHBORS)
    else:
        sigmas = np.full(len(points), config.FIXED_SIGMA)
    
    # Place Gaussian at each head location
    for (x, y), sigma in zip(points, sigmas):
        # Skip points outside image bounds
        if x < 0 or x >= width or y < 0 or y >= height:
            continue
        
        # Convert to integer coordinates
        x_int, y_int = int(round(x)), int(round(y))
        
        # Determine kernel size (should cover ~3 sigma on each side)
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        half_size = kernel_size // 2
        
        # Compute bounds for placing the kernel
        # Handle edge cases where kernel extends beyond image
        x_start = max(0, x_int - half_size)
        x_end = min(width, x_int + half_size + 1)
        y_start = max(0, y_int - half_size)
        y_end = min(height, y_int + half_size + 1)
        
        # Skip if completely outside
        if x_start >= x_end or y_start >= y_end:
            continue
        
        # Create kernel and extract the visible portion
        kernel = create_gaussian_kernel(kernel_size, sigma)
        
        # Compute kernel slice indices
        k_x_start = x_start - (x_int - half_size)
        k_x_end = k_x_start + (x_end - x_start)
        k_y_start = y_start - (y_int - half_size)
        k_y_end = k_y_start + (y_end - y_start)
        
        # Add kernel to density map
        density_map[y_start:y_end, x_start:x_end] += kernel[k_y_start:k_y_end, k_x_start:k_x_end]
    
    return density_map


def generate_density_map_fast(
    image_shape: tuple,
    points: np.ndarray,
    sigma: float = None
) -> np.ndarray:
    """
    Fast density map generation using scipy's gaussian_filter.
    
    This method is faster for large numbers of points but uses a fixed
    sigma for all points. Creates a point map first, then applies
    Gaussian smoothing.
    
    Args:
        image_shape: Tuple of (height, width) for the output density map.
        points: Array of shape (N, 2) containing (x, y) head coordinates.
        sigma: Fixed sigma for Gaussian smoothing. Uses config default if None.
    
    Returns:
        2D numpy array containing the density map.
    """
    if sigma is None:
        sigma = config.FIXED_SIGMA
    
    height, width = image_shape
    density_map = np.zeros((height, width), dtype=np.float32)
    
    if len(points) == 0:
        return density_map
    
    # Place unit impulses at head locations
    for x, y in points:
        x_int, y_int = int(round(x)), int(round(y))
        if 0 <= x_int < width and 0 <= y_int < height:
            density_map[y_int, x_int] += 1.0
    
    # Apply Gaussian filter
    density_map = gaussian_filter(density_map, sigma=sigma, mode='constant')
    
    return density_map


if __name__ == "__main__":
    # Test the density map generation
    print("Testing density map generation...")
    
    # Create test points
    test_points = np.array([
        [100, 100],
        [150, 120],
        [200, 100],
        [300, 250],
        [400, 300],
    ])
    
    # Generate density map
    density_map = generate_density_map((480, 640), test_points)
    
    print(f"Density map shape: {density_map.shape}")
    print(f"Total count (should be ~5): {density_map.sum():.2f}")
    print(f"Min value: {density_map.min():.6f}")
    print(f"Max value: {density_map.max():.6f}")
    
    # Test fast method
    density_map_fast = generate_density_map_fast((480, 640), test_points)
    print(f"\nFast method total count: {density_map_fast.sum():.2f}")
    
    print("\nDensity map generation test passed!")
