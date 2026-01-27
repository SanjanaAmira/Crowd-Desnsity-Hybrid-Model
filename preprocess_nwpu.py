"""
Preprocessing Script for NWPU-Crowd Dataset.

This script generates density maps from NWPU-Crowd .mat annotations:
1. Parse .mat files to extract head point coordinates
2. Generate Gaussian density maps (adaptive sigma)
3. Save density maps as .npy files
4. Create processed split files for training

NWPU-Crowd Dataset Structure:
- images_part1 to images_part5: JPEG images
- mats/: .mat annotation files (UCF-QNRF compatible format)
- train.txt, val.txt, test.txt: official splits

Run this script before training CSRNet:
    python preprocess_nwpu.py

Author: Thesis Implementation - Phase 2 Part 2
"""

import os
import sys
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


# ============================================================================
# NWPU-CROWD SPECIFIC CONFIGURATION
# ============================================================================

# Path to NWPU-Crowd dataset
NWPU_DATA_DIR = os.path.join(config.DATA_DIR, "NWPU-Crowd")

# Output directories
NWPU_DENSITY_MAPS_DIR = os.path.join(NWPU_DATA_DIR, "density_maps")
NWPU_SPLITS_DIR = os.path.join(NWPU_DATA_DIR, "splits")

# Image directories (images are split across multiple folders)
IMAGE_PARTS = ["images_part1", "images_part2", "images_part3", "images_part4", "images_part5"]


def find_image_path(image_id: str) -> str:
    """
    Find the full path to an image given its ID.
    
    NWPU-Crowd images are distributed across images_part1 to images_part5.
    Image files are named as {image_id}.jpg
    
    Args:
        image_id: Image identifier (e.g., "0001")
    
    Returns:
        Full path to the image file, or None if not found.
    """
    for part in IMAGE_PARTS:
        img_path = os.path.join(NWPU_DATA_DIR, part, f"{image_id}.jpg")
        if os.path.exists(img_path):
            return img_path
    
    # Also try .jpeg extension
    for part in IMAGE_PARTS:
        img_path = os.path.join(NWPU_DATA_DIR, part, f"{image_id}.jpeg")
        if os.path.exists(img_path):
            return img_path
    
    return None


def parse_nwpu_mat(mat_path: str) -> np.ndarray:
    """
    Parse NWPU-Crowd .mat annotation file for head positions.
    
    NWPU-Crowd uses UCF-QNRF compatible format with 'annPoints' key
    containing Nx2 array of (x, y) head coordinates.
    
    Args:
        mat_path: Path to .mat annotation file.
    
    Returns:
        Numpy array of shape (N, 2) with head positions [(x, y), ...].
        Returns empty array if no annotations found.
    """
    try:
        mat = loadmat(mat_path)
        
        # NWPU-Crowd format: 'annPoints' contains head coordinates
        if 'annPoints' in mat:
            points = mat['annPoints']
            if len(points) > 0:
                return points.astype(np.float32)
        
        # Alternative format checks
        for key in ['gt', 'location', 'points']:
            if key in mat:
                points = mat[key]
                if len(points) > 0:
                    return points.astype(np.float32)
        
    except Exception as e:
        print(f"Error parsing {mat_path}: {e}")
    
    return np.array([]).reshape(0, 2)


def generate_density_map_fixed_sigma(
    image_shape: tuple,
    points: np.ndarray,
    sigma: float = 15.0
) -> np.ndarray:
    """
    Generate density map using fixed Gaussian sigma.
    
    This is a simpler approach that works well for CSRNet.
    Each head location is represented by a Gaussian blob.
    The sum of the density map equals the number of people.
    
    Args:
        image_shape: (height, width) of the image.
        points: Nx2 array of head coordinates [(x, y), ...].
        sigma: Standard deviation of Gaussian kernel.
    
    Returns:
        Density map of shape (height, width).
    """
    height, width = image_shape
    density_map = np.zeros((height, width), dtype=np.float32)
    
    if len(points) == 0:
        return density_map
    
    # Place a point at each head location
    for point in points:
        x, y = int(point[0]), int(point[1])
        
        # Clip to image bounds
        x = min(max(0, x), width - 1)
        y = min(max(0, y), height - 1)
        
        density_map[y, x] += 1.0
    
    # Apply Gaussian filter
    density_map = gaussian_filter(density_map, sigma=sigma, mode='constant')
    
    # Normalize to preserve count
    if density_map.sum() > 0:
        density_map = density_map * len(points) / density_map.sum()
    
    return density_map


def generate_density_map_adaptive(
    image_shape: tuple,
    points: np.ndarray,
    k: int = 3,
    sigma_coefficient: float = 0.3,
    min_sigma: float = 4.0,
    max_sigma: float = 30.0
) -> np.ndarray:
    """
    Generate density map using adaptive Gaussian sigma based on k-nearest neighbors.
    
    For each head, sigma is computed as:
        sigma = coefficient * average_distance_to_k_nearest_neighbors
    
    This adapts the Gaussian spread based on local crowd density:
    - Dense regions: smaller sigma for precise localization
    - Sparse regions: larger sigma for better coverage
    
    Args:
        image_shape: (height, width) of the image.
        points: Nx2 array of head coordinates [(x, y), ...].
        k: Number of nearest neighbors for sigma calculation.
        sigma_coefficient: Multiplier for average distance.
        min_sigma: Minimum allowed sigma value.
        max_sigma: Maximum allowed sigma value.
    
    Returns:
        Density map of shape (height, width).
    """
    height, width = image_shape
    density_map = np.zeros((height, width), dtype=np.float32)
    
    num_points = len(points)
    if num_points == 0:
        return density_map
    
    # If only one point, use fixed sigma
    if num_points == 1:
        point = points[0]
        x, y = int(point[0]), int(point[1])
        x = min(max(0, x), width - 1)
        y = min(max(0, y), height - 1)
        
        # Create single point density map
        pt_map = np.zeros((height, width), dtype=np.float32)
        pt_map[y, x] = 1.0
        density_map = gaussian_filter(pt_map, sigma=min_sigma, mode='constant')
        return density_map
    
    # Compute pairwise distances for adaptive sigma
    from scipy.spatial import distance
    
    dist_matrix = distance.cdist(points, points, metric='euclidean')
    
    # For each point, find k nearest neighbors (excluding self)
    for i, point in enumerate(points):
        x, y = point[0], point[1]
        
        # Get distances to other points
        distances = dist_matrix[i]
        
        # Sort and get k nearest (excluding self at index 0)
        sorted_distances = np.sort(distances)
        k_distances = sorted_distances[1:min(k + 1, len(sorted_distances))]
        
        # Compute adaptive sigma
        if len(k_distances) > 0:
            avg_distance = np.mean(k_distances)
            sigma = sigma_coefficient * avg_distance
            sigma = np.clip(sigma, min_sigma, max_sigma)
        else:
            sigma = min_sigma
        
        # Create Gaussian for this point
        x_int, y_int = int(x), int(y)
        x_int = min(max(0, x_int), width - 1)
        y_int = min(max(0, y_int), height - 1)
        
        # Create a small density map for this point
        pt_map = np.zeros((height, width), dtype=np.float32)
        pt_map[y_int, x_int] = 1.0
        
        # Apply Gaussian filter with adaptive sigma
        pt_density = gaussian_filter(pt_map, sigma=sigma, mode='constant')
        
        # Normalize to sum to 1 and add to total density
        if pt_density.sum() > 0:
            pt_density = pt_density / pt_density.sum()
        
        density_map += pt_density
    
    return density_map


def load_split_file(split_name: str) -> list:
    """
    Load image IDs from NWPU-Crowd split file.
    
    Split file format:
    - train.txt / val.txt: "image_id luminance_label scene_level"
    - test.txt: "image_id" only
    
    Args:
        split_name: One of 'train', 'val', 'test'.
    
    Returns:
        List of image IDs.
    """
    split_path = os.path.join(NWPU_DATA_DIR, f"{split_name}.txt")
    
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Split file not found: {split_path}")
    
    image_ids = []
    with open(split_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                image_ids.append(parts[0])  # First element is image_id
    
    return image_ids


def process_split(split_name: str, use_fixed_sigma: bool = True) -> list:
    """
    Process all images in a split: generate density maps and save.
    
    Args:
        split_name: One of 'train', 'val', 'test'.
        use_fixed_sigma: If True, use fixed sigma; else use adaptive.
    
    Returns:
        List of processed sample info dicts.
    """
    print(f"\n{'=' * 60}")
    print(f"Processing {split_name} split")
    print(f"{'=' * 60}")
    
    image_ids = load_split_file(split_name)
    print(f"Found {len(image_ids)} images in {split_name} split")
    
    processed_samples = []
    skipped_count = 0
    
    for image_id in tqdm(image_ids, desc=f"Processing {split_name}"):
        # Find image path
        image_path = find_image_path(image_id)
        if image_path is None:
            skipped_count += 1
            continue
        
        # Find annotation path
        mat_path = os.path.join(NWPU_DATA_DIR, "mats", f"{image_id}.mat")
        if not os.path.exists(mat_path):
            skipped_count += 1
            continue
        
        # Load image to get dimensions
        try:
            img = Image.open(image_path)
            width, height = img.size
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            skipped_count += 1
            continue
        
        # Parse annotations
        points = parse_nwpu_mat(mat_path)
        count = len(points)
        
        # Generate density map
        if use_fixed_sigma:
            density_map = generate_density_map_fixed_sigma(
                (height, width), points, sigma=15.0
            )
        else:
            density_map = generate_density_map_adaptive(
                (height, width), points,
                k=3, sigma_coefficient=0.3
            )
        
        # Verify sum equals count (within tolerance)
        density_sum = density_map.sum()
        if count > 0 and abs(density_sum - count) > 0.1:
            # Normalize to ensure count preservation
            density_map = density_map * (count / density_sum)
        
        # Save density map
        density_path = os.path.join(NWPU_DENSITY_MAPS_DIR, f"{image_id}.npy")
        np.save(density_path, density_map)
        
        # Store sample info
        processed_samples.append({
            'image_id': image_id,
            'image_path': image_path,
            'density_path': density_path,
            'count': count
        })
    
    print(f"Processed: {len(processed_samples)} | Skipped: {skipped_count}")
    
    return processed_samples


def save_processed_splits(samples_dict: dict):
    """
    Save processed split files with image_path,density_path,count format.
    
    Args:
        samples_dict: Dictionary mapping split_name to list of samples.
    """
    os.makedirs(NWPU_SPLITS_DIR, exist_ok=True)
    
    for split_name, samples in samples_dict.items():
        split_path = os.path.join(NWPU_SPLITS_DIR, f"{split_name}.txt")
        
        with open(split_path, 'w') as f:
            for sample in samples:
                line = f"{sample['image_path']},{sample['density_path']},{sample['count']}\n"
                f.write(line)
        
        print(f"Saved {split_name} split: {len(samples)} samples -> {split_path}")


def print_statistics(samples_dict: dict):
    """Print dataset statistics."""
    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)
    
    for split_name, samples in samples_dict.items():
        counts = [s['count'] for s in samples]
        
        print(f"\n{split_name.upper()} Split:")
        print(f"  Total samples: {len(samples)}")
        
        if counts:
            print(f"  Count range: {min(counts)} - {max(counts)}")
            print(f"  Mean count: {np.mean(counts):.1f}")
            print(f"  Median count: {np.median(counts):.1f}")
            print(f"  Std count: {np.std(counts):.1f}")
            
            # Count distribution
            zero_count = sum(1 for c in counts if c == 0)
            sparse = sum(1 for c in counts if 0 < c <= 50)
            medium = sum(1 for c in counts if 50 < c <= 500)
            dense = sum(1 for c in counts if c > 500)
            
            print(f"  Distribution:")
            print(f"    Empty (0):       {zero_count}")
            print(f"    Sparse (1-50):   {sparse}")
            print(f"    Medium (51-500): {medium}")
            print(f"    Dense (500+):    {dense}")


def main():
    """
    Main preprocessing pipeline for NWPU-Crowd dataset.
    """
    print("=" * 60)
    print("NWPU-Crowd Dataset Preprocessing for CSRNet")
    print("=" * 60)
    print(f"\nDataset path: {NWPU_DATA_DIR}")
    
    # Check dataset exists
    if not os.path.exists(NWPU_DATA_DIR):
        print(f"\nError: NWPU-Crowd dataset not found at {NWPU_DATA_DIR}")
        print("Please download and extract the dataset first.")
        sys.exit(1)
    
    # Create output directories
    os.makedirs(NWPU_DENSITY_MAPS_DIR, exist_ok=True)
    os.makedirs(NWPU_SPLITS_DIR, exist_ok=True)
    
    print(f"\nDensity maps will be saved to: {NWPU_DENSITY_MAPS_DIR}")
    print(f"Split files will be saved to: {NWPU_SPLITS_DIR}")
    
    # Process each split
    samples_dict = {}
    
    # Process train split
    samples_dict['train'] = process_split('train', use_fixed_sigma=True)
    
    # Process val split
    samples_dict['val'] = process_split('val', use_fixed_sigma=True)
    
    # Process test split (no ground truth for test, but we still need the split)
    # Note: Test set has no annotations in NWPU-Crowd (online evaluation only)
    # We'll create density maps if annotations exist, otherwise skip
    try:
        samples_dict['test'] = process_split('test', use_fixed_sigma=True)
    except Exception as e:
        print(f"\nNote: Test split processing skipped: {e}")
        print("NWPU-Crowd test set is for online evaluation only.")
        samples_dict['test'] = []
    
    # Save processed split files
    print("\n" + "=" * 60)
    print("Saving processed split files")
    print("=" * 60)
    save_processed_splits(samples_dict)
    
    # Print statistics
    print_statistics(samples_dict)
    
    # Final summary
    print("\n" + "=" * 60)
    print("Preprocessing Complete!")
    print("=" * 60)
    print(f"\nDensity maps saved to: {NWPU_DENSITY_MAPS_DIR}")
    print(f"Split files saved to: {NWPU_SPLITS_DIR}")
    print(f"\nNext steps:")
    print("  1. python train_csrnet.py --epochs 100")
    print("  2. python evaluate_csrnet.py --visualize")


if __name__ == "__main__":
    main()
