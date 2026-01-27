"""
Dataset Preprocessing Script for ShanghaiTech Crowd Counting.

This script performs the following steps:
1. Downloads the ShanghaiTech dataset from Kaggle using kagglehub
2. Parses .mat annotation files to extract head positions
3. Generates density maps using Gaussian kernels
4. Creates train/val/test splits
5. Saves all preprocessed data

ShanghaiTech Dataset Structure:
- Part_A: Dense crowds (avg 501 people per image)
- Part_B: Sparse crowds (avg 123 people per image)

Each image has a corresponding .mat file with 'image_info' containing
head positions as (x, y) coordinates.

Run this script before training:
    python preprocess.py

Author: Thesis Implementation
"""

import os
import sys
import glob
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm
import random
import re

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from utils.density_generator import generate_density_map


def download_dataset():
    """
    Download the ShanghaiTech dataset from Kaggle.
    
    Returns:
        Path to the downloaded dataset directory.
    """
    print("=" * 60)
    print("Step 1: Downloading ShanghaiTech Dataset from Kaggle")
    print("=" * 60)
    
    try:
        import kagglehub
    except ImportError:
        print("Error: kagglehub not installed. Please run:")
        print("  pip install kagglehub")
        sys.exit(1)
    
    print(f"Downloading: {config.KAGGLE_DATASET}")
    print("This may take a few minutes...")
    
    # Download dataset
    dataset_path = kagglehub.dataset_download(config.KAGGLE_DATASET)
    
    print(f"Dataset downloaded to: {dataset_path}")
    return dataset_path


def explore_dataset(dataset_path: str) -> dict:
    """
    Explore the downloaded ShanghaiTech dataset structure.
    
    Args:
        dataset_path: Path to the downloaded dataset.
    
    Returns:
        Dictionary with discovered paths and structure info.
    """
    print("\n" + "=" * 60)
    print("Step 2: Exploring Dataset Structure")
    print("=" * 60)
    
    structure = {
        'part_a_train': None,
        'part_a_test': None,
        'part_b_train': None,
        'part_b_test': None,
        'total_images': 0,
        'total_annotations': 0
    }
    
    # Find Part_A and Part_B directories
    for root, dirs, files in os.walk(dataset_path):
        # Count images and annotations
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                structure['total_images'] += 1
            if file.endswith('.mat'):
                structure['total_annotations'] += 1
        
        # Find specific directories
        root_lower = root.lower().replace('\\', '/')
        if 'part_a' in root_lower and 'train' in root_lower and 'images' in root_lower:
            structure['part_a_train'] = os.path.dirname(root)
        elif 'part_a' in root_lower and 'test' in root_lower and 'images' in root_lower:
            structure['part_a_test'] = os.path.dirname(root)
        elif 'part_b' in root_lower and 'train' in root_lower and 'images' in root_lower:
            structure['part_b_train'] = os.path.dirname(root)
        elif 'part_b' in root_lower and 'test' in root_lower and 'images' in root_lower:
            structure['part_b_test'] = os.path.dirname(root)
    
    print(f"\nDataset structure:")
    print(f"  Total images found: {structure['total_images']}")
    print(f"  Total annotations: {structure['total_annotations']}")
    print(f"  Part_A train: {'Found' if structure['part_a_train'] else 'Not found'}")
    print(f"  Part_A test: {'Found' if structure['part_a_test'] else 'Not found'}")
    print(f"  Part_B train: {'Found' if structure['part_b_train'] else 'Not found'}")
    print(f"  Part_B test: {'Found' if structure['part_b_test'] else 'Not found'}")
    
    return structure


def parse_mat_annotation(mat_path: str) -> np.ndarray:
    """
    Parse ShanghaiTech .mat annotation file for head positions.
    
    ShanghaiTech annotations have structure:
    - image_info[0,0]['location'][0,0] contains Nx2 array of (x, y) positions
    
    Args:
        mat_path: Path to .mat file.
    
    Returns:
        Numpy array of shape (N, 2) with head positions.
    """
    try:
        from scipy.io import loadmat
    except ImportError:
        print("Error: scipy not installed for .mat file reading")
        return np.array([]).reshape(0, 2)
    
    try:
        mat = loadmat(mat_path)
        
        # ShanghaiTech format: image_info -> location
        if 'image_info' in mat:
            # Navigate the nested structure
            image_info = mat['image_info']
            # image_info is typically (1,1) containing a struct
            if image_info.shape == (1, 1):
                inner = image_info[0, 0]
                # Look for 'location' field
                if 'location' in inner.dtype.names:
                    locations = inner['location']
                    if locations.shape == (1, 1):
                        points = locations[0, 0]
                        if len(points) > 0:
                            return points.astype(np.float32)
        
        # Alternative format with 'annPoints'
        if 'annPoints' in mat:
            points = mat['annPoints']
            if len(points) > 0:
                return points.astype(np.float32)
        
        # Try direct 'location' key
        if 'location' in mat:
            points = mat['location']
            if len(points) > 0:
                return points.astype(np.float32)
        
        # Try 'gt' key (another common format)
        if 'gt' in mat:
            points = mat['gt']
            if len(points) > 0:
                return points.astype(np.float32)
                
    except Exception as e:
        print(f"Error parsing {mat_path}: {e}")
    
    return np.array([]).reshape(0, 2)


def find_frames_and_annotations(dataset_path: str, use_part: str = 'B') -> list:
    """
    Find all images and their corresponding annotations.
    
    Args:
        dataset_path: Path to the downloaded dataset.
        use_part: 'A' for dense crowds, 'B' for sparse crowds, 'both' for all.
    
    Returns:
        List of dicts with 'image_path', 'points', 'count', and 'split' keys.
    """
    print("\n" + "=" * 60)
    print(f"Step 3: Finding Frames and Annotations (Part {use_part})")
    print("=" * 60)
    
    frames_data = []
    
    # Define search patterns based on selected part
    search_dirs = []
    
    for root, dirs, files in os.walk(dataset_path):
        root_lower = root.lower().replace('\\', '/')
        
        # Filter by part selection
        if use_part.upper() == 'A':
            if 'part_a' not in root_lower:
                continue
        elif use_part.upper() == 'B':
            if 'part_b' not in root_lower:
                continue
        # else 'both' - don't filter
        
        # Find image directories
        if 'images' in root_lower:
            search_dirs.append(root)
    
    print(f"Found {len(search_dirs)} image directories to process")
    
    for img_dir in search_dirs:
        # Determine if train or test split
        is_train = 'train' in img_dir.lower()
        split = 'train' if is_train else 'test'
        
        # Find corresponding ground_truth directory
        gt_dir = img_dir.replace('images', 'ground_truth').replace('images', 'ground-truth')
        if not os.path.exists(gt_dir):
            gt_dir = img_dir.replace('images', 'ground_truth')
        if not os.path.exists(gt_dir):
            # Try sibling directory
            parent = os.path.dirname(img_dir)
            gt_dir = os.path.join(parent, 'ground_truth')
        if not os.path.exists(gt_dir):
            gt_dir = os.path.join(parent, 'ground-truth')
        
        # Get all images in directory
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(glob.glob(os.path.join(img_dir, ext)))
        
        print(f"\n  Processing: {img_dir}")
        print(f"  GT dir: {gt_dir}")
        print(f"  Found {len(image_files)} images, split: {split}")
        
        for img_path in tqdm(image_files, desc=f"Processing {split}"):
            # Find corresponding annotation
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            
            # ShanghaiTech naming: IMG_1.jpg -> GT_IMG_1.mat
            mat_patterns = [
                os.path.join(gt_dir, f"GT_{base_name}.mat"),
                os.path.join(gt_dir, f"{base_name}.mat"),
                os.path.join(gt_dir, f"GT_{base_name.upper()}.mat"),
            ]
            
            points = np.array([]).reshape(0, 2)
            for mat_path in mat_patterns:
                if os.path.exists(mat_path):
                    points = parse_mat_annotation(mat_path)
                    break
            
            frame_info = {
                'image_path': img_path,
                'points': points,
                'count': len(points),
                'split': split
            }
            
            frames_data.append(frame_info)
    
    # Print summary
    train_count = sum(1 for f in frames_data if f['split'] == 'train')
    test_count = sum(1 for f in frames_data if f['split'] == 'test')
    annotated = sum(1 for f in frames_data if f['count'] > 0)
    
    print(f"\n\nProcessed {len(frames_data)} frames")
    print(f"  Train: {train_count}")
    print(f"  Test: {test_count}")
    print(f"  With annotations: {annotated}")
    if annotated > 0:
        avg_count = np.mean([f['count'] for f in frames_data if f['count'] > 0])
        print(f"  Average count: {avg_count:.1f}")
    
    return frames_data


def generate_density_maps(frames_data: list):
    """
    Generate density maps for all frames and save as .npy files.
    
    Args:
        frames_data: List of frame info dicts.
    
    Returns:
        Updated frames_data with 'density_path' added.
    """
    print("\n" + "=" * 60)
    print("Step 4: Generating Density Maps")
    print("=" * 60)
    
    os.makedirs(config.DENSITY_MAPS_DIR, exist_ok=True)
    
    valid_frames = []
    
    for frame in tqdm(frames_data, desc="Generating density maps"):
        img_path = frame['image_path']
        points = frame['points']
        
        # Load image to get dimensions
        try:
            img = Image.open(img_path)
            width, height = img.size  # PIL is (W, H)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            continue
        
        # Skip frames without annotations
        if len(points) == 0:
            continue
        
        # Generate density map
        density_map = generate_density_map((height, width), points)
        
        # Create unique filename
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        # Add random suffix to avoid name collisions
        unique_id = hash(img_path) % 100000
        density_path = os.path.join(config.DENSITY_MAPS_DIR, f"{base_name}_{unique_id}.npy")
        
        # Save density map
        np.save(density_path, density_map)
        
        # Update frame info
        frame['density_path'] = density_path
        valid_frames.append(frame)
    
    print(f"\nGenerated {len(valid_frames)} density maps")
    print(f"Saved to: {config.DENSITY_MAPS_DIR}")
    
    return valid_frames


def create_splits(frames_data: list):
    """
    Create train/val/test splits and save split files.
    
    ShanghaiTech already has train/test split.
    We'll create val split from training data.
    
    Args:
        frames_data: List of frame info dicts with density_path added.
    """
    print("\n" + "=" * 60)
    print("Step 5: Creating Train/Val/Test Splits")
    print("=" * 60)
    
    os.makedirs(config.SPLITS_DIR, exist_ok=True)
    
    # Set random seed for reproducibility
    random.seed(config.RANDOM_SEED)
    
    # Separate existing train and test
    train_data = [f for f in frames_data if f['split'] == 'train' and 'density_path' in f]
    test_data = [f for f in frames_data if f['split'] == 'test' and 'density_path' in f]
    
    # Create validation split from training data (15%)
    random.shuffle(train_data)
    val_size = int(len(train_data) * 0.15)
    val_data = train_data[:val_size]
    train_data = train_data[val_size:]
    
    # Save split files
    splits = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }
    
    for split_name, split_data in splits.items():
        split_path = os.path.join(config.SPLITS_DIR, f"{split_name}.txt")
        with open(split_path, 'w') as f:
            for frame in split_data:
                line = f"{frame['image_path']},{frame['density_path']},{frame['count']}\n"
                f.write(line)
        print(f"  {split_name}: {len(split_data)} samples -> {split_path}")


def copy_dataset_to_local(dataset_path: str):
    """
    Copy downloaded dataset to local data directory.
    
    Args:
        dataset_path: Path to downloaded dataset.
    """
    print("\n" + "=" * 60)
    print("Copying dataset to local directory")
    print("=" * 60)
    
    os.makedirs(config.RAW_DATA_DIR, exist_ok=True)
    
    # Just create a symlink or reference instead of copying everything
    # (ShanghaiTech can be large)
    ref_file = os.path.join(config.RAW_DATA_DIR, "dataset_path.txt")
    with open(ref_file, 'w') as f:
        f.write(dataset_path)
    
    print(f"Dataset reference saved to: {ref_file}")
    print(f"Original dataset at: {dataset_path}")


def main():
    """
    Main preprocessing pipeline for ShanghaiTech dataset.
    """
    print("=" * 60)
    print("ShanghaiTech Dataset Preprocessing Pipeline")
    print("=" * 60)
    print(f"\nProject root: {config.PROJECT_ROOT}")
    print(f"Random seed: {config.RANDOM_SEED}")
    
    # Create directories
    config.create_directories()
    
    # Step 1: Download dataset
    dataset_path = download_dataset()
    
    # Save reference
    copy_dataset_to_local(dataset_path)
    
    # Step 2: Explore dataset structure
    structure = explore_dataset(dataset_path)
    
    # Step 3: Find frames and parse annotations
    # Use Part B (sparse crowds) - more suitable for LCDNet
    # Part A has very dense crowds (500+ people) which may be too challenging
    frames_data = find_frames_and_annotations(dataset_path, use_part='B')
    
    if len(frames_data) == 0:
        print("\nError: No frames found!")
        # Try Part A as fallback
        print("Trying Part A...")
        frames_data = find_frames_and_annotations(dataset_path, use_part='A')
    
    if len(frames_data) == 0:
        print("\nError: No frames processed!")
        sys.exit(1)
    
    # Step 4: Generate density maps
    frames_data = generate_density_maps(frames_data)
    
    if len(frames_data) == 0:
        print("\nError: No density maps generated!")
        sys.exit(1)
    
    # Step 5: Create splits
    create_splits(frames_data)
    
    # Final summary
    print("\n" + "=" * 60)
    print("Preprocessing Complete!")
    print("=" * 60)
    print(f"\nOutput directories:")
    print(f"  Density maps: {config.DENSITY_MAPS_DIR}")
    print(f"  Split files: {config.SPLITS_DIR}")
    print(f"\nTotal frames processed: {len(frames_data)}")
    print(f"\nNext step: Run training with")
    print("  python train.py --epochs 25")


if __name__ == "__main__":
    main()
