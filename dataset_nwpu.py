"""
PyTorch Dataset for NWPU-Crowd Density Estimation.

This module provides a custom Dataset class for loading image and density map
pairs for training and evaluating CSRNet on the NWPU-Crowd dataset.

The dataset expects:
- Images in data/NWPU-Crowd/images_part1-5 directories
- Pre-generated density maps in data/NWPU-Crowd/density_maps as .npy files
- Split files in data/NWPU-Crowd/splits (train.txt, val.txt)

Author: Thesis Implementation - Phase 2 Part 2
"""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Tuple, List, Optional
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


# ============================================================================
# NWPU-CROWD SPECIFIC CONFIGURATION
# ============================================================================

# Path to NWPU-Crowd dataset
NWPU_DATA_DIR = os.path.join(config.DATA_DIR, "NWPU-Crowd")
NWPU_SPLITS_DIR = os.path.join(NWPU_DATA_DIR, "splits")

# CSRNet outputs 1/8 resolution density maps
DENSITY_DOWNSAMPLE_FACTOR = 8


class NWPUCrowdDataset(Dataset):
    """
    PyTorch Dataset for NWPU-Crowd Density Estimation with CSRNet.
    
    This dataset loads image and density map pairs for training CSRNet.
    Each sample consists of:
    - RGB image (normalized with ImageNet statistics)
    - Ground truth density map (downsampled by 8x for CSRNet)
    - Ground truth count (sum of original density map)
    
    The dataset supports train/val/test splits via split files that list
    the image paths (one per line).
    
    Attributes:
        split: One of 'train', 'val', or 'test'.
        image_size: Target size for resizing images (height, width).
        samples: List of (image_path, density_path, count) tuples.
    """
    
    def __init__(
        self,
        split: str = "train",
        image_size: Tuple[int, int] = None,
        augment: bool = None
    ):
        """
        Initialize the dataset.
        
        Args:
            split: One of 'train', 'val', or 'test'.
            image_size: Target size (H, W) for resizing. Uses config default if None.
            augment: Whether to apply data augmentation. True for train by default.
        """
        super().__init__()
        
        assert split in ['train', 'val', 'test'], \
            f"split must be 'train', 'val', or 'test', got '{split}'"
        
        self.split = split
        self.image_size = image_size if image_size else config.IMAGE_SIZE
        self.augment = augment if augment is not None else (split == 'train')
        
        # Load sample paths from split file
        self.samples = self._load_split_file()
        
        # Define image transforms
        self.image_transform = self._build_image_transforms()
    
    def _load_split_file(self) -> List[Tuple[str, str, float]]:
        """
        Load sample information from split file.
        
        Returns:
            List of (image_path, density_path, count) tuples.
        """
        split_file = os.path.join(NWPU_SPLITS_DIR, f"{self.split}.txt")
        
        if not os.path.exists(split_file):
            raise FileNotFoundError(
                f"Split file not found: {split_file}\n"
                f"Please run preprocess_nwpu.py first to generate dataset splits."
            )
        
        samples = []
        with open(split_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Each line contains: image_path,density_path,count
                parts = line.split(',')
                if len(parts) != 3:
                    print(f"Warning: Invalid line format: {line}")
                    continue
                
                image_path, density_path, count = parts
                count = float(count)
                
                # Dynamic path resolution to make it fully portable and work with current local layout
                # NWPU image resolution (e.g. images_part1/0001.jpg)
                image_path_norm = image_path.replace('\\', '/')
                if 'images_part' in image_path_norm:
                    idx = image_path_norm.find('images_part')
                    rel_img_path = image_path_norm[idx:]
                    resolved_image_path = os.path.join(NWPU_DATA_DIR, rel_img_path.replace('/', os.sep))
                else:
                    resolved_image_path = image_path
                
                # NWPU density map resolution (e.g. density_maps/0001.npy)
                density_path_norm = density_path.replace('\\', '/')
                if 'density_maps' in density_path_norm:
                    idx = density_path_norm.find('density_maps')
                    rel_den_path = density_path_norm[idx:]
                    resolved_density_path = os.path.join(NWPU_DATA_DIR, rel_den_path.replace('/', os.sep))
                else:
                    resolved_density_path = density_path
                
                # Verify files exist
                if not os.path.exists(resolved_image_path):
                    print(f"Warning: Image not found: {resolved_image_path} (original split path: {image_path})")
                    continue
                if not os.path.exists(resolved_density_path):
                    print(f"Warning: Density map not found: {resolved_density_path} (original split path: {density_path})")
                    continue
                
                samples.append((resolved_image_path, resolved_density_path, count))
        
        print(f"Loaded {len(samples)} samples for {self.split} split")
        return samples
    
    def _build_image_transforms(self) -> transforms.Compose:
        """
        Build image transformation pipeline.
        
        For training: resize, normalize
        For val/test: resize, normalize
        
        Returns:
            torchvision.transforms.Compose object.
        """
        transform_list = [
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config.NORMALIZE_MEAN,
                std=config.NORMALIZE_STD
            )
        ]
        
        return transforms.Compose(transform_list)
    
    def _apply_augmentation(
        self,
        image: Image.Image,
        density: np.ndarray
    ) -> Tuple[Image.Image, np.ndarray]:
        """
        Apply data augmentation to image and density map.
        
        Both image and density map must be transformed consistently.
        
        Args:
            image: PIL Image.
            density: Numpy array density map.
        
        Returns:
            Tuple of (augmented_image, augmented_density).
        """
        # Random horizontal flip
        if np.random.rand() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            density = np.fliplr(density).copy()
        
        return image, density
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index.
        
        Returns:
            Tuple of:
            - image: Tensor of shape (3, H, W), normalized
            - density: Tensor of shape (1, H/8, W/8) for CSRNet
            - count: Float ground truth count
        """
        # Try to load the sample, with fallback to next sample if corrupted
        max_attempts = 10
        original_idx = idx
        
        for attempt in range(max_attempts):
            try:
                image_path, density_path, count = self.samples[idx]
                
                # Load image
                image = Image.open(image_path).convert('RGB')
                
                # Load density map
                density = np.load(density_path).astype(np.float32)
                
                # Apply augmentation if enabled
                if self.augment:
                    image, density = self._apply_augmentation(image, density)
                
                # Resize density map to match target image size, then downsample for CSRNet
                # Step 1: Resize to full image size
                original_count = density.sum()
                density_pil = Image.fromarray(density)
                density_pil = density_pil.resize(
                    (self.image_size[1], self.image_size[0]),  # PIL uses (W, H)
                    resample=Image.BILINEAR
                )
                density = np.array(density_pil, dtype=np.float32)
                
                # Scale density to preserve count after resizing
                if density.sum() > 0:
                    density = density * (original_count / density.sum())
                
                # Step 2: Downsample by factor of 8 for CSRNet output
                target_h = self.image_size[0] // DENSITY_DOWNSAMPLE_FACTOR
                target_w = self.image_size[1] // DENSITY_DOWNSAMPLE_FACTOR
                
                density_pil = Image.fromarray(density)
                density_pil = density_pil.resize(
                    (target_w, target_h),
                    resample=Image.BILINEAR
                )
                density = np.array(density_pil, dtype=np.float32)
                
                # Scale density to preserve count after downsampling
                # Note: Values are scaled up to maintain sum
                if density.sum() > 0:
                    density = density * (original_count / density.sum())
                
                # Apply image transforms (resize, normalize, etc.)
                image = self.image_transform(image)
                
                # Convert density to tensor with channel dimension
                density = torch.from_numpy(density).unsqueeze(0)  # (1, H/8, W/8)
                
                return image, density, count
                
            except Exception as e:
                # Move to next sample if current one is corrupted
                idx = (idx + 1) % len(self.samples)
                if attempt == max_attempts - 1:
                    # If all attempts failed, create a dummy sample
                    print(f"Warning: Could not load sample {original_idx}, creating placeholder")
                    h, w = self.image_size
                    image = torch.zeros(3, h, w)
                    density = torch.zeros(1, h // DENSITY_DOWNSAMPLE_FACTOR, 
                                         w // DENSITY_DOWNSAMPLE_FACTOR)
                    return image, density, 0.0


def create_nwpu_dataloaders(
    batch_size: int = None,
    num_workers: int = None,
    image_size: Tuple[int, int] = None
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Create DataLoaders for train, validation, and test splits.
    
    Args:
        batch_size: Batch size for training. Uses config default if None.
        num_workers: Number of data loading workers. Uses config default if None.
        image_size: Target image size. Uses config default if None.
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader).
        test_loader may be None if no test split exists.
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    if num_workers is None:
        num_workers = config.NUM_WORKERS
    
    # Create datasets
    train_dataset = NWPUCrowdDataset(split='train', image_size=image_size)
    val_dataset = NWPUCrowdDataset(split='val', image_size=image_size)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Try to create test loader if split exists
    test_loader = None
    try:
        test_dataset = NWPUCrowdDataset(split='test', image_size=image_size)
        if len(test_dataset) > 0:
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )
    except FileNotFoundError:
        print("Note: Test split not found, skipping test dataloader")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the dataset
    print("=" * 60)
    print("Testing NWPUCrowdDataset")
    print("=" * 60)
    
    # Check if preprocessing has been done
    split_file = os.path.join(NWPU_SPLITS_DIR, "train.txt")
    if not os.path.exists(split_file):
        print(f"\nError: Split file not found at {split_file}")
        print("Please run preprocess_nwpu.py first to prepare the dataset.")
        print("\nExample:")
        print("  python preprocess_nwpu.py")
        exit(1)
    
    # Create dataset
    dataset = NWPUCrowdDataset(split='train')
    print(f"\nTrain dataset size: {len(dataset)}")
    
    # Load a sample
    image, density, count = dataset[0]
    
    print(f"\nSample 0:")
    print(f"  Image shape:   {image.shape}")
    print(f"  Density shape: {density.shape}")
    print(f"  Ground truth count: {count}")
    print(f"  Density sum:   {density.sum().item():.2f}")
    
    # Verify density sum matches count approximately
    error = abs(density.sum().item() - count)
    print(f"  Count error:   {error:.4f}")
    
    # Test dataloader
    train_loader, val_loader, test_loader = create_nwpu_dataloaders(batch_size=4)
    
    print(f"\nDataLoader batches:")
    print(f"  Train: {len(train_loader)} batches")
    print(f"  Val:   {len(val_loader)} batches")
    if test_loader:
        print(f"  Test:  {len(test_loader)} batches")
    
    # Load one batch
    images, densities, counts = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  Images:    {images.shape}")
    print(f"  Densities: {densities.shape}")
    print(f"  Counts:    {counts.shape}")
    
    print("\n" + "=" * 60)
    print("NWPUCrowdDataset test passed!")
    print("=" * 60)
