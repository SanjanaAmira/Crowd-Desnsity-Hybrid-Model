"""
Routing Dataset for Training the Routing Classifier.

This module provides a PyTorch Dataset class that loads images and generates
routing labels based on ground truth crowd counts.

Label Generation:
- label = 0 (LCDNet) if GT count <= threshold
- label = 1 (CSRNet) if GT count > threshold

The default threshold is 100, configurable in config_routing.py.

Author: Thesis Implementation - Phase 2 Part 3
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms
from typing import Tuple, List, Optional
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from routing import config_routing
except ImportError:
    import config_routing


class RoutingDataset(Dataset):
    """
    PyTorch Dataset for training the routing classifier.
    
    This dataset loads images from NWPU-Crowd and generates routing labels
    based on ground truth counts. Each sample consists of:
    - image: RGB image tensor (3, 224, 224), ImageNet normalized
    - label: 0 for LCDNet (sparse), 1 for CSRNet (dense)
    - count: Ground truth person count (for analysis)
    
    Attributes:
        split: Dataset split ('train', 'val', or 'test').
        threshold: Routing threshold for label generation.
        samples: List of (image_path, count, label) tuples.
        transform: Image transformation pipeline.
    """
    
    def __init__(
        self,
        split: str = "train",
        threshold: int = None,
        augment: bool = None
    ):
        """
        Initialize the routing dataset.
        
        Args:
            split: One of 'train', 'val', or 'test'.
            threshold: Routing threshold. Uses config default if None.
            augment: Whether to apply data augmentation. True for train by default.
        """
        super(RoutingDataset, self).__init__()
        
        self.split = split.lower()
        self.threshold = threshold if threshold is not None else config_routing.ROUTING_THRESHOLD
        
        # Default augmentation for training only
        if augment is None:
            augment = (self.split == "train")
        self.augment = augment
        
        # Load samples
        self.samples = self._load_samples()
        
        # Build transforms
        self.transform = self._build_transforms()
        
        # Print dataset statistics
        self._print_stats()
    
    def _load_samples(self) -> List[Tuple[str, float, int]]:
        """
        Load sample paths and generate routing labels.
        
        Returns:
            List of (image_path, gt_count, routing_label) tuples.
        """
        # Get split file path
        if self.split == "train":
            split_file = config_routing.NWPU_TRAIN_TXT
        elif self.split == "val":
            split_file = config_routing.NWPU_VAL_TXT
        elif self.split == "test":
            split_file = config_routing.NWPU_TEST_TXT
        else:
            raise ValueError(f"Unknown split: {self.split}")
        
        samples = []
        
        with open(split_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 1:
                    continue
                
                img_id = parts[0]  # e.g., "0001"
                
                # Find image file (check all image directories)
                img_path = self._find_image(img_id)
                if img_path is None:
                    continue
                
                # Load ground truth count from JSON
                json_path = os.path.join(config_routing.NWPU_JSONS_DIR, f"{img_id}.json")
                gt_count = self._load_count(json_path)
                
                # Generate routing label
                label = config_routing.get_routing_label(gt_count)
                
                samples.append((img_path, gt_count, label))
        
        return samples
    
    def _find_image(self, img_id: str) -> Optional[str]:
        """Find the image file across all image directories."""
        for img_dir in config_routing.NWPU_IMAGE_DIRS:
            # Try common extensions
            for ext in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']:
                img_path = os.path.join(img_dir, f"{img_id}{ext}")
                if os.path.exists(img_path):
                    return img_path
        return None
    
    def _load_count(self, json_path: str) -> float:
        """Load ground truth count from JSON annotation file."""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            return float(data.get('human_num', 0))
        except (FileNotFoundError, json.JSONDecodeError):
            return 0.0
    
    def _build_transforms(self) -> transforms.Compose:
        """Build image transformation pipeline."""
        transform_list = []
        
        # Resize to router input size
        transform_list.append(transforms.Resize(config_routing.ROUTER_INPUT_SIZE))
        
        # Data augmentation for training
        if self.augment:
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.1,
                    hue=0.05
                ),
            ])
        
        # Convert to tensor and normalize
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        return transforms.Compose(transform_list)
    
    def _print_stats(self):
        """Print dataset statistics."""
        total = len(self.samples)
        sparse_count = sum(1 for _, _, label in self.samples if label == 0)
        dense_count = total - sparse_count
        
        print(f"\n{self.split.upper()} Routing Dataset:")
        print(f"  Total samples: {total}")
        print(f"  Sparse (LCDNet, label=0): {sparse_count} ({100*sparse_count/total:.1f}%)")
        print(f"  Dense (CSRNet, label=1): {dense_count} ({100*dense_count/total:.1f}%)")
        print(f"  Threshold: {self.threshold}")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, float]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index.
        
        Returns:
            Tuple of:
            - image: Tensor of shape (3, 224, 224), normalized
            - label: Routing label (0 or 1)
            - count: Ground truth count (for analysis)
        """
        img_path, gt_count, label = self.samples[idx]
        
        # Load and transform image
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        return image, label, gt_count


def create_routing_dataloaders(
    batch_size: int = None,
    num_workers: int = None,
    threshold: int = None
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Create DataLoaders for train, validation, and test splits.
    
    Args:
        batch_size: Batch size for training. Uses config default if None.
        num_workers: Number of data loading workers. Uses config default if None.
        threshold: Routing threshold. Uses config default if None.
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader).
        test_loader may be None if no test split exists.
    """
    if batch_size is None:
        batch_size = config_routing.ROUTER_BATCH_SIZE
    if num_workers is None:
        num_workers = config_routing.NUM_WORKERS
    
    # Create datasets
    train_dataset = RoutingDataset(split="train", threshold=threshold, augment=True)
    val_dataset = RoutingDataset(split="val", threshold=threshold, augment=False)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Test loader (optional)
    test_loader = None
    try:
        test_dataset = RoutingDataset(split="test", threshold=threshold, augment=False)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    except Exception as e:
        print(f"Warning: Could not create test loader: {e}")
    
    print(f"\nDataLoaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    if test_loader:
        print(f"  Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the dataset
    print("Testing RoutingDataset...")
    print("=" * 60)
    
    # Create validation dataset (smaller for quick test)
    dataset = RoutingDataset(split="val", augment=False)
    
    # Test single sample
    image, label, count = dataset[0]
    print(f"\nSample 0:")
    print(f"  Image shape: {image.shape}")
    print(f"  Label: {label} ({config_routing.get_model_name(label)})")
    print(f"  GT Count: {count}")
    
    # Test dataloader
    print("\nTesting DataLoader...")
    train_loader, val_loader, _ = create_routing_dataloaders(batch_size=8)
    
    for batch_idx, (images, labels, counts) in enumerate(val_loader):
        print(f"\nBatch {batch_idx}:")
        print(f"  Images: {images.shape}")
        print(f"  Labels: {labels}")
        print(f"  Counts: {counts[:5]}...")
        if batch_idx >= 1:
            break
    
    print("=" * 60)
    print("Dataset test passed!")
