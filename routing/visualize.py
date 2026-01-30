"""
Visualization Utilities for Hybrid Crowd Density Estimation.

This module provides functions to visualize:
- Routing decisions with color coding
- Predicted density maps
- Side-by-side comparison of hybrid vs single-model predictions

Usage:
    python routing/visualize.py --num_samples 10

Author: Thesis Implementation - Phase 2 Part 3
"""

import os
import sys
import argparse
import random
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from routing import config_routing
from routing.hybrid_inference import HybridDensityEstimator

# Set up matplotlib backend
import matplotlib
matplotlib.use('Agg')


def load_sample_images(
    split: str = "val",
    num_samples: int = 10,
    random_seed: int = 42
) -> List[Tuple[str, float]]:
    """Load random sample images for visualization."""
    import json
    
    # Get split file
    if split == "train":
        split_file = config_routing.NWPU_TRAIN_TXT
    elif split == "val":
        split_file = config_routing.NWPU_VAL_TXT
    else:
        split_file = config_routing.NWPU_TEST_TXT
    
    samples = []
    
    with open(split_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 1:
                continue
            
            img_id = parts[0]
            
            # Find image
            img_path = None
            for img_dir in config_routing.NWPU_IMAGE_DIRS:
                for ext in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']:
                    path = os.path.join(img_dir, f"{img_id}{ext}")
                    if os.path.exists(path):
                        img_path = path
                        break
                if img_path:
                    break
            
            if img_path is None:
                continue
            
            # Get GT count from JSON
            json_path = os.path.join(config_routing.NWPU_JSONS_DIR, f"{img_id}.json")
            try:
                with open(json_path, 'r') as jf:
                    data = json.load(jf)
                gt_count = float(data.get('human_num', 0))
            except (FileNotFoundError, ValueError):
                gt_count = 0.0
            
            samples.append((img_path, gt_count))
    
    # Random selection
    random.seed(random_seed)
    if len(samples) > num_samples:
        samples = random.sample(samples, num_samples)
    
    return samples


def visualize_routing_decision(
    estimator: HybridDensityEstimator,
    image_path: str,
    gt_count: float,
    output_dir: str
) -> str:
    """
    Visualize a single routing decision with density map.
    
    Returns:
        Path to saved visualization.
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Get prediction
    result = estimator.predict(image, return_density_map=True)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title(f"Original Image\nGT Count: {gt_count:.0f}")
    axes[0].axis('off')
    
    # Density map
    density_map = result['density_map']
    axes[1].imshow(density_map, cmap='jet')
    axes[1].set_title(f"Density Map\nPred Count: {result['count']:.1f}")
    axes[1].axis('off')
    
    # Routing info
    color = 'blue' if result['model'] == 'LCDNet' else 'red'
    axes[2].text(0.5, 0.7, result['model'], fontsize=32, ha='center', va='center',
                 color='white', fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))
    axes[2].text(0.5, 0.4, f"Confidence: {result['routing_prob']:.1%}",
                 fontsize=16, ha='center', va='center')
    axes[2].text(0.5, 0.25, f"Error: {abs(result['count'] - gt_count):.1f}",
                 fontsize=14, ha='center', va='center')
    axes[2].text(0.5, 0.1, f"Time: {result['total_time']*1000:.1f} ms",
                 fontsize=12, ha='center', va='center')
    axes[2].axis('off')
    axes[2].set_xlim(0, 1)
    axes[2].set_ylim(0, 1)
    
    plt.suptitle(f"Hybrid Routing Decision", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save
    img_name = os.path.basename(image_path).split('.')[0]
    output_path = os.path.join(output_dir, f"routing_{img_name}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def create_summary_grid(
    estimator: HybridDensityEstimator,
    samples: List[Tuple[str, float]],
    output_path: str
):
    """Create a grid summary of routing decisions."""
    n_samples = len(samples)
    n_cols = min(5, n_samples)
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1:
        axes = [axes]
    if n_cols == 1:
        axes = [[ax] for ax in axes]
    
    for idx, (img_path, gt_count) in enumerate(samples):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row][col] if n_rows > 1 else axes[col]
        
        # Load and predict
        image = Image.open(img_path).convert('RGB')
        result = estimator.predict(image)
        
        # Display
        ax.imshow(image)
        
        # Color border based on model
        color = 'blue' if result['model'] == 'LCDNet' else 'red'
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
        
        # Title
        ax.set_title(f"GT: {gt_count:.0f} | Pred: {result['count']:.1f}\n"
                    f"{result['model']} ({result['routing_prob']:.0%})",
                    fontsize=10)
        ax.axis('off')
    
    # Hide empty subplots
    for idx in range(n_samples, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row][col] if n_rows > 1 else axes[col]
        ax.axis('off')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', label='LCDNet (sparse)'),
        Patch(facecolor='red', label='CSRNet (dense)')
    ]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    plt.suptitle("Hybrid Routing Decisions Summary", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Summary grid saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize hybrid routing decisions")
    parser.add_argument(
        '--num_samples', type=int, default=10,
        help='Number of samples to visualize (default: 10)'
    )
    parser.add_argument(
        '--split', type=str, default='val', choices=['train', 'val', 'test'],
        help='Dataset split to use (default: val)'
    )
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Output directory for visualizations'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for sample selection'
    )
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(config_routing.ROUTER_LOG_DIR, "visualizations")
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("HYBRID ROUTING VISUALIZATION")
    print("=" * 60)
    print(f"Split: {args.split}")
    print(f"Samples: {args.num_samples}")
    print(f"Output: {args.output_dir}")
    
    # Load estimator
    print("\nLoading hybrid estimator...")
    try:
        estimator = HybridDensityEstimator()
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nMake sure you have trained the router first:")
        print("  python routing/train_router.py")
        return
    
    # Load samples
    print(f"\nLoading {args.num_samples} sample images...")
    samples = load_sample_images(args.split, args.num_samples, args.seed)
    print(f"Loaded {len(samples)} samples")
    
    # Create individual visualizations
    print("\nGenerating individual visualizations...")
    for img_path, gt_count in samples:
        output_path = visualize_routing_decision(
            estimator, img_path, gt_count, args.output_dir
        )
        print(f"  Saved: {os.path.basename(output_path)}")
    
    # Create summary grid
    print("\nGenerating summary grid...")
    grid_path = os.path.join(args.output_dir, "routing_summary_grid.png")
    create_summary_grid(estimator, samples, grid_path)
    
    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
