"""
Evaluation Script for CSRNet on NWPU-Crowd Dataset.

This script evaluates a trained CSRNet model and reports:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

It also provides per-sample analysis and optional visualization.

Usage:
    python evaluate_csrnet.py
    python evaluate_csrnet.py --checkpoint checkpoints/csrnet/csrnet_best.pth
    python evaluate_csrnet.py --visualize --num_samples 5

Prerequisites:
    - Run preprocess_nwpu.py to prepare the dataset
    - Run train_csrnet.py to train the model

Author: Thesis Implementation - Phase 2 Part 2
"""

import os
import sys
import argparse
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from dataset_nwpu import NWPUCrowdDataset, create_nwpu_dataloaders
from models.csrnet import CSRNet, create_csrnet
from utils.metrics import compute_mae, compute_mse, compute_rmse, density_to_count


# ============================================================================
# CSRNET-SPECIFIC CONFIGURATION
# ============================================================================

# Checkpoint directory for CSRNet
CSRNET_CHECKPOINTS_DIR = os.path.join(config.CHECKPOINTS_DIR, "csrnet")
CSRNET_LOGS_DIR = os.path.join(config.LOGS_DIR, "csrnet")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate CSRNet on NWPU-Crowd validation set'
    )
    parser.add_argument(
        '--checkpoint', type=str,
        default=os.path.join(CSRNET_CHECKPOINTS_DIR, 'csrnet_best.pth'),
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--split', type=str, default='val',
        choices=['train', 'val'],
        help='Split to evaluate on (default: val)'
    )
    parser.add_argument(
        '--device', type=str, default=config.DEVICE,
        help=f'Device to run evaluation on (default: {config.DEVICE})'
    )
    parser.add_argument(
        '--batch_size', type=int, default=1,
        help='Batch size (default: 1 for accurate per-sample analysis)'
    )
    parser.add_argument(
        '--visualize', action='store_true',
        help='Visualize sample predictions'
    )
    parser.add_argument(
        '--num_samples', type=int, default=5,
        help='Number of samples to visualize (default: 5)'
    )
    return parser.parse_args()


def load_model(checkpoint_path: str, device: str) -> CSRNet:
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file.
        device: Device to load model on.
    
    Returns:
        Loaded CSRNet model in eval mode.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Please train the model first with: python train_csrnet.py"
        )
    
    print(f"Loading model from: {checkpoint_path}")
    
    # Create model (without pretrained weights, as we'll load from checkpoint)
    model = CSRNet(pretrained=False)
    model = model.to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Print training info from checkpoint
    if 'epoch' in checkpoint:
        print(f"  Trained for {checkpoint['epoch'] + 1} epochs")
    if 'best_val_mae' in checkpoint:
        print(f"  Best validation MAE: {checkpoint['best_val_mae']:.2f}")
    
    model.eval()
    return model


def evaluate(model: CSRNet, data_loader, device: str) -> dict:
    """
    Evaluate model on dataset.
    
    Args:
        model: Trained CSRNet model.
        data_loader: DataLoader for evaluation data.
        device: Device to run inference on.
    
    Returns:
        Dictionary with evaluation results including:
        - 'mae': Mean Absolute Error
        - 'mse': Mean Squared Error
        - 'rmse': Root Mean Squared Error
        - 'pred_counts': List of predicted counts
        - 'gt_counts': List of ground truth counts
    """
    print("\nEvaluating on dataset...")
    
    model.eval()
    pred_counts = []
    gt_counts = []
    
    with torch.no_grad():
        for images, density_maps, counts in tqdm(data_loader, desc="Evaluating"):
            images = images.to(device)
            
            # Forward pass
            pred_density = model(images)
            
            # Convert density maps to counts
            batch_pred_counts = density_to_count(pred_density)
            
            # Store results
            if isinstance(batch_pred_counts, np.ndarray):
                pred_counts.extend(batch_pred_counts.tolist())
            else:
                pred_counts.append(batch_pred_counts)
            
            gt_counts.extend(counts.numpy().tolist())
    
    # Compute metrics
    mae = compute_mae(pred_counts, gt_counts)
    mse = compute_mse(pred_counts, gt_counts)
    rmse = compute_rmse(pred_counts, gt_counts)
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'pred_counts': pred_counts,
        'gt_counts': gt_counts
    }


def print_results(results: dict, split: str):
    """
    Print evaluation results in a formatted manner.
    
    Args:
        results: Dictionary with evaluation results.
        split: Name of the dataset split.
    """
    print("\n" + "=" * 60)
    print(f"CSRNet EVALUATION RESULTS on {split.upper()} SET")
    print("=" * 60)
    print(f"\n  Mean Absolute Error (MAE):     {results['mae']:.2f}")
    print(f"  Mean Squared Error (MSE):      {results['mse']:.2f}")
    print(f"  Root Mean Squared Error (RMSE): {results['rmse']:.2f}")
    print("\n" + "-" * 60)
    
    # Additional statistics
    pred = np.array(results['pred_counts'])
    gt = np.array(results['gt_counts'])
    errors = np.abs(pred - gt)
    
    print(f"\n  Additional Statistics:")
    print(f"  ─────────────────────")
    print(f"    Total test samples:    {len(pred)}")
    print(f"    Min GT count:          {gt.min():.0f}")
    print(f"    Max GT count:          {gt.max():.0f}")
    print(f"    Mean GT count:         {gt.mean():.1f}")
    print(f"    Median error:          {np.median(errors):.2f}")
    print(f"    Std of errors:         {np.std(errors):.2f}")
    print(f"    Max error:             {errors.max():.2f}")
    print(f"    Min error:             {errors.min():.2f}")
    
    # Error distribution
    print(f"\n  Error Distribution:")
    print(f"  ─────────────────────")
    for threshold in [5.0, 10.0, 20.0, 50.0, 100.0]:
        pct = (errors < threshold).mean() * 100
        print(f"    Errors < {threshold:5.0f}: {pct:5.1f}%")
    
    print("\n" + "=" * 60)


def visualize_samples(
    model: CSRNet,
    dataset: NWPUCrowdDataset,
    device: str,
    num_samples: int = 5,
    save_path: str = None
):
    """
    Visualize sample predictions with images, ground truth, and predictions.
    
    Args:
        model: Trained CSRNet model.
        dataset: Dataset to sample from.
        device: Device for inference.
        num_samples: Number of samples to visualize.
        save_path: Path to save visualization. Shows plot if None.
    """
    print(f"\nVisualizing {num_samples} sample predictions...")
    
    # Select random samples
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(14, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    model.eval()
    
    for i, idx in enumerate(indices):
        image, gt_density, gt_count = dataset[idx]
        
        # Predict
        with torch.no_grad():
            image_batch = image.unsqueeze(0).to(device)
            pred_density = model(image_batch)[0]
        
        pred_count = density_to_count(pred_density)
        
        # Convert image for display
        image_np = image.permute(1, 2, 0).numpy()
        # Denormalize
        image_np = image_np * np.array(config.NORMALIZE_STD) + np.array(config.NORMALIZE_MEAN)
        image_np = np.clip(image_np, 0, 1)
        
        # Plot
        # Original image
        axes[i, 0].imshow(image_np)
        axes[i, 0].set_title(f'Input Image')
        axes[i, 0].axis('off')
        
        # Ground truth density
        gt_map = gt_density[0].numpy()
        axes[i, 1].imshow(gt_map, cmap='jet')
        axes[i, 1].set_title(f'GT Density (Count: {gt_count:.0f})')
        axes[i, 1].axis('off')
        
        # Predicted density
        pred_map = pred_density[0].cpu().numpy()
        axes[i, 2].imshow(pred_map, cmap='jet')
        axes[i, 2].set_title(f'Pred Density (Count: {pred_count:.1f})')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def save_detailed_results(results: dict, split: str, save_path: str):
    """
    Save detailed per-sample results to a file.
    
    Args:
        results: Dictionary with evaluation results.
        split: Name of the dataset split.
        save_path: Path to save results.
    """
    with open(save_path, 'w') as f:
        f.write("CSRNet Evaluation Results on NWPU-Crowd\n")
        f.write(f"Split: {split.upper()}\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"MAE:  {results['mae']:.2f}\n")
        f.write(f"MSE:  {results['mse']:.2f}\n")
        f.write(f"RMSE: {results['rmse']:.2f}\n\n")
        
        f.write("-" * 60 + "\n")
        f.write("Per-sample results:\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Index':>6} {'GT Count':>10} {'Pred Count':>12} {'Error':>10}\n")
        f.write("-" * 60 + "\n")
        
        for i, (gt, pred) in enumerate(zip(results['gt_counts'], results['pred_counts'])):
            error = abs(gt - pred)
            f.write(f"{i:>6} {gt:>10.1f} {pred:>12.1f} {error:>10.2f}\n")
    
    print(f"Saved detailed results to: {save_path}")


def main():
    """Main entry point for evaluation."""
    args = parse_args()
    
    print("=" * 60)
    print("CSRNet Evaluation on NWPU-Crowd")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Split:      {args.split}")
    print(f"Device:     {args.device}")
    
    # Check if data exists
    from dataset_nwpu import NWPU_SPLITS_DIR
    split_file = os.path.join(NWPU_SPLITS_DIR, f"{args.split}.txt")
    if not os.path.exists(split_file):
        print(f"\nError: Split file not found!")
        print(f"Please run preprocessing first:")
        print(f"  python preprocess_nwpu.py")
        sys.exit(1)
    
    # Load model
    try:
        model = load_model(args.checkpoint, args.device)
    except FileNotFoundError as e:
        print(f"\n{e}")
        sys.exit(1)
    
    # Create data loader
    print("\nLoading data...")
    dataset = NWPUCrowdDataset(split=args.split, augment=False)
    from torch.utils.data import DataLoader
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    print(f"Samples: {len(dataset)}")
    
    # Evaluate
    results = evaluate(model, data_loader, args.device)
    
    # Print results
    print_results(results, args.split)
    
    # Save detailed results
    os.makedirs(CSRNET_LOGS_DIR, exist_ok=True)
    results_path = os.path.join(CSRNET_LOGS_DIR, f'csrnet_{args.split}_results.txt')
    save_detailed_results(results, args.split, results_path)
    
    # Visualize if requested
    if args.visualize:
        vis_path = os.path.join(CSRNET_LOGS_DIR, f'csrnet_{args.split}_predictions.png')
        visualize_samples(
            model,
            dataset,
            args.device,
            num_samples=args.num_samples,
            save_path=vis_path
        )
    
    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)
    print(f"\nFinal Results on {args.split.upper()} set:")
    print(f"  MAE:  {results['mae']:.2f}")
    print(f"  MSE:  {results['mse']:.2f}")
    print(f"  RMSE: {results['rmse']:.2f}")


if __name__ == "__main__":
    main()
