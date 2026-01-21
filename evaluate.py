"""
Evaluation Script for LCDNet Crowd Density Estimation.

This script evaluates a trained LCDNet model on the test set and reports:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

It also provides per-sample analysis and optional visualization.

Usage:
    python evaluate.py
    python evaluate.py --checkpoint checkpoints/best_model.pth
    python evaluate.py --visualize

Prerequisites:
    - Run preprocess.py to prepare the dataset
    - Run train.py to train the model

Author: Thesis Implementation
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
from dataset import create_dataloaders, UCSDCrowdDataset
from models.lcdnet import LCDNet, create_model
from utils.metrics import compute_mae, compute_mse, compute_rmse, density_to_count


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate LCDNet on test set'
    )
    parser.add_argument(
        '--checkpoint', type=str,
        default=os.path.join(config.CHECKPOINTS_DIR, 'best_model.pth'),
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--device', type=str, default=config.DEVICE,
        help=f'Device to run evaluation on (default: {config.DEVICE})'
    )
    parser.add_argument(
        '--batch_size', type=int, default=config.BATCH_SIZE,
        help=f'Batch size (default: {config.BATCH_SIZE})'
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


def load_model(checkpoint_path: str, device: str) -> LCDNet:
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file.
        device: Device to load model on.
    
    Returns:
        Loaded LCDNet model in eval mode.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Please train the model first with: python train.py"
        )
    
    print(f"Loading model from: {checkpoint_path}")
    
    # Create model
    model = create_model(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Print training info from checkpoint
    if 'epoch' in checkpoint:
        print(f"  Trained for {checkpoint['epoch'] + 1} epochs")
    if 'best_val_mae' in checkpoint:
        print(f"  Best validation MAE: {checkpoint['best_val_mae']:.4f}")
    
    model.eval()
    return model


def evaluate(model: LCDNet, test_loader, device: str) -> dict:
    """
    Evaluate model on test set.
    
    Args:
        model: Trained LCDNet model.
        test_loader: DataLoader for test data.
        device: Device to run inference on.
    
    Returns:
        Dictionary with evaluation results including:
        - 'mae': Mean Absolute Error
        - 'mse': Mean Squared Error
        - 'rmse': Root Mean Squared Error
        - 'pred_counts': List of predicted counts
        - 'gt_counts': List of ground truth counts
    """
    print("\nEvaluating on test set...")
    
    model.eval()
    pred_counts = []
    gt_counts = []
    
    with torch.no_grad():
        for images, density_maps, counts in tqdm(test_loader, desc="Testing"):
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


def print_results(results: dict):
    """
    Print evaluation results in a formatted manner.
    
    Args:
        results: Dictionary with evaluation results.
    """
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"\n  Mean Absolute Error (MAE):     {results['mae']:.4f}")
    print(f"  Mean Squared Error (MSE):      {results['mse']:.4f}")
    print(f"  Root Mean Squared Error (RMSE): {results['rmse']:.4f}")
    print("\n" + "-" * 60)
    
    # Additional statistics
    pred = np.array(results['pred_counts'])
    gt = np.array(results['gt_counts'])
    errors = np.abs(pred - gt)
    
    print(f"\n  Additional Statistics:")
    print(f"  ─────────────────────")
    print(f"    Total test samples:    {len(pred)}")
    print(f"    Min GT count:          {gt.min():.1f}")
    print(f"    Max GT count:          {gt.max():.1f}")
    print(f"    Mean GT count:         {gt.mean():.1f}")
    print(f"    Median error:          {np.median(errors):.4f}")
    print(f"    Std of errors:         {np.std(errors):.4f}")
    print(f"    Max error:             {errors.max():.4f}")
    print(f"    Min error:             {errors.min():.4f}")
    
    # Error distribution
    print(f"\n  Error Distribution:")
    print(f"  ─────────────────────")
    for threshold in [1.0, 2.0, 5.0, 10.0]:
        pct = (errors < threshold).mean() * 100
        print(f"    Errors < {threshold:4.1f}: {pct:5.1f}%")
    
    print("\n" + "=" * 60)


def visualize_samples(
    model: LCDNet,
    dataset: UCSDCrowdDataset,
    device: str,
    num_samples: int = 5,
    save_path: str = None
):
    """
    Visualize sample predictions with images, ground truth, and predictions.
    
    Args:
        model: Trained model.
        dataset: Test dataset.
        device: Device for inference.
        num_samples: Number of samples to visualize.
        save_path: Path to save visualization. Shows plot if None.
    """
    print(f"\nVisualizing {num_samples} sample predictions...")
    
    # Select random samples
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
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
        axes[i, 1].imshow(gt_density[0].numpy(), cmap='jet')
        axes[i, 1].set_title(f'GT Density (Count: {gt_count:.1f})')
        axes[i, 1].axis('off')
        
        # Predicted density
        axes[i, 2].imshow(pred_density[0].cpu().numpy(), cmap='jet')
        axes[i, 2].set_title(f'Pred Density (Count: {pred_count:.1f})')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def save_detailed_results(results: dict, save_path: str):
    """
    Save detailed per-sample results to a file.
    
    Args:
        results: Dictionary with evaluation results.
        save_path: Path to save results.
    """
    with open(save_path, 'w') as f:
        f.write("LCDNet Evaluation Results\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"MAE:  {results['mae']:.4f}\n")
        f.write(f"MSE:  {results['mse']:.4f}\n")
        f.write(f"RMSE: {results['rmse']:.4f}\n\n")
        
        f.write("-" * 60 + "\n")
        f.write("Per-sample results:\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Index':>6} {'GT Count':>10} {'Pred Count':>12} {'Error':>10}\n")
        f.write("-" * 60 + "\n")
        
        for i, (gt, pred) in enumerate(zip(results['gt_counts'], results['pred_counts'])):
            error = abs(gt - pred)
            f.write(f"{i:>6} {gt:>10.2f} {pred:>12.2f} {error:>10.2f}\n")
    
    print(f"Saved detailed results to: {save_path}")


def main():
    """Main entry point for evaluation."""
    args = parse_args()
    
    print("=" * 60)
    print("LCDNet Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    
    # Check if data exists
    split_file = os.path.join(config.SPLITS_DIR, "test.txt")
    if not os.path.exists(split_file):
        print(f"\nError: Test data not found!")
        print(f"Please run preprocessing first:")
        print(f"  python preprocess.py")
        sys.exit(1)
    
    # Load model
    try:
        model = load_model(args.checkpoint, args.device)
    except FileNotFoundError as e:
        print(f"\n{e}")
        sys.exit(1)
    
    # Create data loaders
    print("\nLoading test data...")
    _, _, test_loader = create_dataloaders(batch_size=args.batch_size)
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Evaluate
    results = evaluate(model, test_loader, args.device)
    
    # Print results
    print_results(results)
    
    # Save detailed results
    results_path = os.path.join(config.LOGS_DIR, 'evaluation_results.txt')
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    save_detailed_results(results, results_path)
    
    # Visualize if requested
    if args.visualize:
        test_dataset = UCSDCrowdDataset(split='test', augment=False)
        vis_path = os.path.join(config.LOGS_DIR, 'sample_predictions.png')
        visualize_samples(
            model,
            test_dataset,
            args.device,
            num_samples=args.num_samples,
            save_path=vis_path
        )
    
    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)
    print(f"\nFinal Results:")
    print(f"  MAE:  {results['mae']:.4f}")
    print(f"  MSE:  {results['mse']:.4f}")
    print(f"  RMSE: {results['rmse']:.4f}")


if __name__ == "__main__":
    main()
