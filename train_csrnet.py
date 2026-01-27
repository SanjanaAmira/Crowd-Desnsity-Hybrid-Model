"""
Training Script for CSRNet on NWPU-Crowd Dataset.

This script trains CSRNet for dense crowd density estimation. It includes:
- Training loop with MSE loss
- Optional counting loss (L1) for improved count accuracy
- Validation after each epoch
- Learning rate scheduling with ReduceLROnPlateau
- Early stopping based on validation MAE
- Checkpoint saving (best model based on validation MAE)
- Progress logging

Usage:
    python train_csrnet.py
    python train_csrnet.py --epochs 100 --batch_size 4 --lr 1e-5

Prerequisites:
    - Run preprocess_nwpu.py first to prepare the dataset

Author: Thesis Implementation - Phase 2 Part 2
"""

import os
import sys
import argparse
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from dataset_nwpu import create_nwpu_dataloaders
from models.csrnet import CSRNet, create_csrnet
from utils.metrics import compute_mae, compute_mse, density_to_count


# ============================================================================
# CSRNET-SPECIFIC CONFIGURATION
# ============================================================================

# CSRNet typically uses smaller learning rate due to pretrained VGG frontend
CSRNET_LR = 1e-5

# Checkpoint directory for CSRNet
CSRNET_CHECKPOINTS_DIR = os.path.join(config.CHECKPOINTS_DIR, "csrnet")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train CSRNet on NWPU-Crowd dataset for dense crowd counting'
    )
    parser.add_argument(
        '--epochs', type=int, default=100,
        help='Number of training epochs (default: 100)'
    )
    parser.add_argument(
        '--batch_size', type=int, default=4,
        help='Batch size (default: 4, smaller due to large model)'
    )
    parser.add_argument(
        '--lr', type=float, default=CSRNET_LR,
        help=f'Learning rate (default: {CSRNET_LR})'
    )
    parser.add_argument(
        '--weight_decay', type=float, default=1e-4,
        help='Weight decay for Adam optimizer (default: 1e-4)'
    )
    parser.add_argument(
        '--count_loss_weight', type=float, default=0.1,
        help='Weight for counting loss term (default: 0.1, higher for NWPU-Crowd)'
    )
    parser.add_argument(
        '--freeze_frontend', action='store_true',
        help='Freeze VGG-16 frontend weights (train only backend)'
    )
    parser.add_argument(
        '--resume', type=str, default=None,
        help='Path to checkpoint to resume training from'
    )
    parser.add_argument(
        '--device', type=str, default=config.DEVICE,
        help=f'Device to train on (default: {config.DEVICE})'
    )
    return parser.parse_args()


class CSRNetTrainer:
    """
    Training manager for CSRNet on NWPU-Crowd dataset.
    
    Handles the complete training pipeline including:
    - Model training and validation
    - Combined loss (MSE + count loss)
    - Checkpoint saving and loading
    - Learning rate scheduling
    - Early stopping
    - Logging and progress tracking
    
    Attributes:
        model: CSRNet model instance.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        criterion: Loss function (MSE for density).
        optimizer: Adam optimizer.
        scheduler: Learning rate scheduler.
        device: Training device (cuda/cpu).
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        learning_rate: float,
        weight_decay: float,
        count_loss_weight: float,
        device: str
    ):
        """
        Initialize the trainer.
        
        Args:
            model: CSRNet model to train.
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            learning_rate: Initial learning rate.
            weight_decay: Weight decay for regularization.
            count_loss_weight: Weight for counting loss term.
            device: Device to train on.
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.count_loss_weight = count_loss_weight
        
        # Loss function: MSE between predicted and ground truth density maps
        # MSE is standard for density map regression
        self.criterion = nn.MSELoss()
        
        # Optimizer: Adam with weight decay
        # Lower learning rate for pretrained VGG frontend
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler: reduce LR when validation MAE plateaus
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            min_lr=1e-7
        )
        
        # Training state
        self.best_val_mae = float('inf')
        self.epochs_without_improvement = 0
        self.current_epoch = 0
        
        # Logging
        self.train_losses = []
        self.val_losses = []
        self.val_maes = []
        
        # Create checkpoint directory
        os.makedirs(CSRNET_CHECKPOINTS_DIR, exist_ok=True)
    
    def train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Uses combined loss:
        - MSE loss for pixel-wise density prediction
        - L1 loss for count prediction (sum of density map)
        
        Returns:
            Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, (images, density_maps, counts) in enumerate(pbar):
            # Move data to device
            images = images.to(self.device)
            density_maps = density_maps.to(self.device)
            counts = counts.to(self.device).float()
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            pred_density = self.model(images)
            
            # Compute density loss (MSE)
            density_loss = self.criterion(pred_density, density_maps)
            
            # Compute counting loss (L1 between predicted and GT counts)
            # This helps the model focus on getting accurate counts
            pred_counts = pred_density.sum(dim=[1, 2, 3])
            count_loss = torch.nn.functional.l1_loss(pred_counts, counts)
            
            # Combined loss
            loss = density_loss + self.count_loss_weight * count_loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            # Track loss
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.6f}',
                'cnt_loss': f'{count_loss.item():.2f}'
            })
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self) -> tuple:
        """
        Validate the model on validation set.
        
        Returns:
            Tuple of (average_loss, MAE, MSE).
        """
        self.model.eval()
        total_loss = 0.0
        pred_counts = []
        gt_counts = []
        
        with torch.no_grad():
            for images, density_maps, counts in tqdm(self.val_loader, desc="Validating"):
                images = images.to(self.device)
                density_maps = density_maps.to(self.device)
                
                # Forward pass
                pred_density = self.model(images)
                
                # Compute loss (MSE only for validation metric)
                loss = self.criterion(pred_density, density_maps)
                total_loss += loss.item()
                
                # Convert density maps to counts
                batch_pred_counts = density_to_count(pred_density)
                
                # Handle different return types
                if isinstance(batch_pred_counts, np.ndarray):
                    pred_counts.extend(batch_pred_counts.tolist())
                else:
                    pred_counts.append(batch_pred_counts)
                
                gt_counts.extend(counts.numpy().tolist())
        
        avg_loss = total_loss / len(self.val_loader)
        mae = compute_mae(pred_counts, gt_counts)
        mse = compute_mse(pred_counts, gt_counts)
        
        return avg_loss, mae, mse
    
    def save_checkpoint(self, filename: str, is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            filename: Name for the checkpoint file.
            is_best: Whether this is the best model so far.
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_mae': self.best_val_mae,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_maes': self.val_maes,
        }
        
        path = os.path.join(CSRNET_CHECKPOINTS_DIR, filename)
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = os.path.join(CSRNET_CHECKPOINTS_DIR, 'csrnet_best.pth')
            torch.save(checkpoint, best_path)
            print(f"  ✓ Saved best model with MAE: {self.best_val_mae:.4f}")
    
    def load_checkpoint(self, path: str):
        """
        Load model checkpoint.
        
        Args:
            path: Path to checkpoint file.
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch'] + 1
        self.best_val_mae = checkpoint['best_val_mae']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.val_maes = checkpoint.get('val_maes', [])
        
        print(f"Resumed from epoch {self.current_epoch} with best MAE: {self.best_val_mae:.4f}")
    
    def train(self, num_epochs: int, resume_path: str = None):
        """
        Main training loop.
        
        Args:
            num_epochs: Total number of epochs to train.
            resume_path: Path to checkpoint to resume from.
        """
        print("\n" + "=" * 60)
        print("Starting CSRNet Training on NWPU-Crowd")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"Batch size: {self.train_loader.batch_size}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Count loss weight: {self.count_loss_weight}")
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print("=" * 60)
        
        # Resume if checkpoint provided
        if resume_path and os.path.exists(resume_path):
            self.load_checkpoint(resume_path)
        
        start_time = time.time()
        early_stop_patience = 25
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, val_mae, val_mse = self.validate()
            self.val_losses.append(val_loss)
            self.val_maes.append(val_mae)
            
            # Update learning rate scheduler
            self.scheduler.step(val_mae)
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"\nEpoch {epoch}/{num_epochs - 1}:")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss:   {val_loss:.6f}")
            print(f"  Val MAE:    {val_mae:.2f}")
            print(f"  Val MSE:    {val_mse:.2f}")
            print(f"  LR:         {current_lr:.2e}")
            print(f"  Time:       {epoch_time:.1f}s")
            
            # Check for improvement
            is_best = val_mae < self.best_val_mae
            if is_best:
                self.best_val_mae = val_mae
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            # Save checkpoint
            if epoch % 5 == 0 or is_best:
                self.save_checkpoint(f'csrnet_epoch_{epoch}.pth', is_best)
            
            # Early stopping
            if self.epochs_without_improvement >= early_stop_patience:
                print(f"\nEarly stopping: No improvement for {early_stop_patience} epochs")
                break
        
        # Training complete
        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"Total training time: {total_time / 60:.1f} minutes")
        print(f"Best validation MAE: {self.best_val_mae:.2f}")
        print(f"Checkpoint saved to: {CSRNET_CHECKPOINTS_DIR}")
        print(f"\nTo evaluate, run:")
        print(f"  python evaluate_csrnet.py")


def main():
    """Main entry point for training."""
    args = parse_args()
    
    # Print configuration
    print("=" * 60)
    print("CSRNet Training Configuration")
    print("=" * 60)
    print(f"Epochs:           {args.epochs}")
    print(f"Batch size:       {args.batch_size}")
    print(f"Learning rate:    {args.lr}")
    print(f"Weight decay:     {args.weight_decay}")
    print(f"Count loss weight:{args.count_loss_weight}")
    print(f"Device:           {args.device}")
    print(f"Freeze frontend:  {args.freeze_frontend}")
    if args.resume:
        print(f"Resume from:      {args.resume}")
    print("=" * 60)
    
    # Check if data is ready
    from dataset_nwpu import NWPU_SPLITS_DIR
    split_file = os.path.join(NWPU_SPLITS_DIR, "train.txt")
    if not os.path.exists(split_file):
        print(f"\nError: Training data not found!")
        print(f"Please run preprocessing first:")
        print(f"  python preprocess_nwpu.py")
        sys.exit(1)
    
    # Create data loaders
    print("\nLoading datasets...")
    train_loader, val_loader, _ = create_nwpu_dataloaders(
        batch_size=args.batch_size
    )
    
    # Create model
    print("\nInitializing CSRNet model...")
    model = create_csrnet(args.device, pretrained=True)
    
    # Optionally freeze frontend
    if args.freeze_frontend:
        print("Freezing VGG-16 frontend weights...")
        model.freeze_frontend()
    
    # Create trainer
    trainer = CSRNetTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        count_loss_weight=args.count_loss_weight,
        device=args.device
    )
    
    # Train
    trainer.train(
        num_epochs=args.epochs,
        resume_path=args.resume
    )


if __name__ == "__main__":
    main()
