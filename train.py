"""
Training Script for LCDNet Crowd Density Estimation.

This script trains the LCDNet model on the UCSD Crowd Dataset for crowd
density estimation. It includes:
- Training loop with MSE loss
- Validation after each epoch
- Learning rate scheduling
- Early stopping
- Checkpoint saving
- Progress logging

Usage:
    python train.py
    python train.py --epochs 50 --batch_size 4 --lr 1e-4

Prerequisites:
    - Run preprocess.py first to prepare the dataset
    - GPU recommended for faster training

Author: Thesis Implementation
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
from dataset import create_dataloaders
from models.lcdnet import LCDNet, create_model
from utils.metrics import compute_mae, compute_mse, density_to_count


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train LCDNet for crowd density estimation'
    )
    parser.add_argument(
        '--epochs', type=int, default=config.NUM_EPOCHS,
        help=f'Number of training epochs (default: {config.NUM_EPOCHS})'
    )
    parser.add_argument(
        '--batch_size', type=int, default=config.BATCH_SIZE,
        help=f'Batch size (default: {config.BATCH_SIZE})'
    )
    parser.add_argument(
        '--lr', type=float, default=config.LEARNING_RATE,
        help=f'Learning rate (default: {config.LEARNING_RATE})'
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


class Trainer:
    """
    Training manager for LCDNet.
    
    Handles the complete training pipeline including:
    - Model training and validation
    - Loss computation and optimization
    - Checkpoint saving and loading
    - Learning rate scheduling
    - Early stopping
    - Logging and progress tracking
    
    Attributes:
        model: LCDNet model instance.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        criterion: Loss function (MSE).
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
        device: str
    ):
        """
        Initialize the trainer.
        
        Args:
            model: LCDNet model to train.
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            learning_rate: Initial learning rate.
            device: Device to train on.
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss function: MSE between predicted and ground truth density maps
        # MSE penalizes large errors more, encouraging accurate density estimation
        self.criterion = nn.MSELoss()
        
        # Optimizer: Adam with weight decay for regularization
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # Learning rate scheduler: reduce LR when validation loss plateaus
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config.LR_SCHEDULER_FACTOR,
            patience=config.LR_SCHEDULER_PATIENCE
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
        os.makedirs(config.CHECKPOINTS_DIR, exist_ok=True)
    
    def train_epoch(self) -> float:
        """
        Train for one epoch.
        
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
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            pred_density = self.model(images)
            
            # Compute loss
            loss = self.criterion(pred_density, density_maps)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            self.optimizer.step()
            
            # Track loss
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.6f}',
                'avg_loss': f'{total_loss / num_batches:.6f}'
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
                
                # Compute loss
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
        
        path = os.path.join(config.CHECKPOINTS_DIR, filename)
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = os.path.join(config.CHECKPOINTS_DIR, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"  Saved best model with MAE: {self.best_val_mae:.4f}")
    
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
        print("Starting Training")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"Batch size: {self.train_loader.batch_size}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Model parameters: {self.model.count_parameters():,}")
        print("=" * 60)
        
        # Resume if checkpoint provided
        if resume_path and os.path.exists(resume_path):
            self.load_checkpoint(resume_path)
        
        start_time = time.time()
        
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
            print(f"  Val Loss: {val_loss:.6f}")
            print(f"  Val MAE: {val_mae:.4f}")
            print(f"  Val MSE: {val_mse:.4f}")
            print(f"  LR: {current_lr:.2e}")
            print(f"  Time: {epoch_time:.1f}s")
            
            # Check for improvement
            is_best = val_mae < self.best_val_mae
            if is_best:
                self.best_val_mae = val_mae
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            # Save checkpoint
            if epoch % config.CHECKPOINT_INTERVAL == 0 or is_best:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth', is_best)
            
            # Early stopping
            if self.epochs_without_improvement >= config.EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping: No improvement for {config.EARLY_STOPPING_PATIENCE} epochs")
                break
        
        # Training complete
        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"Total training time: {total_time / 60:.1f} minutes")
        print(f"Best validation MAE: {self.best_val_mae:.4f}")
        print(f"Final checkpoint saved to: {config.CHECKPOINTS_DIR}")
        print(f"\nTo evaluate, run:")
        print(f"  python evaluate.py")


def main():
    """Main entry point for training."""
    args = parse_args()
    
    # Print configuration
    print("=" * 60)
    print("LCDNet Training Configuration")
    print("=" * 60)
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Device: {args.device}")
    if args.resume:
        print(f"Resume from: {args.resume}")
    print("=" * 60)
    
    # Check if data is ready
    split_file = os.path.join(config.SPLITS_DIR, "train.txt")
    if not os.path.exists(split_file):
        print(f"\nError: Training data not found!")
        print(f"Please run preprocessing first:")
        print(f"  python preprocess.py")
        sys.exit(1)
    
    # Create data loaders
    print("\nLoading datasets...")
    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=args.batch_size
    )
    
    # Create model
    print("\nInitializing model...")
    model = create_model(args.device)
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.lr,
        device=args.device
    )
    
    # Train
    trainer.train(
        num_epochs=args.epochs,
        resume_path=args.resume
    )


if __name__ == "__main__":
    main()
