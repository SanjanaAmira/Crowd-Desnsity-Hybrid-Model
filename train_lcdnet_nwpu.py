"""
Fine-tuning Script for LCDNet on NWPU-Crowd Sparse Subset.

This script fine-tunes the pre-trained LCDNet model (trained on ShanghaiTech Part B)
on the sparse subset of NWPU-Crowd (ground truth count <= 100).
It adapts the model to the NWPU-Crowd domain to close the cross-dataset performance gap.

Usage:
    python train_lcdnet_nwpu.py --epochs 50 --batch_size 8 --lr 1e-5

Author: Thesis Implementation - Phase 3 Stage 2
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
# Set DENSITY_DOWNSAMPLE_FACTOR to 1 before importing dataset class
import dataset_nwpu
dataset_nwpu.DENSITY_DOWNSAMPLE_FACTOR = 1
from dataset_nwpu import NWPUCrowdDataset
from models.lcdnet import create_model
from utils.metrics import compute_mae, compute_mse, density_to_count


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Fine-tune LCDNet on NWPU-Crowd sparse images'
    )
    parser.add_argument(
        '--epochs', type=int, default=50,
        help='Number of training epochs (default: 50)'
    )
    parser.add_argument(
        '--batch_size', type=int, default=8,
        help='Batch size (default: 8)'
    )
    parser.add_argument(
        '--lr', type=float, default=1e-5,
        help='Learning rate for fine-tuning (default: 1e-5)'
    )
    parser.add_argument(
        '--weight_decay', type=float, default=1e-5,
        help='Weight decay for regularization (default: 1e-5)'
    )
    parser.add_argument(
        '--count_loss_weight', type=float, default=0.01,
        help='Weight for counting loss term (default: 0.01)'
    )
    parser.add_argument(
        '--pretrained_path', type=str,
        default=os.path.join(config.CHECKPOINTS_DIR, "best_model.pth"),
        help='Path to pretrained ShanghaiTech LCDNet weights'
    )
    parser.add_argument(
        '--device', type=str, default=config.DEVICE,
        help=f'Device to train on (default: {config.DEVICE})'
    )
    return parser.parse_args()


class LCDNetNWPUTrainer:
    """
    Fine-tuning manager for LCDNet on NWPU-Crowd sparse subset.
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
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.count_loss_weight = count_loss_weight
        
        self.criterion = nn.MSELoss()
        
        # Fine-tuning optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )
        
        self.best_val_mae = float('inf')
        self.epochs_without_improvement = 0
        self.current_epoch = 0
        
        self.train_losses = []
        self.val_losses = []
        self.val_maes = []
        
        os.makedirs(config.CHECKPOINTS_DIR, exist_ok=True)
        
    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, (images, density_maps, counts) in enumerate(pbar):
            images = images.to(self.device)
            density_maps = density_maps.to(self.device)
            counts = counts.to(self.device).float()
            
            self.optimizer.zero_grad()
            
            # Forward pass (outputs B, 1, H, W)
            pred_density = self.model(images)
            
            # Density loss (MSE)
            density_loss = self.criterion(pred_density, density_maps)
            
            # Counting loss (L1)
            pred_counts = pred_density.sum(dim=[1, 2, 3])
            count_loss = torch.nn.functional.l1_loss(pred_counts, counts)
            
            loss = density_loss + self.count_loss_weight * count_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.6f}',
                'cnt_loss': f'{count_loss.item():.2f}'
            })
            
        return total_loss / num_batches
        
    def validate(self) -> tuple:
        self.model.eval()
        total_loss = 0.0
        pred_counts = []
        gt_counts = []
        
        with torch.no_grad():
            for images, density_maps, counts in tqdm(self.val_loader, desc="Validating"):
                images = images.to(self.device)
                density_maps = density_maps.to(self.device)
                
                pred_density = self.model(images)
                loss = self.criterion(pred_density, density_maps)
                total_loss += loss.item()
                
                batch_pred_counts = density_to_count(pred_density)
                if isinstance(batch_pred_counts, np.ndarray):
                    pred_counts.extend(batch_pred_counts.tolist())
                else:
                    pred_counts.append(batch_pred_counts)
                    
                gt_counts.extend(counts.numpy().tolist())
                
        avg_loss = total_loss / len(self.val_loader)
        mae = compute_mae(pred_counts, gt_counts)
        mse = compute_mse(pred_counts, gt_counts)
        return avg_loss, mae, mse
        
    def save_checkpoint(self, is_best: bool = False):
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
        
        path = os.path.join(config.CHECKPOINTS_DIR, f"lcdnet_nwpu_epoch_{self.current_epoch}.pth")
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = os.path.join(config.CHECKPOINTS_DIR, "best_model_nwpu_sparse.pth")
            torch.save(checkpoint, best_path)
            print(f"  * Saved best LCDNet NWPU sparse model with MAE: {self.best_val_mae:.4f}")
            
    def train(self, num_epochs: int):
        print(f"Fine-tuning LCDNet sparse model on {self.device}...")
        early_stop_patience = 15
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            val_loss, val_mae, val_mse = self.validate()
            self.val_losses.append(val_loss)
            self.val_maes.append(val_mae)
            
            self.scheduler.step(val_mae)
            
            epoch_time = time.time() - epoch_start
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"\nEpoch {epoch}/{num_epochs - 1}:")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss:   {val_loss:.6f}")
            print(f"  Val MAE:    {val_mae:.2f}")
            print(f"  Val MSE:    {val_mse:.2f}")
            print(f"  LR:         {current_lr:.2e}")
            print(f"  Time:       {epoch_time:.1f}s")
            
            is_best = val_mae < self.best_val_mae
            if is_best:
                self.best_val_mae = val_mae
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
                
            if epoch % 5 == 0 or is_best:
                self.save_checkpoint(is_best)
                
            if self.epochs_without_improvement >= early_stop_patience:
                print(f"\nEarly stopping: No improvement for {early_stop_patience} epochs.")
                break
                
        print("\nLCDNet Fine-tuning Complete!")
        print(f"Best Validation MAE on Sparse: {self.best_val_mae:.2f}")


def main():
    args = parse_args()
    
    print("=" * 60)
    print("LCDNet NWPU Sparse Fine-tuning Configuration")
    print("=" * 60)
    print(f"Epochs:           {args.epochs}")
    print(f"Batch size:       {args.batch_size}")
    print(f"Learning rate:    {args.lr}")
    print(f"Weight decay:     {args.weight_decay}")
    print(f"Count loss weight:{args.count_loss_weight}")
    print(f"Pretrained weights:{args.pretrained_path}")
    print(f"Device:           {args.device}")
    print("=" * 60)
    
    # 1. Load and filter datasets for GT count <= 100
    print("\nLoading and filtering NWPU datasets for sparse subset (GT <= 100)...")
    train_dataset = NWPUCrowdDataset(split='train')
    val_dataset = NWPUCrowdDataset(split='val')
    
    # Filter
    train_dataset.samples = [s for s in train_dataset.samples if s[2] <= 100]
    val_dataset.samples = [s for s in val_dataset.samples if s[2] <= 100]
    
    print(f"Filtered Train samples (GT <= 100): {len(train_dataset)}")
    print(f"Filtered Val samples (GT <= 100): {len(val_dataset)}")
    
    # Create PyTorch DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    # 2. Create and load model
    print("\nCreating LCDNet model...")
    model = create_model(args.device)
    
    if os.path.exists(args.pretrained_path):
        print(f"Loading pretrained weights from {args.pretrained_path}...")
        ckpt = torch.load(args.pretrained_path, map_location=args.device)
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        print(f"Warning: Pretrained weights not found at {args.pretrained_path}. Starting from scratch!")
        
    # 3. Trainer
    trainer = LCDNetNWPUTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        count_loss_weight=args.count_loss_weight,
        device=args.device
    )
    
    # Train
    trainer.train(args.epochs)


if __name__ == "__main__":
    main()
