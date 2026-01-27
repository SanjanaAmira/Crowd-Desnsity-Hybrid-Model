"""
Improved Training Script for CSRNet on NWPU-Crowd Dataset.

Key improvements over original train_csrnet.py:
1. SGD optimizer with momentum (like original CSRNet paper)
2. SSIM loss for sharper density maps
3. Higher learning rate with proper scheduling
4. Better gradient clipping

Usage:
    python train_csrnet_v2.py --epochs 100 --lr 1e-4

Author: Thesis Implementation - Phase 2 Part 2 (Improved)
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from dataset_nwpu import create_nwpu_dataloaders
from models.csrnet import CSRNet, create_csrnet
from utils.metrics import compute_mae, compute_mse, density_to_count


# Checkpoint directory
CSRNET_CHECKPOINTS_DIR = os.path.join(config.CHECKPOINTS_DIR, "csrnet")


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index Loss for sharper density maps.
    
    SSIM measures structural similarity between images, helping
    preserve spatial structures in the predicted density maps.
    """
    def __init__(self, window_size=11, sigma=1.5):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.channel = 1
        self.window = self._create_window(window_size, sigma)
        
    def _create_window(self, window_size, sigma):
        """Create Gaussian window for SSIM."""
        gauss = torch.Tensor([
            np.exp(-(x - window_size//2)**2 / (2*sigma**2)) 
            for x in range(window_size)
        ])
        gauss = gauss / gauss.sum()
        window = gauss.unsqueeze(1) @ gauss.unsqueeze(0)
        window = window.unsqueeze(0).unsqueeze(0)
        return window
    
    def forward(self, pred, target):
        """Compute SSIM loss (1 - SSIM)."""
        if self.window.device != pred.device:
            self.window = self.window.to(pred.device)
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        mu_pred = torch.nn.functional.conv2d(
            pred, self.window, padding=self.window_size//2, groups=1
        )
        mu_target = torch.nn.functional.conv2d(
            target, self.window, padding=self.window_size//2, groups=1
        )
        
        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_pred_target = mu_pred * mu_target
        
        sigma_pred_sq = torch.nn.functional.conv2d(
            pred * pred, self.window, padding=self.window_size//2, groups=1
        ) - mu_pred_sq
        sigma_target_sq = torch.nn.functional.conv2d(
            target * target, self.window, padding=self.window_size//2, groups=1
        ) - mu_target_sq
        sigma_pred_target = torch.nn.functional.conv2d(
            pred * target, self.window, padding=self.window_size//2, groups=1
        ) - mu_pred_target
        
        ssim_map = ((2*mu_pred_target + C1) * (2*sigma_pred_target + C2)) / \
                   ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))
        
        return 1 - ssim_map.mean()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Improved CSRNet Training')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-6,
                       help='Learning rate (1e-6 recommended for SGD)')
    parser.add_argument('--momentum', type=float, default=0.95)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--ssim_weight', type=float, default=0.1,
                       help='Weight for SSIM loss (0.1 recommended)')
    parser.add_argument('--count_weight', type=float, default=0.001,
                       help='Weight for count loss')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--device', type=str, default=config.DEVICE)
    return parser.parse_args()


class ImprovedTrainer:
    """Improved trainer with SGD, SSIM loss, and better scheduling."""
    
    def __init__(self, model, train_loader, val_loader, args):
        self.model = model.to(args.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = args.device
        self.ssim_weight = args.ssim_weight
        self.count_weight = args.count_weight
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.ssim_loss = SSIMLoss()
        
        # SGD optimizer like original CSRNet paper
        self.optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
        
        # Step LR scheduler: reduce by 0.1 every 30 epochs
        self.scheduler = StepLR(self.optimizer, step_size=30, gamma=0.1)
        
        # State
        self.best_mae = float('inf')
        self.current_epoch = 0
        
        os.makedirs(CSRNET_CHECKPOINTS_DIR, exist_ok=True)
    
    def train_epoch(self):
        """Train for one epoch with combined loss."""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for images, density_maps, counts in pbar:
            images = images.to(self.device)
            density_maps = density_maps.to(self.device)
            counts = counts.to(self.device).float()
            
            self.optimizer.zero_grad()
            
            # Forward pass
            pred_density = self.model(images)
            
            # Combined loss: MSE + SSIM + Count
            mse = self.mse_loss(pred_density, density_maps)
            
            # Normalize density maps for SSIM (0-1 range)
            pred_norm = pred_density / (pred_density.max() + 1e-8)
            target_norm = density_maps / (density_maps.max() + 1e-8)
            ssim = self.ssim_loss(pred_norm, target_norm)
            
            # Count loss
            pred_counts = pred_density.sum(dim=[1, 2, 3])
            count_loss = torch.nn.functional.l1_loss(pred_counts, counts)
            
            # Total loss
            loss = mse + self.ssim_weight * ssim + self.count_weight * count_loss
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'mse': f'{mse.item():.4f}',
                'cnt': f'{count_loss.item():.1f}'
            })
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        """Validate and compute MAE."""
        self.model.eval()
        pred_counts = []
        gt_counts = []
        
        with torch.no_grad():
            for images, density_maps, counts in tqdm(self.val_loader, desc="Validating"):
                images = images.to(self.device)
                pred_density = self.model(images)
                
                batch_pred = density_to_count(pred_density)
                if isinstance(batch_pred, np.ndarray):
                    pred_counts.extend(batch_pred.tolist())
                else:
                    pred_counts.append(batch_pred)
                gt_counts.extend(counts.numpy().tolist())
        
        mae = compute_mae(pred_counts, gt_counts)
        mse = compute_mse(pred_counts, gt_counts)
        return mae, mse
    
    def save_checkpoint(self, filename, is_best=False):
        """Save checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_mae': self.best_mae,
        }
        path = os.path.join(CSRNET_CHECKPOINTS_DIR, filename)
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = os.path.join(CSRNET_CHECKPOINTS_DIR, 'csrnet_best_v2.pth')
            torch.save(checkpoint, best_path)
            print(f"  ✓ New best MAE: {self.best_mae:.2f}")
    
    def load_checkpoint(self, path):
        """Load checkpoint and properly resume training."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Resume optimizer state if available
        if 'optimizer_state_dict' in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except:
                print("Warning: Could not load optimizer state, using fresh optimizer")
        
        # Resume from saved epoch
        self.current_epoch = checkpoint.get('epoch', 0) + 1
        self.best_mae = checkpoint.get('best_mae', checkpoint.get('best_val_mae', float('inf')))
        
        print(f"Resumed from epoch {self.current_epoch}, best MAE: {self.best_mae:.2f}")

    
    def train(self, num_epochs, resume_path=None):
        """Main training loop."""
        print("\n" + "=" * 60)
        print("CSRNet V2 Training (SGD + SSIM Loss)")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"SSIM weight: {self.ssim_weight}")
        print(f"Count weight: {self.count_weight}")
        print("=" * 60)
        
        if resume_path and os.path.exists(resume_path):
            self.load_checkpoint(resume_path)
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_mae, val_mse = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            
            # Print results
            lr = self.optimizer.param_groups[0]['lr']
            print(f"\nEpoch {epoch}/{num_epochs-1}:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val MAE:    {val_mae:.2f}")
            print(f"  Val MSE:    {val_mse:.2f}")
            print(f"  LR:         {lr:.2e}")
            
            # Save best
            is_best = val_mae < self.best_mae
            if is_best:
                self.best_mae = val_mae
            
            if epoch % 5 == 0 or is_best:
                self.save_checkpoint(f'csrnet_v2_epoch_{epoch}.pth', is_best)
        
        print("\n" + "=" * 60)
        print(f"Training Complete! Best MAE: {self.best_mae:.2f}")
        print("=" * 60)


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Improved CSRNet Training Configuration")
    print("=" * 60)
    print(f"Epochs:      {args.epochs}")
    print(f"Batch size:  {args.batch_size}")
    print(f"LR:          {args.lr}")
    print(f"Momentum:    {args.momentum}")
    print(f"SSIM weight: {args.ssim_weight}")
    print("=" * 60)
    
    # Load data
    print("\nLoading datasets...")
    train_loader, val_loader, _ = create_nwpu_dataloaders(batch_size=args.batch_size)
    
    # Create model
    print("Initializing model...")
    model = create_csrnet(args.device, pretrained=True)
    
    # Train
    trainer = ImprovedTrainer(model, train_loader, val_loader, args)
    trainer.train(args.epochs, args.resume)


if __name__ == "__main__":
    main()
