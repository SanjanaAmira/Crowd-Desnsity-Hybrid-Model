"""
Training Script for the Routing Classifier.

This script trains a MobileNetV2-based binary classifier to route images
to either LCDNet (sparse scenes) or CSRNet (dense scenes).

Training Configuration:
- Loss: CrossEntropyLoss
- Optimizer: Adam (lr=1e-4, weight_decay=1e-5)
- Scheduler: ReduceLROnPlateau
- Early Stopping: Patience=10 epochs

Usage:
    python routing/train_router.py                    # Full training
    python routing/train_router.py --epochs 5        # Custom epochs
    python routing/train_router.py --quick_test      # Quick validation

Author: Thesis Implementation - Phase 2 Part 3
"""

import os
import sys
import time
import argparse
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from routing import config_routing
from routing.router import create_router
from routing.dataset_routing import create_routing_dataloaders


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False
    
    def __call__(self, val_acc: float) -> bool:
        if self.best_score is None:
            self.best_score = val_acc
        elif val_acc < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = val_acc
            self.counter = 0
        return self.should_stop


def train_epoch(
    model: nn.Module,
    train_loader,
    criterion: nn.Module,
    optimizer,
    device: str
) -> tuple:
    """
    Train for one epoch.
    
    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels, _) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Progress indicator
        if (batch_idx + 1) % 10 == 0:
            print(f"    Batch {batch_idx + 1}/{len(train_loader)}, "
                  f"Loss: {loss.item():.4f}", end='\r')
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def validate(
    model: nn.Module,
    val_loader,
    criterion: nn.Module,
    device: str
) -> tuple:
    """
    Validate the model.
    
    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels, _ in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def save_checkpoint(
    model: nn.Module,
    optimizer,
    epoch: int,
    val_acc: float,
    path: str
):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc
    }, path)
    print(f"  Checkpoint saved: {path}")


def train_router(args):
    """Main training function."""
    
    print("=" * 60)
    print("ROUTING CLASSIFIER TRAINING")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {config_routing.DEVICE}")
    print(f"Routing threshold: {config_routing.ROUTING_THRESHOLD}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print("=" * 60)
    
    # Create directories
    config_routing.create_routing_directories()
    
    # Set device
    device = config_routing.DEVICE
    
    # Create model
    print("\n[1/4] Creating model...")
    model = create_router(device=device)
    
    # Create dataloaders
    print("\n[2/4] Loading data...")
    train_loader, val_loader, _ = create_routing_dataloaders(
        batch_size=args.batch_size
    )
    
    # Loss function and optimizer
    print("\n[3/4] Setting up training...")
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=config_routing.ROUTER_WEIGHT_DECAY
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=config_routing.ROUTER_LR_FACTOR,
        patience=config_routing.ROUTER_LR_PATIENCE,
        verbose=True
    )
    early_stopping = EarlyStopping(patience=config_routing.ROUTER_EARLY_STOPPING)
    
    # Training log
    log_file = config_routing.ROUTER_TRAINING_LOG
    with open(log_file, 'w') as f:
        f.write("Routing Classifier Training Log\n")
        f.write("=" * 60 + "\n")
        f.write(f"Threshold: {config_routing.ROUTING_THRESHOLD}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Learning rate: {args.lr}\n\n")
        f.write("Epoch\tTrain_Loss\tTrain_Acc\tVal_Loss\tVal_Acc\n")
        f.write("-" * 60 + "\n")
    
    # Training loop
    print("\n[4/4] Training...")
    print("-" * 60)
    
    best_val_acc = 0.0
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )
        
        epoch_time = time.time() - epoch_start
        
        # Print progress
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}% | "
              f"Time: {epoch_time:.1f}s")
        
        # Log to file
        with open(log_file, 'a') as f:
            f.write(f"{epoch}\t{train_loss:.4f}\t{train_acc:.2f}\t"
                   f"{val_loss:.4f}\t{val_acc:.2f}\n")
        
        # Update scheduler
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                model, optimizer, epoch, val_acc,
                config_routing.ROUTER_BEST_PATH
            )
        
        # Early stopping
        if early_stopping(val_acc):
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break
    
    total_time = time.time() - start_time
    
    # Training summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {config_routing.ROUTER_BEST_PATH}")
    print(f"Training log: {log_file}")
    print("=" * 60)
    
    # Append summary to log
    with open(log_file, 'a') as f:
        f.write("\n" + "=" * 60 + "\n")
        f.write(f"Training completed at: {datetime.now()}\n")
        f.write(f"Total time: {total_time/60:.1f} minutes\n")
        f.write(f"Best validation accuracy: {best_val_acc:.2f}%\n")
    
    return best_val_acc


def main():
    parser = argparse.ArgumentParser(description="Train routing classifier")
    parser.add_argument(
        '--epochs', type=int, default=config_routing.ROUTER_EPOCHS,
        help=f'Number of training epochs (default: {config_routing.ROUTER_EPOCHS})'
    )
    parser.add_argument(
        '--batch_size', type=int, default=config_routing.ROUTER_BATCH_SIZE,
        help=f'Batch size (default: {config_routing.ROUTER_BATCH_SIZE})'
    )
    parser.add_argument(
        '--lr', type=float, default=config_routing.ROUTER_LR,
        help=f'Learning rate (default: {config_routing.ROUTER_LR})'
    )
    parser.add_argument(
        '--quick_test', action='store_true',
        help='Run quick test with 2 epochs'
    )
    
    args = parser.parse_args()
    
    if args.quick_test:
        args.epochs = 2
        print("Running quick test mode (2 epochs)...")
    
    train_router(args)


if __name__ == "__main__":
    main()
