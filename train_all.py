"""
Sequential Training Wrapper for Phase 3 Stage 2.

This script sequentially fine-tunes LCDNet on NWPU-Crowd sparse images and trains
MobileCount on NWPU-Crowd dense images. Running them sequentially prevents
GPU out-of-memory (OOM) errors.

Usage:
    python train_all.py --epochs 50 --batch_size 8
"""

import sys
import subprocess
import argparse


def main():
    parser = argparse.ArgumentParser(description="Sequential training script for Phase 3 Stage 2")
    parser.add_argument(
        '--epochs', type=int, default=50,
        help='Number of epochs to train each model (default: 50)'
    )
    parser.add_argument(
        '--batch_size', type=int, default=8,
        help='Batch size for training (default: 8)'
    )
    args = parser.parse_args()

    # 1. Fine-tune LCDNet on Sparse Subset
    print("=" * 70)
    print(f"STEP 1: Fine-tuning LCDNet (Sparse Subset, GT <= 100) for {args.epochs} epochs")
    print("=" * 70)
    
    cmd_lcd = [
        sys.executable, "train_lcdnet_nwpu.py",
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--lr", "1e-5"
    ]
    
    try:
        subprocess.run(cmd_lcd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] LCDNet fine-tuning failed with exit code: {e.returncode}")
        sys.exit(e.returncode)

    # 2. Train MobileCount on Dense Subset
    print("\n" + "=" * 70)
    print(f"STEP 2: Training MobileCount (Dense Subset, GT > 100) for {args.epochs} epochs")
    print("=" * 70)
    
    cmd_mc = [
        sys.executable, "train_lightweight.py",
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--lr", "1e-4"
    ]
    
    try:
        subprocess.run(cmd_mc, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] MobileCount training failed with exit code: {e.returncode}")
        sys.exit(e.returncode)

    print("\n" + "=" * 70)
    print("ALL TRAINING STAGES COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
