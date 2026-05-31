"""
Knowledge-distillation training for MobileCount.

Teacher : CSRNet (16.2M params, frozen, NOT in deployed pipeline).
Student : MobileCount (884K params).

Total loss:
    L = alpha * L_density(student, GT)
      + beta  * L_distill(student, teacher.detach())
      + gamma * L_count(student, GT)

L_density and L_distill are MSE on density maps (same 1/8 res for both).
L_count is L1 on summed counts.

Initialised from the existing MobileCount checkpoint, fine-tuned for a
few epochs (default 12) -- enough to see if KD helps.

Resulting checkpoint:
    checkpoints/mobilecount_distilled.pth   (kept separate; old one is intact)

Usage:
    python train_mobilecount_distill.py --epochs 12 --batch_size 8
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from dataset_nwpu import NWPUCrowdDataset
from models.mobilecount import MobileCount
from models.csrnet import CSRNet
from utils.metrics import compute_mae, compute_mse, density_to_count


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--alpha', type=float, default=0.5,
                   help='Weight on L_density (student vs GT)')
    p.add_argument('--beta',  type=float, default=0.5,
                   help='Weight on L_distill (student vs teacher)')
    p.add_argument('--gamma', type=float, default=0.05,
                   help='Weight on L1 count loss')
    p.add_argument('--student_init',
                   default=os.path.join(config.CHECKPOINTS_DIR,
                                        'mobilecount_best.pth'),
                   help='Initial student weights (warm start)')
    p.add_argument('--teacher_ckpt',
                   default=os.path.join(config.CHECKPOINTS_DIR,
                                        'csrnet', 'csrnet_best.pth'),
                   help='Teacher checkpoint (frozen)')
    p.add_argument('--save_path',
                   default=os.path.join(config.CHECKPOINTS_DIR,
                                        'mobilecount_distilled.pth'))
    p.add_argument('--filter_dense', action='store_true',
                   help='Only train on GT > 100 (matches existing MC training)')
    p.add_argument('--device', default=config.DEVICE)
    return p.parse_args()


def load_state_dict(path, device):
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        return ckpt['model_state_dict']
    return ckpt


def make_loaders(batch_size, filter_dense):
    train_ds = NWPUCrowdDataset(split='train')
    val_ds   = NWPUCrowdDataset(split='val')
    if filter_dense:
        train_ds.samples = [s for s in train_ds.samples if s[2] > 100]
        val_ds.samples   = [s for s in val_ds.samples   if s[2] > 100]
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True)
    return train_loader, val_loader


def main():
    args = parse_args()
    device = args.device
    print("=" * 60)
    print("MobileCount Knowledge-Distillation Training")
    print("=" * 60)
    print(f"alpha (GT density)  : {args.alpha}")
    print(f"beta  (teacher KD)  : {args.beta}")
    print(f"gamma (count L1)    : {args.gamma}")
    print(f"epochs={args.epochs}  bs={args.batch_size}  lr={args.lr}")
    print(f"Student init: {args.student_init}")
    print(f"Teacher    : {args.teacher_ckpt}")
    print(f"Save to    : {args.save_path}")
    print(f"Filter dense (>100): {args.filter_dense}")

    train_loader, val_loader = make_loaders(args.batch_size, args.filter_dense)

    # Student
    student = MobileCount(pretrained=False)
    if os.path.exists(args.student_init):
        student.load_state_dict(load_state_dict(args.student_init, device))
        print("Loaded student warm-start.")
    student.to(device).train()

    # Teacher (frozen)
    teacher = CSRNet(pretrained=False)
    teacher.load_state_dict(load_state_dict(args.teacher_ckpt, device))
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    optimizer = optim.AdamW(student.parameters(),
                            lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
    mse = nn.MSELoss()

    best_val_mae = float('inf')
    history = []

    for epoch in range(args.epochs):
        student.train()
        epoch_loss = 0.0
        n_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for images, gt_density, gt_counts in pbar:
            images = images.to(device, non_blocking=True)
            gt_density = gt_density.to(device, non_blocking=True)
            gt_counts = gt_counts.to(device, non_blocking=True).float()

            with torch.no_grad():
                t_density = teacher(images)  # B,1,H/8,W/8

            s_density = student(images)      # B,1,H/8,W/8

            # Match resolutions if necessary (should already match for 384 in)
            if s_density.shape[-2:] != t_density.shape[-2:]:
                t_density = nn.functional.interpolate(
                    t_density, size=s_density.shape[-2:],
                    mode='bilinear', align_corners=False)

            l_dens = mse(s_density, gt_density)
            l_kd   = mse(s_density, t_density)
            pred_counts = s_density.sum(dim=[1, 2, 3])
            l_cnt = nn.functional.l1_loss(pred_counts, gt_counts)

            loss = (args.alpha * l_dens + args.beta * l_kd
                    + args.gamma * l_cnt)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({'loss': f"{loss.item():.4f}",
                              'l_dens': f"{l_dens.item():.4f}",
                              'l_kd': f"{l_kd.item():.4f}",
                              'l_cnt': f"{l_cnt.item():.1f}"})

        scheduler.step()

        # Validation
        student.eval()
        preds, gts = [], []
        val_loss = 0.0
        with torch.no_grad():
            for images, gt_density, gt_counts in tqdm(val_loader, desc="val"):
                images = images.to(device)
                gt_density = gt_density.to(device)
                s_density = student(images)
                val_loss += mse(s_density, gt_density).item()
                p = density_to_count(s_density)
                if isinstance(p, np.ndarray):
                    preds.extend(p.tolist())
                else:
                    preds.append(p)
                gts.extend(gt_counts.numpy().tolist())
        val_mae = compute_mae(preds, gts)
        val_mse = compute_mse(preds, gts)
        val_loss /= max(1, len(val_loader))
        train_loss = epoch_loss / max(1, n_batches)
        history.append({'epoch': epoch, 'train_loss': train_loss,
                        'val_loss': val_loss, 'val_mae': val_mae,
                        'val_mse': val_mse,
                        'lr': optimizer.param_groups[0]['lr']})
        print(f"\nEpoch {epoch}: train={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_MAE={val_mae:.2f}  "
              f"val_MSE={val_mse:.2f}  lr={optimizer.param_groups[0]['lr']:.2e}")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            torch.save({'epoch': epoch,
                        'model_state_dict': student.state_dict(),
                        'best_val_mae': best_val_mae,
                        'args': vars(args)},
                       args.save_path)
            print(f"  -> NEW BEST distilled MobileCount (MAE {best_val_mae:.2f})")

    print(f"\nDone. Best distilled MAE: {best_val_mae:.2f}")
    print(f"Saved to: {args.save_path}")


if __name__ == "__main__":
    main()
