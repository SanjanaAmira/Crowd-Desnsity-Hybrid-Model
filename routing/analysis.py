"""
Router calibration + failure-case analysis for Phase 3.

Calibration (item #11):
- Reliability diagram (predicted confidence vs empirical accuracy).
- Expected Calibration Error (ECE).

Failure cases (item #13):
- Top-K worst predictions for each model:
  LCDNet, MobileCount, Hybrid-Hard.
- Output: a CSV per model + a markdown summary.

Author: Thesis - Phase 3 Final Defense Prep
"""

import os
import sys
import json
import argparse
import csv
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from routing import config_routing
from routing.router import load_router
from models.lcdnet import LCDNet
from models.mobilecount import MobileCount


def load_samples():
    samples = []
    with open(config_routing.NWPU_VAL_TXT, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            img_id = parts[0]
            img_path = None
            for img_dir in config_routing.NWPU_IMAGE_DIRS:
                for ext in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']:
                    p = os.path.join(img_dir, f"{img_id}{ext}")
                    if os.path.exists(p):
                        img_path = p
                        break
                if img_path:
                    break
            if img_path is None:
                continue
            json_path = os.path.join(config_routing.NWPU_JSONS_DIR, f"{img_id}.json")
            try:
                with open(json_path, 'r') as jf:
                    gt = float(json.load(jf).get('human_num', 0))
            except Exception:
                gt = 0.0
            samples.append((img_path, gt, img_id))
    return samples


def gt_label(gt: float) -> int:
    """Ground-truth routing label (0=sparse, 1=dense) using the configured T."""
    return 0 if gt <= config_routing.ROUTING_THRESHOLD else 1


def build_predictions(samples, device):
    """One forward pass per model per image. Returns list of dicts."""
    print("Running models on val (one pass per model)...")

    router = load_router(device=device)
    router.eval()

    lcd = LCDNet()
    ckpt = torch.load(config_routing.LCDNET_CHECKPOINT, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        lcd.load_state_dict(ckpt['model_state_dict'])
    else:
        lcd.load_state_dict(ckpt)
    lcd.to(device).eval()

    mc = MobileCount(pretrained=False)
    ckpt = torch.load(config_routing.DENSE_CHECKPOINT, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        mc.load_state_dict(ckpt['model_state_dict'])
    else:
        mc.load_state_dict(ckpt)
    mc.to(device).eval()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    router_tf = transforms.Compose([
        transforms.Resize(config_routing.ROUTER_INPUT_SIZE),
        transforms.ToTensor(), normalize])
    density_tf = transforms.Compose([
        transforms.Resize(config_routing.DENSITY_INPUT_SIZE),
        transforms.ToTensor(), normalize])

    out = []
    with torch.no_grad():
        for img_path, gt, img_id in tqdm(samples, desc="forward"):
            img = Image.open(img_path).convert('RGB')
            r_in = router_tf(img).unsqueeze(0).to(device)
            d_in = density_tf(img).unsqueeze(0).to(device)
            probs = F.softmax(router(r_in), dim=1)
            p_lcd = probs[0, 0].item()
            p_csr = probs[0, 1].item()
            pred_label = int(p_csr >= 0.5)
            lcd_count = lcd(d_in).sum().item()
            mc_count = mc(d_in).sum().item()
            hybrid_count = mc_count if pred_label == 1 else lcd_count
            out.append({
                'img_id': img_id,
                'path': img_path,
                'gt': gt,
                'gt_label': gt_label(gt),
                'p_lcd': p_lcd,
                'p_csr': p_csr,
                'pred_label': pred_label,
                'router_correct': int(pred_label == gt_label(gt)),
                'lcd_count': lcd_count,
                'mc_count': mc_count,
                'hybrid_count': hybrid_count,
                'lcd_err': abs(lcd_count - gt),
                'mc_err': abs(mc_count - gt),
                'hybrid_err': abs(hybrid_count - gt),
            })

    del router, lcd, mc
    if device == 'cuda':
        torch.cuda.empty_cache()
    return out


# -----------------------------------------------------------------------------
# Calibration
# -----------------------------------------------------------------------------

def reliability_curve(records, n_bins: int = 10):
    """For binary classifier, compute confidence vs accuracy per bin.
    Confidence = max(p_lcd, p_csr); correctness = (pred_label == gt_label)."""
    confs = np.array([max(r['p_lcd'], r['p_csr']) for r in records])
    correct = np.array([r['router_correct'] for r in records], dtype=float)
    bins = np.linspace(0.5, 1.0, n_bins + 1)  # confidence is in [0.5, 1]
    bin_idx = np.digitize(confs, bins, right=False) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    out = []
    ece = 0.0
    n = len(confs)
    for b in range(n_bins):
        mask = bin_idx == b
        if not mask.any():
            continue
        c_mean = confs[mask].mean()
        a_mean = correct[mask].mean()
        weight = mask.sum() / n
        ece += weight * abs(c_mean - a_mean)
        out.append({
            'bin_low': float(bins[b]),
            'bin_high': float(bins[b + 1]),
            'count': int(mask.sum()),
            'avg_confidence': float(c_mean),
            'accuracy': float(a_mean),
        })
    return out, float(ece)


def plot_reliability(curve, ece, out_path):
    if not curve:
        return
    confs = [c['avg_confidence'] for c in curve]
    accs  = [c['accuracy'] for c in curve]
    counts = [c['count'] for c in curve]

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0.5, 1.0], [0.5, 1.0], '--', color='gray', label='Perfect calibration')
    ax.plot(confs, accs, 'o-', color='C0', label=f'Router (ECE={ece:.3f})')
    for c, a, n in zip(confs, accs, counts):
        ax.annotate(f"n={n}", (c, a), textcoords='offset points',
                    xytext=(4, 4), fontsize=8)
    ax.set_xlabel("Predicted confidence (max softmax)")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("Router reliability diagram (NWPU val)")
    ax.set_xlim(0.45, 1.02); ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Failure-case dump
# -----------------------------------------------------------------------------

def dump_topk(records, key, k, out_csv):
    items = sorted(records, key=lambda r: r[key], reverse=True)[:k]
    cols = ['img_id', 'gt', 'gt_label',
            'p_lcd', 'p_csr', 'pred_label', 'router_correct',
            'lcd_count', 'mc_count', 'hybrid_count',
            'lcd_err', 'mc_err', 'hybrid_err', 'path']
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in items:
            w.writerow({c: r[c] for c in cols})
    return items


def write_md_summary(out_dir, ece, calib, top_lcd, top_mc, top_hyb, n_total,
                     router_acc):
    md = []
    md.append("# Phase 3 — Router Calibration & Failure Cases\n")
    md.append(f"NWPU val, {n_total} images. Router accuracy on val: "
              f"{router_acc*100:.2f}%. **ECE = {ece:.3f}** "
              f"(perfect=0; lower is better).\n")
    md.append("## Reliability bins\n")
    md.append("| Conf bin | Count | Avg conf | Empirical acc |")
    md.append("| :------- | ----: | -------: | ------------: |")
    for c in calib:
        md.append(f"| {c['bin_low']:.2f}–{c['bin_high']:.2f} | {c['count']} | "
                  f"{c['avg_confidence']:.3f} | {c['accuracy']:.3f} |")
    md.append("\n![reliability](reliability_diagram.png)\n")

    def block(title, items):
        md.append(f"\n## {title}\n")
        md.append("| img_id | GT | GT lbl | p_lcd | p_csr | pred lbl | "
                  "LCD | MC | Hybrid | LCD err | MC err | Hyb err |")
        md.append("|---|---:|:---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|")
        for r in items:
            md.append(f"| {r['img_id']} | {r['gt']:.0f} | {r['gt_label']} | "
                      f"{r['p_lcd']:.2f} | {r['p_csr']:.2f} | {r['pred_label']} | "
                      f"{r['lcd_count']:.0f} | {r['mc_count']:.0f} | "
                      f"{r['hybrid_count']:.0f} | "
                      f"{r['lcd_err']:.0f} | {r['mc_err']:.0f} | "
                      f"{r['hybrid_err']:.0f} |")

    block("Top-10 worst LCDNet predictions",  top_lcd)
    block("Top-10 worst MobileCount predictions", top_mc)
    block("Top-10 worst Hybrid-Hard predictions", top_hyb)

    md.append("\n## Notes\n")
    md.append("- 'GT lbl': ground-truth routing label (0=sparse ≤100, 1=dense >100).\n"
              "- 'pred lbl': what the router predicted (argmax).\n"
              "- A miss-route pushes the worst case to the wrong model and "
              "dominates the failure list.\n")

    out = os.path.join(out_dir, "phase3_router_calibration_failures.md")
    with open(out, 'w', encoding='utf-8') as f:
        f.write("\n".join(md))
    return out


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--out_dir', default=config_routing.ROUTER_LOG_DIR)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = config_routing.DEVICE
    print(f"Device: {device}")

    samples = load_samples()
    records = build_predictions(samples, device)
    print(f"Done: {len(records)} samples")

    router_acc = float(np.mean([r['router_correct'] for r in records]))
    print(f"Router accuracy on val: {router_acc*100:.2f}%")

    # Calibration
    calib, ece = reliability_curve(records, n_bins=10)
    print(f"ECE: {ece:.4f}")
    plot_path = os.path.join(args.out_dir, "reliability_diagram.png")
    plot_reliability(calib, ece, plot_path)
    print(f"Saved reliability diagram: {plot_path}")

    # Failure CSVs
    top_lcd = dump_topk(records, 'lcd_err',    args.top_k,
                        os.path.join(args.out_dir, "phase3_top_failures_lcdnet.csv"))
    top_mc  = dump_topk(records, 'mc_err',     args.top_k,
                        os.path.join(args.out_dir, "phase3_top_failures_mobilecount.csv"))
    top_hyb = dump_topk(records, 'hybrid_err', args.top_k,
                        os.path.join(args.out_dir, "phase3_top_failures_hybrid.csv"))

    md_path = write_md_summary(args.out_dir, ece, calib, top_lcd, top_mc,
                               top_hyb, len(records), router_acc)
    print(f"Saved markdown: {md_path}")


if __name__ == "__main__":
    main()
