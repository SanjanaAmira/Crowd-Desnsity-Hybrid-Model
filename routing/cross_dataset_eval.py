"""
Cross-dataset evaluation: trained on NWPU-Crowd, tested on ShanghaiTech.

Evaluates LCDNet (alone), MobileCount (alone), and Hybrid-Hard on the
ShanghaiTech Part A test split (182 images) and Part B test split (316 images).
Ground-truth counts come from the matlab .mat annotation files (not density
maps -- we only need the head count).

Reports MAE / RMSE / stratified MAE per dataset, plus oracle upper bound.

Usage:
    python routing/cross_dataset_eval.py
    python routing/cross_dataset_eval.py --part A
    python routing/cross_dataset_eval.py --part B --max_samples 50

Author: Thesis - Phase 3 Final Defense Prep
"""

import os
import sys
import json
import argparse
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

try:
    from scipy.io import loadmat
except ImportError:
    print("scipy is required: pip install scipy")
    sys.exit(1)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from routing import config_routing
from routing.router import load_router
from models.lcdnet import LCDNet
from models.mobilecount import MobileCount
import config as project_config


SHT_ROOT = os.path.join(project_config.DATA_DIR, "ShanghaiTech")


def shanghaitech_count(mat_path: str) -> float:
    """Extract head count from a ShanghaiTech .mat ground-truth file."""
    m = loadmat(mat_path)
    # The struct field name differs by part / file:
    # most use 'image_info' -> [0,0]['number'][0,0]; older formats use 'annPoints'.
    if 'image_info' in m:
        info = m['image_info']
        try:
            num = info[0, 0]['number'][0, 0]
            return float(num)
        except Exception:
            try:
                pts = info[0, 0]['location'][0, 0]
                return float(len(pts))
            except Exception:
                pass
    if 'annPoints' in m:
        return float(len(m['annPoints']))
    raise RuntimeError(f"Unknown .mat format: {mat_path}")


def list_split(part: str, max_samples: Optional[int]):
    """Return list of (image_path, gt_count) for ShanghaiTech part A or B test."""
    base = os.path.join(SHT_ROOT, f"part_{part}", "test_data")
    img_dir = os.path.join(base, "images")
    gt_dir = os.path.join(base, "ground-truth")
    samples = []
    for fn in sorted(os.listdir(img_dir)):
        if not fn.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        stem = os.path.splitext(fn)[0]
        mat_path = os.path.join(gt_dir, f"GT_{stem}.mat")
        if not os.path.exists(mat_path):
            continue
        try:
            gt = shanghaitech_count(mat_path)
        except Exception as e:
            print(f"skip {fn}: {e}")
            continue
        samples.append((os.path.join(img_dir, fn), gt))
        if max_samples and len(samples) >= max_samples:
            break
    return samples


def density_bin(gt: float) -> str:
    if gt <= 100:
        return "sparse"
    elif gt <= 500:
        return "medium"
    else:
        return "dense"


def stratified(records):
    bins = defaultdict(list)
    for r in records:
        bins[density_bin(r['gt'])].append(abs(r['pred'] - r['gt']))
    out = {}
    for k, v in bins.items():
        if v:
            arr = np.asarray(v)
            out[k] = {'count': len(arr),
                      'mae': float(arr.mean()),
                      'rmse': float(np.sqrt((arr**2).mean()))}
    return out


def metrics_from_errs(errs):
    a = np.asarray(errs)
    return {'mae': float(a.mean()),
            'rmse': float(np.sqrt((a**2).mean())),
            'n': len(a)}


def evaluate(samples, device):
    """One forward pass per model; collect per-image preds for all schemes."""
    print("Loading models...")
    router = load_router(device=device); router.eval()
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

    lcd_records, mc_records, hyb_records, oracle_records = [], [], [], []
    routing_stats = defaultdict(int)
    oracle_picks = defaultdict(int)

    with torch.no_grad():
        for img_path, gt in tqdm(samples, desc="forward"):
            img = Image.open(img_path).convert('RGB')
            r_in = router_tf(img).unsqueeze(0).to(device)
            d_in = density_tf(img).unsqueeze(0).to(device)
            probs = F.softmax(router(r_in), dim=1)
            p_lcd = probs[0, 0].item()
            p_csr = probs[0, 1].item()
            pred_label = 0 if p_lcd >= p_csr else 1

            lcd_count = lcd(d_in).sum().item()
            mc_count = mc(d_in).sum().item()
            hybrid_count = mc_count if pred_label == 1 else lcd_count
            choice = "MobileCount" if pred_label == 1 else "LCDNet"
            routing_stats[choice] += 1

            lcd_records.append({'gt': gt, 'pred': lcd_count})
            mc_records.append({'gt': gt, 'pred': mc_count})
            hyb_records.append({'gt': gt, 'pred': hybrid_count})

            # Oracle pick = whichever single model gets closer
            if abs(lcd_count - gt) <= abs(mc_count - gt):
                oracle_records.append({'gt': gt, 'pred': lcd_count})
                oracle_picks['LCDNet'] += 1
            else:
                oracle_records.append({'gt': gt, 'pred': mc_count})
                oracle_picks['MobileCount'] += 1

    def block(records, label):
        errs = [abs(r['pred'] - r['gt']) for r in records]
        m = metrics_from_errs(errs)
        m['label'] = label
        m['stratified'] = stratified(records)
        return m

    return {
        'LCDNet':       block(lcd_records, 'LCDNet (alone)'),
        'MobileCount':  block(mc_records, 'MobileCount (alone)'),
        'Hybrid-Hard':  block(hyb_records, 'Hybrid-Hard'),
        'Oracle':       block(oracle_records, 'Oracle Router'),
        'routing_stats': dict(routing_stats),
        'oracle_picks':  dict(oracle_picks),
    }


def write_report(part, results, n, out_txt):
    lines = []
    bar = "=" * 78
    sub = "-" * 78
    lines += [bar,
              f"PHASE 3 CROSS-DATASET EVAL  -  ShanghaiTech Part {part} test "
              f"({n} images)",
              bar,
              "Models trained on NWPU-Crowd (no fine-tuning on ShanghaiTech).",
              ""]
    lines += [f"{'Model':<22}{'MAE':>10}{'RMSE':>10}",
              sub]
    for k in ['LCDNet', 'MobileCount', 'Hybrid-Hard', 'Oracle']:
        r = results[k]
        lines.append(f"{r['label']:<22}{r['mae']:>10.2f}{r['rmse']:>10.2f}")
    lines += [sub, ""]

    lines += ["STRATIFIED MAE",
              sub,
              f"{'Model':<22}{'Sparse (<=100)':>16}"
              f"{'Medium (100-500)':>20}{'Dense (>500)':>18}",
              sub]
    for k in ['LCDNet', 'MobileCount', 'Hybrid-Hard', 'Oracle']:
        r = results[k]
        s = r['stratified']
        cells = []
        for b in ['sparse', 'medium', 'dense']:
            c = s.get(b)
            cells.append(f"{c['mae']:.1f} (n={c['count']})" if c else "  -  ")
        lines.append(f"{r['label']:<22}{cells[0]:>16}{cells[1]:>20}{cells[2]:>18}")
    lines += [sub, ""]

    lines.append(f"Routing distribution (Hybrid-Hard):")
    for m, n_ in results['routing_stats'].items():
        lines.append(f"   {m:<14} {n_:>4} images ({100*n_/n:5.1f}%)")
    lines.append("")
    lines.append(f"Oracle picks:")
    for m, n_ in results['oracle_picks'].items():
        lines.append(f"   {m:<14} {n_:>4} images ({100*n_/n:5.1f}%)")
    lines += ["", bar]

    text = "\n".join(lines)
    print(text)
    os.makedirs(os.path.dirname(out_txt), exist_ok=True)
    with open(out_txt, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Saved: {out_txt}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', choices=['A', 'B', 'both'], default='both')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--out_dir', default=config_routing.ROUTER_LOG_DIR)
    args = parser.parse_args()

    device = config_routing.DEVICE
    print(f"Device: {device}")

    parts = ['A', 'B'] if args.part == 'both' else [args.part]

    all_summary = {}
    for part in parts:
        samples = list_split(part, args.max_samples)
        print(f"\nShanghaiTech Part {part} test: {len(samples)} samples")
        if not samples:
            print(f"No samples for Part {part}, skipping.")
            continue
        results = evaluate(samples, device)
        out_txt = os.path.join(
            args.out_dir, f"phase3_cross_dataset_part{part}.txt")
        write_report(part, results, len(samples), out_txt)
        all_summary[f'part_{part}'] = {
            'n': len(samples),
            'results': {k: {kk: vv for kk, vv in v.items()
                            if kk != 'predictions'}
                        for k, v in results.items()
                        if isinstance(v, dict) and 'mae' in v}
        }

    out_json = os.path.join(args.out_dir, "phase3_cross_dataset_summary.json")
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(all_summary, f, indent=2, default=str)
    print(f"\nJSON summary: {out_json}")


if __name__ == "__main__":
    main()
