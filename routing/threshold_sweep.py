"""
Threshold sweeps for the hybrid system.

Two modes:
  --mode router_prob (cheap, no retraining):
      Sweep the router probability threshold p* used at inference time
      (i.e., route to MobileCount only if P(dense) >= p*). This shifts the
      operating point on the existing router without retraining.

  --mode soft_margin (cheap, no retraining):
      Sweep the soft-fusion margin (1 - margin = bypass confidence). Higher
      margin -> more fusion. Lower -> more pure hard routing.

For each setting, recomputes the hybrid MAE / RMSE / stratified-MAE on the
already-cached LCDNet and MobileCount predictions on NWPU val.

This avoids re-running the density models 500x per sweep point.

Usage:
    python routing/threshold_sweep.py --mode router_prob
    python routing/threshold_sweep.py --mode soft_margin
"""

import os
import sys
import json
import argparse
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from routing import config_routing
from routing.router import load_router
from models.lcdnet import LCDNet
from models.mobilecount import MobileCount


def density_bin(gt: float) -> str:
    if gt <= 100:
        return "sparse"
    elif gt <= 500:
        return "medium"
    else:
        return "dense"


# -----------------------------------------------------------------------------
# Load samples + cache router probs and per-model predictions ONCE
# -----------------------------------------------------------------------------

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
            samples.append((img_path, gt))
    return samples


def build_cache(samples, device):
    """For each image, run router + LCDNet + MobileCount once and cache:
    - router probs (p_lcd, p_csr)
    - LCDNet predicted count
    - MobileCount predicted count
    - LCDNet density map sum + MC density map sum
    - LCDNet and MobileCount density maps for fusion
    """
    print("Building cache (one forward pass per model per image)...")
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

    cache = []
    with torch.no_grad():
        for img_path, gt in tqdm(samples, desc="Caching"):
            img = Image.open(img_path).convert('RGB')
            r_in = router_tf(img).unsqueeze(0).to(device)
            d_in = density_tf(img).unsqueeze(0).to(device)
            probs = F.softmax(router(r_in), dim=1)
            p_lcd = probs[0, 0].item()
            p_csr = probs[0, 1].item()
            lcd_d = lcd(d_in)
            mc_d = mc(d_in)
            # Up-sample MC to LCDNet's resolution + normalise sum (same as in
            # hybrid_inference.py)
            mc_sum = mc_d.sum().item()
            mc_up = F.interpolate(mc_d, size=lcd_d.shape[-2:],
                                  mode='bilinear', align_corners=False)
            mc_up_sum = mc_up.sum().item()
            if mc_up_sum > 0:
                mc_up = mc_up * (mc_sum / mc_up_sum)
            cache.append({
                'gt': gt,
                'p_lcd': p_lcd,
                'p_csr': p_csr,
                'lcd_count': lcd_d.sum().item(),
                'mc_count': mc_sum,
                # save fused-count by fusion alpha
                'lcd_d': lcd_d.cpu(),
                'mc_up': mc_up.cpu(),
                'path': img_path,
            })
    del router, lcd, mc
    if device == 'cuda':
        torch.cuda.empty_cache()
    return cache


# -----------------------------------------------------------------------------
# Sweep helpers
# -----------------------------------------------------------------------------

def metrics_from_errors(errs):
    errs = np.asarray(errs)
    return {
        'mae': float(errs.mean()),
        'rmse': float(np.sqrt((errs**2).mean())),
        'n': len(errs),
    }


def stratified(records):
    bins = defaultdict(list)
    for r in records:
        bins[density_bin(r['gt'])].append(abs(r['pred'] - r['gt']))
    return {k: metrics_from_errors(v) for k, v in bins.items()}


def sweep_router_prob(cache, thresholds):
    """For each threshold p*, route to MobileCount if p_csr >= p*. Always
    LCDNet otherwise (i.e., shift the decision boundary on the existing router)."""
    results = []
    for p_star in thresholds:
        records = []
        used_mc = 0
        for c in cache:
            if c['p_csr'] >= p_star:
                pred = c['mc_count']
                used_mc += 1
            else:
                pred = c['lcd_count']
            records.append({'gt': c['gt'], 'pred': pred})
        errs = [abs(r['pred'] - r['gt']) for r in records]
        m = metrics_from_errors(errs)
        m['threshold'] = p_star
        m['mc_share'] = used_mc / len(cache)
        m['stratified'] = stratified(records)
        results.append(m)
    return results


def sweep_soft_margin(cache, margins):
    """For each soft margin m, bypass mode if max(p)>=1-m, else fuse with
    weights (p_lcd, p_csr)."""
    results = []
    for m_val in margins:
        records = []
        n_fuse = n_lcd = n_mc = 0
        for c in cache:
            p_lcd, p_csr = c['p_lcd'], c['p_csr']
            if p_lcd >= 1.0 - m_val:
                pred = c['lcd_count']; n_lcd += 1
            elif p_csr >= 1.0 - m_val:
                pred = c['mc_count']; n_mc += 1
            else:
                fused = (p_lcd * c['lcd_d'] + p_csr * c['mc_up'])
                pred = fused.sum().item()
                n_fuse += 1
            records.append({'gt': c['gt'], 'pred': pred})
        errs = [abs(r['pred'] - r['gt']) for r in records]
        out = metrics_from_errors(errs)
        out['soft_margin'] = m_val
        out['n_fuse'] = n_fuse
        out['n_lcd_hard'] = n_lcd
        out['n_mc_hard'] = n_mc
        out['stratified'] = stratified(records)
        results.append(out)
    return results


def sweep_fusion_alpha(cache, alphas):
    """For ambiguous images (default soft 0.05 margin window), try fixed alpha
    in fused = alpha*LCD + (1-alpha)*MC. Edge cases (high-confidence) routed
    hard."""
    results = []
    margin = 0.05
    for alpha in alphas:
        records = []
        for c in cache:
            p_lcd, p_csr = c['p_lcd'], c['p_csr']
            if p_lcd >= 1.0 - margin:
                pred = c['lcd_count']
            elif p_csr >= 1.0 - margin:
                pred = c['mc_count']
            else:
                fused = (alpha * c['lcd_d'] + (1 - alpha) * c['mc_up'])
                pred = fused.sum().item()
            records.append({'gt': c['gt'], 'pred': pred})
        errs = [abs(r['pred'] - r['gt']) for r in records]
        out = metrics_from_errors(errs)
        out['alpha'] = alpha
        out['stratified'] = stratified(records)
        results.append(out)
    return results


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def print_sweep(results, key, label):
    print(f"\n{label} sweep")
    print(f"{'value':>10} {'MAE':>10} {'RMSE':>10}  sparse | medium | dense")
    for r in results:
        s = r['stratified']
        sp = s.get('sparse', {}).get('mae', 0.0)
        md = s.get('medium', {}).get('mae', 0.0)
        dn = s.get('dense',  {}).get('mae', 0.0)
        print(f"{r[key]:>10.3f} {r['mae']:>10.2f} {r['rmse']:>10.2f}   "
              f"{sp:6.1f} | {md:6.1f} | {dn:6.1f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='all',
                        choices=['router_prob', 'soft_margin', 'fusion_alpha', 'all'])
    parser.add_argument('--out', default=os.path.join(
        config_routing.ROUTER_LOG_DIR, "phase3_threshold_sweep.json"))
    parser.add_argument('--out_txt', default=os.path.join(
        config_routing.ROUTER_LOG_DIR, "phase3_threshold_sweep.txt"))
    parser.add_argument('--dense_ckpt', default=None,
                        help='Override dense (MobileCount) checkpoint')
    parser.add_argument('--tag', default=None,
                        help='Tag appended to output filenames')
    args = parser.parse_args()

    if args.tag:
        args.out = args.out.replace('.json', f'_{args.tag}.json')
        args.out_txt = args.out_txt.replace('.txt', f'_{args.tag}.txt')
    if args.dense_ckpt:
        config_routing.DENSE_CHECKPOINT = args.dense_ckpt
        print(f"OVERRIDE dense ckpt: {args.dense_ckpt}")

    device = config_routing.DEVICE
    print(f"Device: {device}")

    samples = load_samples()
    print(f"Loaded {len(samples)} val samples")

    # Build cache
    cache = build_cache(samples, device)

    payload = {}

    if args.mode in ('router_prob', 'all'):
        # Sweep p* in 0.1 .. 0.9 (step 0.05)
        thresholds = [round(x, 2) for x in np.arange(0.1, 0.91, 0.05)]
        rp = sweep_router_prob(cache, thresholds)
        payload['router_prob'] = rp
        print_sweep(rp, 'threshold', "ROUTER PROB THRESHOLD (route to MC if P(dense) >= p*)")

    if args.mode in ('soft_margin', 'all'):
        margins = [0.0, 0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
        sm = sweep_soft_margin(cache, margins)
        payload['soft_margin'] = sm
        print_sweep(sm, 'soft_margin', "SOFT FUSION MARGIN")

    if args.mode in ('fusion_alpha', 'all'):
        alphas = [round(x, 2) for x in np.arange(0.0, 1.01, 0.1)]
        fa = sweep_fusion_alpha(cache, alphas)
        payload['fusion_alpha'] = fa
        print_sweep(fa, 'alpha', "FIXED FUSION ALPHA (alpha*LCD + (1-alpha)*MC)")

    # Write JSON
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    # tensors are not in payload, so JSON is fine
    with open(args.out, 'w') as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nSaved JSON: {args.out}")

    # Write text summary too
    lines = []
    bar = "=" * 78
    sub = "-" * 78
    lines += [bar, "PHASE 3 THRESHOLD SWEEPS (no retraining)", bar]

    def block(name, results, key):
        lines.append("")
        lines.append(name)
        lines.append(sub)
        lines.append(f"{'value':>10} {'MAE':>10} {'RMSE':>10}  sparse | medium | dense  | extra")
        lines.append(sub)
        for r in results:
            s = r['stratified']
            sp = s.get('sparse', {}).get('mae', 0.0)
            md = s.get('medium', {}).get('mae', 0.0)
            dn = s.get('dense',  {}).get('mae', 0.0)
            extra = ""
            if 'mc_share' in r:
                extra = f" | MC share {r['mc_share']*100:5.1f}%"
            if 'n_fuse' in r:
                extra = f" | fused {r['n_fuse']}"
            lines.append(f"{r[key]:>10.3f} {r['mae']:>10.2f} {r['rmse']:>10.2f}   "
                         f"{sp:6.1f} | {md:6.1f} | {dn:6.1f}{extra}")

    if 'router_prob' in payload:
        block("ROUTER PROB THRESHOLD (route to MC if P(dense) >= p*)",
              payload['router_prob'], 'threshold')
    if 'soft_margin' in payload:
        block("SOFT FUSION MARGIN", payload['soft_margin'], 'soft_margin')
    if 'fusion_alpha' in payload:
        block("FIXED FUSION ALPHA  (fused = alpha*LCD + (1-alpha)*MC)",
              payload['fusion_alpha'], 'alpha')

    # Best-of summary
    lines.append("")
    lines.append(bar)
    lines.append("BEST-OF SUMMARY")
    lines.append(sub)
    for k in ['router_prob', 'soft_margin', 'fusion_alpha']:
        if k in payload:
            best = min(payload[k], key=lambda r: r['mae'])
            key = {'router_prob': 'threshold',
                   'soft_margin': 'soft_margin',
                   'fusion_alpha': 'alpha'}[k]
            lines.append(f"{k:14s}  best {key}={best[key]:.3f}  -> MAE {best['mae']:.2f}, RMSE {best['rmse']:.2f}")
    lines.append(bar)

    with open(args.out_txt, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"Saved text: {args.out_txt}")


if __name__ == "__main__":
    main()
