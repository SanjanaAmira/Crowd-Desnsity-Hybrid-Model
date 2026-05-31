"""
Full Phase-3 Defense Evaluation Script.

Generates the full evaluation suite for the LCDNet + MobileCount + Router
hybrid system. All comparisons are made WITHOUT CSRNet in the deployed pipeline
(CSRNet is only retained as a literature reference if needed).

This script produces:
  1. Full NWPU val evaluation (all 500 images, not just 50)
  2. Per-model results: LCDNet only, MobileCount only, Hybrid-Hard, Hybrid-Soft
  3. Oracle Router baseline (upper bound assuming a perfect router)
  4. Stratified MAE by density bin (sparse / medium / dense)
  5. Edge-deployment metrics: parameters, FLOPs/MACs, GPU latency, CPU latency,
     peak GPU memory
  6. Bootstrap 95% confidence intervals on MAE

Outputs are written to logs/phase3_full_evaluation.txt and a JSON dump for
plotting later.

Usage:
    python routing/full_evaluation.py
    python routing/full_evaluation.py --max_samples 100   # quick smoke test

Author: Thesis - Phase 3 Final Defense Prep
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from routing import config_routing
from routing.hybrid_inference import HybridDensityEstimator
from models.lcdnet import LCDNet
from models.mobilecount import MobileCount
from models.csrnet import CSRNet
import config


# ---------------------------------------------------------------------------
# Sample loading
# ---------------------------------------------------------------------------

def load_samples(split: str = "val",
                 max_samples: Optional[int] = None) -> List[Tuple[str, float]]:
    """Load (image_path, gt_count) tuples from an NWPU split."""
    if split == "train":
        split_file = config_routing.NWPU_TRAIN_TXT
    elif split == "val":
        split_file = config_routing.NWPU_VAL_TXT
    elif split == "test":
        split_file = config_routing.NWPU_TEST_TXT
    else:
        raise ValueError(f"Unknown split: {split}")

    samples = []
    with open(split_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            img_id = parts[0]

            img_path = None
            for img_dir in config_routing.NWPU_IMAGE_DIRS:
                for ext in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']:
                    candidate = os.path.join(img_dir, f"{img_id}{ext}")
                    if os.path.exists(candidate):
                        img_path = candidate
                        break
                if img_path:
                    break
            if img_path is None:
                continue

            json_path = os.path.join(config_routing.NWPU_JSONS_DIR, f"{img_id}.json")
            try:
                with open(json_path, 'r') as jf:
                    gt_count = float(json.load(jf).get('human_num', 0))
            except (FileNotFoundError, json.JSONDecodeError):
                gt_count = 0.0

            samples.append((img_path, gt_count))
            if max_samples and len(samples) >= max_samples:
                break
    return samples


# ---------------------------------------------------------------------------
# Single-model evaluator (LCDNet / MobileCount only)
# ---------------------------------------------------------------------------

class SingleModelEvaluator:
    """Light wrapper for evaluating a single density model."""

    def __init__(self, model_name: str, checkpoint_path: str, device: str):
        self.model_name = model_name
        self.device = device

        name = model_name.lower()
        if name == "lcdnet":
            self.model = LCDNet()
        elif name == "csrnet":
            self.model = CSRNet(pretrained=False)
        elif name == "mobilecount":
            self.model = MobileCount(pretrained=False)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        ckpt = torch.load(checkpoint_path, map_location=device)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            self.model.load_state_dict(ckpt['model_state_dict'])
        else:
            self.model.load_state_dict(ckpt)

        self.model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.transform = transforms.Compose([
            transforms.Resize(config_routing.DENSITY_INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def predict(self, image: Image.Image) -> Tuple[float, float]:
        start = time.time()
        x = self.transform(image).unsqueeze(0).to(self.device)
        density = self.model(x)
        count = density.sum().item()
        return count, time.time() - start


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(errors: np.ndarray) -> Dict[str, float]:
    return {
        'mae': float(np.mean(errors)),
        'mse': float(np.mean(errors ** 2)),
        'rmse': float(np.sqrt(np.mean(errors ** 2))),
        'median_ae': float(np.median(errors)),
    }


def bootstrap_mae_ci(errors: np.ndarray,
                     n_resamples: int = 2000,
                     ci: float = 0.95,
                     seed: int = 0) -> Tuple[float, float]:
    """Return (lower, upper) bound on MAE at given confidence."""
    rng = np.random.default_rng(seed)
    n = len(errors)
    if n == 0:
        return 0.0, 0.0
    means = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, n)
        means[i] = errors[idx].mean()
    lo = np.quantile(means, (1 - ci) / 2)
    hi = np.quantile(means, 1 - (1 - ci) / 2)
    return float(lo), float(hi)


def density_bin(gt_count: float) -> str:
    """Categorise an image by ground-truth density."""
    if gt_count <= 100:
        return "sparse"
    elif gt_count <= 500:
        return "medium"
    else:
        return "dense"


def stratified_mae(predictions: List[Dict]) -> Dict[str, Dict[str, float]]:
    """MAE per density bin."""
    bins = defaultdict(list)
    for p in predictions:
        bins[density_bin(p['gt'])].append(p['error'])
    out = {}
    for k, errs in bins.items():
        if errs:
            arr = np.asarray(errs)
            out[k] = {
                'count': len(arr),
                'mae': float(arr.mean()),
                'rmse': float(np.sqrt((arr ** 2).mean())),
            }
    return out


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------

def evaluate_single(samples, model_name, checkpoint, device, label=None):
    label = label or model_name
    ev = SingleModelEvaluator(model_name, checkpoint, device)
    preds, errs, times = [], [], []
    for img_path, gt in tqdm(samples, desc=label):
        img = Image.open(img_path).convert('RGB')
        pred, t = ev.predict(img)
        err = abs(pred - gt)
        preds.append({'path': img_path, 'gt': gt, 'pred': pred, 'error': err,
                      'model': model_name})
        errs.append(err)
        times.append(t)
    errs = np.asarray(errs)
    metrics = compute_metrics(errs)
    metrics.update({
        'avg_time_ms': float(np.mean(times) * 1000),
        'mae_ci95': bootstrap_mae_ci(errs),
        'stratified': stratified_mae(preds),
        'predictions': preds,
        'model_label': label,
    })
    del ev
    if device == 'cuda':
        torch.cuda.empty_cache()
    return metrics


def evaluate_hybrid(samples, estimator, mode, soft_margin=0.05):
    label = "Hybrid-Hard" if mode == "hard" else "Hybrid-Soft"
    preds, errs, times = [], [], []
    routing_stats = defaultdict(int)
    for img_path, gt in tqdm(samples, desc=label):
        img = Image.open(img_path).convert('RGB')
        result = estimator.predict(img, mode=mode, soft_margin=soft_margin)
        pred = result['count']
        err = abs(pred - gt)
        preds.append({'path': img_path, 'gt': gt, 'pred': pred, 'error': err,
                      'model': result['model'],
                      'routing_prob': result['routing_prob']})
        errs.append(err)
        times.append(result['total_time'])
        routing_stats[result['model']] += 1
    errs = np.asarray(errs)
    metrics = compute_metrics(errs)
    total = len(samples)
    metrics.update({
        'avg_time_ms': float(np.mean(times) * 1000),
        'mae_ci95': bootstrap_mae_ci(errs),
        'stratified': stratified_mae(preds),
        'routing_stats': dict(routing_stats),
        'routing_pct': {k: 100 * v / total for k, v in routing_stats.items()},
        'predictions': preds,
        'model_label': label,
    })
    return metrics


def evaluate_oracle(lcd_preds, mc_preds):
    """Oracle router: pick the better of LCDNet vs MobileCount per image."""
    assert len(lcd_preds) == len(mc_preds)
    preds, errs = [], []
    pick_stats = defaultdict(int)
    for a, b in zip(lcd_preds, mc_preds):
        assert a['path'] == b['path'], "prediction lists not aligned"
        if a['error'] <= b['error']:
            chosen = a
            pick_stats['LCDNet'] += 1
        else:
            chosen = b
            pick_stats['MobileCount'] += 1
        preds.append({**chosen, 'oracle_pick': chosen['model']})
        errs.append(chosen['error'])
    errs = np.asarray(errs)
    metrics = compute_metrics(errs)
    total = len(preds)
    metrics.update({
        'avg_time_ms': None,  # not meaningful
        'mae_ci95': bootstrap_mae_ci(errs),
        'stratified': stratified_mae(preds),
        'pick_stats': dict(pick_stats),
        'pick_pct': {k: 100 * v / total for k, v in pick_stats.items()},
        'predictions': preds,
        'model_label': 'Oracle Router',
    })
    return metrics


# ---------------------------------------------------------------------------
# Edge-deployment metrics (params, FLOPs, latency CPU/GPU, memory)
# ---------------------------------------------------------------------------

def _try_count_macs(model, input_shape):
    """Try fvcore first, then thop, then return None."""
    try:
        from fvcore.nn import FlopCountAnalysis
        x = torch.randn(*input_shape)
        flops = FlopCountAnalysis(model, x)
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        return float(flops.total())
    except Exception:
        pass
    try:
        from thop import profile
        x = torch.randn(*input_shape)
        macs, _ = profile(model, inputs=(x,), verbose=False)
        return float(macs)
    except Exception:
        return None


def _measure_latency(model, input_shape, device, runs=30, warmup=5):
    model.eval().to(device)
    x = torch.randn(*input_shape).to(device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(runs):
            _ = model(x)
        if device == 'cuda':
            torch.cuda.synchronize()
    return (time.time() - start) / runs * 1000.0


def measure_efficiency(device: str) -> Dict[str, Dict]:
    """Measure params / MACs / GPU+CPU latency / peak memory per model."""
    out = {}

    specs = [
        ('LCDNet',      LCDNet(),                   (1, 3, 384, 384)),
        ('MobileCount', MobileCount(pretrained=False), (1, 3, 384, 384)),
        ('Router',      _build_router(),            (1, 3, 224, 224)),
    ]

    for name, model, shape in specs:
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        macs = _try_count_macs(model, shape)
        cpu_lat = _measure_latency(model, shape, 'cpu', runs=10, warmup=3)
        gpu_lat = None
        peak_mem = None
        if device == 'cuda':
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                gpu_lat = _measure_latency(model, shape, 'cuda', runs=30, warmup=5)
                peak_mem = float(torch.cuda.max_memory_allocated()) / (1024 ** 2)
            except Exception as e:
                print(f"  GPU measurement for {name} failed: {e}")
        out[name] = {
            'params': params,
            'params_M': params / 1e6,
            'macs': macs,
            'gflops': (macs / 1e9 if macs is not None else None),
            'cpu_latency_ms': cpu_lat,
            'gpu_latency_ms': gpu_lat,
            'gpu_peak_mem_MB': peak_mem,
            'input_shape': list(shape),
        }
        del model
        if device == 'cuda':
            torch.cuda.empty_cache()
    return out


def _build_router():
    """Build a fresh router with random weights for FLOP/latency timing only."""
    from routing.router import RoutingClassifier
    return RoutingClassifier(num_classes=2, pretrained=False)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def fmt_ci(ci):
    lo, hi = ci
    return f"[{lo:.2f}, {hi:.2f}]"


def write_report(results: List[Dict],
                 efficiency: Dict[str, Dict],
                 split: str,
                 n_samples: int,
                 output_txt: str,
                 output_json: str):
    os.makedirs(os.path.dirname(output_txt) or '.', exist_ok=True)
    lines = []
    bar = "=" * 78
    sub = "-" * 78

    lines += [bar,
              "PHASE 3 FINAL DEFENSE EVALUATION  -  LCDNet + MobileCount + Router",
              bar,
              f"Date:               {datetime.now():%Y-%m-%d %H:%M:%S}",
              f"Split:              {split}",
              f"Samples evaluated:  {n_samples}",
              f"Device:             {config_routing.DEVICE}",
              f"Routing threshold:  {config_routing.ROUTING_THRESHOLD}",
              ""]

    # Headline table
    lines += ["MAIN RESULTS",
              sub,
              f"{'Model':<22}{'MAE':>10}{'RMSE':>10}{'95% CI MAE':>22}{'Time (ms)':>14}",
              sub]
    for r in results:
        ms = f"{r['avg_time_ms']:.1f}" if r['avg_time_ms'] is not None else "  -  "
        lines.append(f"{r['model_label']:<22}{r['mae']:>10.2f}{r['rmse']:>10.2f}"
                     f"{fmt_ci(r['mae_ci95']):>22}{ms:>14}")
    lines += [sub, ""]

    # Stratified table
    lines += ["STRATIFIED MAE BY DENSITY BIN",
              sub,
              f"{'Model':<22}{'Sparse (<=100)':>18}"
              f"{'Medium (100-500)':>20}{'Dense (>500)':>18}",
              sub]
    for r in results:
        cells = []
        for b in ['sparse', 'medium', 'dense']:
            s = r['stratified'].get(b)
            cells.append(f"{s['mae']:.1f} (n={s['count']})" if s else "   -   ")
        lines.append(f"{r['model_label']:<22}{cells[0]:>18}{cells[1]:>20}{cells[2]:>18}")
    lines += [sub, ""]

    # Hybrid routing stats
    for r in results:
        if 'routing_pct' in r:
            lines.append(f"Routing distribution ({r['model_label']}):")
            for m, pct in r['routing_pct'].items():
                n = r['routing_stats'][m]
                lines.append(f"   {m:<28} {n:>4} images  ({pct:5.1f}%)")
            lines.append("")
        if 'pick_pct' in r:
            lines.append(f"Oracle picks ({r['model_label']}):")
            for m, pct in r['pick_pct'].items():
                n = r['pick_stats'][m]
                lines.append(f"   {m:<28} {n:>4} images  ({pct:5.1f}%)")
            lines.append("")

    # Efficiency
    lines += [bar,
              "EDGE-DEPLOYMENT METRICS (per component)",
              sub,
              f"{'Component':<14}{'Params (M)':>12}{'GFLOPs':>10}"
              f"{'GPU (ms)':>12}{'CPU (ms)':>12}{'GPU Mem (MB)':>16}",
              sub]
    total_params = 0
    for name, e in efficiency.items():
        gflops = f"{e['gflops']:.2f}" if e['gflops'] is not None else "  -  "
        gpu = f"{e['gpu_latency_ms']:.1f}" if e['gpu_latency_ms'] is not None else "  -  "
        cpu = f"{e['cpu_latency_ms']:.1f}"
        mem = f"{e['gpu_peak_mem_MB']:.1f}" if e['gpu_peak_mem_MB'] is not None else "  -  "
        lines.append(f"{name:<14}{e['params_M']:>12.3f}{gflops:>10}{gpu:>12}"
                     f"{cpu:>12}{mem:>16}")
        total_params += e['params']
    lines += [sub,
              f"{'TOTAL':<14}{total_params/1e6:>12.3f}",
              ""]

    lines += [bar, "INTERPRETATION NOTES", bar,
              "* LCDNet alone is reported on the FULL val (sparse+medium+dense).",
              "  Its strong story is in the sparse stratum; the dense bin is",
              "  expected to be poor because LCDNet was domain-adapted on sparse.",
              "* Oracle Router is the upper bound for ANY routing strategy that",
              "  picks one of {LCDNet, MobileCount}. Gap between Hybrid and Oracle",
              "  is the cost of router errors.",
              "* MAE 95% CI uses a 2000-iteration bootstrap.",
              "* CSRNet is intentionally excluded from the deployed pipeline.",
              bar]

    text = "\n".join(lines)
    print(text)
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"\nText report saved to: {output_txt}")

    # Strip predictions for JSON (too big), keep summaries
    summary = {
        'meta': {
            'date': datetime.now().isoformat(),
            'split': split,
            'samples': n_samples,
            'device': config_routing.DEVICE,
            'routing_threshold': config_routing.ROUTING_THRESHOLD,
        },
        'results': [{k: v for k, v in r.items() if k != 'predictions'}
                    for r in results],
        'efficiency': efficiency,
    }
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"JSON summary saved to: {output_json}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='val',
                        choices=['train', 'val', 'test'])
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Limit samples (for smoke test)')
    parser.add_argument('--soft_margin', type=float, default=0.05)
    parser.add_argument('--skip_efficiency', action='store_true',
                        help='Skip the params/FLOPs/latency benchmarking step')
    parser.add_argument('--efficiency_only', action='store_true',
                        help='Only run the efficiency benchmark, skip evaluation')
    parser.add_argument('--output_txt', type=str, default=None)
    parser.add_argument('--output_json', type=str, default=None)
    args = parser.parse_args()

    output_txt = args.output_txt or os.path.join(
        config_routing.ROUTER_LOG_DIR, "phase3_full_evaluation.txt")
    output_json = args.output_json or os.path.join(
        config_routing.ROUTER_LOG_DIR, "phase3_full_evaluation.json")

    device = config_routing.DEVICE
    print("=" * 78)
    print("PHASE 3 FULL EVALUATION  (LCDNet + MobileCount + Router, no CSRNet)")
    print("=" * 78)
    print(f"Split: {args.split} | max_samples: {args.max_samples or 'All'}")

    if args.efficiency_only:
        print("\nMeasuring edge-deployment metrics only...")
        efficiency = measure_efficiency(device)
        # Pretty-print just the efficiency block
        bar = "=" * 78
        sub = "-" * 78
        lines = [bar, "EDGE-DEPLOYMENT METRICS (per component)", sub,
                 f"{'Component':<14}{'Params (M)':>12}{'GFLOPs':>10}"
                 f"{'GPU (ms)':>12}{'CPU (ms)':>12}{'GPU Mem (MB)':>16}",
                 sub]
        total_params = 0
        for name, e in efficiency.items():
            gflops = f"{e['gflops']:.2f}" if e['gflops'] is not None else "  -  "
            gpu = f"{e['gpu_latency_ms']:.1f}" if e['gpu_latency_ms'] is not None else "  -  "
            cpu = f"{e['cpu_latency_ms']:.1f}"
            mem = f"{e['gpu_peak_mem_MB']:.1f}" if e['gpu_peak_mem_MB'] is not None else "  -  "
            lines.append(f"{name:<14}{e['params_M']:>12.3f}{gflops:>10}{gpu:>12}"
                         f"{cpu:>12}{mem:>16}")
            total_params += e['params']
        lines += [sub, f"{'TOTAL':<14}{total_params/1e6:>12.3f}", bar]
        eff_only_txt = output_txt.replace('.txt', '_efficiency.txt')
        eff_only_json = output_json.replace('.json', '_efficiency.json')
        text = "\n".join(lines)
        print(text)
        os.makedirs(os.path.dirname(eff_only_txt) or '.', exist_ok=True)
        with open(eff_only_txt, 'w', encoding='utf-8') as f:
            f.write(text)
        with open(eff_only_json, 'w', encoding='utf-8') as f:
            json.dump(efficiency, f, indent=2, default=str)
        print(f"\nSaved: {eff_only_txt}")
        return

    samples = load_samples(args.split, args.max_samples)
    print(f"Loaded {len(samples)} samples from NWPU {args.split}.\n")

    results = []

    # 1. LCDNet only
    lcd = evaluate_single(samples, "LCDNet",
                          config_routing.LCDNET_CHECKPOINT, device,
                          label="LCDNet (alone)")
    results.append(lcd)

    # 2. MobileCount only
    mc = evaluate_single(samples, "MobileCount",
                         config_routing.DENSE_CHECKPOINT, device,
                         label="MobileCount (alone)")
    results.append(mc)

    # 3. Hybrid-Hard
    estimator = HybridDensityEstimator(device=device)
    hh = evaluate_hybrid(samples, estimator, mode="hard")
    results.append(hh)

    # 4. Hybrid-Soft (fusion)
    hs = evaluate_hybrid(samples, estimator, mode="soft",
                         soft_margin=args.soft_margin)
    results.append(hs)

    # 5. Oracle Router upper bound
    oracle = evaluate_oracle(lcd['predictions'], mc['predictions'])
    results.append(oracle)

    # 6. Efficiency
    efficiency = {}
    if not args.skip_efficiency:
        print("\nMeasuring edge-deployment metrics (params, FLOPs, latency)...")
        efficiency = measure_efficiency(device)

    # Report
    write_report(results, efficiency, args.split, len(samples),
                 output_txt, output_json)


if __name__ == "__main__":
    main()
