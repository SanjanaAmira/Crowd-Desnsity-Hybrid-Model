"""
Evaluation Script for the Hybrid Crowd Density Estimation System.

This script evaluates and compares three approaches:
1. LCDNet-only: All images through LCDNet
2. CSRNet-only: All images through CSRNet
3. Hybrid: Router decides per image

Metrics Reported:
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- Average inference time per image
- Routing statistics (% routed to each model)

Usage:
    python routing/evaluate_hybrid.py                     # Full evaluation
    python routing/evaluate_hybrid.py --split val        # Specific split
    python routing/evaluate_hybrid.py --max_samples 100  # Limit samples

Author: Thesis Implementation - Phase 2 Part 3
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from routing import config_routing
from routing.hybrid_inference import HybridDensityEstimator

# Import models directly for single-model evaluation
from models.lcdnet import LCDNet
from models.csrnet import CSRNet
from torchvision import transforms
import config


class SingleModelEvaluator:
    """Evaluator for single model (LCDNet or CSRNet) testing."""
    
    def __init__(self, model_name: str, checkpoint_path: str, device: str):
        self.model_name = model_name
        self.device = device
        
        # Load model
        if model_name.lower() == "lcdnet":
            self.model = LCDNet()
        elif model_name.lower() == "csrnet":
            self.model = CSRNet(pretrained=False)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model = self.model.to(device)
        self.model.eval()
        
        # Freeze
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Transform
        self.transform = transforms.Compose([
            transforms.Resize(config_routing.DENSITY_INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def predict(self, image: Image.Image) -> Tuple[float, float]:
        """Predict count and return (count, inference_time)."""
        start = time.time()
        
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            density = self.model(input_tensor)
            count = density.sum().item()
        
        elapsed = time.time() - start
        return count, elapsed


def load_evaluation_samples(
    split: str = "val",
    max_samples: Optional[int] = None
) -> List[Tuple[str, float]]:
    """
    Load evaluation samples from NWPU-Crowd.
    
    Returns:
        List of (image_path, gt_count) tuples.
    """
    # Get split file
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
            if len(parts) < 1:
                continue
            
            img_id = parts[0]
            
            # Find image
            img_path = None
            for img_dir in config_routing.NWPU_IMAGE_DIRS:
                for ext in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']:
                    path = os.path.join(img_dir, f"{img_id}{ext}")
                    if os.path.exists(path):
                        img_path = path
                        break
                if img_path:
                    break
            
            if img_path is None:
                continue
            
            # Get GT count from JSON
            json_path = os.path.join(config_routing.NWPU_JSONS_DIR, f"{img_id}.json")
            try:
                with open(json_path, 'r') as jf:
                    data = json.load(jf)
                gt_count = float(data.get('human_num', 0))
            except (FileNotFoundError, json.JSONDecodeError):
                gt_count = 0.0
            
            samples.append((img_path, gt_count))
            
            if max_samples and len(samples) >= max_samples:
                break
    
    return samples


def evaluate_single_model(
    samples: List[Tuple[str, float]],
    model_name: str,
    checkpoint_path: str,
    device: str
) -> Dict:
    """Evaluate a single model on all samples."""
    evaluator = SingleModelEvaluator(model_name, checkpoint_path, device)
    
    errors = []
    times = []
    predictions = []
    
    print(f"\nEvaluating {model_name}...")
    for img_path, gt_count in tqdm(samples, desc=model_name):
        image = Image.open(img_path).convert('RGB')
        pred_count, elapsed = evaluator.predict(image)
        
        error = abs(pred_count - gt_count)
        errors.append(error)
        times.append(elapsed)
        predictions.append({
            'path': img_path,
            'gt': gt_count,
            'pred': pred_count,
            'error': error
        })
    
    # Compute metrics
    errors = np.array(errors)
    mae = np.mean(errors)
    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)
    avg_time = np.mean(times) * 1000  # ms
    
    return {
        'model': model_name,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'avg_time_ms': avg_time,
        'predictions': predictions
    }


def evaluate_hybrid(
    samples: List[Tuple[str, float]],
    estimator: HybridDensityEstimator
) -> Dict:
    """Evaluate hybrid system on all samples."""
    errors = []
    times = []
    predictions = []
    routing_stats = defaultdict(int)
    
    print("\nEvaluating Hybrid...")
    for img_path, gt_count in tqdm(samples, desc="Hybrid"):
        image = Image.open(img_path).convert('RGB')
        result = estimator.predict(image)
        
        pred_count = result['count']
        elapsed = result['total_time']
        model_used = result['model']
        
        error = abs(pred_count - gt_count)
        errors.append(error)
        times.append(elapsed)
        routing_stats[model_used] += 1
        
        predictions.append({
            'path': img_path,
            'gt': gt_count,
            'pred': pred_count,
            'error': error,
            'model': model_used,
            'routing_prob': result['routing_prob']
        })
    
    # Compute metrics
    errors = np.array(errors)
    mae = np.mean(errors)
    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)
    avg_time = np.mean(times) * 1000  # ms
    
    total = len(samples)
    routing_pct = {k: 100 * v / total for k, v in routing_stats.items()}
    
    return {
        'model': 'Hybrid',
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'avg_time_ms': avg_time,
        'routing_stats': dict(routing_stats),
        'routing_pct': routing_pct,
        'predictions': predictions
    }


def print_results(results: List[Dict], output_file: str = None):
    """Print and save evaluation results."""
    lines = []
    
    header = "=" * 70
    lines.append(header)
    lines.append("HYBRID CROWD DENSITY ESTIMATION - EVALUATION RESULTS")
    lines.append(header)
    lines.append(f"Evaluation date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Device: {config_routing.DEVICE}")
    lines.append(f"Routing threshold: {config_routing.ROUTING_THRESHOLD}")
    lines.append("")
    
    # Summary table
    lines.append("-" * 70)
    lines.append(f"{'Model':<15} {'MAE':>10} {'MSE':>12} {'RMSE':>10} {'Time (ms)':>12}")
    lines.append("-" * 70)
    
    for r in results:
        lines.append(f"{r['model']:<15} {r['mae']:>10.2f} {r['mse']:>12.2f} "
                    f"{r['rmse']:>10.2f} {r['avg_time_ms']:>12.1f}")
    
    lines.append("-" * 70)
    
    # Hybrid routing statistics
    hybrid_result = next((r for r in results if r['model'] == 'Hybrid'), None)
    if hybrid_result and 'routing_pct' in hybrid_result:
        lines.append("\nRouting Statistics (Hybrid):")
        for model, pct in hybrid_result['routing_pct'].items():
            count = hybrid_result['routing_stats'][model]
            lines.append(f"  {model}: {count} images ({pct:.1f}%)")
    
    lines.append("")
    lines.append(header)
    
    # Print to console
    output = "\n".join(lines)
    print(output)
    
    # Save to file
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            f.write(output)
            
            # Add per-sample details
            f.write("\n\n" + "=" * 70 + "\n")
            f.write("PER-SAMPLE RESULTS (HYBRID)\n")
            f.write("=" * 70 + "\n\n")
            
            if hybrid_result:
                f.write(f"{'Image':<20} {'GT':>8} {'Pred':>8} {'Error':>8} {'Model':>10}\n")
                f.write("-" * 60 + "\n")
                
                for pred in hybrid_result['predictions'][:100]:  # First 100
                    img_name = os.path.basename(pred['path'])[:18]
                    f.write(f"{img_name:<20} {pred['gt']:>8.1f} {pred['pred']:>8.1f} "
                           f"{pred['error']:>8.2f} {pred['model']:>10}\n")
        
        print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate hybrid crowd density estimation")
    parser.add_argument(
        '--split', type=str, default='val', choices=['train', 'val', 'test'],
        help='Dataset split to evaluate on (default: val)'
    )
    parser.add_argument(
        '--max_samples', type=int, default=None,
        help='Maximum number of samples to evaluate (for quick testing)'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output file for results (default: logs/hybrid_evaluation_results.txt)'
    )
    args = parser.parse_args()
    
    if args.output is None:
        args.output = config_routing.HYBRID_EVAL_RESULTS
    
    device = config_routing.DEVICE
    
    print("=" * 70)
    print("HYBRID CROWD DENSITY ESTIMATION - EVALUATION")
    print("=" * 70)
    print(f"Split: {args.split}")
    print(f"Device: {device}")
    print(f"Max samples: {args.max_samples or 'All'}")
    
    # Load samples
    print("\nLoading evaluation samples...")
    samples = load_evaluation_samples(args.split, args.max_samples)
    print(f"Loaded {len(samples)} samples")
    
    results = []
    
    # Evaluate LCDNet-only
    try:
        lcdnet_result = evaluate_single_model(
            samples, "LCDNet", config_routing.LCDNET_CHECKPOINT, device
        )
        results.append(lcdnet_result)
    except Exception as e:
        print(f"Warning: Could not evaluate LCDNet: {e}")
    
    # Evaluate CSRNet-only
    try:
        csrnet_result = evaluate_single_model(
            samples, "CSRNet", config_routing.CSRNET_CHECKPOINT, device
        )
        results.append(csrnet_result)
    except Exception as e:
        print(f"Warning: Could not evaluate CSRNet: {e}")
    
    # Evaluate Hybrid
    try:
        estimator = HybridDensityEstimator(device=device)
        hybrid_result = evaluate_hybrid(samples, estimator)
        results.append(hybrid_result)
    except Exception as e:
        print(f"Warning: Could not evaluate Hybrid: {e}")
        import traceback
        traceback.print_exc()
    
    # Print and save results
    if results:
        print_results(results, args.output)
    else:
        print("No results to display.")


if __name__ == "__main__":
    main()
