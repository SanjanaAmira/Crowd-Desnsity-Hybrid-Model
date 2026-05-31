"""
Qualitative density-map figures for the Phase-3 thesis report and slides.

For a hand-picked set of NWPU-Crowd val images (one each from sparse, medium,
dense, plus a router-failure case), produce a 1x5 panel:

    [ input ] [ GT density ] [ LCDNet pred ] [ MobileCount pred ] [ Hybrid pred ]

Each panel is annotated with the count.

Author: Thesis - Phase 3 Final Defense Prep
"""

import os
import sys
import json
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from routing import config_routing
from routing.router import load_router
from models.lcdnet import LCDNet
from models.mobilecount import MobileCount


# Image IDs hand-picked to show diverse scenarios.
# (id, label) — choose so we cover sparse/medium/dense/router-fail.
DEFAULT_IMAGE_IDS = [
    ("3115", "sparse  (GT=62)"),         # router rightly chose LCDNet
    ("3110", "medium  (GT=240)"),        # routed to MobileCount
    ("3146", "dense   (GT=5951)"),       # extreme dense, hybrid still fails
    ("3234", "router-fail (GT=12924, p_lcd=.99)"),  # router catastrophic
]


def find_image(img_id):
    for img_dir in config_routing.NWPU_IMAGE_DIRS:
        for ext in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']:
            p = os.path.join(img_dir, f"{img_id}{ext}")
            if os.path.exists(p):
                return p
    return None


def load_gt(img_id):
    json_path = os.path.join(config_routing.NWPU_JSONS_DIR, f"{img_id}.json")
    try:
        with open(json_path, 'r') as f:
            return float(json.load(f).get('human_num', 0))
    except Exception:
        return 0.0


def load_gt_density(img_id):
    """Load NWPU density map if available."""
    p = os.path.join(config_routing.NWPU_DATA_DIR, "density_maps", f"{img_id}.npy")
    if os.path.exists(p):
        return np.load(p)
    return None


def to_density_input(img, device):
    tf = transforms.Compose([
        transforms.Resize(config_routing.DENSITY_INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return tf(img).unsqueeze(0).to(device)


def to_router_input(img, device):
    tf = transforms.Compose([
        transforms.Resize(config_routing.ROUTER_INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return tf(img).unsqueeze(0).to(device)


def gamma_norm(arr, gamma=0.5):
    """Make low-density regions visible by gamma-correcting the heatmap."""
    a = np.clip(arr, 0, None)
    a = a / (a.max() + 1e-12)
    return a ** gamma


def make_panel(records, out_path):
    n_rows = len(records)
    fig, axes = plt.subplots(n_rows, 5, figsize=(20, 4 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]
    col_titles = ["Input", "GT density", "LCDNet pred", "MobileCount pred",
                  "Hybrid pred"]
    for j, ct in enumerate(col_titles):
        axes[0, j].set_title(ct, fontsize=14)

    for i, r in enumerate(records):
        ax = axes[i]
        # Input
        ax[0].imshow(r['img'])
        ax[0].set_ylabel(r['label'], fontsize=12)
        ax[0].set_xticks([]); ax[0].set_yticks([])

        # GT density
        if r['gt_density'] is not None:
            ax[1].imshow(gamma_norm(r['gt_density']), cmap='jet')
            ax[1].set_xlabel(f"GT={r['gt']:.0f}", fontsize=11)
        else:
            ax[1].text(0.5, 0.5, f"(no GT density map)\nGT={r['gt']:.0f}",
                       ha='center', va='center')
        ax[1].set_xticks([]); ax[1].set_yticks([])

        # LCDNet
        ax[2].imshow(gamma_norm(r['lcd_density']), cmap='jet')
        ax[2].set_xlabel(f"pred={r['lcd_count']:.0f}  err={abs(r['lcd_count']-r['gt']):.0f}",
                         fontsize=11)
        ax[2].set_xticks([]); ax[2].set_yticks([])

        # MobileCount
        ax[3].imshow(gamma_norm(r['mc_density']), cmap='jet')
        ax[3].set_xlabel(f"pred={r['mc_count']:.0f}  err={abs(r['mc_count']-r['gt']):.0f}",
                         fontsize=11)
        ax[3].set_xticks([]); ax[3].set_yticks([])

        # Hybrid
        ax[4].imshow(gamma_norm(r['hybrid_density']), cmap='jet')
        ax[4].set_xlabel(
            f"chose {r['hybrid_choice']}  pred={r['hybrid_count']:.0f}  "
            f"err={abs(r['hybrid_count']-r['gt']):.0f}", fontsize=11)
        ax[4].set_xticks([]); ax[4].set_yticks([])

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ids', nargs='*', default=None,
                        help="space-separated img ids (uses defaults if absent)")
    parser.add_argument('--out',
                        default=os.path.join(config_routing.ROUTER_LOG_DIR,
                                             "phase3_qualitative_panel.png"))
    parser.add_argument('--dense_ckpt', default=None,
                        help='Override MobileCount checkpoint')
    args = parser.parse_args()

    if args.dense_ckpt:
        config_routing.DENSE_CHECKPOINT = args.dense_ckpt
        print(f"OVERRIDE dense ckpt: {args.dense_ckpt}")

    device = config_routing.DEVICE
    if args.ids:
        chosen = [(i, "") for i in args.ids]
    else:
        chosen = DEFAULT_IMAGE_IDS

    # load all models once
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

    records = []
    with torch.no_grad():
        for img_id, label in chosen:
            img_path = find_image(img_id)
            if img_path is None:
                print(f"Skip {img_id}: image not found")
                continue
            img = Image.open(img_path).convert('RGB')
            gt = load_gt(img_id)
            gt_d = load_gt_density(img_id)

            r_in = to_router_input(img, device)
            d_in = to_density_input(img, device)
            probs = F.softmax(router(r_in), dim=1)
            p_lcd = probs[0, 0].item()
            p_csr = probs[0, 1].item()
            pred_label = 0 if p_lcd >= p_csr else 1

            lcd_d = lcd(d_in)
            mc_d = mc(d_in)

            if pred_label == 0:
                hybrid_d = lcd_d
                choice = "LCDNet"
            else:
                hybrid_d = mc_d
                choice = "MobileCount"

            r_label = label or f"id={img_id}"
            records.append({
                'img_id': img_id,
                'label': r_label,
                'img': np.asarray(img),
                'gt': gt,
                'gt_density': gt_d,
                'lcd_density': lcd_d.squeeze().cpu().numpy(),
                'lcd_count':   lcd_d.sum().item(),
                'mc_density': mc_d.squeeze().cpu().numpy(),
                'mc_count':   mc_d.sum().item(),
                'hybrid_density': hybrid_d.squeeze().cpu().numpy(),
                'hybrid_count':   hybrid_d.sum().item(),
                'hybrid_choice': choice,
                'p_lcd': p_lcd, 'p_csr': p_csr,
            })

    make_panel(records, args.out)


if __name__ == "__main__":
    main()
