"""
Bundle only the essential checkpoints for sharing with teammates.

Creates two zips in the project root:
  - checkpoints_deploy.zip   (LCDNet + distilled MC + baseline MC + router)  ~54MB
  - checkpoints_full.zip     (deploy set + CSRNet teacher)                   ~240MB

Usage:
    python bundle_checkpoints.py            # builds the deploy zip
    python bundle_checkpoints.py --full     # also builds the full zip (with teacher)
"""

import os
import sys
import zipfile
import argparse

ROOT = os.path.dirname(os.path.abspath(__file__))
CK = os.path.join(ROOT, "checkpoints")

DEPLOY = [
    ("best_model_nwpu_sparse.pth", "LCDNet (sparse, deployed)"),
    ("mobilecount_distilled.pth",  "MobileCount distilled (dense, deployed)"),
    ("mobilecount_best.pth",       "MobileCount baseline (comparison)"),
    (os.path.join("router", "router_best.pth"), "Router"),
]
TEACHER = [(os.path.join("csrnet", "csrnet_best.pth"), "CSRNet (KD teacher only)")]


def build(zip_name, items):
    out = os.path.join(ROOT, zip_name)
    total = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for rel, desc in items:
            src = os.path.join(CK, rel)
            if not os.path.exists(src):
                print(f"  MISSING: {rel}  ({desc})  -- skipped")
                continue
            arc = os.path.join("checkpoints", rel)
            z.write(src, arc)
            mb = os.path.getsize(src) / 1e6
            total += mb
            print(f"  + {rel:42s} {mb:6.1f} MB  {desc}")
    print(f"  => {zip_name}  ({total:.1f} MB of files)\n")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="also build the full zip including the CSRNet teacher")
    args = ap.parse_args()

    print("Building deploy bundle (no CSRNet teacher)...")
    build("checkpoints_deploy.zip", DEPLOY)

    if args.full:
        print("Building full bundle (with CSRNet teacher)...")
        build("checkpoints_full.zip", DEPLOY + TEACHER)

    print("Done. Upload the zip(s) to Google Drive / OneDrive and share the link.")


if __name__ == "__main__":
    main()
