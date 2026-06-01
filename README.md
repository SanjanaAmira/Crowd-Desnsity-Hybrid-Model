# Hybrid Crowd Density Estimation

> A lightweight hybrid crowd-counting system for edge deployment.  
> **LCDNet** (sparse) + **MobileCount** (dense, knowledge-distilled) + **MobileNetV2 Router**

---

## Overview

Standard crowd-counting models are either accurate but heavy (CSRNet, 16.2M params) or fast but weak. This project proposes a **routing-based hybrid** that dispatches each image to the most suitable lightweight specialist, recovering much of the accuracy of a large model while keeping the deployed footprint tiny.

A learned router classifies each scene as *sparse* or *dense* and routes it to:
- **LCDNet** — depthwise-separable encoder-decoder, fine-tuned on NWPU sparse scenes
- **MobileCount** — MobileNetV2-backbone density estimator, improved via knowledge distillation from CSRNet

CSRNet is used **only as a training-time teacher** and is never deployed.

---

## Results

### NWPU-Crowd Validation (500 images)

| Model | MAE | RMSE | Params | Latency (RTX 3050) |
|:---|:---:|:---:|:---:|:---:|
| LCDNet alone | 348.0 | 1033.9 | 0.92M | 40.7 ms |
| MobileCount (baseline) | 213.2 | 803.6 | 0.88M | 30.5 ms |
| MobileCount (distilled) | 198.7 | 778.5 | 0.88M | 30.5 ms |
| Hybrid-Hard (baseline) | 183.0 | 787.7 | 4.35M | 53.4 ms |
| **Hybrid-Hard (distilled, p\*=0.85)** | **180.9** | **763.6** | **4.35M** | **~55 ms** |
| Oracle Router (upper bound) | 166.6 | 751.1 | — | — |

### Stratified MAE (distilled MobileCount)

| Model | Sparse ≤100 (n=181) | Medium 100–500 (n=229) | Dense >500 (n=90) |
|:---|:---:|:---:|:---:|
| LCDNet alone | **21.5** | 169.1 | 1460.0 |
| MobileCount (distilled) | 84.5 | 98.4 | 683.6 |
| **Hybrid-Hard** | **33.6** | 99.5 | 692.8 |
| Oracle Router | 14.2 | 83.8 | 683.6 |

### Cross-Dataset (no fine-tuning)

| Dataset | LCDNet | MobileCount | **Hybrid** | Oracle |
|:---|:---:|:---:|:---:|:---:|
| ShanghaiTech Part B (316 imgs) | 74.1 | 41.7 | **35.2** | 29.3 |
| ShanghaiTech Part A (182 imgs) | 340.2 | 132.3 | 133.1 | 125.7 |

### Edge Metrics

| Component | Params | GFLOPs | GPU (ms) | CPU (ms) |
|:---|:---:|:---:|:---:|:---:|
| LCDNet | 0.917M | 14.59 | 23.1 | 208.8 |
| MobileCount | 0.884M | 1.07 | 3.4 | 13.5 |
| Router | 2.552M | 0.33 | 4.8 | 8.9 |
| **Total** | **4.354M** | — | — | — |

---

## Architecture

```
Input Image
     │
     ▼
┌─────────────────────┐
│  Router             │  MobileNetV2 · 224×224 · 2-class
│  P(sparse/dense)    │
└──────────┬──────────┘
           │ threshold p* = 0.85
     ┌─────┴──────┐
     │            │
     ▼            ▼
 LCDNet      MobileCount
 (sparse)    (dense, distilled)
 0.92M       0.88M · 384×384
     │            │
     └─────┬──────┘
           ▼
     density map
     sum() → count
```

---

## Project Structure

```
Thesis/
├── models/
│   ├── lcdnet.py               # LCDNet architecture
│   ├── mobilecount.py          # MobileCount architecture
│   └── csrnet.py               # CSRNet (KD teacher only)
├── routing/
│   ├── router.py               # MobileNetV2 routing classifier
│   ├── hybrid_inference.py     # end-to-end hybrid pipeline
│   ├── full_evaluation.py      # main evaluation suite
│   ├── threshold_sweep.py      # threshold / margin / alpha sweeps
│   ├── analysis.py             # router calibration + failure cases
│   ├── qualitative_figures.py  # density-map visualizations
│   └── cross_dataset_eval.py   # ShanghaiTech evaluation
├── train.py                    # LCDNet training (ShanghaiTech)
├── train_csrnet.py             # CSRNet training
├── train_lightweight.py        # MobileCount training
├── train_lcdnet_nwpu.py        # LCDNet fine-tuning on NWPU sparse
├── train_mobilecount_distill.py # Knowledge distillation
├── dataset.py                  # ShanghaiTech dataset
├── dataset_nwpu.py             # NWPU-Crowd dataset
├── config.py                   # global config
├── bundle_checkpoints.py       # package checkpoints for sharing
├── THESIS_REPORT.md            # full technical report
└── progress_report.md          # defense prep notes
```

---

## Setup

```bash
# Clone the repo
git clone https://github.com/navidnawaj/Crowd-Desnsity-Hybrid-Model.git
cd Crowd-Desnsity-Hybrid-Model

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# Install dependencies (GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install thop scipy matplotlib tqdm
```

---

## Checkpoints

Checkpoints are **not stored in this repo** (too large for GitHub). Download from the shared link provided by the team, then unzip into the project root:

```
checkpoints/
  best_model_nwpu_sparse.pth      ← LCDNet (sparse specialist)
  mobilecount_distilled.pth       ← MobileCount distilled (use this)
  mobilecount_best.pth            ← MobileCount baseline
  router/router_best.pth          ← Router
  csrnet/csrnet_best.pth          ← CSRNet (KD teacher, not deployed)
```

To bundle checkpoints for sharing:
```bash
python bundle_checkpoints.py          # deploy set only (~50 MB)
python bundle_checkpoints.py --full   # includes CSRNet teacher (~240 MB)
```

---

## Evaluation

```bash
# Full NWPU val evaluation (500 images) — baseline MobileCount
python routing/full_evaluation.py

# With distilled MobileCount
python routing/full_evaluation.py --dense_ckpt checkpoints/mobilecount_distilled.pth --tag distilled

# Threshold sweep
python routing/threshold_sweep.py --mode all

# Router calibration + failure cases
python routing/analysis.py

# Cross-dataset (ShanghaiTech A + B)
python routing/cross_dataset_eval.py

# Qualitative density-map figures
python routing/qualitative_figures.py
```

---

## Training

```bash
# 1. Train LCDNet on ShanghaiTech
python train.py

# 2. Fine-tune LCDNet on NWPU sparse subset
python train_lcdnet_nwpu.py

# 3. Train MobileCount on NWPU dense subset
python train_lightweight.py --epochs 80

# 4. Train Router
python routing/train_router.py

# 5. Knowledge distillation (CSRNet teacher → MobileCount student)
python train_mobilecount_distill.py --epochs 12
```

---

## Key Findings

- **Routing works:** hybrid (MAE 180.9) beats both specialists alone (198.7 / 348.0)
- **Sparse scenes are where routing matters most:** sparse MAE drops from 84.5 → 33.6
- **Knowledge distillation:** CSRNet teacher improved MobileCount by **−14.5 MAE** at zero inference cost
- **Soft fusion ≈ hard routing:** no measurable benefit from blending density maps
- **Generalizes:** ShanghaiTech Part B MAE 35.2 without any fine-tuning
- **Router calibration:** ECE 0.095, 88.8% accuracy — well-calibrated above 95% confidence

---

## Datasets

| Dataset | Source | Use |
|:---|:---|:---|
| NWPU-Crowd | [NWPU-Crowd](https://gjy3035.github.io/NWPU-Crowd-Sample-Code/) | Primary training + evaluation |
| ShanghaiTech | [ShanghaiTech](https://github.com/desenzhou/ShanghaiTechDataset) | Cross-dataset evaluation |

---

## Authors

- **Navid** — Phase 1–3 implementation, training, evaluation
- **Sumaiya Shahara** — team member

---

## License

For academic use only. All model weights and results are produced as part of a university thesis project.
