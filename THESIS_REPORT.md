# Hybrid Crowd Density Estimation — Comprehensive Technical Report

**Author:** Navid
**Project:** Lightweight Hybrid Crowd Counting for Edge Deployment
**Repository:** Crowd-Density-Hybrid-Model
**Hardware:** NVIDIA GeForce RTX 3050 (8 GB)
**Primary dataset:** NWPU-Crowd · **Cross-dataset:** ShanghaiTech Part A & B

---

## 1. Executive Summary

We present a fully lightweight hybrid crowd-counting system that combines two
small density estimators behind a learned router. The deployed pipeline totals
**~4.35M parameters** and runs at **~55 ms/image on an RTX 3050** (≈210 ms on a
single CPU thread), making it suitable for resource-constrained edge devices.

A MobileNetV2-based router classifies each scene as *sparse* or *dense* and
dispatches it to **LCDNet** (sparse specialist) or a **knowledge-distilled
MobileCount** (dense specialist). CSRNet — a heavy 16.2M-parameter model — is
used **only as a training-time teacher** for distillation and is never deployed.

**Best configuration on the full NWPU-Crowd validation set (500 images):**

| Metric | Value |
| :--- | :--- |
| Hybrid MAE (distilled MC, p\*=0.85) | **180.9** |
| MobileCount alone (distilled) | 198.7 |
| LCDNet alone | 348.0 |
| Oracle Router (upper bound) | 166.6 |
| Total parameters | 4.35M |
| Latency (RTX 3050) | ~55 ms/image |

**Generalization (no fine-tuning) — ShanghaiTech Part B:** Hybrid MAE **35.2**,
beating both single components.

---

## 2. Problem Statement & Motivation

Crowd counting via density-map regression must work across a huge range of
scene densities — from a handful of people to tens of thousands. A single
model tuned for sparse scenes fails on dense crowds and vice-versa. Heavy
state-of-the-art models (e.g. CSRNet, 16.2M params) achieve strong accuracy
but are impractical for edge deployment.

**Our hypothesis:** a *router* can dispatch each image to the most suitable
lightweight specialist, recovering much of the accuracy of a large model while
keeping the deployed footprint tiny.

**Design goals:**
1. Total deployed parameters well under a heavy single model.
2. Per-image latency low enough for edge hardware.
3. Accuracy that beats either lightweight specialist used alone.

---

## 3. System Architecture

```
                    Input image
                         │
              ┌──────────▼───────────┐
              │  Router (MobileNetV2) │   224×224 input, 2-class
              │  P(sparse), P(dense)  │
              └──────────┬───────────┘
                  argmax / threshold p*
              ┌──────────┴───────────┐
       sparse │                      │ dense
        ┌─────▼──────┐        ┌──────▼─────────┐
        │   LCDNet   │        │  MobileCount   │   384×384 input
        │  (0.92M)   │        │  (distilled,   │
        │            │        │   0.88M)       │
        └─────┬──────┘        └──────┬─────────┘
              │   density map (1/8)  │
              └──────────┬───────────┘
                         ▼
                  sum() → count
```

### 3.1 Components

| Component | Backbone / design | Params | Input | Output |
| :--- | :--- | :---: | :---: | :--- |
| **Router** | MobileNetV2 + 2-class head | 2.55M | 224×224 | P(sparse), P(dense) |
| **LCDNet** | Depthwise-separable encoder-decoder, skip connections | 0.92M | 384×384 | density map (full res) |
| **MobileCount** | MobileNetV2 features 0–13 + dilated context + skip fusion | 0.88M | 384×384 | density map (1/8 res, 48×48) |

**Total deployed: ~4.35M parameters.**

### 3.2 Routing logic

- **Hard routing (primary):** route to the model with the higher router
  probability. A tunable threshold `p*` controls the decision boundary
  (route to MobileCount only if `P(dense) ≥ p*`).
- **Soft fusion (sensitivity analysis):** for low-confidence images, run both
  models and blend density maps weighted by router probabilities. (Found to
  give no measurable benefit — see §7.3.)

### 3.3 Routing label definition

An image is labelled *sparse* if ground-truth count ≤ 100, else *dense*
(threshold T = 100, consistent with crowd-counting literature). This labelling
trains the router and defines the sparse/medium/dense strata used throughout.

---

## 4. Datasets

| Dataset | Split | Images | Use |
| :--- | :--- | :---: | :--- |
| NWPU-Crowd | train | 3,109 | training all components |
| NWPU-Crowd | val | 500 | primary evaluation |
| ShanghaiTech Part A | test | 182 | cross-dataset (denser) |
| ShanghaiTech Part B | test | 316 | cross-dataset (sparser) |

Ground-truth counts: NWPU from JSON `human_num`; ShanghaiTech from `.mat`
head-annotation files.

**Density strata (NWPU val):** sparse (≤100): 181 images · medium (100–500):
229 · dense (>500): 90.

---

## 5. Methodology

### 5.1 Density-map training (LCDNet, MobileCount)

- Gaussian-kernel density maps; adaptive sigma (k-NN) with bounds.
- Loss: MSE on density maps + small L1 count term.
- Adam / AdamW optimizers, ReduceLROnPlateau / cosine LR.
- ImageNet normalization; 384×384 inputs.

### 5.2 LCDNet domain adaptation

LCDNet was fine-tuned on the NWPU **sparse** subset, dropping sparse-scene MAE
from ~273 to ~21 — establishing it as the sparse specialist.

### 5.3 Router training

- MobileNetV2 pretrained backbone, custom 2-class head.
- 25 epochs, batch 32, Adam lr 1e-4, augmentation (flip, color jitter).
- **Validation accuracy: 88.8%.**

### 5.4 Knowledge distillation (CSRNet → MobileCount)

Combined loss:

```
L = α · MSE(student, GT)
  + β · MSE(student, teacher.detach())
  + γ · L1(count_student, count_GT)
```

with `α = 0.5, β = 0.5, γ = 0.05`. Both teacher and student emit 1/8-resolution
(48×48) density maps, so distillation is a direct map-to-map match. 12 epochs,
AdamW, cosine LR, warm-started from the baseline MobileCount. The teacher
(CSRNet, 16.2M) is frozen and discarded after training.

---

## 6. Edge-Deployment Metrics

| Component | Params (M) | GFLOPs | GPU latency (ms) | CPU latency (ms) | GPU peak mem (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| LCDNet | 0.917 | 14.59 | 23.1 | 208.8 | 368.8 |
| MobileCount | 0.884 | 1.07 | 3.4 | 13.5 | 63.6 |
| Router | 2.552 | 0.33 | 4.8 | 8.9 | 52.6 |
| **TOTAL** | **4.354** | — | — | — | — |

**End-to-end hybrid (RTX 3050):**
- Sparse path (Router → LCDNet): ~28 ms
- Dense path (Router → MobileCount): ~8 ms
- Average over val (Hybrid-Hard, incl. image load + transforms): ~55 ms/image

FLOPs measured with `thop`; latency averaged over 30 GPU / 10 CPU runs after warmup.

---

## 7. Results

All NWPU numbers are on the **full 500-image validation set**. 95% confidence
intervals are from a 2000-iteration bootstrap.

### 7.1 Main results (NWPU val)

| Model | MAE | RMSE | 95% CI MAE | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: |
| LCDNet (alone) | 348.0 | 1033.9 | [269.5, 443.1] | 40.7 |
| MobileCount (baseline) | 213.2 | 803.6 | [155.4, 292.6] | 30.5 |
| **MobileCount (distilled)** | **198.7** | 778.5 | [143.8, 275.1] | 30.5 |
| Hybrid-Hard (baseline MC) | 183.0 | 787.7 | [126.2, 260.3] | 53.4 |
| **Hybrid-Hard (distilled MC)** | **182.4** | 764.0 | [127.8, 258.6] | 55.0 |
| **Hybrid-Hard (distilled, p\*=0.85)** | **180.9** | 763.6 | — | ~55 |
| Hybrid-Soft (fusion) | 182.6 | 763.8 | [127.9, 259.0] | 56.0 |
| Oracle Router (upper bound) | 166.6 | 751.1 | [112.9, 241.5] | — |

**Key reading:** the hybrid beats both deployed specialists (LCDNet 348,
distilled MobileCount 199). The gap to the oracle (167) is the cost of
imperfect routing (~14 MAE).

### 7.2 Stratified MAE by density (NWPU val, distilled MC)

| Model | Sparse (≤100, n=181) | Medium (100–500, n=229) | Dense (>500, n=90) |
| :--- | :---: | :---: | :---: |
| LCDNet (alone) | **21.5** | 169.1 | 1460.0 |
| MobileCount (distilled) | 84.5 | 98.4 | 683.6 |
| Hybrid-Hard | 33.6 | 99.5 | 692.8 |
| Oracle Router | 14.2 | 83.8 | 683.6 |

The hybrid's advantage is concentrated in the **sparse stratum**, where routing
to LCDNet cuts MAE from 84.5 (MobileCount) to 33.6.

### 7.3 Threshold & fusion sweeps (distilled MC, no retraining)

Router probability threshold `p*` (route to MobileCount if `P(dense) ≥ p*`):

| p\* | MAE | RMSE | Sparse | MC share |
| :---: | :---: | :---: | :---: | :---: |
| 0.50 (argmax) | 182.45 | 764.0 | 33.6 | 66.6% |
| 0.70 | 181.58 | 763.8 | 29.6 | 65.0% |
| 0.75 | 180.93 | 763.6 | 27.4 | 63.8% |
| **0.85** | **180.92** | **763.6** | 27.7 | 63.0% |
| 0.90 | 185.05 | 766.3 | 27.4 | 61.2% |

**Best p\* = 0.85 → MAE 180.92**, a free 1.5-point gain with no retraining.

- **Soft-fusion margin sweep:** all settings within MAE 182.9–183.7 →
  soft fusion provides no measurable benefit over hard routing.
- **Fixed fusion-α sweep:** best α≈0.5 gives MAE 183.5; still no gain.

### 7.4 Knowledge-distillation progression (NWPU val MAE)

| Epoch | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
| :---: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| MAE | 239.6 | 217.6 | 216.2 | 208.9 | 227.9 | **200.2** | 203.8 | 204.4 | 215.6 | **198.7** | 198.8 | 200.2 |

**Best epoch 9, MAE 198.68** (from baseline 213.22 → **−14.5 MAE**). Distillation
also improved standalone MobileCount sparse MAE from 127.5 → 84.5.

### 7.5 Cross-dataset evaluation (trained on NWPU, no fine-tuning)

**ShanghaiTech Part B (316 images, predominantly sparse):**

| Model | MAE | RMSE |
| :--- | :---: | :---: |
| LCDNet alone | 74.06 | 111.12 |
| MobileCount alone | 41.68 | 53.53 |
| **Hybrid-Hard** | **35.20** | **50.75** |
| Oracle Router | 29.32 | 45.09 |

**ShanghaiTech Part A (182 images, predominantly dense):**

| Model | MAE | RMSE |
| :--- | :---: | :---: |
| LCDNet alone | 340.19 | 485.79 |
| MobileCount alone | 132.33 | 213.11 |
| Hybrid-Hard | 133.08 | 212.76 |
| Oracle Router | 125.71 | 206.50 |

On Part B the hybrid clearly wins; on Part A it ties with MobileCount because
the router sends 95% of (dense) images to MobileCount anyway. This confirms the
routing mechanism transfers to an unseen dataset.

### 7.6 Router calibration

- Validation accuracy: **88.8%** (NWPU val, T=100).
- **Expected Calibration Error (ECE): 0.095** (10-bin).
- Well-calibrated in the high-confidence regime: 442/500 images fall in the
  [0.95, 1.0] confidence bin with 92.8% empirical accuracy. Errors concentrate
  in the 65–95% confidence range.

---

## 8. Failure-Case Analysis

Top worst hybrid predictions on NWPU val (with baseline MC):

| img_id | GT | LCDNet | MobileCount | Hybrid | Cause |
| :--- | :---: | :---: | :---: | :---: | :--- |
| 3234 | 12,924 | 10 | 158 | 10 | **Router catastrophic miss** — 99% confident "sparse", routed to LCDNet |
| 3408 | 9,728 | 78 | 2,428 | 2,428 | Both models saturate at extreme density |
| 3353 | 7,122 | 28 | 2,077 | 2,077 | Density beyond training range |
| 3587 | 6,799 | 60 | 2,301 | 2,301 | Same |
| 3146 | 5,951 | 35 | 2,265 | 2,265 | Same |

**Observations:**
1. Image 3234 alone contributes ~25 MAE to the 500-image average. Excluding
   it drops hybrid MAE from ~183 to ~158.
2. Beyond ~3,000 people, **both** lightweight models saturate and undercount —
   a fundamental capacity limit of sub-1M-param density heads, not a routing
   error. This is the clearest target for future work.

---

## 9. Discussion

**Why the hybrid only modestly beats distilled MobileCount alone (180.9 vs 198.7):**
The aggregate MAE is dominated by a handful of extreme-density images where
*both* models fail. The hybrid's real, consistent win is in the sparse and
boundary regions (sparse MAE 33.6 vs 84.5) and in cross-dataset robustness
(ShanghaiTech B: 35.2 vs 41.7). The contribution is best framed as **accuracy-
aware routing for edge deployment**, not as a new accuracy record.

**Efficiency framing.** Against published NWPU methods (MAE ~40–90), our
absolute MAE is higher; the contribution is the **~4.35M-parameter,
edge-deployable** footprint with a tunable accuracy/latency operating point,
not state-of-the-art accuracy.

**Knowledge distillation worked as intended:** −14.5 MAE on the standalone
student with zero added deployment cost (the student is still 0.88M params; the
16.2M teacher is discarded).

---

## 10. Limitations

1. **Extreme density (>3,000):** both specialists saturate; aggregate MAE/RMSE
   is dominated by these tails.
2. **Router errors:** ~11% misroute rate; one catastrophic miss (3234) skews
   the mean. Calibrated confidence could trigger a safe fallback.
3. **No physical edge deployment:** efficiency claims rest on params/FLOPs/
   latency measured on an RTX 3050 and CPU, not on a Jetson/Raspberry Pi.
4. **Soft fusion adds complexity without benefit** — recommended to drop.

---

## 11. Future Work

- DM-Count or Bayesian Loss + multi-scale crops to lift dense-scene accuracy.
- Confidence-aware fallback for low-confidence router decisions.
- Physical edge-device benchmark (Jetson Nano / Raspberry Pi) with power draw.
- Distill into an even smaller student; quantization (INT8) for further speedup.

---

## 12. Reproducibility — Scripts & Artifacts

| File | Purpose |
| :--- | :--- |
| `routing/full_evaluation.py` | full eval suite (main, oracle, stratified, CI, edge metrics) |
| `routing/threshold_sweep.py` | router-threshold / soft-margin / fusion-α sweeps |
| `routing/analysis.py` | router calibration (ECE) + top-K failure cases |
| `routing/qualitative_figures.py` | side-by-side density-map panels |
| `routing/cross_dataset_eval.py` | ShanghaiTech A/B evaluation |
| `train_mobilecount_distill.py` | CSRNet→MobileCount knowledge distillation |
| `bundle_checkpoints.py` | package essential checkpoints for sharing |

### Result files (`logs/`)

| File | Contents |
| :--- | :--- |
| `phase3_full_evaluation.txt` | full-val results (baseline MC) |
| `phase3_full_evaluation_distilled.txt` | full-val results (distilled MC) |
| `phase3_threshold_sweep.txt` / `_distilled.txt` | sweeps |
| `phase3_router_calibration_failures.md` | calibration + failure dumps |
| `phase3_qualitative_panel.png` / `_distilled.png` | density-map figures |
| `phase3_cross_dataset_partA.txt` / `partB.txt` | ShanghaiTech results |
| `distill_train.log` | distillation training log |
| `reliability_diagram.png` | router reliability diagram |

### Checkpoints (deployed pipeline)

| File | Role | Params |
| :--- | :--- | :---: |
| `checkpoints/best_model_nwpu_sparse.pth` | LCDNet (sparse) | 0.92M |
| `checkpoints/mobilecount_distilled.pth` | **MobileCount (dense, deployed)** | 0.88M |
| `checkpoints/router/router_best.pth` | Router | 2.55M |
| `checkpoints/csrnet/csrnet_best.pth` | CSRNet (KD teacher, **not deployed**) | 16.2M |

Checkpoints are excluded from git (size). Use `bundle_checkpoints.py` to
package them; share via Drive/OneDrive.

### Environment

- PyTorch 2.6.0 + cu124, torchvision 0.21.0
- CUDA 12.x, NVIDIA RTX 3050 (8 GB)
- `thop`, `fvcore` (FLOPs), `scipy` (.mat reading)

---

## 13. Conclusion

The hybrid LCDNet + distilled-MobileCount + router system delivers a
**4.35M-parameter, edge-deployable** crowd counter that **outperforms either
lightweight specialist alone** (NWPU val MAE 180.9 vs 198.7 / 348.0), recovers
sparse-scene accuracy via routing (sparse MAE 33.6), generalizes to an unseen
dataset without fine-tuning (ShanghaiTech B MAE 35.2), and runs at ~55 ms/image
on commodity GPU hardware. Knowledge distillation from CSRNet improved the
deployed dense model by 14.5 MAE at zero inference cost. The work is positioned
as an efficiency/edge-deployment contribution with a tunable accuracy–latency
operating point.
