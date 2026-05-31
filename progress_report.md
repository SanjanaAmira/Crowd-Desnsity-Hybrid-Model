# Thesis Progress Update — Phase 3 (Final Defense Prep)

**Author:** Navid
**System:** LCDNet (sparse) + MobileCount (dense) + MobileNetV2 Router

CSRNet has been removed from the deployed pipeline as discussed. The full
system is end-to-end lightweight (~4.35M parameters total) and suitable for
edge deployment. CSRNet is only retained as a literature reference (and
optionally as a training-time teacher for knowledge distillation, see TODO).

---

## Headline Results (full NWPU-Crowd val, 500 images)

| Model                  | MAE     | RMSE     | 95% CI MAE       | CPU latency (ms) |
| :--------------------- | :-----: | :------: | :--------------: | :--------------: |
| LCDNet (alone)         | 348.0   | 1033.9   | [269.5, 443.1]   | 239.0            |
| MobileCount (alone)    | 213.2   | 803.6    | [155.4, 292.6]   | 30.8             |
| **Hybrid-Hard**        | **183.0** | 787.7  | [126.2, 260.3]   | 121.5            |
| Hybrid-Soft (fusion)   | 183.0   | 787.7    | [126.5, 260.4]   | 138.4            |
| Oracle Router (UB)     | 167.7   | 778.4    | [111.8, 245.2]   | —                |

* 95% CI from 2000-iteration bootstrap.
* Hybrid latency includes router + chosen counter.
* Oracle Router is the upper bound for any routing strategy that picks one of
  {LCDNet, MobileCount} per image.

## Stratified MAE by density bin

| Model              | Sparse (≤100, n=181) | Medium (100–500, n=229) | Dense (>500, n=90) |
| :----------------- | :------------------: | :---------------------: | :----------------: |
| LCDNet (alone)     | **21.5**             | 169.1                   | 1460.0             |
| MobileCount (alone)| 127.5                | 85.5                    | 710.6              |
| **Hybrid-Hard**    | 35.4                 | **89.2**                | 718.7              |
| Oracle Router      | 18.5                 | 72.2                    | 710.6              |

## Edge-deployment metrics

| Component   | Params (M) | GFLOPs | CPU latency (ms) |
| :---------- | :--------: | :----: | :--------------: |
| LCDNet      | 0.917      | 14.59  | 215.2            |
| MobileCount | 0.884      | 1.07   | 14.4             |
| Router      | 2.552      | 0.33   | 8.8              |
| **TOTAL**   | **4.354**  |        |                  |

## Routing distribution (Hybrid-Hard, 500 images)

* MobileCount path: 333 images (66.6%)
* LCDNet path:      167 images (33.4%)

## Routing distribution (Hybrid-Soft)

* MobileCount only: 296 (59.2%)
* LCDNet only:      146 (29.2%)
* Soft fusion:       58 (11.6%)

## Router

* Backbone: MobileNetV2 (~2.55M params)
* Validation accuracy: 88.8% on NWPU val (threshold T=100)
* Cost of router errors: ~15 MAE (gap between Hybrid 183 and Oracle 168)

---

## Defense Narrative

We present a fully lightweight hybrid crowd-counting system, totalling
**~4.35M parameters**, suitable for edge deployment. A MobileNetV2-based
router classifies scenes as sparse or dense and dispatches to **LCDNet** or
**MobileCount** accordingly. On NWPU-Crowd (full validation, 500 images), the
hybrid achieves **MAE 183.0**, outperforming both component models in
isolation (LCDNet 348.0, MobileCount 213.2). Stratified analysis shows the
hybrid recovers most of LCDNet's strength on sparse scenes (35 vs 22 MAE)
while retaining MobileCount's behaviour on dense scenes. Compared to the
oracle upper bound (MAE 167.7), the gap of ~15 MAE represents the cost of
imperfect routing.

We additionally fine-tuned LCDNet on the NWPU sparse subset, reducing
sparse-scene MAE from 273 to ~21.

---

## Remaining work (1-week roadmap)

* [x] #1  Full NWPU val evaluation (500 images)
* [x] #2  Comparison table without CSRNet
* [x] #3  Oracle Router upper bound
* [x] #4  Stratified MAE (sparse / medium / dense)
* [x] #5  Re-framed efficiency claim (vs MobileCount, edge-deployment)
* [x] #6  Edge metrics (params, FLOPs, CPU latency)
* [ ] #7  Knowledge distillation MobileCount ← CSRNet (training-time teacher)
* [ ] #8  Better MobileCount training (multi-scale, DM-Count loss, longer)
* [ ] #9  Routing threshold sweep (T = 50, 75, 100, 150, 200)
* [ ] #10 Better fusion weights (1D grid search on val)
* [ ] #11 Router calibration (reliability diagrams)
* [ ] #12 Cross-dataset eval (train NWPU, test ShanghaiTech)
* [ ] #13 Failure case analysis (worst 5 errors)
* [ ] #14 Qualitative density-map figures
* [ ] #15 Bootstrap CI on differences (already done as per-model CIs)
* [ ] #16 Final progress report + slides

Files of interest:

* `routing/full_evaluation.py` — full evaluation suite
* `logs/phase3_full_evaluation.txt` — current main results
* `logs/phase3_full_evaluation_efficiency.txt` — edge metrics with FLOPs
