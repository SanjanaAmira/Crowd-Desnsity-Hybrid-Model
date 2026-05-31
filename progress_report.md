# Thesis Progress Update: Hybrid Crowd Density Estimation

Dear Professor,

I have completed the training and evaluation of the lightweight dense counter (**MobileCount**) on the NWPU-Crowd dataset for Phase 3. 

Below is a summary of the latest validation results (evaluated on a 50-sample benchmark):

### Performance & Efficiency Summary
| Model / Pipeline | MAE (Error) | Parameters | Inference Speed | Role / Status |
| :--- | :---: | :---: | :---: | :--- |
| **CSRNet (Baseline)** | 95.29 | 16.2M | 42.5 ms | Original Heavy Dense Counter |
| **MobileCount** | 169.97 | **884K** | **26.2 ms** | Lightweight Counter (**94.5% smaller, 38% faster**) |
| **Hybrid-Soft (Our System)** | **158.14** | *Dynamic* | **~30-40 ms** | Final Router + LCDNet + MobileCount (with Fusion) |

### Key System Updates
* **Domain Adaptation:** Fine-tuned LCDNet on NWPU sparse subset (sparse MAE dropped from 273.45 to ~21.0).
* **Confidence-Aware Soft Routing & Fusion:** High-confidence cases ($\ge 95\%$) trigger a single-model bypass. Uncertain cases execute both models and perform soft weighted density map fusion ($\alpha \cdot \text{LCDNet} + \beta \cdot \text{MobileCount}$).

### Request for Approval
By switching from CSRNet to MobileCount, we trade off some counting accuracy (MAE 158.14 vs. 95.29) for a **$94.5\%$ reduction in model size** and a **$38\%$ speedup**, making the system deployable on resource-constrained edge devices. 

Could you please confirm if this accuracy-vs-efficiency trade-off is approved for our thesis scope? If so, I will proceed to the final stage of routing optimization.

Best regards,  
Navid
