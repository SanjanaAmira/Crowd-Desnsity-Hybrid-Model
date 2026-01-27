# Crowd Density Estimation: LCDNet + CSRNet

A complete, thesis-ready pipeline for training and evaluating crowd density estimation models:
- **LCDNet** (Lightweight Crowd Density Network) on **ShanghaiTech** for sparse scenes
- **CSRNet** (Congested Scene Recognition Network) on **NWPU-Crowd** for dense scenes

## 📁 Project Structure

```
LCDnet/
├── data/
│   ├── raw/                    # Downloaded ShanghaiTech dataset
│   ├── density_maps/           # Generated density maps (.npy)
│   ├── splits/                 # Train/val/test split files
│   └── NWPU-Crowd/             # NWPU-Crowd dataset
│       ├── images_part1-5/     # NWPU images
│       ├── mats/               # Annotations (.mat files)
│       ├── density_maps/       # Generated density maps
│       └── splits/             # Processed split files
├── models/
│   ├── __init__.py
│   ├── lcdnet.py              # LCDNet architecture (Phase 1)
│   └── csrnet.py              # CSRNet architecture (Phase 2)
├── utils/
│   ├── __init__.py
│   ├── density_generator.py   # Gaussian kernel density map generation
│   └── metrics.py             # MAE, MSE evaluation metrics
├── checkpoints/
│   ├── best_model.pth         # LCDNet checkpoint
│   └── csrnet/                # CSRNet checkpoints
├── logs/                      # Training logs and evaluation results
├── config.py                  # Hyperparameters and paths
├── dataset.py                 # ShanghaiTech Dataset class
├── dataset_nwpu.py            # NWPU-Crowd Dataset class
├── preprocess.py              # ShanghaiTech preprocessing
├── preprocess_nwpu.py         # NWPU-Crowd preprocessing
├── train.py                   # LCDNet training
├── train_csrnet.py            # CSRNet training
├── evaluate.py                # LCDNet evaluation
├── evaluate_csrnet.py         # CSRNet evaluation
└── README.md                  # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd c:\Users\Navid\Desktop\LCDnet
pip install -r requirements.txt
```

### 2. Preprocess Dataset

Downloads the UCSD dataset from Kaggle and generates density maps:

```bash
python preprocess.py
```

This will:
- Download the UCSD Anomaly Detection Dataset via `kagglehub`
- Generate ground truth density maps using Gaussian kernels
- Create train/val/test splits (70/15/15)

### 3. Train the Model

```bash
python train.py
```

Training options:
```bash
python train.py --epochs 100 --batch_size 8 --lr 1e-4
python train.py --resume checkpoints/checkpoint_epoch_50.pth  # Resume training
```

### 4. Evaluate

```bash
python evaluate.py
```

Evaluation options:
```bash
python evaluate.py --visualize  # Show sample predictions
python evaluate.py --checkpoint checkpoints/best_model.pth
```

---

## 🚀 Phase 2: CSRNet on NWPU-Crowd (Dense Scenes)

### 1. Preprocess NWPU-Crowd Dataset

Make sure the NWPU-Crowd dataset is in `data/NWPU-Crowd/` with images, mats, and split files.

```bash
python preprocess_nwpu.py
```

This will:
- Parse `.mat` annotation files to extract head positions
- Generate density maps using Gaussian kernels
- Create processed split files for training

### 2. Train CSRNet

```bash
python train_csrnet.py
```

Training options:
```bash
python train_csrnet.py --epochs 100 --batch_size 4 --lr 1e-5
python train_csrnet.py --freeze_frontend  # Train only backend (faster)
python train_csrnet.py --resume checkpoints/csrnet/csrnet_epoch_50.pth
```

### 3. Evaluate CSRNet

```bash
python evaluate_csrnet.py
```

Evaluation options:
```bash
python evaluate_csrnet.py --visualize --num_samples 5
python evaluate_csrnet.py --checkpoint checkpoints/csrnet/csrnet_best.pth
```

---

## 🏗️ Architecture

### LCDNet (Lightweight - for Sparse Scenes)

**LCDNet** uses an encoder-decoder architecture with:

- **Depthwise Separable Convolutions**: Reduces parameters by ~8x compared to standard convolutions
- **Encoder**: 4 stages with progressive downsampling
- **Decoder**: 4 stages with skip connections for spatial detail recovery
- **Output**: Single-channel density map at original resolution

```
Input (3, 256, 256)
    ↓
Encoder: [32] → [64] → [128] → [256]
    ↓
Decoder: [256] → [128] → [64] → [32]
    ↓
Output (1, 256, 256) - Density Map
```

### CSRNet (Dense - for Congested Scenes)

**CSRNet** uses a two-stage architecture designed for dense crowds:

- **VGG-16 Frontend**: First 10 conv layers from pretrained VGG-16 for robust feature extraction
- **Dilated Backend**: 6 dilated convolution layers with dilation rate 2 for enlarged receptive field
- **Output**: 1/8 resolution density map (sum = estimated count)

```
Input (3, 384, 384)
    ↓
VGG-16 Frontend: conv1_1 → conv4_3 (3 max pools)
    ↓
Feature Maps (512, 48, 48)
    ↓
Dilated Backend: 512 → 512 → 512 → 256 → 128 → 64
    ↓
Output (1, 48, 48) - Density Map
    ↓
Sum pixels → Estimated Count
```



## 📊 Metrics

The pipeline evaluates using standard crowd counting metrics:

| Metric | Description |
|--------|-------------|
| **MAE** | Mean Absolute Error - average count error per image |
| **MSE** | Mean Squared Error - penalizes large errors more |
| **RMSE** | Root MSE - same units as count for interpretability |

## ⚙️ Configuration

All hyperparameters are in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `IMAGE_SIZE` | (256, 256) | Training image resolution |
| `BATCH_SIZE` | 8 | Training batch size |
| `LEARNING_RATE` | 1e-4 | Adam optimizer learning rate |
| `NUM_EPOCHS` | 100 | Training epochs |
| `TRAIN_RATIO` | 0.70 | Training split ratio |
| `INITIAL_FILTERS` | 32 | Base filter count |

## 📚 Understanding the Code

### Density Map Generation

Each annotated head position is represented as a 2D Gaussian. The density map's sum equals the total person count:

```python
from utils.density_generator import generate_density_map

# Generate density map from head positions
density_map = generate_density_map(
    image_shape=(480, 640),
    points=head_positions  # (N, 2) array of x, y coordinates
)
person_count = density_map.sum()  # Equals number of people
```

### Model Inference

```python
from models.lcdnet import create_model

model = create_model("cuda")
model.load_state_dict(torch.load("checkpoints/best_model.pth")['model_state_dict'])
model.eval()

with torch.no_grad():
    density_map = model(image_tensor)
    estimated_count = density_map.sum().item()
```

## 📝 Notes for Thesis Defense

1. **Why density maps?** Unlike detection-based methods, density estimation handles extreme crowding where individuals overlap and can't be detected separately.

2. **Why depthwise separable convolutions?** They provide 8x parameter reduction while maintaining accuracy, making the model efficient for deployment.

3. **Why adaptive sigma?** Using nearest-neighbor distance for Gaussian kernel size produces better density maps in varying crowd densities.

4. **Why MSE loss?** MSE penalizes large errors quadratically, encouraging the model to avoid extreme prediction errors.

## 🔬 Reproducibility

- Random seed is fixed in `config.py` (`RANDOM_SEED = 42`)
- Data splits are deterministic given the same seed
- Model initialization uses PyTorch defaults

## 📄 License

This code is provided for educational and research purposes as part of an undergraduate thesis project.

## 🙏 Acknowledgments

- UCSD Anomaly Detection Dataset from Kaggle
- LCDNet architecture inspired by lightweight crowd counting literature
