# LCDNet: Crowd Density Estimation on UCSD Dataset

A complete, thesis-ready pipeline for training and evaluating **LCDNet** (Lightweight Crowd Density Network) on the **UCSD Crowd Dataset** for crowd density estimation.

## 📁 Project Structure

```
LCDnet/
├── data/
│   ├── raw/                    # Downloaded UCSD dataset
│   ├── density_maps/           # Generated density maps (.npy)
│   └── splits/                 # Train/val/test split files
├── models/
│   ├── __init__.py
│   └── lcdnet.py              # LCDNet architecture
├── utils/
│   ├── __init__.py
│   ├── density_generator.py   # Gaussian kernel density map generation
│   └── metrics.py             # MAE, MSE evaluation metrics
├── checkpoints/               # Saved model checkpoints
├── logs/                      # Training logs and evaluation results
├── config.py                  # Hyperparameters and paths
├── dataset.py                 # PyTorch Dataset class
├── preprocess.py              # Dataset preprocessing script
├── train.py                   # Training script
├── evaluate.py                # Evaluation script
├── requirements.txt           # Python dependencies
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

## 🏗️ Architecture

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
