"""
Configuration file for LCDNet Crowd Density Estimation.

This file centralizes all hyperparameters, paths, and settings for the pipeline.
Modify these values to experiment with different configurations.

Author: Thesis Implementation
"""

import os

# ============================================================================
# PATH CONFIGURATION
# ============================================================================

# Base project directory (automatically detected)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Data directories
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
DENSITY_MAPS_DIR = os.path.join(DATA_DIR, "density_maps")
SPLITS_DIR = os.path.join(DATA_DIR, "splits")

# Model and checkpoint directories
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================

# Kaggle dataset identifier for UCSD Anomaly Detection Dataset
KAGGLE_DATASET = "karthiknm1/ucsd-anomaly-detection-dataset"

# Train/Validation/Test split ratios (must sum to 1.0)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Random seed for reproducibility
RANDOM_SEED = 42

# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================

# Target image size for training (height, width)
# Smaller sizes use less memory but may lose detail
IMAGE_SIZE = (256, 256)

# ImageNet normalization statistics (standard for pretrained backbones)
# Even without pretrained weights, this normalization is commonly used
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# ============================================================================
# DENSITY MAP GENERATION
# ============================================================================

# Gaussian kernel parameters for density map generation
# Sigma determines the spread of each Gaussian blob centered at head locations
# Adaptive sigma: sigma = SIGMA_COEFFICIENT * average_nearest_neighbor_distance
SIGMA_COEFFICIENT = 0.3

# Fixed sigma fallback when adaptive calculation isn't possible
# (e.g., when only one person in frame)
FIXED_SIGMA = 15.0

# Minimum and maximum sigma bounds to prevent extreme values
MIN_SIGMA = 4.0
MAX_SIGMA = 30.0

# Number of nearest neighbors for adaptive sigma calculation
K_NEAREST_NEIGHBORS = 3

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

# LCDNet configuration
# Number of initial filters (doubled at each encoder stage)
INITIAL_FILTERS = 32

# Number of encoder/decoder stages
NUM_STAGES = 4

# Dropout rate for regularization
DROPOUT_RATE = 0.2

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================

# Number of training epochs
NUM_EPOCHS = 100

# Batch size (reduce if running out of GPU memory)
BATCH_SIZE = 8

# Learning rate for Adam optimizer
LEARNING_RATE = 1e-4

# Weight decay for regularization
WEIGHT_DECAY = 1e-5

# Learning rate scheduler parameters
# Reduce LR by factor when validation loss plateaus
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 10

# Early stopping patience (stop if no improvement for this many epochs)
EARLY_STOPPING_PATIENCE = 20

# How often to print training progress (in batches)
LOG_INTERVAL = 10

# How often to save checkpoints (in epochs)
CHECKPOINT_INTERVAL = 5

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================

# Automatically use GPU if available
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Number of data loading workers
# Set to 0 on Windows to avoid multiprocessing issues
NUM_WORKERS = 0 if os.name == 'nt' else 4


def create_directories():
    """
    Create all necessary directories for the project.
    Call this before running any scripts.
    """
    directories = [
        DATA_DIR,
        RAW_DATA_DIR,
        DENSITY_MAPS_DIR,
        SPLITS_DIR,
        CHECKPOINTS_DIR,
        LOGS_DIR,
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created/verified directory: {directory}")


if __name__ == "__main__":
    # When run directly, create all directories and print configuration
    print("LCDNet Configuration")
    print("=" * 50)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Device: {DEVICE}")
    print(f"Image Size: {IMAGE_SIZE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print("=" * 50)
    create_directories()
