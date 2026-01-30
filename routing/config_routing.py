"""
Configuration for Hybrid Routing System.

This file contains all settings for the routing classifier and hybrid inference.
Modify these values to experiment with different configurations.

Author: Thesis Implementation - Phase 2 Part 3
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ============================================================================
# ROUTING THRESHOLD
# ============================================================================

# Routing decision threshold based on ground truth count
# If GT count <= ROUTING_THRESHOLD -> route to LCDNet (label=0)
# If GT count > ROUTING_THRESHOLD -> route to CSRNet (label=1)
#
# Justification for T=100:
# - Literature considers <100 as "sparse" crowds
# - LCDNet excels at low-medium density (ShanghaiTech style)
# - CSRNet excels at high density (NWPU-Crowd dense scenes)
# - Creates balanced training split for router
ROUTING_THRESHOLD = 100

# ============================================================================
# ROUTER MODEL CONFIGURATION
# ============================================================================

# Router input size (MobileNetV2 standard)
ROUTER_INPUT_SIZE = (224, 224)

# Number of output classes (0=LCDNet, 1=CSRNet)
NUM_CLASSES = 2

# Dropout rate in classification head
ROUTER_DROPOUT = 0.3

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================

# Number of training epochs
ROUTER_EPOCHS = 25

# Batch size for router training
ROUTER_BATCH_SIZE = 32

# Learning rate for Adam optimizer
ROUTER_LR = 1e-4

# Weight decay for regularization
ROUTER_WEIGHT_DECAY = 1e-5

# Learning rate scheduler patience
ROUTER_LR_PATIENCE = 5

# Learning rate reduction factor
ROUTER_LR_FACTOR = 0.5

# Early stopping patience
ROUTER_EARLY_STOPPING = 10

# ============================================================================
# CHECKPOINT PATHS
# ============================================================================

# Router checkpoint directory
ROUTER_CHECKPOINT_DIR = os.path.join(config.CHECKPOINTS_DIR, "router")

# Router best model path
ROUTER_BEST_PATH = os.path.join(ROUTER_CHECKPOINT_DIR, "router_best.pth")

# LCDNet checkpoint path (trained on ShanghaiTech)
LCDNET_CHECKPOINT = os.path.join(config.CHECKPOINTS_DIR, "best_model.pth")

# CSRNet checkpoint path (trained on NWPU-Crowd)
CSRNET_CHECKPOINT = os.path.join(config.CHECKPOINTS_DIR, "csrnet", "csrnet_best.pth")

# ============================================================================
# DATASET PATHS (NWPU-Crowd)
# ============================================================================

NWPU_DATA_DIR = os.path.join(config.DATA_DIR, "NWPU-Crowd")
NWPU_JSONS_DIR = os.path.join(NWPU_DATA_DIR, "jsons")
NWPU_TRAIN_TXT = os.path.join(NWPU_DATA_DIR, "train.txt")
NWPU_VAL_TXT = os.path.join(NWPU_DATA_DIR, "val.txt")
NWPU_TEST_TXT = os.path.join(NWPU_DATA_DIR, "test.txt")

# Image directories (NWPU stores images in parts)
NWPU_IMAGE_DIRS = [
    os.path.join(NWPU_DATA_DIR, f"images_part{i}") for i in range(1, 6)
]

# ============================================================================
# LOGGING
# ============================================================================

ROUTER_LOG_DIR = config.LOGS_DIR
ROUTER_TRAINING_LOG = os.path.join(ROUTER_LOG_DIR, "router_training.log")
HYBRID_EVAL_RESULTS = os.path.join(ROUTER_LOG_DIR, "hybrid_evaluation_results.txt")

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================

DEVICE = config.DEVICE
NUM_WORKERS = config.NUM_WORKERS

# ============================================================================
# DENSITY MODEL INPUT SIZE (for hybrid inference)
# ============================================================================

DENSITY_INPUT_SIZE = config.IMAGE_SIZE  # (384, 384)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_routing_directories():
    """Create all necessary directories for routing system."""
    directories = [ROUTER_CHECKPOINT_DIR, ROUTER_LOG_DIR]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def get_routing_label(gt_count: float) -> int:
    """
    Get routing label from ground truth count.
    
    Args:
        gt_count: Ground truth person count.
        
    Returns:
        0 for LCDNet (sparse), 1 for CSRNet (dense).
    """
    return 0 if gt_count <= ROUTING_THRESHOLD else 1


def get_model_name(label: int) -> str:
    """Get model name from routing label."""
    return "LCDNet" if label == 0 else "CSRNet"


if __name__ == "__main__":
    print("Routing Configuration")
    print("=" * 50)
    print(f"Routing Threshold: {ROUTING_THRESHOLD}")
    print(f"Router Input Size: {ROUTER_INPUT_SIZE}")
    print(f"Device: {DEVICE}")
    print(f"LCDNet Checkpoint: {LCDNET_CHECKPOINT}")
    print(f"CSRNet Checkpoint: {CSRNET_CHECKPOINT}")
    print("=" * 50)
    create_routing_directories()
