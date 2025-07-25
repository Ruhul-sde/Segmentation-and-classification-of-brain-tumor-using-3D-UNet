
"""
Enhanced configuration file for BraTS 2024 brain tumor segmentation training
"""

import torch
import os
from pathlib import Path

class Config:
    """Base configuration for BraTS 2024 training"""
    
    # Model parameters
    MODEL_NAME = "Enhanced_UNet3D"
    IN_CHANNELS = 4  # BraTS modalities: T1c, T1n, T2f, T2w
    OUT_CHANNELS = 4  # Background, Necrotic, Edema, Enhancing
    FEATURES = [32, 64, 128, 256, 512]
    
    # Training parameters
    EPOCHS = 100
    BATCH_SIZE = 2
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    
    # Data parameters
    IMAGE_SIZE = (128, 128, 128)  # Resized for memory efficiency
    NUM_WORKERS = 4
    PIN_MEMORY = True
    CACHE_RATE = 0.5  # Percentage of data to cache
    
    # Augmentation parameters
    AUGMENTATION_PROB = 0.5
    ROTATION_RANGE = 15
    FLIP_PROB = 0.5
    NOISE_STD = 0.1
    INTENSITY_SCALE = (0.9, 1.1)
    
    # Loss function weights
    LOSS_WEIGHTS = {
        'dice': 0.5,
        'cross_entropy': 0.3,
        'focal': 0.2
    }
    
    # Optimizer parameters
    OPTIMIZER = "AdamW"
    BETAS = (0.9, 0.999)
    EPS = 1e-8
    
    # Scheduler parameters
    SCHEDULER = "CosineAnnealingWarmRestarts"
    T_0 = 10
    T_MULT = 2
    ETA_MIN = 1e-6
    
    # Early stopping
    EARLY_STOPPING_PATIENCE = 20
    MONITOR_METRIC = "val_dice"
    
    # Paths
    DATA_ROOT = "data/BraTS2024"
    CHECKPOINT_DIR = "results/checkpoints"
    MODEL_SAVE_PATH = "results/models"
    LOG_DIR = "results/logs"
    VISUALIZATION_DIR = "results/visualizations"
    REPORT_DIR = "results/reports"
    
    # Experiment tracking
    USE_WANDB = True
    USE_TENSORBOARD = True
    PROJECT_NAME = "brain-tumor-segmentation-brats2024"
    
    # Device configuration
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MIXED_PRECISION = True if torch.cuda.is_available() else False
    
    # Validation parameters
    VAL_INTERVAL = 1
    VAL_SPLIT = 0.2
    
    # Metrics to track
    METRICS = [
        'dice_score',
        'hausdorff_distance',
        'sensitivity',
        'specificity',
        'jaccard_index',
        'volume_similarity'
    ]
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        dirs = [
            cls.CHECKPOINT_DIR,
            cls.MODEL_SAVE_PATH,
            cls.LOG_DIR,
            cls.VISUALIZATION_DIR,
            cls.REPORT_DIR
        ]
        
        for directory in dirs:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def print_config(cls):
        """Print configuration summary"""
        print("=" * 50)
        print("BraTS 2024 Training Configuration")
        print("=" * 50)
        print(f"Model: {cls.MODEL_NAME}")
        print(f"Input Channels: {cls.IN_CHANNELS}")
        print(f"Output Channels: {cls.OUT_CHANNELS}")
        print(f"Image Size: {cls.IMAGE_SIZE}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Learning Rate: {cls.LEARNING_RATE}")
        print(f"Epochs: {cls.EPOCHS}")
        print(f"Device: {cls.DEVICE}")
        print(f"Mixed Precision: {cls.MIXED_PRECISION}")
        print(f"Data Root: {cls.DATA_ROOT}")
        print("=" * 50)

class FastTraining(Config):
    """Fast training configuration for testing"""
    EPOCHS = 20
    BATCH_SIZE = 1
    IMAGE_SIZE = (64, 64, 64)
    NUM_WORKERS = 2
    CACHE_RATE = 0.1
    VAL_INTERVAL = 5
    EARLY_STOPPING_PATIENCE = 10

class HighQuality(Config):
    """High quality training configuration"""
    EPOCHS = 300
    BATCH_SIZE = 1  # Larger images require smaller batch
    IMAGE_SIZE = (192, 192, 128)
    LEARNING_RATE = 5e-5
    FEATURES = [64, 128, 256, 512, 1024]
    CACHE_RATE = 0.8
    EARLY_STOPPING_PATIENCE = 50
    
    # Enhanced augmentation
    AUGMENTATION_PROB = 0.8
    ROTATION_RANGE = 20
    NOISE_STD = 0.05

class LightWeight(Config):
    """Lightweight model configuration"""
    FEATURES = [16, 32, 64, 128, 256]
    BATCH_SIZE = 4
    LEARNING_RATE = 2e-4
    IMAGE_SIZE = (96, 96, 96)

class ProductionConfig(Config):
    """Production-ready configuration"""
    EPOCHS = 150
    BATCH_SIZE = 2
    IMAGE_SIZE = (128, 128, 128)
    MIXED_PRECISION = True
    CACHE_RATE = 0.6
    
    # Robust training settings
    EARLY_STOPPING_PATIENCE = 30
    LEARNING_RATE = 8e-5
    
    # Enhanced loss weighting
    LOSS_WEIGHTS = {
        'dice': 0.6,
        'cross_entropy': 0.25,
        'focal': 0.15
    }

# BraTS-specific constants
BRATS_MODALITIES = ['t1c', 't1n', 't2f', 't2w']
BRATS_LABELS = {
    0: 'Background',
    1: 'Necrotic/Non-enhancing tumor core', 
    2: 'Peritumoral edema/Invaded tissue',
    4: 'GD-enhancing tumor'  # Note: BraTS uses label 4, not 3
}

BRATS_REGIONS = {
    'WT': [1, 2, 4],  # Whole tumor
    'TC': [1, 4],     # Tumor core  
    'ET': [4]         # Enhancing tumor
}

# Color scheme for visualization
BRATS_COLORS = {
    0: (0, 0, 0, 0),         # Background - transparent
    1: (255, 0, 0, 180),     # Necrotic - red
    2: (0, 255, 0, 180),     # Edema - green
    4: (0, 0, 255, 180)      # Enhancing - blue
}
