
"""
Configuration file for brain tumor segmentation training
"""

import torch

class Config:
    # Model parameters
    MODEL_NAME = "UNet3D_Enhanced"
    IN_CHANNELS = 1
    OUT_CHANNELS = 4  # Background, Necrotic Core, Peritumoral Edema, Enhancing Tumor
    FEATURES = [32, 64, 128, 256, 512]  # Enhanced feature hierarchy
    USE_DEEP_SUPERVISION = True
    USE_ATTENTION = True
    DROPOUT_RATE = 0.2
    
    # Training parameters
    EPOCHS = 100
    BATCH_SIZE = 2  # Adjust based on GPU memory
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    
    # Data parameters
    IMAGE_SIZE = (128, 128, 128)
    TRAIN_VAL_SPLIT = 0.2
    NUM_WORKERS = 4
    PIN_MEMORY = True
    
    # Augmentation parameters
    AUGMENT_PROB = 0.5
    ROTATION_LIMIT = 15
    NOISE_VARIANCE = (10.0, 50.0)
    BRIGHTNESS_CONTRAST_LIMIT = 0.2
    
    # Loss function parameters
    DICE_WEIGHT = 0.5
    FOCAL_WEIGHT = 0.5
    FOCAL_ALPHA = 0.25
    FOCAL_GAMMA = 2.0
    
    # Optimization parameters
    OPTIMIZER = "AdamW"
    SCHEDULER = "ReduceLROnPlateau"
    SCHEDULER_PATIENCE = 10
    SCHEDULER_FACTOR = 0.5
    
    # Mixed precision training
    USE_AMP = torch.cuda.is_available()
    
    # Regularization
    DROPOUT_RATE = 0.2
    BATCH_NORM_MOMENTUM = 0.1
    
    # Early stopping
    EARLY_STOPPING_PATIENCE = 20
    EARLY_STOPPING_MIN_DELTA = 1e-4
    
    # Checkpointing
    SAVE_BEST_MODEL = True
    SAVE_CHECKPOINT_EVERY = 10
    CHECKPOINT_DIR = "checkpoints"
    
    # Evaluation metrics
    METRICS = ["dice", "hausdorff", "sensitivity", "specificity"]
    
    # Class names
    CLASS_NAMES = ["Background", "Necrotic Core", "Peritumoral Edema", "Enhancing Tumor"]
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Logging
    LOG_LEVEL = "INFO"
    LOG_FILE = "training.log"
    TENSORBOARD_LOG_DIR = "runs"
    
    # Data paths
    DATA_DIR = "data"
    SYNTHETIC_DATA_DIR = "synthetic_data"
    MODEL_SAVE_PATH = "models"
    
    @classmethod
    def get_config_dict(cls):
        """Return configuration as dictionary"""
        return {
            attr: getattr(cls, attr) 
            for attr in dir(cls) 
            if not attr.startswith('_') and not callable(getattr(cls, attr))
        }
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("=" * 50)
        print("TRAINING CONFIGURATION")
        print("=" * 50)
        config_dict = cls.get_config_dict()
        for key, value in config_dict.items():
            if not key.startswith('get_') and not key.startswith('print_'):
                print(f"{key}: {value}")
        print("=" * 50)

# Hyperparameter configurations for different scenarios
class FastTraining(Config):
    """Configuration for fast training (debugging/testing)"""
    EPOCHS = 10
    BATCH_SIZE = 1
    IMAGE_SIZE = (64, 64, 64)
    FEATURES = [16, 32, 64]

class HighQuality(Config):
    """Configuration for high-quality training"""
    EPOCHS = 200
    BATCH_SIZE = 1  # Large models need smaller batch sizes
    FEATURES = [64, 128, 256, 512]
    LEARNING_RATE = 5e-5

class LightWeight(Config):
    """Configuration for lightweight model"""
    FEATURES = [16, 32, 64, 128]
    BATCH_SIZE = 4
    LEARNING_RATE = 2e-4
