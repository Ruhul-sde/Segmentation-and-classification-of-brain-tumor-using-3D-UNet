
"""
Environment configuration and validation
"""

import os
import sys
import torch
import numpy as np
import logging
from datetime import datetime

def setup_environment():
    """Setup the development environment"""
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Setup logging
    setup_logging()
    
    # Validate environment
    validate_dependencies()
    
    return device

def setup_logging():
    """Setup logging configuration"""
    os.makedirs('logs', exist_ok=True)
    
    log_filename = f"logs/app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Logging initialized")
    
    return logger

def validate_dependencies():
    """Validate that all required dependencies are available"""
    required_packages = {
        'numpy': 'numpy',
        'torch': 'torch',
        'torchvision': 'torchvision',
        'matplotlib': 'matplotlib.pyplot',
        'plotly': 'plotly.graph_objects',
        'nibabel': 'nibabel',
        'sklearn': 'sklearn',
        'cv2': 'cv2',
        'PIL': 'PIL',
        'flask': 'flask',
        'scipy': 'scipy',
        'pandas': 'pandas'
    }
    
    missing_packages = []
    available_packages = []
    
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
            available_packages.append(package_name)
        except ImportError:
            missing_packages.append(package_name)
    
    print(f"\n✓ Available packages ({len(available_packages)}): {', '.join(available_packages)}")
    
    if missing_packages:
        print(f"\n✗ Missing packages ({len(missing_packages)}): {', '.join(missing_packages)}")
        print("Please install missing packages using the Dependencies tool in Replit")
        return False
    else:
        print("\n✓ All required dependencies are available!")
        return True

def get_system_info():
    """Get system information"""
    info = {
        'python_version': sys.version,
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'numpy_version': np.__version__,
        'platform': sys.platform
    }
    
    return info

def print_system_info():
    """Print system information"""
    info = get_system_info()
    
    print("\n" + "="*50)
    print("SYSTEM INFORMATION")
    print("="*50)
    
    for key, value in info.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    
    print("="*50)

if __name__ == "__main__":
    print("Setting up environment...")
    device = setup_environment()
    print_system_info()
    print(f"\nEnvironment setup complete! Using device: {device}")
