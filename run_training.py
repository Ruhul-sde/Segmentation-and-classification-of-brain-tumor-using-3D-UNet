
#!/usr/bin/env python3
"""
Simple script to run brain tumor segmentation training
"""

import os
import sys
from config import Config, FastTraining, HighQuality, LightWeight

def main():
    print("Brain Tumor Segmentation Training")
    print("=" * 40)
    
    # Choose configuration
    print("Select training configuration:")
    print("1. Fast Training (for testing)")
    print("2. Standard Training")
    print("3. High Quality Training")
    print("4. Lightweight Model")
    
    choice = input("Enter choice (1-4) [default: 2]: ").strip()
    
    if choice == "1":
        config = FastTraining
        print("Using Fast Training configuration")
    elif choice == "3":
        config = HighQuality
        print("Using High Quality configuration")
    elif choice == "4":
        config = LightWeight
        print("Using Lightweight configuration")
    else:
        config = Config
        print("Using Standard configuration")
    
    # Print configuration
    config.print_config()
    
    # Create directories
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
    
    # Run training with synthetic data
    print("\nStarting training with synthetic data...")
    
    try:
        from train_model import main as train_main
        import argparse
        
        # Create args for train_model
        sys.argv = [
            'train_model.py',
            '--create_synthetic',
            '--num_samples', '100',
            '--epochs', str(config.EPOCHS),
            '--batch_size', str(config.BATCH_SIZE),
            '--lr', str(config.LEARNING_RATE)
        ]
        
        train_main()
        
    except Exception as e:
        print(f"Error during training: {e}")
        print("Make sure all dependencies are installed correctly.")
        print("You can also run: python train_model.py --create_synthetic")

if __name__ == "__main__":
    main()
