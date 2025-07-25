
#!/usr/bin/env python3
"""
Training script for brain tumor segmentation model
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import argparse
import logging
from main import UNet3D, ResidualConv3D, AttentionGate3D
from training import BrainTumorTrainer, BrainTumorDataset, create_data_transforms

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_synthetic_data(num_samples=100, save_dir="synthetic_data"):
    """Create synthetic brain tumor data for training demonstration"""
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/images", exist_ok=True)
    os.makedirs(f"{save_dir}/masks", exist_ok=True)
    
    image_paths = []
    mask_paths = []
    
    for i in range(num_samples):
        # Create synthetic 3D brain image (128x128x128)
        image = np.random.randn(128, 128, 128) * 0.1 + 0.5
        
        # Add some brain-like structure
        center = (64, 64, 64)
        xx, yy, zz = np.meshgrid(
            np.arange(128) - center[0],
            np.arange(128) - center[1], 
            np.arange(128) - center[2],
            indexing='ij'
        )
        brain_mask = (xx**2 + yy**2 + zz**2) < 50**2
        image[brain_mask] += 0.3
        
        # Create synthetic tumor
        mask = np.zeros((128, 128, 128), dtype=np.uint8)
        if i % 2 == 0:  # 50% have tumors
            tumor_center = (
                np.random.randint(40, 88),
                np.random.randint(40, 88),
                np.random.randint(40, 88)
            )
            tumor_size = np.random.randint(8, 20)
            
            xx_t, yy_t, zz_t = np.meshgrid(
                np.arange(128) - tumor_center[0],
                np.arange(128) - tumor_center[1],
                np.arange(128) - tumor_center[2],
                indexing='ij'
            )
            tumor_mask = (xx_t**2 + yy_t**2 + zz_t**2) < tumor_size**2
            
            # Multi-class tumor (necrotic core, edema, enhancing tumor)
            mask[tumor_mask] = 1  # Background tumor
            core_mask = (xx_t**2 + yy_t**2 + zz_t**2) < (tumor_size*0.6)**2
            mask[core_mask] = 2  # Necrotic core
            enhancing_mask = (xx_t**2 + yy_t**2 + zz_t**2) < (tumor_size*0.3)**2
            mask[enhancing_mask] = 3  # Enhancing tumor
            
            # Add tumor signal to image
            image[tumor_mask] += 0.4
            image[core_mask] -= 0.2
            image[enhancing_mask] += 0.6
        
        # Save as .npy files (simulating NIfTI data)
        image_path = f"{save_dir}/images/brain_{i:03d}.npy"
        mask_path = f"{save_dir}/masks/mask_{i:03d}.npy"
        
        np.save(image_path, image.astype(np.float32))
        np.save(mask_path, mask.astype(np.uint8))
        
        image_paths.append(image_path)
        mask_paths.append(mask_path)
    
    logger.info(f"Created {num_samples} synthetic samples in {save_dir}")
    return image_paths, mask_paths

class SyntheticDataset(torch.utils.data.Dataset):
    """Dataset for synthetic data (numpy arrays)"""
    
    def __init__(self, image_paths, mask_paths, augment=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.augment = augment
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load numpy arrays
        image = np.load(self.image_paths[idx])
        mask = np.load(self.mask_paths[idx])
        
        # Simple augmentation
        if self.augment and np.random.rand() > 0.5:
            # Random flip
            if np.random.rand() > 0.5:
                image = np.flip(image, axis=0).copy()
                mask = np.flip(mask, axis=0).copy()
            if np.random.rand() > 0.5:
                image = np.flip(image, axis=1).copy()
                mask = np.flip(mask, axis=1).copy()
            
            # Add noise
            image += np.random.normal(0, 0.05, image.shape)
        
        # Normalize image
        image = (image - np.mean(image)) / (np.std(image) + 1e-8)
        
        # Convert to tensors
        image = torch.from_numpy(image).float().unsqueeze(0)  # Add channel dim
        mask = torch.from_numpy(mask).long()
        
        return {"image": image, "mask": mask}

def main():
    parser = argparse.ArgumentParser(description='Train Brain Tumor Segmentation Model')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--data_dir', type=str, default='synthetic_data', help='Data directory')
    parser.add_argument('--create_synthetic', action='store_true', help='Create synthetic data')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of synthetic samples')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create or load data
    if args.create_synthetic:
        logger.info("Creating synthetic data...")
        image_paths, mask_paths = create_synthetic_data(args.num_samples, args.data_dir)
    else:
        # Load existing data paths
        data_dir = Path(args.data_dir)
        image_paths = sorted(list((data_dir / "images").glob("*.npy")))
        mask_paths = sorted(list((data_dir / "masks").glob("*.npy")))
        
        if len(image_paths) == 0:
            logger.error("No data found. Use --create_synthetic to generate synthetic data.")
            return
    
    # Split data
    split_idx = int(0.8 * len(image_paths))
    train_images = image_paths[:split_idx]
    train_masks = mask_paths[:split_idx]
    val_images = image_paths[split_idx:]
    val_masks = mask_paths[split_idx:]
    
    logger.info(f"Training samples: {len(train_images)}, Validation samples: {len(val_images)}")
    
    # Create datasets and loaders
    train_dataset = SyntheticDataset(train_images, train_masks, augment=True)
    val_dataset = SyntheticDataset(val_images, val_masks, augment=False)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=2, 
        pin_memory=True if device.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Create model
    model = UNet3D(in_channels=1, out_channels=4)
    
    # Initialize trainer
    trainer = BrainTumorTrainer(model, device, args.lr)
    
    # Start training
    logger.info("Starting training...")
    trainer.train(train_loader, val_loader, args.epochs, "best_brain_tumor_model.pth")
    
    # Plot training history
    trainer.plot_training_history()
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()
