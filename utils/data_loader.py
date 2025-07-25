
"""
Data loading utilities for brain tumor segmentation
"""

import os
import numpy as np
import nibabel as nib
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from scipy import ndimage
import matplotlib.pyplot as plt

class BrainTumorDataset(Dataset):
    """Dataset class for brain tumor segmentation"""
    
    def __init__(self, data_dir, transform=None, augment=False):
        self.data_dir = data_dir
        self.transform = transform
        self.augment = augment
        self.image_files = self._get_image_files()
    
    def _get_image_files(self):
        """Get list of image files from data directory"""
        image_files = []
        if os.path.exists(self.data_dir):
            for file in os.listdir(self.data_dir):
                if file.endswith(('.nii', '.nii.gz', '.png', '.jpg', '.jpeg')):
                    image_files.append(os.path.join(self.data_dir, file))
        return image_files
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        
        # Load image
        if image_path.endswith(('.nii', '.nii.gz')):
            image = self._load_nifti(image_path)
        else:
            image = self._load_standard_image(image_path)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Create dummy segmentation for demonstration
        segmentation = np.zeros_like(image, dtype=np.int64)
        
        return {
            'image': torch.FloatTensor(image).unsqueeze(0),
            'segmentation': torch.LongTensor(segmentation),
            'path': image_path
        }
    
    def _load_nifti(self, path):
        """Load NIfTI image"""
        try:
            img = nib.load(path)
            data = img.get_fdata()
            return self._preprocess_3d(data)
        except Exception as e:
            print(f"Error loading NIfTI file {path}: {e}")
            return np.zeros((128, 128, 128))
    
    def _load_standard_image(self, path):
        """Load standard image formats"""
        try:
            img = Image.open(path).convert('L')
            data = np.array(img)
            # Create 3D volume from 2D image
            data_3d = np.stack([data] * 128, axis=-1)
            return self._preprocess_3d(data_3d)
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            return np.zeros((128, 128, 128))
    
    def _preprocess_3d(self, data):
        """Preprocess 3D image data"""
        # Normalize
        if data.max() > data.min():
            data = (data - data.min()) / (data.max() - data.min())
        
        # Resize to standard size
        target_shape = (128, 128, 128)
        if data.shape != target_shape:
            zoom_factors = [t/s for t, s in zip(target_shape, data.shape)]
            data = ndimage.zoom(data, zoom_factors, order=1)
        
        return data.astype(np.float32)

def create_synthetic_data(output_dir, num_samples=10):
    """Create synthetic brain tumor data for testing"""
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(num_samples):
        # Create synthetic brain volume
        brain = np.random.randn(128, 128, 128) * 0.1 + 0.5
        
        # Add tumor region
        center = np.random.randint(32, 96, 3)
        radius = np.random.randint(10, 25)
        
        xx, yy, zz = np.meshgrid(
            np.arange(128) - center[0],
            np.arange(128) - center[1],
            np.arange(128) - center[2],
            indexing='ij'
        )
        
        tumor_mask = (xx**2 + yy**2 + zz**2) < radius**2
        brain[tumor_mask] += np.random.uniform(0.3, 0.7)
        
        # Clip values
        brain = np.clip(brain, 0, 1)
        
        # Save as numpy array
        np.save(os.path.join(output_dir, f'synthetic_brain_{i:03d}.npy'), brain)
    
    print(f"Created {num_samples} synthetic brain volumes in {output_dir}")

def get_data_loader(data_dir, batch_size=2, shuffle=True, num_workers=0):
    """Create data loader for training/testing"""
    dataset = BrainTumorDataset(data_dir)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
