
import numpy as np
import torch
import nibabel as nib
from pathlib import Path
import cv2
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

def load_nifti_volume(file_path):
    """Load NIfTI volume and return numpy array"""
    try:
        nii_img = nib.load(file_path)
        volume = nii_img.get_fdata()
        return volume.astype(np.float32)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def preprocess_volume(volume, target_shape=(128, 128, 128)):
    """Preprocess volume: normalize and resize"""
    # Normalize
    volume = (volume - np.mean(volume)) / (np.std(volume) + 1e-8)
    
    # Resize if needed
    if volume.shape != target_shape:
        # Simple resize (you might want to use more sophisticated methods)
        volume = np.resize(volume, target_shape)
    
    return volume

def augment_volume_3d(volume, mask=None):
    """Apply 3D augmentations"""
    # Random rotation
    if np.random.rand() > 0.5:
        k = np.random.randint(1, 4)
        volume = np.rot90(volume, k, axes=(0, 1))
        if mask is not None:
            mask = np.rot90(mask, k, axes=(0, 1))
    
    # Random flip
    for axis in [0, 1, 2]:
        if np.random.rand() > 0.5:
            volume = np.flip(volume, axis=axis)
            if mask is not None:
                mask = np.flip(mask, axis=axis)
    
    # Add noise
    noise = np.random.normal(0, 0.05, volume.shape)
    volume = volume + noise
    
    # Intensity scaling
    scale = np.random.uniform(0.9, 1.1)
    volume = volume * scale
    
    if mask is not None:
        return volume.copy(), mask.copy()
    return volume.copy()

def create_2d_slices_from_3d(volume, mask=None, num_slices=32):
    """Extract 2D slices from 3D volume for 2D training"""
    depth = volume.shape[2]
    slice_indices = np.linspace(0, depth-1, num_slices, dtype=int)
    
    slices = []
    mask_slices = []
    
    for idx in slice_indices:
        slice_2d = volume[:, :, idx]
        slices.append(slice_2d)
        
        if mask is not None:
            mask_slice = mask[:, :, idx]
            mask_slices.append(mask_slice)
    
    if mask is not None:
        return np.array(slices), np.array(mask_slices)
    return np.array(slices)

def get_2d_augmentations():
    """Get 2D augmentation pipeline"""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.RandomBrightnessContrast(p=0.3),
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
        A.Normalize(mean=0.0, std=1.0),
        ToTensorV2()
    ])

def calculate_class_weights(masks):
    """Calculate class weights for imbalanced datasets"""
    unique, counts = np.unique(masks, return_counts=True)
    total_pixels = np.sum(counts)
    
    weights = {}
    for class_id, count in zip(unique, counts):
        weights[class_id] = total_pixels / (len(unique) * count)
    
    return weights

def validate_data_integrity(image_paths, mask_paths):
    """Validate that all images and masks can be loaded and have correct shapes"""
    valid_pairs = []
    
    for img_path, mask_path in zip(image_paths, mask_paths):
        try:
            if img_path.endswith('.npy'):
                img = np.load(img_path)
                mask = np.load(mask_path)
            else:
                img = load_nifti_volume(img_path)
                mask = load_nifti_volume(mask_path)
            
            if img is not None and mask is not None:
                if img.shape == mask.shape:
                    valid_pairs.append((img_path, mask_path))
                else:
                    print(f"Shape mismatch: {img_path} ({img.shape}) vs {mask_path} ({mask.shape})")
            else:
                print(f"Failed to load: {img_path} or {mask_path}")
        
        except Exception as e:
            print(f"Error validating {img_path}: {e}")
    
    return valid_pairs

def create_train_val_split(image_paths, mask_paths, val_split=0.2, random_state=42):
    """Create train/validation split"""
    # Validate data first
    valid_pairs = validate_data_integrity(image_paths, mask_paths)
    
    if len(valid_pairs) == 0:
        raise ValueError("No valid image-mask pairs found!")
    
    images, masks = zip(*valid_pairs)
    
    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        images, masks, test_size=val_split, random_state=random_state
    )
    
    return list(train_imgs), list(val_imgs), list(train_masks), list(val_masks)

def get_dataset_statistics(image_paths):
    """Calculate dataset statistics"""
    all_intensities = []
    shapes = []
    
    for path in image_paths:
        try:
            if path.endswith('.npy'):
                img = np.load(path)
            else:
                img = load_nifti_volume(path)
            
            if img is not None:
                all_intensities.extend(img.flatten())
                shapes.append(img.shape)
        except Exception as e:
            print(f"Error processing {path}: {e}")
    
    all_intensities = np.array(all_intensities)
    
    stats = {
        'mean': np.mean(all_intensities),
        'std': np.std(all_intensities),
        'min': np.min(all_intensities),
        'max': np.max(all_intensities),
        'unique_shapes': list(set(shapes)),
        'total_samples': len(image_paths)
    }
    
    return stats
