
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import nibabel as nib
from pathlib import Path
import albumentations as A
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
from monai.losses import DiceLoss, FocalLoss
from losses import CombinedLoss3D, DeepSupervisionLoss3D
from monai.metrics import DiceMetric
from monai.transforms import (
    Compose, LoadImage, AddChannel, Spacing, Orientation,
    ScaleIntensity, RandSpatialCrop, RandRotate90, RandFlip,
    ToTensor, EnsureChannelFirst
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BrainTumorDataset(Dataset):
    """Optimized dataset for brain tumor segmentation"""
    
    def __init__(self, image_paths, mask_paths, transforms=None, cache_data=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transforms = transforms
        self.cache_data = cache_data
        self.cached_data = {}
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        if self.cache_data and idx in self.cached_data:
            return self.cached_data[idx]
            
        # Load image and mask
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        # Load NIfTI files
        image = nib.load(image_path).get_fdata()
        mask = nib.load(mask_path).get_fdata()
        
        # Normalize image
        image = (image - np.mean(image)) / (np.std(image) + 1e-8)
        
        # Convert to tensor
        image = torch.from_numpy(image).float().unsqueeze(0)  # Add channel dim
        mask = torch.from_numpy(mask).long()
        
        if self.transforms:
            # Apply transforms (if using MONAI transforms)
            data_dict = {"image": image, "label": mask}
            data_dict = self.transforms(data_dict)
            image, mask = data_dict["image"], data_dict["label"]
        
        sample = {"image": image, "mask": mask}
        
        if self.cache_data:
            self.cached_data[idx] = sample
            
        return sample

class ImprovedUNet3D(nn.Module):
    """Improved 3D U-Net with attention and residual connections"""
    
    def __init__(self, in_channels=1, out_channels=4, features=[32, 64, 128, 256, 512]):
        super(ImprovedUNet3D, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout3d(0.2)
        
        # Down part of U-Net with residual connections
        for feature in features:
            self.downs.append(ResidualBlock3D(in_channels, feature))
            in_channels = feature
            
        # Up part of U-Net with attention
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose3d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(AttentionBlock3D(feature*2, feature))
            self.ups.append(ResidualBlock3D(feature*2, feature))
            
        self.bottleneck = ResidualBlock3D(features[-1], features[-1]*2)
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)
        
    def forward(self, x):
        skip_connections = []
        
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
            x = self.dropout(x)
            
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        for idx in range(0, len(self.ups), 3):
            x = self.ups[idx](x)  # Upsampling
            skip_connection = skip_connections[idx//3]
            
            # Apply attention
            x = self.ups[idx+1](x, skip_connection)
            
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='trilinear', align_corners=False)
                
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+2](concat_skip)  # Residual block
            
        return self.final_conv(x)

class ResidualBlock3D(nn.Module):
    """3D Residual block with batch normalization"""
    
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm3d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out

class AttentionBlock3D(nn.Module):
    """3D Attention block for U-Net"""
    
    def __init__(self, F_g, F_l):
        super(AttentionBlock3D, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_l, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_l)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_l, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_l)
        )
        
        self.psi = nn.Sequential(
            nn.Conv3d(F_l, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class CombinedLoss(nn.Module):
    """Enhanced combined loss function for better segmentation"""
    
    def __init__(self, alpha=0.5, beta=0.3, gamma=0.2):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.dice_loss = DiceLoss(to_onehot_y=True, softmax=True)
        self.focal_loss = FocalLoss(to_onehot_y=True)
        
    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        
        # Add boundary loss for better edge detection
        pred_soft = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 4, 1, 2, 3).float()
        
        # Simple boundary loss using gradients
        grad_pred = torch.abs(pred_soft[:, :, 1:, :, :] - pred_soft[:, :, :-1, :, :]).mean()
        grad_target = torch.abs(target_one_hot[:, :, 1:, :, :] - target_one_hot[:, :, :-1, :, :]).mean()
        boundary_loss = F.mse_loss(grad_pred, grad_target)
        
        return self.alpha * dice + self.beta * focal + self.gamma * boundary_loss

class BrainTumorTrainer:
    """Optimized trainer for brain tumor segmentation"""
    
    def __init__(self, model, device, learning_rate=1e-4):
        self.model = model.to(device)
        self.device = device
        self.criterion = CombinedLoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        self.scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
        self.best_dice = 0.0
        self.train_losses = []
        self.val_losses = []
        self.dice_scores = []
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc="Training")
        for batch_idx, batch in enumerate(progress_bar):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        dice_metric = DiceMetric(include_background=True, reduction="mean")
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, masks)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                
                total_loss += loss.item()
                
                # Calculate Dice score
                predictions = torch.argmax(outputs, dim=1, keepdim=True)
                dice_metric(y_pred=predictions, y=masks.unsqueeze(1))
        
        avg_loss = total_loss / len(val_loader)
        dice_score = dice_metric.aggregate().item()
        dice_metric.reset()
        
        return avg_loss, dice_score
    
    def train(self, train_loader, val_loader, epochs, save_path="best_model.pth"):
        logger.info(f"Starting training for {epochs} epochs")
        
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch+1}/{epochs}")
            
            # Training
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss, dice_score = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.dice_scores.append(dice_score)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Dice: {dice_score:.4f}")
            
            # Save best model
            if dice_score > self.best_dice:
                self.best_dice = dice_score
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'dice_score': dice_score,
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                    'dice_scores': self.dice_scores
                }, save_path)
                logger.info(f"New best model saved with Dice: {dice_score:.4f}")
    
    def plot_training_history(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Training Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Dice score plot
        ax2.plot(self.dice_scores, label='Dice Score', color='green')
        ax2.set_title('Validation Dice Score')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Dice Score')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

def create_data_transforms():
    """Create data augmentation transforms"""
    train_transforms = Compose([
        LoadImage(keys=["image", "label"]),
        AddChannel(keys=["image", "label"]),
        Spacing(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        Orientation(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensity(keys=["image"]),
        RandSpatialCrop(keys=["image", "label"], roi_size=(128, 128, 128), random_size=False),
        RandRotate90(keys=["image", "label"], prob=0.5, spatial_axes=[0, 2]),
        RandFlip(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlip(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlip(keys=["image", "label"], prob=0.5, spatial_axis=2),
        ToTensor(keys=["image", "label"])
    ])
    
    val_transforms = Compose([
        LoadImage(keys=["image", "label"]),
        AddChannel(keys=["image", "label"]),
        Spacing(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        Orientation(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensity(keys=["image"]),
        RandSpatialCrop(keys=["image", "label"], roi_size=(128, 128, 128), random_size=False),
        ToTensor(keys=["image", "label"])
    ])
    
    return train_transforms, val_transforms

def train_model(data_dir, epochs=100, batch_size=2, learning_rate=1e-4):
    """Main training function"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create model
    model = ImprovedUNet3D(in_channels=1, out_channels=4)
    
    # Get data paths (you'll need to implement this based on your data structure)
    # image_paths, mask_paths = get_data_paths(data_dir)
    # train_images, val_images, train_masks, val_masks = train_test_split(
    #     image_paths, mask_paths, test_size=0.2, random_state=42
    # )
    
    # Create data loaders
    # train_transforms, val_transforms = create_data_transforms()
    # train_dataset = BrainTumorDataset(train_images, train_masks, train_transforms)
    # val_dataset = BrainTumorDataset(val_images, val_masks, val_transforms)
    
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Initialize trainer
    trainer = BrainTumorTrainer(model, device, learning_rate)
    
    # Start training
    # trainer.train(train_loader, val_loader, epochs)
    
    # Plot results
    # trainer.plot_training_history()
    
    logger.info("Training completed!")
    return model, trainer

if __name__ == "__main__":
    # Example usage
    data_directory = "/path/to/your/data"
    model, trainer = train_model(data_directory, epochs=50, batch_size=2)
