
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CombinedLoss3D(nn.Module):
    """Advanced combined loss function for 3D medical image segmentation"""
    
    def __init__(self, alpha=0.5, beta=0.3, gamma=0.2, smooth=1e-5):
        super(CombinedLoss3D, self).__init__()
        self.alpha = alpha  # Dice loss weight
        self.beta = beta    # Focal loss weight  
        self.gamma = gamma  # Boundary loss weight
        self.smooth = smooth
        
    def dice_loss(self, pred, target):
        """Multi-class Dice loss for 3D segmentation"""
        pred = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 4, 1, 2, 3).float()
        
        intersection = (pred * target_one_hot).sum(dim=[2, 3, 4])
        union = pred.sum(dim=[2, 3, 4]) + target_one_hot.sum(dim=[2, 3, 4])
        
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice_score.mean()
        
        return dice_loss
    
    def focal_loss(self, pred, target, alpha=0.25, gamma=2.0):
        """Focal loss for handling class imbalance"""
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()
    
    def boundary_loss(self, pred, target):
        """Boundary-aware loss for better edge segmentation"""
        pred_soft = F.softmax(pred, dim=1)
        
        # Compute gradients to detect boundaries
        def compute_gradient_3d(tensor):
            grad_x = torch.abs(tensor[:, :, 1:, :, :] - tensor[:, :, :-1, :, :])
            grad_y = torch.abs(tensor[:, :, :, 1:, :] - tensor[:, :, :, :-1, :])
            grad_z = torch.abs(tensor[:, :, :, :, 1:] - tensor[:, :, :, :, :-1])
            
            # Pad to maintain original size
            grad_x = F.pad(grad_x, (0, 0, 0, 0, 0, 1))
            grad_y = F.pad(grad_y, (0, 0, 0, 1, 0, 0))
            grad_z = F.pad(grad_z, (0, 1, 0, 0, 0, 0))
            
            return grad_x + grad_y + grad_z
        
        # Convert target to one-hot for gradient computation
        target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 4, 1, 2, 3).float()
        
        pred_boundary = compute_gradient_3d(pred_soft)
        target_boundary = compute_gradient_3d(target_one_hot)
        
        boundary_loss = F.mse_loss(pred_boundary, target_boundary)
        return boundary_loss
    
    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        boundary = self.boundary_loss(pred, target)
        
        total_loss = self.alpha * dice + self.beta * focal + self.gamma * boundary
        
        return total_loss, {
            'dice_loss': dice.item(),
            'focal_loss': focal.item(), 
            'boundary_loss': boundary.item(),
            'total_loss': total_loss.item()
        }

class TverskyLoss3D(nn.Module):
    """Tversky loss for handling extreme class imbalance"""
    
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-5):
        super(TverskyLoss3D, self).__init__()
        self.alpha = alpha  # Controls false positives
        self.beta = beta    # Controls false negatives
        self.smooth = smooth
        
    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 4, 1, 2, 3).float()
        
        TP = (pred * target_one_hot).sum(dim=[2, 3, 4])
        FP = (pred * (1 - target_one_hot)).sum(dim=[2, 3, 4])
        FN = ((1 - pred) * target_one_hot).sum(dim=[2, 3, 4])
        
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        tversky_loss = 1 - tversky.mean()
        
        return tversky_loss

class DeepSupervisionLoss3D(nn.Module):
    """Deep supervision loss for training multiple output scales"""
    
    def __init__(self, weights=[1.0, 0.8, 0.6, 0.4], loss_fn=None):
        super(DeepSupervisionLoss3D, self).__init__()
        self.weights = weights
        self.loss_fn = loss_fn or CombinedLoss3D()
        
    def forward(self, predictions, target):
        if isinstance(predictions, tuple):
            main_pred, deep_preds = predictions
            total_loss = self.loss_fn(main_pred, target)[0] * self.weights[0]
            
            for i, pred in enumerate(deep_preds):
                if i < len(self.weights) - 1:
                    # Resize target to match prediction size
                    target_resized = F.interpolate(
                        target.float().unsqueeze(1), 
                        size=pred.shape[2:], 
                        mode='nearest'
                    ).squeeze(1).long()
                    
                    loss_val = self.loss_fn(pred, target_resized)[0]
                    total_loss += loss_val * self.weights[i + 1]
            
            return total_loss
        else:
            return self.loss_fn(predictions, target)[0]
