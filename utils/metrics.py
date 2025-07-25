
"""
Evaluation metrics for brain tumor segmentation
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import directed_hausdorff

class SegmentationMetrics:
    """Evaluation metrics for segmentation tasks"""
    
    @staticmethod
    def dice_coefficient(pred, target, smooth=1e-6):
        """Calculate Dice coefficient"""
        pred = torch.sigmoid(pred) if pred.requires_grad else pred
        pred = (pred > 0.5).float()
        
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return dice.item()
    
    @staticmethod
    def iou_score(pred, target, smooth=1e-6):
        """Calculate Intersection over Union (IoU)"""
        pred = torch.sigmoid(pred) if pred.requires_grad else pred
        pred = (pred > 0.5).float()
        
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        
        iou = (intersection + smooth) / (union + smooth)
        return iou.item()
    
    @staticmethod
    def sensitivity(pred, target, smooth=1e-6):
        """Calculate sensitivity (recall)"""
        pred = torch.sigmoid(pred) if pred.requires_grad else pred
        pred = (pred > 0.5).float()
        
        tp = (pred * target).sum()
        fn = ((1 - pred) * target).sum()
        
        sensitivity = (tp + smooth) / (tp + fn + smooth)
        return sensitivity.item()
    
    @staticmethod
    def specificity(pred, target, smooth=1e-6):
        """Calculate specificity"""
        pred = torch.sigmoid(pred) if pred.requires_grad else pred
        pred = (pred > 0.5).float()
        
        tn = ((1 - pred) * (1 - target)).sum()
        fp = (pred * (1 - target)).sum()
        
        specificity = (tn + smooth) / (tn + fp + smooth)
        return specificity.item()
    
    @staticmethod
    def hausdorff_distance(pred, target):
        """Calculate Hausdorff distance"""
        try:
            pred_np = pred.detach().cpu().numpy() if torch.is_tensor(pred) else pred
            target_np = target.detach().cpu().numpy() if torch.is_tensor(target) else target
            
            pred_points = np.where(pred_np > 0.5)
            target_points = np.where(target_np > 0.5)
            
            if len(pred_points[0]) == 0 or len(target_points[0]) == 0:
                return float('inf')
            
            pred_coords = np.column_stack(pred_points)
            target_coords = np.column_stack(target_points)
            
            return max(
                directed_hausdorff(pred_coords, target_coords)[0],
                directed_hausdorff(target_coords, pred_coords)[0]
            )
        except:
            return float('inf')
    
    @staticmethod
    def compute_all_metrics(pred, target):
        """Compute all segmentation metrics"""
        metrics = {}
        
        metrics['dice'] = SegmentationMetrics.dice_coefficient(pred, target)
        metrics['iou'] = SegmentationMetrics.iou_score(pred, target)
        metrics['sensitivity'] = SegmentationMetrics.sensitivity(pred, target)
        metrics['specificity'] = SegmentationMetrics.specificity(pred, target)
        metrics['hausdorff'] = SegmentationMetrics.hausdorff_distance(pred, target)
        
        return metrics

class LossMetrics:
    """Loss functions for segmentation"""
    
    @staticmethod
    def dice_loss(pred, target, smooth=1e-6):
        """Dice loss function"""
        pred = torch.sigmoid(pred)
        
        intersection = (pred * target).sum(dim=(2, 3, 4))
        union = pred.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4))
        
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()
    
    @staticmethod
    def focal_loss(pred, target, alpha=0.25, gamma=2.0):
        """Focal loss function"""
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()
    
    @staticmethod
    def combined_loss(pred, target, dice_weight=0.5, focal_weight=0.5):
        """Combined dice and focal loss"""
        dice = LossMetrics.dice_loss(pred, target)
        focal = LossMetrics.focal_loss(pred, target)
        
        return dice_weight * dice + focal_weight * focal
