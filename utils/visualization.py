
"""
Visualization utilities for brain tumor segmentation
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import base64
import io

class MedicalVisualizer:
    """Medical imaging visualization utilities"""
    
    @staticmethod
    def plot_volume_slices(volume, segmentation=None, title="Brain Volume", figsize=(15, 10)):
        """Plot multiple slices of a 3D volume"""
        fig, axes = plt.subplots(3, 4, figsize=figsize)
        fig.suptitle(title, fontsize=16)
        
        # Select slices
        depth, height, width = volume.shape
        slice_indices = {
            'axial': np.linspace(depth//4, 3*depth//4, 4, dtype=int),
            'sagittal': np.linspace(height//4, 3*height//4, 4, dtype=int),
            'coronal': np.linspace(width//4, 3*width//4, 4, dtype=int)
        }
        
        # Plot axial slices
        for i, slice_idx in enumerate(slice_indices['axial']):
            axes[0, i].imshow(volume[slice_idx, :, :], cmap='gray')
            if segmentation is not None:
                overlay = np.ma.masked_where(segmentation[slice_idx, :, :] == 0, 
                                           segmentation[slice_idx, :, :])
                axes[0, i].imshow(overlay, cmap='jet', alpha=0.6)
            axes[0, i].set_title(f'Axial {slice_idx}')
            axes[0, i].axis('off')
        
        # Plot sagittal slices
        for i, slice_idx in enumerate(slice_indices['sagittal']):
            axes[1, i].imshow(volume[:, slice_idx, :], cmap='gray')
            if segmentation is not None:
                overlay = np.ma.masked_where(segmentation[:, slice_idx, :] == 0, 
                                           segmentation[:, slice_idx, :])
                axes[1, i].imshow(overlay, cmap='jet', alpha=0.6)
            axes[1, i].set_title(f'Sagittal {slice_idx}')
            axes[1, i].axis('off')
        
        # Plot coronal slices
        for i, slice_idx in enumerate(slice_indices['coronal']):
            axes[2, i].imshow(volume[:, :, slice_idx], cmap='gray')
            if segmentation is not None:
                overlay = np.ma.masked_where(segmentation[:, :, slice_idx] == 0, 
                                           segmentation[:, :, slice_idx])
                axes[2, i].imshow(overlay, cmap='jet', alpha=0.6)
            axes[2, i].set_title(f'Coronal {slice_idx}')
            axes[2, i].axis('off')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def create_3d_volume_plot(volume, threshold=0.5, title="3D Brain Volume"):
        """Create 3D plotly visualization of brain volume"""
        # Downsample for performance
        volume_small = volume[::2, ::2, ::2]
        
        # Create 3D scatter plot
        x, y, z = np.where(volume_small > threshold)
        values = volume_small[x, y, z]
        
        fig = go.Figure(data=go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=2,
                color=values,
                colorscale='Viridis',
                opacity=0.6,
                colorbar=dict(title="Intensity")
            )
        ))
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z"
            ),
            width=800,
            height=600
        )
        
        return fig
    
    @staticmethod
    def plot_training_metrics(metrics_dict, save_path=None):
        """Plot training metrics"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Loss', 'Dice Score', 'Learning Rate', 'Validation Metrics')
        )
        
        epochs = list(range(len(metrics_dict.get('train_loss', []))))
        
        # Plot training loss
        if 'train_loss' in metrics_dict:
            fig.add_trace(
                go.Scatter(x=epochs, y=metrics_dict['train_loss'], name='Train Loss'),
                row=1, col=1
            )
        if 'val_loss' in metrics_dict:
            fig.add_trace(
                go.Scatter(x=epochs, y=metrics_dict['val_loss'], name='Val Loss'),
                row=1, col=1
            )
        
        # Plot dice score
        if 'dice_score' in metrics_dict:
            fig.add_trace(
                go.Scatter(x=epochs, y=metrics_dict['dice_score'], name='Dice Score'),
                row=1, col=2
            )
        
        # Plot learning rate
        if 'learning_rate' in metrics_dict:
            fig.add_trace(
                go.Scatter(x=epochs, y=metrics_dict['learning_rate'], name='LR'),
                row=2, col=1
            )
        
        # Plot validation metrics
        if 'val_dice' in metrics_dict:
            fig.add_trace(
                go.Scatter(x=epochs, y=metrics_dict['val_dice'], name='Val Dice'),
                row=2, col=2
            )
        
        fig.update_layout(
            title="Training Metrics",
            height=800,
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    @staticmethod
    def fig_to_base64(fig):
        """Convert matplotlib figure to base64 string"""
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.read()).decode()
        plt.close(fig)
        return img_str
