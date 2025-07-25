
"""
Advanced visualization utilities for brain tumor segmentation with modern approaches
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import base64
import io
from pathlib import Path
import json
from sklearn.metrics import confusion_matrix, classification_report
from scipy import ndimage
import nibabel as nib

# Set modern plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class ModernMedicalVisualizer:
    """Advanced medical imaging visualization with modern styling"""
    
    def __init__(self, output_dir="results/visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Color schemes for different tumor classes
        self.class_colors = {
            0: 'rgba(0,0,0,0)',      # Background - transparent
            1: 'rgba(255,0,0,0.7)',   # Necrotic core - red
            2: 'rgba(0,255,0,0.7)',   # Edema - green  
            3: 'rgba(0,0,255,0.7)'    # Enhancing tumor - blue
        }
        
        self.class_names = {
            0: 'Background',
            1: 'Necrotic Core',
            2: 'Peritumoral Edema', 
            3: 'Enhancing Tumor'
        }
    
    def create_multimodal_visualization(self, images, segmentation, patient_id="Unknown"):
        """Create comprehensive multimodal BraTS visualization"""
        modality_names = ['T1c', 'T1n', 'T2f', 'T2w']
        
        fig = make_subplots(
            rows=3, cols=4,
            subplot_titles=[f'{mod} - Axial' for mod in modality_names] + 
                          [f'{mod} - Sagittal' for mod in modality_names] + 
                          [f'{mod} - Coronal' for mod in modality_names],
            vertical_spacing=0.05,
            horizontal_spacing=0.05
        )
        
        # Get middle slices
        mid_axial = images.shape[2] // 2
        mid_sagittal = images.shape[3] // 2  
        mid_coronal = images.shape[4] // 2
        
        for i, modality in enumerate(modality_names):
            # Axial view
            img_axial = images[i, :, :, mid_axial]
            seg_axial = segmentation[:, :, mid_axial]
            
            fig.add_trace(
                go.Heatmap(z=img_axial, colorscale='gray', showscale=False),
                row=1, col=i+1
            )
            
            # Sagittal view  
            img_sagittal = images[i, :, mid_sagittal, :]
            fig.add_trace(
                go.Heatmap(z=img_sagittal, colorscale='gray', showscale=False),
                row=2, col=i+1
            )
            
            # Coronal view
            img_coronal = images[i, mid_coronal, :, :]
            fig.add_trace(
                go.Heatmap(z=img_coronal, colorscale='gray', showscale=False),
                row=3, col=i+1
            )
        
        fig.update_layout(
            title=f"BraTS 2024 Multimodal Visualization - Patient: {patient_id}",
            height=900,
            template="plotly_white"
        )
        
        return fig
    
    def create_segmentation_overlay(self, image, segmentation, modality="T1c"):
        """Create modern segmentation overlay visualization"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Brain Tumor Segmentation - {modality}', fontsize=16, fontweight='bold')
        
        # Get slices
        mid_axial = image.shape[2] // 2
        mid_sagittal = image.shape[0] // 2
        mid_coronal = image.shape[1] // 2
        
        views = [
            (image[:, :, mid_axial], segmentation[:, :, mid_axial], 'Axial'),
            (image[mid_sagittal, :, :], segmentation[mid_sagittal, :, :], 'Sagittal'),
            (image[:, mid_coronal, :], segmentation[:, mid_coronal, :], 'Coronal')
        ]
        
        for i, (img_slice, seg_slice, view_name) in enumerate(views):
            # Original image
            axes[0, i].imshow(img_slice, cmap='gray', aspect='equal')
            axes[0, i].set_title(f'{view_name} - Original', fontweight='bold')
            axes[0, i].axis('off')
            
            # Overlay segmentation
            axes[1, i].imshow(img_slice, cmap='gray', aspect='equal')
            
            # Add segmentation overlay for each class
            for class_id in range(1, 4):
                mask = seg_slice == class_id
                if np.any(mask):
                    colored_mask = np.zeros((*mask.shape, 4))
                    color = self.class_colors[class_id]
                    colored_mask[mask] = [
                        int(color.split('(')[1].split(',')[0])/255,  # R
                        int(color.split(',')[1])/255,                 # G  
                        int(color.split(',')[2])/255,                 # B
                        0.6                                           # A
                    ]
                    axes[1, i].imshow(colored_mask, aspect='equal')
            
            axes[1, i].set_title(f'{view_name} - Segmentation', fontweight='bold')
            axes[1, i].axis('off')
        
        # Add legend
        legend_elements = []
        for class_id in range(1, 4):
            color = self.class_colors[class_id]
            rgb = [int(color.split('(')[1].split(',')[0])/255,
                   int(color.split(',')[1])/255,
                   int(color.split(',')[2])/255]
            legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=rgb, 
                                               label=self.class_names[class_id]))
        
        fig.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(0.98, 0.5))
        
        plt.tight_layout()
        return fig
    
    def create_3d_tumor_reconstruction(self, segmentation, spacing=(1.0, 1.0, 1.0)):
        """Create 3D tumor reconstruction using marching cubes"""
        from skimage import measure
        
        fig = go.Figure()
        
        # Process each tumor class
        for class_id in range(1, 4):
            mask = segmentation == class_id
            if np.sum(mask) < 100:  # Skip if too few voxels
                continue
                
            try:
                # Use marching cubes to create mesh
                verts, faces, _, _ = measure.marching_cubes(
                    mask.astype(float), level=0.5, spacing=spacing
                )
                
                # Get color for this class
                color_str = self.class_colors[class_id]
                rgb_vals = [
                    int(color_str.split('(')[1].split(',')[0]),
                    int(color_str.split(',')[1]),
                    int(color_str.split(',')[2])
                ]
                color = f'rgb({rgb_vals[0]}, {rgb_vals[1]}, {rgb_vals[2]})'
                
                fig.add_trace(go.Mesh3d(
                    x=verts[:, 0],
                    y=verts[:, 1], 
                    z=verts[:, 2],
                    i=faces[:, 0],
                    j=faces[:, 1],
                    k=faces[:, 2],
                    color=color,
                    opacity=0.8,
                    name=self.class_names[class_id]
                ))
                
            except Exception as e:
                print(f"Error creating mesh for class {class_id}: {e}")
        
        fig.update_layout(
            title="3D Tumor Reconstruction",
            scene=dict(
                xaxis_title="X (mm)",
                yaxis_title="Y (mm)", 
                zaxis_title="Z (mm)",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
                aspectmode='cube'
            ),
            width=800,
            height=600,
            template="plotly_white"
        )
        
        return fig
    
    def create_volume_analysis_dashboard(self, segmentation, spacing=(1.0, 1.0, 1.0)):
        """Create comprehensive volume analysis dashboard"""
        # Calculate volumes
        voxel_volume = np.prod(spacing)
        volumes = {}
        
        for class_id in range(1, 4):
            mask = segmentation == class_id
            volume_voxels = np.sum(mask)
            volume_mm3 = volume_voxels * voxel_volume
            volumes[self.class_names[class_id]] = volume_mm3
        
        # Create dashboard
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                'Tumor Volume Distribution', 'Volume by Slice Position',
                'Tumor Compactness Analysis', 'Class Distribution',
                'Surface Area Analysis', 'Clinical Metrics'
            ],
            specs=[[{"type": "pie"}, {"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "scatter"}, {"type": "table"}]]
        )
        
        # Volume distribution pie chart
        if sum(volumes.values()) > 0:
            fig.add_trace(
                go.Pie(
                    labels=list(volumes.keys()),
                    values=list(volumes.values()),
                    hole=0.4,
                    textinfo="label+percent"
                ),
                row=1, col=1
            )
        
        # Volume by slice position
        slice_volumes = []
        slice_positions = range(0, segmentation.shape[2], 5)
        
        for z in slice_positions:
            if z < segmentation.shape[2]:
                slice_vol = np.sum(segmentation[:, :, z] > 0) * spacing[0] * spacing[1]
                slice_volumes.append(slice_vol)
        
        fig.add_trace(
            go.Scatter(
                x=list(slice_positions[:len(slice_volumes)]),
                y=slice_volumes,
                mode='lines+markers',
                name='Tumor Volume per Slice'
            ),
            row=1, col=2
        )
        
        # Compactness analysis (simplified)
        compactness_scores = []
        for class_id in range(1, 4):
            mask = segmentation == class_id
            if np.sum(mask) > 0:
                # Simplified compactness calculation
                volume = np.sum(mask)
                equivalent_radius = (3 * volume / (4 * np.pi)) ** (1/3)
                compactness = volume / (equivalent_radius ** 3)
                compactness_scores.append(compactness)
            else:
                compactness_scores.append(0)
        
        fig.add_trace(
            go.Bar(
                x=list(self.class_names.values())[1:],
                y=compactness_scores,
                name='Compactness Score'
            ),
            row=2, col=1
        )
        
        # Clinical metrics table
        total_tumor_volume = sum(volumes.values())
        metrics_data = [
            ['Total Tumor Volume', f'{total_tumor_volume:.1f} mm³'],
            ['Largest Component', max(volumes.keys(), key=lambda k: volumes[k]) if volumes else 'None'],
            ['Number of Components', str(len([v for v in volumes.values() if v > 0]))],
            ['Necrotic Percentage', f'{(volumes.get("Necrotic Core", 0) / total_tumor_volume * 100):.1f}%' if total_tumor_volume > 0 else '0%'],
            ['Edema Volume', f'{volumes.get("Peritumoral Edema", 0):.1f} mm³'],
            ['Enhancing Volume', f'{volumes.get("Enhancing Tumor", 0):.1f} mm³']
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Metric', 'Value']),
                cells=dict(values=list(zip(*metrics_data)))
            ),
            row=2, col=3
        )
        
        fig.update_layout(
            title="Comprehensive Tumor Volume Analysis",
            height=800,
            template="plotly_white"
        )
        
        return fig
    
    def create_training_dashboard(self, metrics_history):
        """Create modern training dashboard"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Loss Curves', 'Dice Score Progression',
                'Learning Rate Schedule', 'Validation Metrics'
            ]
        )
        
        epochs = list(range(len(metrics_history.get('train_loss', []))))
        
        # Loss curves
        if 'train_loss' in metrics_history:
            fig.add_trace(
                go.Scatter(x=epochs, y=metrics_history['train_loss'], 
                          name='Training Loss', line=dict(color='red')),
                row=1, col=1
            )
        
        if 'val_loss' in metrics_history:
            fig.add_trace(
                go.Scatter(x=epochs, y=metrics_history['val_loss'],
                          name='Validation Loss', line=dict(color='blue')),
                row=1, col=1
            )
        
        # Dice scores
        if 'dice_scores' in metrics_history:
            fig.add_trace(
                go.Scatter(x=epochs, y=metrics_history['dice_scores'],
                          name='Dice Score', line=dict(color='green')),
                row=1, col=2
            )
        
        # Learning rate
        if 'learning_rate' in metrics_history:
            fig.add_trace(
                go.Scatter(x=epochs, y=metrics_history['learning_rate'],
                          name='Learning Rate', line=dict(color='orange')),
                row=2, col=1
            )
        
        fig.update_layout(
            title="Training Progress Dashboard",
            height=800,
            template="plotly_white"
        )
        
        return fig
    
    def create_performance_heatmap(self, confusion_matrices, class_names):
        """Create performance heatmap for each class"""
        fig, axes = plt.subplots(1, len(confusion_matrices), figsize=(15, 4))
        
        if len(confusion_matrices) == 1:
            axes = [axes]
        
        for i, (cm, class_name) in enumerate(zip(confusion_matrices, class_names)):
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
            axes[i].set_title(f'{class_name} Confusion Matrix')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        return fig
    
    def save_visualization(self, fig, filename, format='html'):
        """Save visualization in specified format"""
        filepath = self.output_dir / f"{filename}.{format}"
        
        if hasattr(fig, 'write_html') and format == 'html':
            fig.write_html(str(filepath))
        elif hasattr(fig, 'write_image'):
            fig.write_image(str(filepath), width=1200, height=800, scale=2)
        else:
            # Matplotlib figure
            fig.savefig(str(filepath), dpi=300, bbox_inches='tight')
            plt.close(fig)
        
        return str(filepath)
    
    def generate_medical_report(self, patient_data, results):
        """Generate comprehensive medical report"""
        report_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Brain Tumor Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #f0f0f0; padding: 20px; border-radius: 10px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background: #e9f5ff; border-radius: 5px; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Brain Tumor Segmentation Analysis</h1>
                <p>Patient ID: {patient_data.get('id', 'Unknown')}</p>
                <p>Analysis Date: {patient_data.get('date', 'Unknown')}</p>
                <p>Model: BraTS 2024 Enhanced U-Net</p>
            </div>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <div class="metric">
                    <strong>Total Tumor Volume:</strong> {results.get('total_volume', 0):.1f} mm³
                </div>
                <div class="metric">
                    <strong>Confidence Score:</strong> {results.get('confidence', 0):.2f}
                </div>
                <div class="metric">
                    <strong>Risk Assessment:</strong> {results.get('risk_level', 'Unknown')}
                </div>
            </div>
            
            <div class="section">
                <h2>Detailed Analysis</h2>
                <table>
                    <tr><th>Component</th><th>Volume (mm³)</th><th>Percentage</th></tr>
                    <tr><td>Necrotic Core</td><td>{results.get('necrotic_volume', 0):.1f}</td><td>{results.get('necrotic_pct', 0):.1f}%</td></tr>
                    <tr><td>Peritumoral Edema</td><td>{results.get('edema_volume', 0):.1f}</td><td>{results.get('edema_pct', 0):.1f}%</td></tr>
                    <tr><td>Enhancing Tumor</td><td>{results.get('enhancing_volume', 0):.1f}</td><td>{results.get('enhancing_pct', 0):.1f}%</td></tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Clinical Recommendations</h2>
                <ul>
                    <li>Follow-up imaging recommended in 3-6 months</li>
                    <li>Multidisciplinary team consultation advised</li>
                    <li>Consider advanced imaging techniques for better characterization</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        report_path = self.output_dir / "medical_report.html"
        with open(report_path, 'w') as f:
            f.write(report_html)
        
        return str(report_path)

# Utility functions
def create_modern_colormap():
    """Create modern medical colormap"""
    from matplotlib.colors import ListedColormap
    colors = ['black', '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    return ListedColormap(colors)

def plot_slice_comparison(original, predicted, ground_truth, slice_idx=None):
    """Plot comparison between original, predicted, and ground truth"""
    if slice_idx is None:
        slice_idx = original.shape[2] // 2
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original[:, :, slice_idx], cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(predicted[:, :, slice_idx], cmap=create_modern_colormap())
    axes[1].set_title('Predicted Segmentation')
    axes[1].axis('off')
    
    axes[2].imshow(ground_truth[:, :, slice_idx], cmap=create_modern_colormap())
    axes[2].set_title('Ground Truth')
    axes[2].axis('off')
    
    plt.tight_layout()
    return fig
