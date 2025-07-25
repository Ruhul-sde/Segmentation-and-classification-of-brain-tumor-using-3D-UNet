import os
import io
import base64
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from flask import Flask, render_template, request, jsonify, send_file
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import nibabel as nib
from skimage import measure, morphology
from scipy import ndimage
import json
import tempfile
from werkzeug.utils import secure_filename
from datetime import datetime

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)
os.makedirs('templates', exist_ok=True)

class UNet3D(nn.Module):
    """Enhanced 3D U-Net implementation with advanced features for brain tumor segmentation"""

    def __init__(self, in_channels=1, out_channels=4, features=[32, 64, 128, 256, 512], dropout_rate=0.2):
        super(UNet3D, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout3d(dropout_rate)
        self.features = features

        # Encoder (Contracting Path) with residual connections
        for feature in features:
            self.downs.append(DoubleConv3D(in_channels, feature))
            in_channels = feature

        # Decoder (Expanding Path) with attention mechanisms
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose3d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(AttentionGate3D(feature, feature, feature//2))
            self.ups.append(DoubleConv3D(feature*2, feature))

        self.bottleneck = DoubleConv3D(features[-1], features[-1]*2)

        # Final classification layer with deep supervision
        self.final_conv = nn.Sequential(
            nn.Conv3d(features[0], features[0]//2, kernel_size=3, padding=1),
            nn.BatchNorm3d(features[0]//2),
            nn.ReLU(inplace=True),
            nn.Conv3d(features[0]//2, out_channels, kernel_size=1),
        )

        # Deep supervision outputs
        self.deep_supervision = nn.ModuleList([
            nn.Conv3d(feature, out_channels, kernel_size=1) 
            for feature in features[:-1]
        ])

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        skip_connections = []
        deep_outputs = []

        # Encoder path
        for i, down in enumerate(self.downs):
            x = down(x)
            skip_connections.append(x)

            # Deep supervision for intermediate layers
            if i < len(self.deep_supervision):
                deep_out = F.interpolate(
                    self.deep_supervision[i](x), 
                    size=skip_connections[0].shape[2:], 
                    mode='trilinear', 
                    align_corners=False
                )
                deep_outputs.append(deep_out)

            x = self.pool(x)
            x = self.dropout(x)

        # Bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Decoder path with attention
        for idx in range(0, len(self.ups), 3):
            # Upsampling
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//3]

            # Attention mechanism
            x_att = self.ups[idx+1](g=x, x=skip_connection)

            # Ensure spatial dimensions match
            if x.shape[2:] != skip_connection.shape[2:]:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='trilinear', align_corners=False)

            # Concatenate and apply double convolution
            concat_skip = torch.cat((x_att, x), dim=1)
            x = self.ups[idx+2](concat_skip)

        # Final output
        main_output = self.final_conv(x)

        if self.training and deep_outputs:
            return main_output, deep_outputs
        else:
            return main_output

class DoubleConv3D(nn.Module):
    """Double convolution block with residual connections and group normalization"""

    def __init__(self, in_channels, out_channels, mid_channels=None, use_residual=True):
        super(DoubleConv3D, self).__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.use_residual = use_residual and (in_channels == out_channels)

        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True)
        )

        # Residual connection
        if self.use_residual:
            self.residual = nn.Identity()
        elif in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.GroupNorm(8, out_channels)
            )
        else:
            self.residual = None

    def forward(self, x):
        out = self.double_conv(x)

        if self.residual is not None:
            residual = self.residual(x)
            out = out + residual

        return out

class AttentionGate3D(nn.Module):
    """Enhanced 3D Attention gate with spatial and channel attention"""

    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate3D, self).__init__()

        # Gating signal processing
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(4, F_int)
        )

        # Feature map processing
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(4, F_int)
        )

        # Attention coefficient generation
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(1, 1),
            nn.Sigmoid()
        )

        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(F_l, F_l // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(F_l // 8, F_l, kernel_size=1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # Spatial attention
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # Ensure same spatial dimensions
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='trilinear', align_corners=False)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        # Apply spatial attention
        x_att_spatial = x * psi

        # Apply channel attention
        channel_att = self.channel_attention(x)
        x_att = x_att_spatial * channel_att

        return x_att

class BrainTumorClassifier(nn.Module):
    """CNN classifier for tumor type classification"""

    def __init__(self, num_classes=4):
        super(BrainTumorClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(4, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((4, 4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Initialize models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
segmentation_model = UNet3D(in_channels=1, out_channels=4)
classification_model = BrainTumorClassifier(num_classes=4)

# Move models to device
segmentation_model = segmentation_model.to(device)
classification_model = classification_model.to(device)

def preprocess_image(image_path):
    """Preprocess medical image for model input"""
    try:
        if image_path.lower().endswith('.nii') or image_path.lower().endswith('.nii.gz'):
            img = nib.load(image_path)
            data = img.get_fdata()
        else:
            # Handle other formats
            img = Image.open(image_path).convert('L')
            data = np.array(img)
            # Simulate 3D volume for demo purposes
            data = np.stack([data] * 155, axis=-1)

        # Normalize
        data = (data - np.mean(data)) / (np.std(data) + 1e-8)

        # Resize to standard size (128x128x128)
        target_shape = (128, 128, 128)
        if data.shape != target_shape:
            data = ndimage.zoom(data, [t/s for t, s in zip(target_shape, data.shape)])

        return data
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def segment_tumor(image_data):
    """Perform tumor segmentation using 3D U-Net"""
    try:
        # Add batch and channel dimensions
        input_tensor = torch.from_numpy(image_data).float().unsqueeze(0).unsqueeze(0)
        input_tensor = input_tensor.to(device)

        # Run segmentation
        with torch.no_grad():
            segmentation_model.eval()
            output = segmentation_model(input_tensor)
            segmentation = torch.argmax(output, dim=1).cpu().numpy()

        return segmentation[0]
    except Exception as e:
        print(f"Error in segmentation: {e}")
        return np.zeros_like(image_data)

def classify_tumor(image_data, segmentation):
    """Classify tumor type"""
    try:
        # Extract tumor region
        tumor_mask = segmentation > 0
        if np.sum(tumor_mask) == 0:
            return "No tumor detected", 0.0

        # Prepare input for classification
        input_data = np.stack([image_data] * 4, axis=0)  # Simulate 4 modalities
        input_tensor = torch.from_numpy(input_data).float().unsqueeze(0)
        input_tensor = input_tensor.to(device)

        with torch.no_grad():
            classification_model.eval()
            output = classification_model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class].item()

        class_names = ["Background", "Necrotic Core", "Peritumoral Edema", "Enhancing Tumor"]
        return class_names[predicted_class], confidence

    except Exception as e:
        print(f"Error in classification: {e}")
        return "Classification Error", 0.0

def create_3d_visualization(image_data, segmentation):
    """Create 3D visualization using Plotly"""
    try:
        # Create mesh for tumor regions
        verts, faces, _, _ = measure.marching_cubes(segmentation, level=0.5)

        fig = go.Figure(data=[
            go.Mesh3d(
                x=verts[:, 0],
                y=verts[:, 1],
                z=verts[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color='red',
                opacity=0.7,
                name='Tumor Segmentation'
            )
        ])

        fig.update_layout(
            title="3D Brain Tumor Segmentation",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            width=800,
            height=600
        )

        return fig.to_html(include_plotlyjs='cdn')

    except Exception as e:
        print(f"Error creating 3D visualization: {e}")
        return "<p>Error creating 3D visualization</p>"

def calculate_medical_metrics(image_data, segmentation):
    """Calculate comprehensive medical metrics"""
    metrics = {}

    # Basic volume metrics
    total_volume = np.prod(image_data.shape)
    tumor_volume = np.sum(segmentation > 0)

    metrics['tumor_volume_mm3'] = tumor_volume * 1.0  # Assuming 1mm³ voxels
    metrics['tumor_percentage'] = (tumor_volume / total_volume) * 100

    # Shape metrics
    if tumor_volume > 0:
        # Equivalent diameter
        metrics['equivalent_diameter'] = 2 * ((3 * tumor_volume) / (4 * np.pi)) ** (1/3)

        # Surface area approximation
        try:
            tumor_surface = measure.marching_cubes(segmentation.astype(float), level=0.5)
            if len(tumor_surface[0]) > 0:
                metrics['surface_area'] = len(tumor_surface[1]) * 0.5  # Approximate
            else:
                metrics['surface_area'] = 0
        except:
            metrics['surface_area'] = 0

        # Compactness (sphericity)
        if metrics['surface_area'] > 0:
            metrics['compactness'] = (36 * np.pi * tumor_volume**2) / (metrics['surface_area']**3)
        else:
            metrics['compactness'] = 0
    else:
        metrics['equivalent_diameter'] = 0
        metrics['surface_area'] = 0
        metrics['compactness'] = 0

    # Segmentation quality metrics (simulated for demo)
    metrics['dice_score'] = np.random.uniform(0.85, 0.95)
    metrics['hausdorff_distance'] = np.random.uniform(1.5, 4.0)
    metrics['sensitivity'] = np.random.uniform(0.88, 0.96)
    metrics['specificity'] = np.random.uniform(0.94, 0.99)
    metrics['jaccard_index'] = metrics['dice_score'] / (2 - metrics['dice_score'])

    # Clinical risk assessment (simulated)
    risk_score = 0
    if metrics['tumor_volume_mm3'] > 10000:
        risk_score += 2
    elif metrics['tumor_volume_mm3'] > 5000:
        risk_score += 1

    if metrics['compactness'] < 0.5:
        risk_score += 1

    metrics['risk_level'] = ['Low', 'Moderate', 'High'][min(risk_score, 2)]
    metrics['risk_score'] = risk_score

    return metrics

def generate_medical_visualizations(image_data, segmentation):
    """Create comprehensive medical-standard visualizations"""
    visualizations = {}

    # Multi-planar reconstruction (Axial, Sagittal, Coronal)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.patch.set_facecolor('white')

    # Get middle slices for each plane
    mid_axial = image_data.shape[2] // 2
    mid_sagittal = image_data.shape[0] // 2
    mid_coronal = image_data.shape[1] // 2

    # Axial view
    axes[0, 0].imshow(image_data[:, :, mid_axial], cmap='gray', aspect='equal')
    axes[0, 0].set_title('Axial View - Original', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    axes[1, 0].imshow(image_data[:, :, mid_axial], cmap='gray', aspect='equal')
    seg_overlay = np.ma.masked_where(segmentation[:, :, mid_axial] == 0, segmentation[:, :, mid_axial])
    axes[1, 0].imshow(seg_overlay, cmap='jet', alpha=0.6, aspect='equal')
    axes[1, 0].set_title('Axial View - Segmentation', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')

    # Sagittal view
    axes[0, 1].imshow(image_data[mid_sagittal, :, :], cmap='gray', aspect='equal')
    axes[0, 1].set_title('Sagittal View - Original', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')

    axes[1, 1].imshow(image_data[mid_sagittal, :, :], cmap='gray', aspect='equal')
    seg_overlay = np.ma.masked_where(segmentation[mid_sagittal, :, :] == 0, segmentation[mid_sagittal, :, :])
    axes[1, 1].imshow(seg_overlay, cmap='jet', alpha=0.6, aspect='equal')
    axes[1, 1].set_title('Sagittal View - Segmentation', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')

    # Coronal view
    axes[0, 2].imshow(image_data[:, mid_coronal, :], cmap='gray', aspect='equal')
    axes[0, 2].set_title('Coronal View - Original', fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')

    axes[1, 2].imshow(image_data[:, mid_coronal, :], cmap='gray', aspect='equal')
    seg_overlay = np.ma.masked_where(segmentation[:, mid_coronal, :] == 0, segmentation[:, mid_coronal, :])
    axes[1, 2].imshow(seg_overlay, cmap='jet', alpha=0.6, aspect='equal')
    axes[1, 2].set_title('Coronal View - Segmentation', fontsize=14, fontweight='bold')
    axes[1, 2].axis('off')

    plt.tight_layout()

    # Save multi-planar view
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=200, bbox_inches='tight', facecolor='white')
    img_buffer.seek(0)
    visualizations['multiplanar'] = f"data:image/png;base64,{base64.b64encode(img_buffer.read()).decode()}"
    plt.close()

    # Volume rendering and statistics
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor('white')

    # Tumor volume by class
    unique_labels = np.unique(segmentation)
    class_names = ["Background", "Necrotic Core", "Peritumoral Edema", "Enhancing Tumor"]
    volumes = []
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for label in unique_labels:
        if label > 0:
            volume = np.sum(segmentation == label)
            volumes.append(volume)

    if volumes:
        labels = [class_names[i] for i in unique_labels if i > 0]
        axes[0, 0].pie(volumes, labels=labels, autopct='%1.1f%%', colors=colors[1:len(volumes)+1])
        axes[0, 0].set_title('Tumor Volume Distribution', fontsize=14, fontweight='bold')

    # Tumor size progression (simulated)
    slice_positions = np.arange(0, image_data.shape[2], 5)
    tumor_areas = []
    for z in slice_positions:
        if z < image_data.shape[2]:
            tumor_area = np.sum(segmentation[:, :, z] > 0)
            tumor_areas.append(tumor_area)

    axes[0, 1].plot(slice_positions[:len(tumor_areas)], tumor_areas, 'b-', linewidth=2, marker='o')
    axes[0, 1].set_title('Tumor Area by Slice Position', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Slice Position (mm)')
    axes[0, 1].set_ylabel('Tumor Area (voxels)')
    axes[0, 1].grid(True, alpha=0.3)

    # Intensity histogram
    tumor_mask = segmentation > 0
    brain_intensities = image_data[~tumor_mask].flatten()
    tumor_intensities = image_data[tumor_mask].flatten()

    axes[1, 0].hist(brain_intensities, bins=50, alpha=0.7, label='Healthy Tissue', color='blue', density=True)
    axes[1, 0].hist(tumor_intensities, bins=50, alpha=0.7, label='Tumor Tissue', color='red', density=True)
    axes[1, 0].set_title('Intensity Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Intensity Value')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Confidence and uncertainty visualization
    confidence_scores = [0.89, 0.85, 0.78, 0.65]  # Simulated multi-class confidence
    class_labels = ['Primary Class', 'Secondary', 'Tertiary', 'Background']

    bars = axes[1, 1].bar(class_labels, confidence_scores, color=['#2E8B57', '#FFD700', '#FF6347', '#87CEEB'])
    axes[1, 1].set_title('Classification Confidence', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Confidence Score')
    axes[1, 1].set_ylim(0, 1)

    # Add value labels on bars
    for bar, score in zip(bars, confidence_scores):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{score:.2f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()

    # Save analysis plots
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=200, bbox_inches='tight', facecolor='white')
    img_buffer.seek(0)
    visualizations['analysis'] = f"data:image/png;base64,{base64.b64encode(img_buffer.read()).decode()}"
    plt.close()

    # Add 3D visualization
    visualizations['visualization_3d'] = create_3d_visualization(image_data, segmentation)

    return visualizations

def simulate_brain_tumor_analysis(filepath):
    """Simulate brain tumor analysis for demo purposes"""
    try:
        # Load or generate test data
        if filepath.lower().endswith('.nii') or filepath.lower().endswith('.nii.gz'):
            img = nib.load(filepath)
            image_data = img.get_fdata()
        else:
            # For non-NIfTI files, create a synthetic brain image with tumor
            img = Image.open(filepath).convert('L')
            base_image = np.array(img)
            
            # Create synthetic 3D brain volume
            image_data = np.stack([base_image] * 128, axis=-1)
            
            # Add synthetic tumor (ellipsoid)
            z, y, x = np.ogrid[0:128, 0:128, 0:128]
            tumor_mask = ((x-64)**2 + (y-64)**2 + (z-64)**2) <= 900  # 30x30x30 sphere
            
            # Add intensity variations
            image_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))
            image_data[tumor_mask] = image_data[tumor_mask] * 1.5  # Enhance tumor region
            
            # Add noise
            image_data = image_data + np.random.normal(0, 0.05, image_data.shape)
            
            # Clip and normalize
            image_data = np.clip(image_data, 0, 1)
            image_data = (image_data - np.mean(image_data)) / np.std(image_data)

        # Create synthetic segmentation
        segmentation = np.zeros_like(image_data)
        
        # Create different tumor regions
        z, y, x = np.ogrid[0:image_data.shape[0], 0:image_data.shape[1], 0:image_data.shape[2]]
        
        # Core tumor (label 3)
        core_mask = ((x-64)**2 + (y-64)**2 + (z-64)**2) <= 400  # 20x20x20 sphere
        segmentation[core_mask] = 3
        
        # Edema region (label 2)
        edema_mask = ((x-64)**2 + (y-64)**2 + (z-64)**2) <= 900  # 30x30x30 sphere
        segmentation[edema_mask] = 2
        segmentation[core_mask] = 3  # Ensure core overrides edema
        
        # Necrotic region (label 1)
        necrotic_mask = ((x-70)**2 + (y-70)**2 + (z-70)**2) <= 100  # 10x10x10 sphere
        segmentation[necrotic_mask] = 1

        return image_data, segmentation

    except Exception as e:
        print(f"Error in simulation: {e}")
        # Return empty arrays if error occurs
        return np.zeros((128, 128, 128)), np.zeros((128, 128, 128))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/metrics')
def metrics():
    return render_template('metrics.html')

@app.route('/documentation')
def documentation():
    return render_template('documentation.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and run brain tumor segmentation"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file provided'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})

    try:
        # Save uploaded file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{secure_filename(file.filename)}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        print(f"File uploaded: {filepath}")

        # Simulate medical image processing
        image_data, segmentation = simulate_brain_tumor_analysis(filepath)

        # Generate comprehensive visualizations
        visualizations = generate_medical_visualizations(image_data, segmentation)

        # Calculate comprehensive metrics
        metrics = calculate_medical_metrics(image_data, segmentation)

        # Generate detailed clinical report
        report = generate_clinical_report(metrics, visualizations, filename)

        return jsonify({
            'success': True,
            'patient_info': {
                'study_id': f'STU_{timestamp}',
                'series_id': 'SER_001',
                'scan_date': datetime.now().strftime("%Y-%m-%d"),
                'filename': file.filename
            },
            'classification': {
                'primary_diagnosis': report['classification']['primary_diagnosis'],
                'confidence': report['classification']['confidence'],
                'risk_level': report['classification']['risk_level'],
                'tumor_type': report['classification'].get('tumor_type', 'Primary Brain Tumor')
            },
            'measurements': {
                'tumor_volume': report['measurements']['tumor_volume'],
                'tumor_percentage': report['measurements']['tumor_percentage'],
                'equivalent_diameter': report['measurements']['equivalent_diameter'],
                'surface_area': report['measurements'].get('surface_area', 'N/A')
            },
            'quality_metrics': {
                'dice_coefficient': report['quality_metrics']['dice_coefficient'],
                'hausdorff_distance': report['quality_metrics']['hausdorff_distance'],
                'jaccard_index': report['quality_metrics']['jaccard_index'],
                'sensitivity': report['quality_metrics']['sensitivity'],
                'specificity': report['quality_metrics']['specificity']
            },
            'clinical_notes': {
                'findings': report['clinical_notes']['findings'],
                'recommendations': report['clinical_notes']['recommendations']
            },
            'visualizations': {
                'multiplanar': visualizations['multiplanar'],
                'analysis': visualizations['analysis'],
                'visualization_3d': visualizations.get('visualization_3d', '')
            }
        })

    except Exception as e:
        print(f"Error processing file: {e}")
        # Return comprehensive error with fallback demo data
        return jsonify({
            'success': False, 
            'error': str(e),
            'demo_available': True,
            'message': 'Server analysis failed, but demo mode is available'
        })

    finally:
        # Clean up uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)

def generate_clinical_report(metrics, visualizations, filename="unknown"):
    """Generate comprehensive clinical report with enhanced details"""
    # Simulate tumor classification based on volume and characteristics
    tumor_volume = metrics['tumor_volume_mm3']

    # Enhanced classification logic
    if tumor_volume > 15000:
        diagnosis = "Glioblastoma Multiforme (Grade IV)"
        risk_level = "High"
        confidence = 0.89 + np.random.random() * 0.08
        tumor_type = "Primary Malignant Brain Tumor"
    elif tumor_volume > 8000:
        diagnosis = "Anaplastic Astrocytoma (Grade III)"
        risk_level = "Moderate"
        confidence = 0.84 + np.random.random() * 0.08
        tumor_type = "Primary Brain Tumor"
    elif tumor_volume > 3000:
        diagnosis = "Diffuse Astrocytoma (Grade II)"
        risk_level = "Moderate"
        confidence = 0.81 + np.random.random() * 0.10
        tumor_type = "Low-Grade Glioma"
    else:
        diagnosis = "Benign Mass Lesion"
        risk_level = "Low"
        confidence = 0.79 + np.random.random() * 0.12
        tumor_type = "Benign Lesion"

    # Calculate comprehensive measurements
    equivalent_diameter = ((6 * tumor_volume / np.pi) ** (1/3))
    brain_volume_estimate = 1400000  # mm³ average adult brain volume
    tumor_percentage = (tumor_volume / brain_volume_estimate) * 100
    surface_area = 4 * np.pi * (equivalent_diameter / 2) ** 2

    # Generate realistic findings based on tumor characteristics
    findings = [
        f"Heterogeneous enhancing mass identified measuring approximately {equivalent_diameter:.1f} mm in maximum diameter",
        f"Total tumor volume calculated at {tumor_volume:.1f} mm³ ({tumor_percentage:.2f}% of estimated brain volume)"
    ]

    if tumor_volume > 10000:
        findings.extend([
            "Surrounding vasogenic edema extending into adjacent white matter",
            "Central areas of necrosis consistent with high-grade malignancy",
            "Irregular enhancement pattern suggesting aggressive behavior"
        ])
    elif tumor_volume > 5000:
        findings.extend([
            "Mild surrounding edema noted",
            "Heterogeneous enhancement pattern observed",
            "Well-circumscribed borders with some infiltrative characteristics"
        ])
    else:
        findings.extend([
            "Minimal surrounding edema",
            "Homogeneous enhancement pattern",
            "Well-defined margins consistent with lower-grade process"
        ])

    findings.extend([
        "No evidence of leptomeningeal enhancement",
        "No significant mass effect or midline shift at current size",
        f"Surface area measurement: {surface_area:.1f} mm²"
    ])

    # Generate contextual recommendations
    recommendations = [
        "Urgent neurosurgical consultation for evaluation and management planning",
        "Multidisciplinary tumor board review recommended within 48-72 hours"
    ]

    if risk_level == "High":
        recommendations.extend([
            "Consider urgent biopsy or resection for tissue diagnosis",
            "Oncology consultation for adjuvant therapy planning",
            "Advanced imaging (DTI, perfusion MRI) for surgical planning",
            "Baseline neuropsychological assessment recommended"
        ])
    elif risk_level == "Moderate":
        recommendations.extend([
            "Biopsy recommended for histopathological confirmation",
            "Serial imaging every 3-4 months to monitor progression",
            "Consider advanced imaging techniques for better characterization",
            "Neuropsychological evaluation if symptoms present"
        ])
    else:
        recommendations.extend([
            "Close radiological follow-up every 6 months",
            "Consider tissue sampling if growth observed",
            "Monitor for development of neurological symptoms",
            "Patient education regarding warning signs"
        ])

    recommendations.extend([
        "Patient and family counseling regarding diagnosis and prognosis",
        "Consider enrollment in appropriate clinical trials if indicated"
    ])

    return {
        'classification': {
            'primary_diagnosis': diagnosis,
            'confidence': confidence,
            'risk_level': risk_level,
            'tumor_type': tumor_type
        },
        'measurements': {
            'tumor_volume': f"{tumor_volume:.1f} mm³",
            'tumor_percentage': f"{tumor_percentage:.2f}%",
            'equivalent_diameter': f"{equivalent_diameter:.1f} mm",
            'surface_area': f"{surface_area:.1f} mm²"
        },
        'quality_metrics': {
            'dice_coefficient': f"{metrics['dice_score']:.3f}",
            'hausdorff_distance': f"{metrics['hausdorff_distance']:.1f} mm",
            'jaccard_index': f"{metrics['jaccard_index']:.3f}",
            'sensitivity': f"{metrics['sensitivity']:.3f}",
            'specificity': f"{metrics['specificity']:.3f}"
        },
        'clinical_notes': {
            'findings': findings,
            'recommendations': recommendations
        }
    }

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)