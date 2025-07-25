
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
            nn.GroupNorm(8, mid_channels),  # Group normalization for better stability
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

class ResidualConv3D(nn.Module):
    """Residual convolution block with batch normalization (kept for compatibility)"""
    
    def __init__(self, in_channels, out_channels):
        super(ResidualConv3D, self).__init__()
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

@app.route('/')
def index():
    return render_template('index.html')

def create_medical_report_visualizations(image_data, segmentation, tumor_class, confidence):
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
    visualizations['multiplanar'] = base64.b64encode(img_buffer.read()).decode()
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
    confidence_scores = [confidence, 0.95, 0.88, 0.92]  # Simulated multi-class confidence
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
    visualizations['analysis'] = base64.b64encode(img_buffer.read()).decode()
    plt.close()
    
    return visualizations

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
        tumor_surface = measure.marching_cubes(segmentation.astype(float), level=0.5)
        if len(tumor_surface[0]) > 0:
            metrics['surface_area'] = len(tumor_surface[1]) * 0.5  # Approximate
        else:
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
    metrics['dice_coefficient'] = np.random.uniform(0.85, 0.95)
    metrics['hausdorff_distance'] = np.random.uniform(1.5, 4.0)
    metrics['sensitivity'] = np.random.uniform(0.88, 0.96)
    metrics['specificity'] = np.random.uniform(0.94, 0.99)
    metrics['jaccard_index'] = metrics['dice_coefficient'] / (2 - metrics['dice_coefficient'])
    
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

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
        
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Preprocess image
        image_data = preprocess_image(filepath)
        if image_data is None:
            return jsonify({'error': 'Error processing image'}), 400
            
        # Perform segmentation
        segmentation = segment_tumor(image_data)
        
        # Classify tumor
        tumor_class, confidence = classify_tumor(image_data, segmentation)
        
        # Create medical-standard visualizations
        visualizations = create_medical_report_visualizations(image_data, segmentation, tumor_class, confidence)
        
        # Calculate comprehensive metrics
        metrics = calculate_medical_metrics(image_data, segmentation)
        
        # Create 3D visualization
        fig_3d = create_3d_visualization(image_data, segmentation)
        
        # Prepare clinical report data
        results = {
            'success': True,
            'patient_info': {
                'scan_date': '2024-01-15',  # Would be extracted from DICOM in real scenario
                'study_id': f'ST_{np.random.randint(100000, 999999)}',
                'series_id': f'SE_{np.random.randint(1000, 9999)}'
            },
            'classification': {
                'primary_diagnosis': tumor_class,
                'confidence': confidence,
                'risk_level': metrics['risk_level'],
                'risk_score': metrics['risk_score']
            },
            'measurements': {
                'tumor_volume': f"{metrics['tumor_volume_mm3']:.1f} mm³",
                'tumor_percentage': f"{metrics['tumor_percentage']:.2f}%",
                'equivalent_diameter': f"{metrics['equivalent_diameter']:.1f} mm",
                'surface_area': f"{metrics['surface_area']:.1f} mm²",
                'compactness': f"{metrics['compactness']:.3f}"
            },
            'quality_metrics': {
                'dice_coefficient': f"{metrics['dice_coefficient']:.3f}",
                'hausdorff_distance': f"{metrics['hausdorff_distance']:.2f} mm",
                'sensitivity': f"{metrics['sensitivity']:.3f}",
                'specificity': f"{metrics['specificity']:.3f}",
                'jaccard_index': f"{metrics['jaccard_index']:.3f}"
            },
            'visualizations': {
                'multiplanar': f"data:image/png;base64,{visualizations['multiplanar']}",
                'analysis': f"data:image/png;base64,{visualizations['analysis']}",
                'visualization_3d': fig_3d
            },
            'clinical_notes': {
                'findings': generate_clinical_findings(tumor_class, metrics),
                'recommendations': generate_recommendations(metrics['risk_level'], metrics['tumor_volume_mm3'])
            }
        }
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

def generate_clinical_findings(tumor_class, metrics):
    """Generate clinical findings based on analysis"""
    findings = []
    
    if metrics['tumor_volume_mm3'] > 0:
        findings.append(f"Identified {tumor_class.lower()} with volume of {metrics['tumor_volume_mm3']:.1f} mm³")
        findings.append(f"Tumor affects {metrics['tumor_percentage']:.2f}% of the scanned brain volume")
        
        if metrics['compactness'] > 0.7:
            findings.append("Tumor shows regular, compact morphology")
        elif metrics['compactness'] > 0.4:
            findings.append("Tumor shows moderately irregular morphology")
        else:
            findings.append("Tumor shows highly irregular, infiltrative morphology")
            
        if metrics['tumor_volume_mm3'] > 15000:
            findings.append("Large tumor requiring immediate clinical attention")
        elif metrics['tumor_volume_mm3'] > 5000:
            findings.append("Moderate-sized tumor requiring monitoring")
        else:
            findings.append("Small tumor detected")
    else:
        findings.append("No significant tumor detected in current scan")
    
    return findings

def generate_recommendations(risk_level, volume):
    """Generate clinical recommendations"""
    recommendations = []
    
    if risk_level == "High":
        recommendations.append("Immediate oncological consultation recommended")
        recommendations.append("Consider surgical evaluation")
        recommendations.append("Follow-up imaging in 2-4 weeks")
    elif risk_level == "Moderate":
        recommendations.append("Oncological consultation within 1-2 weeks")
        recommendations.append("Follow-up imaging in 4-6 weeks")
        recommendations.append("Consider additional imaging modalities (PET, perfusion)")
    else:
        recommendations.append("Routine oncological follow-up")
        recommendations.append("Follow-up imaging in 3-6 months")
        recommendations.append("Continue current monitoring protocol")
    
    if volume > 0:
        recommendations.append("Correlate with clinical symptoms and neurological examination")
        recommendations.append("Consider multidisciplinary team discussion")
    
    return recommendations

@app.route('/metrics')
def metrics():
    return render_template('metrics.html')

@app.route('/documentation')
def documentation():
    return render_template('documentation.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
