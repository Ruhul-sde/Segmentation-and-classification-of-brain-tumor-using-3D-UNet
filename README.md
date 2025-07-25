
# Brain Tumor Segmentation and Classification System

A comprehensive 3D U-Net based system for brain tumor segmentation and classification using deep learning.

## Project Structure

```
├── data/                           # Dataset storage
│   ├── raw/                       # Raw medical images (NIfTI, DICOM)
│   ├── processed/                 # Preprocessed data
│   └── synthetic/                 # Synthetic training data
├── models/                        # Trained model files (.pth)
├── checkpoints/                   # Training checkpoints
├── logs/                          # Training and application logs
├── uploads/                       # User uploaded files
├── static/                        # Web assets (CSS, JS, images)
├── templates/                     # HTML templates
├── results/                       # Analysis results and reports
│   ├── visualizations/           # Generated plots and charts
│   └── reports/                  # Medical reports
├── utils/                         # Utility functions
│   ├── data_loader.py            # Data loading utilities
│   ├── visualization.py          # Plotting and visualization
│   └── metrics.py                # Evaluation metrics
├── tests/                         # Unit tests
├── main.py                        # Flask web application
├── training.py                    # Model training script
├── config.py                      # Configuration settings
├── losses.py                      # Loss functions
├── environment.py                 # Environment setup
└── setup_project.py               # Project initialization
```

## Features

- **3D U-Net Architecture**: Enhanced with attention mechanisms and residual connections
- **Web Interface**: User-friendly Flask-based interface for image upload and analysis
- **Medical-Standard Visualizations**: Multi-planar reconstructions, 3D volume rendering
- **Comprehensive Metrics**: Dice coefficient, Hausdorff distance, sensitivity, specificity
- **Clinical Reports**: Automated generation of medical-standard reports
- **Real-time Processing**: Efficient preprocessing and inference pipeline

## Model Architecture

The system uses an enhanced 3D U-Net with:
- Attention gates for improved feature focusing
- Residual connections for better gradient flow
- Deep supervision for enhanced learning
- Group normalization for stable training
- Mixed precision training support

## Usage

### 1. Setup Environment
```bash
python setup_project.py
python validate_setup.py
```

### 2. Run Web Application
```bash
python main.py
```

### 3. Train Model (Optional)
```bash
python training.py
```

## Dependencies

- PyTorch (with CUDA support if available)
- NumPy, SciPy
- Matplotlib, Plotly
- NiBabel (for NIfTI files)
- scikit-image, scikit-learn
- Flask, Werkzeug
- OpenCV, Pillow
- Pandas, Seaborn

## Medical Standards Compliance

The system follows medical imaging standards:
- DICOM compatibility
- Multi-planar reconstruction (MPR)
- Standardized reporting format
- Clinical metrics validation
- Quality assurance protocols

## Performance Metrics

- **Dice Coefficient**: Overlap measurement
- **Hausdorff Distance**: Boundary accuracy
- **Sensitivity**: True positive rate
- **Specificity**: True negative rate
- **Volume Measurements**: Tumor volume quantification

## Configuration

Modify `config.py` to adjust:
- Model architecture parameters
- Training hyperparameters
- Data preprocessing settings
- Evaluation metrics
- File paths and directories

## Development

1. Add test data to `data/raw/`
2. Configure parameters in `config.py`
3. Run training with `python training.py`
4. Evaluate with web interface via `python main.py`

## License

This project is for educational and research purposes.
