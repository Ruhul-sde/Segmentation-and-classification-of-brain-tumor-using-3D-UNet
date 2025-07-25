
#!/usr/bin/env python3
"""
Project setup script for Brain Tumor Segmentation System
Creates necessary directories and initializes the project structure
"""

import os
import sys
import subprocess

def create_directories():
    """Create necessary project directories"""
    directories = [
        'data',
        'data/raw',
        'data/processed',
        'data/synthetic',
        'models',
        'checkpoints',
        'logs',
        'uploads',
        'static',
        'static/css',
        'static/js',
        'static/images',
        'templates',
        'results',
        'results/visualizations',
        'results/reports',
        'utils',
        'tests'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def create_init_files():
    """Create __init__.py files for Python packages"""
    init_files = [
        'utils/__init__.py',
        'tests/__init__.py'
    ]
    
    for init_file in init_files:
        with open(init_file, 'w') as f:
            f.write('# Package initialization file\n')
        print(f"Created: {init_file}")

def create_gitignore():
    """Create .gitignore file"""
    gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# Data files
data/raw/*
data/processed/*
!data/raw/.gitkeep
!data/processed/.gitkeep

# Model files
models/*.pth
models/*.h5
checkpoints/*.pth

# Logs
logs/*.log
*.log

# Uploads
uploads/*
!uploads/.gitkeep

# Results
results/visualizations/*
results/reports/*
!results/visualizations/.gitkeep
!results/reports/.gitkeep

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Temporary files
*.tmp
*.temp
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content.strip())
    print("Created .gitignore")

def create_placeholder_files():
    """Create placeholder files for empty directories"""
    placeholder_dirs = [
        'data/raw',
        'data/processed',
        'uploads',
        'results/visualizations',
        'results/reports'
    ]
    
    for directory in placeholder_dirs:
        placeholder_path = os.path.join(directory, '.gitkeep')
        with open(placeholder_path, 'w') as f:
            f.write('# Placeholder file to keep directory in git\n')

def check_dependencies():
    """Check if all required dependencies are available"""
    required_packages = [
        'numpy', 'scipy', 'matplotlib', 'plotly', 'nibabel',
        'sklearn', 'cv2', 'PIL', 'flask', 'torch', 'pandas'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'PIL':
                import PIL
            elif package == 'sklearn':
                import sklearn
            else:
                __import__(package)
            print(f"✓ {package} is available")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} is missing")
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Please install missing packages using the Dependencies tool")
    else:
        print("\n✓ All required packages are available!")

def main():
    """Main setup function"""
    print("Setting up Brain Tumor Segmentation Project Structure...")
    print("=" * 60)
    
    create_directories()
    print()
    
    create_init_files()
    print()
    
    create_gitignore()
    print()
    
    create_placeholder_files()
    print()
    
    print("Checking dependencies...")
    check_dependencies()
    print()
    
    print("=" * 60)
    print("Project setup complete!")
    print("=" * 60)
    
    print("\nProject Structure:")
    print("├── data/                    # Dataset storage")
    print("│   ├── raw/                # Raw medical images")
    print("│   ├── processed/          # Preprocessed data")
    print("│   └── synthetic/          # Synthetic training data")
    print("├── models/                 # Trained model files")
    print("├── checkpoints/            # Training checkpoints")
    print("├── logs/                   # Training and application logs")
    print("├── uploads/                # User uploaded files")
    print("├── static/                 # Web assets (CSS, JS, images)")
    print("├── templates/              # HTML templates")
    print("├── results/                # Analysis results and reports")
    print("├── utils/                  # Utility functions")
    print("├── tests/                  # Unit tests")
    print("├── main.py                 # Flask web application")
    print("├── training.py             # Model training script")
    print("├── config.py               # Configuration settings")
    print("└── requirements.txt        # Python dependencies")

if __name__ == "__main__":
    main()
