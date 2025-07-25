
#!/usr/bin/env python3
"""
Validate project setup and dependencies
"""

import os
import sys

def check_project_structure():
    """Check if all required directories exist"""
    required_dirs = [
        'data', 'data/raw', 'data/processed', 'data/synthetic',
        'models', 'checkpoints', 'logs', 'uploads', 'static',
        'templates', 'results', 'utils', 'tests'
    ]
    
    missing_dirs = []
    for directory in required_dirs:
        if not os.path.exists(directory):
            missing_dirs.append(directory)
    
    if missing_dirs:
        print(f"Missing directories: {missing_dirs}")
        return False
    else:
        print("✓ All required directories exist")
        return True

def check_core_files():
    """Check if core project files exist"""
    core_files = [
        'main.py', 'config.py', 'training.py', 'losses.py',
        'utils/data_loader.py', 'utils/visualization.py', 'utils/metrics.py'
    ]
    
    missing_files = []
    for file_path in core_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"Missing core files: {missing_files}")
        return False
    else:
        print("✓ All core files exist")
        return True

def test_imports():
    """Test critical imports"""
    try:
        import numpy as np
        import torch
        import flask
        import matplotlib.pyplot as plt
        print("✓ Critical imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def main():
    """Main validation function"""
    print("Validating project setup...")
    print("=" * 40)
    
    structure_ok = check_project_structure()
    files_ok = check_core_files()
    imports_ok = test_imports()
    
    print("=" * 40)
    
    if structure_ok and files_ok and imports_ok:
        print("✓ Project setup validation PASSED")
        print("You can now run: python main.py")
        return True
    else:
        print("✗ Project setup validation FAILED")
        print("Please run: python setup_project.py")
        return False

if __name__ == "__main__":
    main()
