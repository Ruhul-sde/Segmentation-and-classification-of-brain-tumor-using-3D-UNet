
#!/usr/bin/env python3
"""
Application startup script with environment validation
"""

import os
import sys
import subprocess

def main():
    """Main startup function"""
    print("🧠 Brain Tumor Segmentation System")
    print("=" * 50)
    
    # Setup environment
    print("Setting up environment...")
    try:
        from environment import setup_environment, print_system_info
        device = setup_environment()
        print_system_info()
    except ImportError:
        print("Environment module not found, running basic setup...")
        device = "cpu"
    
    # Validate setup
    print("\nValidating project setup...")
    try:
        from validate_setup import main as validate
        if not validate():
            print("Setup validation failed. Running setup...")
            subprocess.run([sys.executable, "setup_project.py"])
    except ImportError:
        print("Validation module not found, assuming setup is correct...")
    
    # Create synthetic data if no real data exists
    data_dir = "data/raw"
    if os.path.exists(data_dir) and len(os.listdir(data_dir)) == 0:
        print("\nNo training data found. Creating synthetic data...")
        try:
            from utils.data_loader import create_synthetic_data
            create_synthetic_data("data/synthetic", num_samples=5)
            print("✓ Synthetic data created")
        except ImportError:
            print("Could not create synthetic data")
    
    # Start the application
    print(f"\n🚀 Starting application on device: {device}")
    print("=" * 50)
    print("Access the web interface at: http://localhost:5000")
    print("Press Ctrl+C to stop the application")
    print("=" * 50)
    
    try:
        import main
    except ImportError as e:
        print(f"Error starting application: {e}")
        print("Please ensure all dependencies are installed")
        sys.exit(1)

if __name__ == "__main__":
    main()
