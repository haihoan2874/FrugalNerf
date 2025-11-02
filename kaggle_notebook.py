#!/usr/bin/env python3
"""
FrugalNeRF Training Notebook for Kaggle
========================================

This notebook provides a complete setup for training FrugalNeRF on Kaggle.
It includes data loading, model training, and rendering.

Usage:
1. Clone the repository
2. Upload your dataset to Kaggle
3. Run this notebook

Author: haihoan2874
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

def setup_environment():
    """Setup the training environment"""
    print("🚀 Setting up FrugalNeRF training environment...")

    # Check GPU availability
    if torch.cuda.is_available():
        print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
        device = torch.device("cuda:0")
    else:
        print("❌ No GPU available, using CPU")
        device = torch.device("cpu")

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    return device

def clone_repository():
    """Clone the FrugalNeRF repository"""
    if not os.path.exists("FrugalNeRF"):
        print("📥 Cloning FrugalNeRF repository...")
        os.system("git clone https://github.com/haihoan2874/FrugalNerf.git")
        print("✅ Repository cloned successfully")
    else:
        print("📁 Repository already exists")

    # Change to repository directory
    os.chdir("FrugalNeRF")

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")

    # Install PyTorch (if not already installed)
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
    except ImportError:
        print("Installing PyTorch...")
        os.system("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

    # Install other dependencies
    dependencies = [
        "tqdm",
        "scikit-image",
        "opencv-python",
        "configargparse",
        "lpips",
        "imageio-ffmpeg"
    ]

    for dep in dependencies:
        try:
            __import__(dep.replace("-", "_"))
            print(f"✅ {dep} already installed")
        except ImportError:
            print(f"Installing {dep}...")
            os.system(f"pip install {dep}")

    print("✅ All dependencies installed")

def setup_dataset():
    """Setup the dataset for training"""
    print("📊 Setting up dataset...")

    # Check if dataset exists
    dataset_paths = [
        "../frugalnerf_data",
        "./frugalnerf_data",
        "/kaggle/input/frugalnerf-dataset"
    ]

    dataset_path = None
    for path in dataset_paths:
        if os.path.exists(path):
            dataset_path = path
            break

    if dataset_path is None:
        print("❌ Dataset not found. Please upload your dataset to Kaggle or place it in the correct directory.")
        print("Expected locations:")
        for path in dataset_paths:
            print(f"  - {path}")
        return None

    print(f"✅ Dataset found at: {dataset_path}")

    # Check dataset structure
    images_path = os.path.join(dataset_path, "images")
    if os.path.exists(images_path):
        num_images = len([f for f in os.listdir(images_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
        print(f"📸 Found {num_images} images in dataset")
    else:
        print("⚠️  Images directory not found")

    return dataset_path

def create_config_file(dataset_path, config_name="your_own_data.txt"):
    """Create configuration file for custom dataset"""
    config_path = f"FrugalNeRF/configs/{config_name}"

    config_content = f"""expname = habitat67_experiment
basedir = ./logs

datadir = {dataset_path}
dataset_type = your_own_data

factor = 2
llffhold = 8

N_samples = 64
N_importance = 128
N_rand = 1024
N_iters = 50000

D = 8
W = 256
chunk = 8192
ckpt = None

lr = 5e-4
lr_decay = 250
decay_step = 25

white_bkgd = False
"""

    with open(config_path, 'w') as f:
        f.write(config_content)

    print(f"✅ Config file created: {config_path}")
    return config_path

def train_model(config_path, gpu_id=0):
    """Train the FrugalNeRF model"""
    print("🎯 Starting FrugalNeRF training...")

    train_command = f"""python FrugalNeRF/train.py \
  --config {config_path} \
  --gpu {gpu_id} \
  --N_iters 50000"""

    print(f"Running: {train_command}")
    result = os.system(train_command)

    if result == 0:
        print("✅ Training completed successfully!")
    else:
        print("❌ Training failed")

    return result == 0

def render_results(config_path, gpu_id=0):
    """Render results after training"""
    print("🎨 Rendering results...")

    render_command = f"""python FrugalNeRF/renderer.py \
  --config {config_path} \
  --render_only \
  --render_test"""

    print(f"Running: {render_command}")
    result = os.system(render_command)

    if result == 0:
        print("✅ Rendering completed successfully!")
    else:
        print("❌ Rendering failed")

    return result == 0

def main():
    """Main training pipeline"""
    print("🎯 FrugalNeRF Kaggle Training Pipeline")
    print("=" * 50)

    # Step 1: Setup environment
    device = setup_environment()

    # Step 2: Clone repository
    clone_repository()

    # Step 3: Install dependencies
    install_dependencies()

    # Step 4: Setup dataset
    dataset_path = setup_dataset()
    if dataset_path is None:
        print("❌ Cannot proceed without dataset")
        return

    # Step 5: Create config file
    config_path = create_config_file(dataset_path)

    # Step 6: Train model
    if train_model(config_path):
        # Step 7: Render results
        render_results(config_path)
    else:
        print("❌ Training failed, skipping rendering")

    print("\n🎉 Pipeline completed!")
    print("Check the 'logs' directory for results")

if __name__ == "__main__":
    main()
