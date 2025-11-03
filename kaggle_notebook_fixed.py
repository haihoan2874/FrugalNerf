#!/usr/bin/env python3
"""
FrugalNeRF Training Notebook for Kaggle - FIXED VERSION
=======================================================

This notebook provides a complete setup for training FrugalNeRF on Kaggle.
It supports multiple dataset types: LLFF, DTU, RealEstate10K, and custom datasets.

FIXED: Proper directory handling for Kaggle environment

Usage:
1. Set DATASET_TYPE and DATASET_PATH variables
2. Upload your dataset to Kaggle
3. Run this notebook

Author: haihoan2874
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# ========== CONFIGURATION SECTION ==========
# Change these variables according to your dataset

DATASET_TYPE = "your_own_data"  # Options: "llff", "dtu", "realestate10k", "your_own_data"
DATASET_PATH = "/kaggle/input/frugalnerf-habitat67"  # Path to your dataset on Kaggle
EXPERIMENT_NAME = "habitat67_experiment"  # Name for this experiment
TRAINING_ITERATIONS = 50000  # Number of training iterations

# Dataset-specific configurations
if DATASET_TYPE == "llff":
    CONFIG_TEMPLATE = "llff_default_2v.txt"
elif DATASET_TYPE == "dtu":
    CONFIG_TEMPLATE = "dtu_default_2v.txt"
elif DATASET_TYPE == "realestate10k":
    CONFIG_TEMPLATE = "realestate10k_defalt_2v.txt"
else:  # your_own_data
    CONFIG_TEMPLATE = "your_own_data.txt"

# ===========================================

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
        result = os.system("git clone https://github.com/haihoan2874/FrugalNerf.git FrugalNeRF")
        if result == 0:
            print("✅ Repository cloned successfully")
        else:
            print("❌ Failed to clone repository")
            return False
    else:
        print("📁 Repository already exists")

    # Change to repository directory
    if os.path.exists("FrugalNeRF"):
        os.chdir("FrugalNeRF")
        print("✅ Changed to FrugalNeRF directory")
        return True
    else:
        print("❌ FrugalNeRF directory not found after cloning")
        return False

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
    print(f"📊 Setting up {DATASET_TYPE} dataset...")

    # Check if dataset exists
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Dataset not found at: {DATASET_PATH}")
        print("Please check your dataset path and ensure it's added to the notebook.")
        return None

    print(f"✅ Dataset found at: {DATASET_PATH}")

    # Dataset-specific checks
    if DATASET_TYPE == "your_own_data":
        # Check for custom dataset structure
        images_path = os.path.join(DATASET_PATH, "images")
        meta_path = os.path.join(DATASET_PATH, "frugal_dataset.txt")

        if os.path.exists(images_path):
            num_images = len([f for f in os.listdir(images_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
            print(f"📸 Found {num_images} images")
        else:
            print("⚠️  Images directory not found")

        if os.path.exists(meta_path):
            print("📄 Metadata file found")
        else:
            print("⚠️  Metadata file not found")

    elif DATASET_TYPE in ["llff", "dtu", "realestate10k"]:
        # Check for standard dataset structure
        scenes = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]
        print(f"📁 Found {len(scenes)} scenes: {scenes[:5]}...")  # Show first 5

    return DATASET_PATH

def create_config_file(dataset_path, config_name):
    """Create configuration file for the selected dataset"""

    if DATASET_TYPE == "your_own_data":
        config_content = f"""expname = {EXPERIMENT_NAME}
basedir = ./logs

datadir = {dataset_path}
dataset_type = your_own_data

factor = 2
llffhold = 8

N_samples = 64
N_importance = 128
N_rand = 1024
N_iters = {TRAINING_ITERATIONS}

D = 8
W = 256
chunk = 8192
ckpt = None

lr = 5e-4
lr_decay = 250
decay_step = 25

white_bkgd = False
"""
    else:
        # Load template config and modify
        template_path = f"configs/{CONFIG_TEMPLATE}"
        if os.path.exists(template_path):
            with open(template_path, 'r') as f:
                config_content = f.read()

            # Replace key parameters
            config_content = config_content.replace("expname = ", f"expname = {EXPERIMENT_NAME}")
            config_content = config_content.replace("N_iters = ", f"N_iters = {TRAINING_ITERATIONS}")
            # Add datadir if not present
            if "datadir =" not in config_content:
                config_content = f"datadir = {dataset_path}\n" + config_content
        else:
            print(f"❌ Template config {template_path} not found")
            return None

    config_path = f"configs/{config_name}"
    with open(config_path, 'w') as f:
        f.write(config_content)

    print(f"✅ Config file created: {config_path}")
    return config_path

def train_model(config_path, gpu_id=0):
    """Train the FrugalNeRF model"""
    print("🎯 Starting FrugalNeRF training...")

    train_command = f"""python train.py \
  --config {config_path} \
  --gpu {gpu_id} \
  --N_iters {TRAINING_ITERATIONS}"""

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

    render_command = f"""python renderer.py \
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
    print(f"Dataset Type: {DATASET_TYPE}")
    print(f"Dataset Path: {DATASET_PATH}")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Iterations: {TRAINING_ITERATIONS}")
    print("=" * 50)

    # Step 1: Setup environment
    device = setup_environment()

    # Step 2: Clone repository
    if not clone_repository():
        print("❌ Failed to clone/setup repository")
        return

    # Step 3: Install dependencies
    install_dependencies()

    # Step 4: Setup dataset
    dataset_path = setup_dataset()
    if dataset_path is None:
        print("❌ Cannot proceed without dataset")
        return

    # Step 5: Create config file
    config_name = f"{DATASET_TYPE}_config.txt"
    config_path = create_config_file(dataset_path, config_name)
    if config_path is None:
        print("❌ Cannot proceed without config file")
        return

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
