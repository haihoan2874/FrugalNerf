# FrugalNeRF

## [Project page](https://linjohnss.github.io/frugalnerf/) | [Paper](https://arxiv.org/abs/2410.16271)

This repository contains a pytorch implementation for the paper: [FrugalNeRF: Fast Convergence for Extreme Few-shot Novel View Synthesis without Learned Priors](https://linjohnss.github.io/frugalnerf/). Our work presents a simple baseline to reconstruct radiance fields in few-shot setting, which achieves **fast** training process without learned priors.<br><br>

![teaser](assets/teaser.png)

## System Requirements

- OS: Windows 10/11 or Ubuntu 20.04+
- GPU: NVIDIA GPU with CUDA support (tested on RTX 3060 and above)
- RAM: 16GB minimum, 32GB recommended
- Storage: At least 10GB free space for datasets and model checkpoints

## Installation

### 1. Install CUDA and Python Dependencies

1. Install [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (11.3 or higher)
2. Install [Conda](https://docs.conda.io/en/latest/miniconda.html)

### 2. Create and Setup Environment

```bash
# Create conda environment
conda create -n frugalnerf python=3.8
conda activate frugalnerf

# Install PyTorch with CUDA support (adjust cuda version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install required packages
pip install tqdm scikit-image opencv-python configargparse lpips imageio-ffmpeg
pip install kornia tensorboard torchmetrics plyfile pandas timm
pip install torch-efficient-distloss

# Install COLMAP (required for dataset processing)
# For Windows: Download and install from https://github.com/colmap/colmap/releases
# For Ubuntu: sudo apt-get install colmap
```

### 3. Verify Installation

```bash
# Check CUDA is available
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

## Dataset Setup

### 1. Download Datasets

- **LLFF Dataset**:

  ```bash
  # Create data directory
  mkdir -p data/nerf_llff_data
  cd data/nerf_llff_data

  # Download and extract LLFF dataset
  # Option 1: Download from original source
  wget https://drive.google.com/drive/folders/14boI-o5hGO9srnWaaogTU5_ji7wkX2S7

  # Option 2: Or download individual scenes
  # fern, flower, fortress, horns, leaves, orchids, room, trex
  ```

- **Directory Structure**:
  ```
  data/
  └── nerf_llff_data/
      ├── fern/
      │   ├── images/
      │   └── poses_bounds.npy
      ├── flower/
      ├── horns/
      └── ...
  ```

### 2. Dataset Preprocessing (Optional for better quality)

```bash
# Generate sparse depth for LLFF dataset (if needed)
python extra/colmap_llff.py --data_dir ./data/nerf_llff_data/horns

# Verify dataset structure
python -c "from dataLoader.llff import LLFFDataset; ds=LLFFDataset('./data/nerf_llff_data/horns')"
```

## Quick Start

### Basic Training & Testing

1. **Training Multiple Views** (Using PowerShell Script):

```powershell
# Train on a scene with 2, 3 and 4 views
.\scripts\run_llff_batch.ps1 -Scenes @("horns") -Views @(2,3,4)

# Only run specific views
.\scripts\run_llff_batch.ps1 -Scenes @("fern") -Views @(2,3)

# Skip training and only render if checkpoints exist
.\scripts\run_llff_batch.ps1 -Scenes @("room") -Views @(2,3,4) -SkipTrain
```

The batch script will:

- Train models for each view configuration
- Export mesh (.ply file)
- Render test images
- Create spiral view videos
- Compute quality metrics

2. **Manual Training** (Single Model):

```bash
# Train with minimal settings (faster)
python train.py --config configs/llff_light_2v.txt --datadir ./data/nerf_llff_data/horns --train_frame_num 0 3 --test_frame_num 6

# Train with better quality settings
python train.py --config configs/llff_default_2v.txt --datadir ./data/nerf_llff_data/horns --train_frame_num 0 3 --test_frame_num 6
```

### Configuration & Outputs

1. **Config Files**:

- `llff_light_2v.txt`: Faster training, lower quality
- `llff_default_2v.txt`: Better quality, slower training
- Key parameters:
  ```bash
  downsample_train = 8.0     # Higher = faster but lower quality (4.0-8.0)
  n_iters = 2000            # Number of training iterations
  batch_size = 8192         # Chunk size for rendering
  ```

2. **Output Structure**:

```
log/
└── scene_Nv_light/          # e.g. fern_2v_light
    ├── scene_Nv_light.th    # Model checkpoint
    ├── imgs_test_all/       # Test view renderings
    ├── imgs_spiral/         # Novel view video frames
    ├── scene_Nv_light.ply   # Exported mesh
    └── mean.txt             # Quality metrics (PSNR/SSIM/LPIPS)
```

3. **Generated Files**:

- Metrics & Logs:

  - `metrics_test_scene_Nv.txt`: Test view metrics
  - `metrics_novel_scene_Nv.txt`: Novel view metrics
  - `run_scene_Nv.txt`: Training log
  - `render_scene_Nv.txt`: Rendering log
  - `spiral_scene_Nv.txt`: Video generation log

- Results CSV:
  ```
  frugal_results.csv  # Summary of all experiments:
  scene,views,expname,psnr,ssim,lpips,train_time_seconds,ckpt
  fern,2,fern_2v_light,19.21,0.59,0.32,1200,/path/to/ckpt
  ...
  ```

## Best Practices & Troubleshooting

1. **Recommended Workflow**:

   - Start with 2-view training to verify setup
   - Use light config for initial testing
   - Progress to 3 and 4 views for better quality
   - Check generated videos in imgs_spiral/
   - Compare metrics in frugal_results.csv

2. **Common Issues**:

   ```bash
   # CUDA Out of Memory
   - Reduce batch_size or chunk size
   - Increase downsample_train

   # Poor Quality
   - Try default config instead of light
   - Add more training views (3 or 4)
   - Verify test frame is between train frames

   # Failed Video Generation
   - Check checkpoint exists
   - Ensure render_spiral flag is set
   - Verify enough GPU memory for rendering
   ```

3. **Dataset Requirements**:
   - Images must be sequential & overlap
   - COLMAP-processed poses_bounds.npy
   - Consistent image dimensions
   - Standard format (JPG/PNG)

<!-- ## Training with your own data
We provide code for training on your own image set:
Calibrating images with the script from [NGP](https://github.com/NVlabs/instant-ngp/blob/master/docs/nerf_dataset_tips.md):
`python dataLoader/colmap2nerf.py --colmap_matcher exhaustive --run_colmap`, then adjust the datadir in `configs/your_own_data.txt`. Please check the `scene_bbox` and `near_far` if you get abnormal results.
     -->

## Citation

If you find our code or paper helps, please consider citing:

```
@inproceedings{lin2024frugalnerf,
  title={FrugalNeRF: Fast Convergence for Extreme Few-shot Novel View Synthesis without Learned Priors},
  author={Chen, Po-Yi and Lin, Yueh-Cheng and Mui, Paul and Lin, Guan-Ting and Liu, Yen-Cheng and Chen, Kai},
  journal={arXiv preprint arXiv:2410.16271},
  year={2024}
}
```

## Acknowledgments

This implementation builds upon several excellent open-source projects:

- [TensoRF](https://github.com/apchenstu/TensoRF) - Base architecture and tensor decomposition
- [LLFF](https://github.com/Fyusion/LLFF) - Dataset processing and pose estimation
- [COLMAP](https://github.com/colmap/colmap) - Structure-from-Motion and camera calibration
- [IBRNet](https://github.com/googleinterns/IBRNet) - Neural rendering concepts

The code is available under the MIT license. Original licenses for referenced projects can be found in the `licenses/` folder.
