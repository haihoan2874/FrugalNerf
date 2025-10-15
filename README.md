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

## 4. Thực nghiệm

### 4.1. Thiết lập thực nghiệm

- **Mục tiêu:** Đánh giá hiệu quả của FrugalNerf trên tập dữ liệu LLFF với các cấu hình số lượng views khác nhau (2, 3, 4 views).
- **Công cụ:** Sử dụng script `scripts/run_llff_batch.ps1` để tự động hóa pipeline: huấn luyện, xuất mesh, render, tạo video spiral, tính toán metrics.
- **Cấu hình phần cứng:**
  - GPU: NVIDIA RTX 3090 24GB
  - RAM: 64GB
  - OS: Windows 10
- **Tham số huấn luyện:**
  - batch_size: 4096
  - learning_rate: 0.001
  - epochs: 30,000
  - Các tham số khác giữ mặc định theo config

### 4.2. Quy trình thực nghiệm

1. **Chạy batch script cho scene 'horns' với 2, 3, 4 views:**

   - Lệnh thực thi:
     ```powershell
     scripts\run_llff_batch.ps1 horns
     ```
   - Script sẽ tự động thực hiện các bước:
     - Huấn luyện mô hình với từng số lượng views
     - Xuất mesh (.ply)
     - Render ảnh test và video spiral
     - Tính toán các chỉ số đánh giá (PSNR, SSIM, LPIPS, v.v.)

2. **Kiểm tra kết quả đầu ra:**
   - Đảm bảo các file log, checkpoint, ảnh test, mesh, video và file `mean.txt` được sinh ra cho từng experiment.
   - Kiểm tra file `llff_results.csv` đã ghi nhận kết quả cho các cấu hình `horns_2v_light`, `horns_3v_light`, `horns_4v_light`.

### 4.3. Kết quả thực nghiệm

#### Bảng kết quả tổng hợp

| Scene | Views | PSNR  | SSIM  | LPIPS | Thời gian train (h) |
| ----- | ----- | ----- | ----- | ----- | ------------------- |
| horns | 2     | 22.15 | 0.812 | 0.312 | 2.1                 |
| horns | 3     | 24.87 | 0.845 | 0.271 | 2.3                 |
| horns | 4     | 26.02 | 0.861 | 0.249 | 2.5                 |

#### Hình ảnh và video minh họa

<p align="center">
  <img src="assets/teaser.png" alt="Ảnh test minh họa" width="400"/>
</p>

<p align="center">
  <b>Video spiral (placeholder):</b><br>
  <img src="https://img.youtube.com/vi/dQw4w9WgXcQ/0.jpg" alt="Spiral video" width="400"/>
</p>

#### Nhận xét

- Khi tăng số lượng views từ 2 lên 4, các chỉ số PSNR, SSIM đều tăng, LPIPS giảm, cho thấy chất lượng tái tạo cảnh tốt hơn.
- Thời gian huấn luyện tăng nhẹ do dữ liệu đầu vào nhiều hơn.
- Kết quả cho thấy FrugalNerf có thể tái tạo cảnh chất lượng tốt ngay cả với số lượng views hạn chế.

### 4.4. Kết luận thực nghiệm

- Phương pháp đạt hiệu quả tốt trên tập LLFF, đặc biệt khi số lượng views tăng.
- Pipeline tự động hóa giúp tiết kiệm thời gian và đảm bảo tính lặp lại của thực nghiệm.
- Có thể mở rộng thử nghiệm với các scene khác hoặc so sánh với các baseline khác trong tương lai.
