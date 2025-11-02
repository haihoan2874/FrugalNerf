# FrugalNeRF

## [Project page](https://linjohnss.github.io/frugalnerf/) | [Paper](https://arxiv.org/abs/2410.16271)

This repository contains a pytorch implementation for the paper: [FrugalNeRF: Fast Convergence for Extreme Few-shot Novel View Synthesis without Learned Priors](https://linjohnss.github.io/frugalnerf/). Our work presents a simple baseline to reconstruct radiance fields in few-shot setting, which achieves **fast** training process without learned proirs.<br><br>

![teaser](assets/teaser.png)

## Installation

#### Tested on Ubuntu 24.04 + Pytorch 2.4.1

Install environment:

```
conda create -n frugalnerf python=3.8
conda activate frugalnerf
pip install torch torchvision
pip install tqdm scikit-image opencv-python configargparse lpips imageio-ffmpeg kornia lpips tensorboard torchmetrics plyfile pandas timm
pip install torch-efficient-distloss
```

## Dataset

Please follow the instructions in [ViP-NeRF](https://github.com/NagabhushanSN95/ViP-NeRF/blob/main/src/database_utils/README.md) to set up various databases.

- [LLFF](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)
- [DTU](https://drive.google.com/file/d/1aTSmJa8Oo2qCc2Ce2kT90MHEA6UTSBKj/view?usp=share_link)
- [RealEstate10K](https://google.github.io/realestate10k/download.html)

<!-- ## Propressing

To generate sparse depth for LLFF dataset:
```bash
# Generate sparse depth for all scenes in LLFF dataset
python extra/colmap_llff.py --data_dir path/to/llff/dataset
``` -->

## Quick Start

The training script is in `train.py`, to train a FrugalNeRF:

For single scene training:

```bash
python train.py --config configs/llff_default_2v.txt --datadir ./data/nerf_llff_data/horns --train_frame_num 20 42 --test_frame_num 0 8 16 24 32 40 48 56
```

For training on entire dataset:

```bash
bash scripts/run_llff_2v.sh
```

## Rendering

```
python train.py --config configs/llff_default_2v.txt --ckpt path/to/your/checkpoint --render_only 1 --render_test 1
```

You can just simply pass `--render_only 1` and `--ckpt path/to/your/checkpoint` to render images from a pre-trained
checkpoint. You may also need to specify what you want to render, like `--render_test 1`, `--render_train 1` or `--render_path 1`.
The rendering results are located in your checkpoint folder.

<!-- ## Training with your own data
We provide code for training on your own image set:
Calibrating images with the script from [NGP](https://github.com/NVlabs/instant-ngp/blob/master/docs/nerf_dataset_tips.md):
`python dataLoader/colmap2nerf.py --colmap_matcher exhaustive --run_colmap`, then adjust the datadir in `configs/your_own_data.txt`. Please check the `scene_bbox` and `near_far` if you get abnormal results.
     -->

## Custom Dataset Setup (MapReduce Processed)

This project includes a custom pipeline for processing Habitat67 dataset using MapReduce and integrating it with FrugalNeRF training.

### Dataset Structure

After MapReduce processing, the dataset should be organized as:

```
habitat67/
├── images/
│   ├── frame_00001.jpg
│   ├── frame_00002.jpg
│   └── ... (161 ground-level images)
└── frugal_dataset.txt  (processed metadata with poses, intrinsics, bounds)
```

### MapReduce Processing Pipeline

1. **Input**: Raw images from Habitat67 dataset
2. **MapReduce Job**: Processes images through `FrugalNeRFMapper` and `FrugalNeRFReducer`
3. **Output**: `frugal_dataset.txt` with processed metadata

### Training with Custom Dataset

```bash
# Train on processed Habitat67 dataset
python train.py --config configs/your_own_data.txt --datadir ./habitat67
```

### DataLoader Integration

The `YourOwnDataset` class in `dataLoader/your_own_data.py` has been modified to:

- Parse `frugal_dataset.txt` format instead of JSON
- Load 161 processed images
- Extract camera poses, intrinsics, and bounds from text format

### MapReduce Components

- **FrugalNeRFMapper.java**: Processes individual images, estimates depth, generates rays
- **FrugalNeRFReducer.java**: Aggregates processed data by scene
- **FrugalNeRFJob.java**: Main job configuration and execution

### Notes

- Dataset contains 161 ground-level images from Habitat67
- Processed through MapReduce for distributed preprocessing
- Compatible with FrugalNeRF's few-shot training pipeline

## Citation

If you find our code or paper helps, please consider citing:

```
@inproceedings{lin2024frugalnerf,
  title={FrugalNeRF: Fast Convergence for Few-shot Novel View Synthesis without Learned Priors},
  author={Chin-Yang Lin and Chung-Ho Wu and Chang-Han Yeh and Shih-Han Yen and Cheng Sun and Yu-Lun Liu},
  booktitle={CVPR},
  year={2025}
}
```

## Acknowledgements

The code is available under the MIT license and draws from [TensoRF](https://github.com/apchenstu/TensoRF), [ViP-NeRF](https://github.com/NagabhushanSN95/ViP-NeRF), which are also licensed under the MIT license. Licenses for these projects can be found in the licenses/ folder.
