import os
import kaggle
import zipfile

# Download the dataset
kaggle.api.authenticate()
kaggle.api.dataset_download_files('arenagrenade/llff-dataset-full', path='./data', unzip=True)

# Move files to the correct location
source_dir = './data/nerf_synthetic'
target_dir = './data/nerf_llff_data'

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# Move all contents to the target directory
for item in os.listdir(source_dir):
    source = os.path.join(source_dir, item)
    target = os.path.join(target_dir, item)
    os.rename(source, target)

print("Dataset downloaded and organized successfully!")