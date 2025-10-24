# COLMAP -> poses_bounds.npy converter

This folder contains a small script to convert a COLMAP TXT model (the output
of model_converter --output_type TXT) into the poses_bounds.npy format
expected by the FrugalNeRF pipeline.

Files:

- colmap_to_poses.py : main converter (Python3). Takes --model_txt (path to folder with cameras.txt and images.txt), optional --image_dir and --out.
- requirements.txt : python requirements (numpy, Pillow)

Quick usage (PowerShell):

1. Install dependencies:
   python -m pip install -r tools\requirements.txt

2. Run COLMAP example commands (replace paths):

   # set workspace paths

   $images = 'C:\\data\\scene\\images'
   $work = 'C:\\data\\scene\\colmap_db'
   New-Item -ItemType Directory -Path $work -Force

   # feature extraction

   colmap feature_extractor --database_path "$work\\database.db" --image_path $images

   # matching

   colmap exhaustive_matcher --database_path "$work\\database.db"

   # mapping/sparse

   $sparse = 'C:\\data\\scene\\sparse'
   New-Item -ItemType Directory -Path $sparse -Force
   colmap mapper --database_path "$work\\database.db" --image_path $images --output_path $sparse

   # convert model to TXT

   colmap model_converter --input_path "$sparse\\0" --output_path "$sparse\\0_txt" --output_type TXT

3. Convert COLMAP TXT to poses_bounds.npy:
   python tools\colmap_to_poses.py --model_txt C:\\data\\scene\\sparse\\0_txt --image_dir C:\\data\\scene\\images --out C:\\data\\scene\\poses_bounds.npy

Notes:

- The script infers focal length from COLMAP camera parameters when possible; otherwise it falls back to a heuristic (0.5 \* min(W,H)).
- The script writes a float32 numpy array of shape (N,17) where N is number of images.
- If you have RealityCapture FBX camera exports instead, tell me and I will provide a FBX->poses converter.
