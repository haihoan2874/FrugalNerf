Sample NeRF Dataset for MapReduce Demo

This directory contains sample data for demonstrating FrugalNeRF MapReduce processing.

Structure:
- images/: Directory containing sample RGB images
- poses/: Directory containing camera pose files (transforms.json)
- README.txt: This file

For the demo, we will:
1. Upload sample images to HDFS
2. Run MapReduce job to process images
3. Convert output to FrugalNeRF format
4. Visualize results
