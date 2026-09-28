# UFrac3D: 3-D Fluid Flow Velocity Prediction in Single Fractures

This repository contains the official PyTorch implementation for the manuscript: **"Attention Residual U-Net for Three-Dimensional Fluid Flow Velocity Prediction in Single Fractures"** 

The repository provides a deep learning surrogate framework designed to predict three-dimensional velocity vector fields inside rough single fractures. The models are trained on synthetic fractal Brownian motion (fBm) geometries and evaluated on both synthetic data and real rock specimens (andesite, granite, and shale).

## Available Architectures

The repository implements six model configurations evaluated in the study. These architectures can be trained to predict the full 3-D velocity vector field using either a single-channel input (binary fracture geometry) or a two-channel input (binary geometry augmented with its Euclidean Distance Transform).

1. **U-Net**: Baseline fully convolutional 3-D encoder-decoder.
2. **AttResUNet**: Incorporates residual connections and attention gates.
3. **AttResUNet-ASPP**: Incorporates an Atrous Spatial Pyramid Pooling module at the bottleneck to resolve multi-scale spatial features.

## Repository Structure

- **training/**: Core machine learning framework.
  - train.py: Main script for model training and validation.
  - evaluate.py: Main script for inference and computation of voxel-wise error metrics (RMSE, SMAPE, Average Angular Error).
  - models.py: PyTorch network definitions for all evaluated architectures.
  - dataset.py: Data loaders, transformations, and input processing pipelines.
- **processing/**: Physics integration, domain analysis, and post-processing tools.
  - permeability.py: Pipeline for calculating macroscopic permeability from predicted flow fields via Darcy's law.
  - physics_baselines.py: Computes analytical physics baselines (standard and local cubic laws).
  - domain_gap.py: Computes Kolmogorov-Smirnov (KS) statistics and Wasserstein distances to quantify structural domain gaps.
  - analysis.py: Aggregates error metrics for generating statistical tables and distributions.
- **data/**: Directory for sample fracture geometries (.mat) and LBM ground truth velocity fields.
- **weights/**: Directory for pre-trained model checkpoints (.pth).

## Installation

Clone the repository and install the required dependencies:

`bash
git clone https://github.com/fadhillaha/UFrac3D.git
cd UFrac3D
pip install -r requirements.txt
`

## Data and Pre-trained Models Accessibility

* **Datasets**: Put the geometry arrays and corresponding LBM velocity fields and place them within the data/ directory.
* **Model Weights**: Download the pre-trained checkpoints and place them within the weights/ directory.

## Usage Instructions

### 1. Model Training
To train the AttResUNet-ASPP model utilizing the two-input configuration (geometry and EDT) to predict the 3-D velocity vector:
`bash
cd training
python train.py --input-folder ../data/geometries --mask-folder ../data/velocities --model attresunet_aspp --in-channels 2 --target-mode vector
`

### 2. Inference and Evaluation
To evaluate a trained model and output voxel-wise statistical metrics:
`bash
cd training
python evaluate.py --model-weights ../weights/aspp2.pth --input-folder ../data/geometries --mask-folder ../data/velocities --model attresunet_aspp --in-channels 2 --target-mode vector
`

