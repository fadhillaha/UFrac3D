# Fluid Flow Prediction using U-Nets in 3D Single Fractures (UFrac3D)

This repository contains the code for a deep-learning study on predicting
3D fluid-flow velocity fields inside singular rock fracture geometries. Two
convolutional networks are provided: a 3D U-Net and an Attention
Residual 3D U-Net. Each takes a 128×128×128 voxel fracture geometry as
input — either one channel (binary geometry) or two channels (binary
geometry + Euclidean Distance Transform) — and predicts the velocity field.

## Repository structure

```
data/             .mat fracture sub-volumes 
  dataset.py      FractureDataset, EDT computation, train/val split
  models.py       UNet3D and AttResUNet architectures
  train.py        Training: AdamW, weighted MAE, ReduceLROnPlateau, AMP, early stopping
  evaluate.py     Per-sample RMSE / RRMSE / sMAPE / MAE, written to CSV
  permeability.py Permeability estimation vs. LBM reference values
  analysis.py     Summary tables and plots from evaluation CSVs
weights/          Trained checkpoints
results/          Output CSVs, tables and plots
requirements.txt
README.md
```

## Data layout

`build_dataframe` pairs files by base name: a `.mat` geometry file (variable
`sub_volume` or `wadah`) in the input folder and a `.csv` velocity file with
the same base name in the mask folder. Targets are square-rooted and scaled
during training; evaluation reverses this and converts lattice units to
m/s.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run each script from the `src/` directory (or add it to `PYTHONPATH`).
Every script supports `--help`.

Train:

```bash
python src/train.py \
  --input-folder data/input --mask-folder data/sim \
  --model unet3d --in-channels 2 \
  --save-path weights/unet3d_2in.pth \
  --history-path results/unet3d_2in_history.csv
```

Evaluate (writes a per-sample metrics CSV):

```bash
python src/evaluate.py \
  --input-folder data/input --mask-folder data/sim \
  --checkpoint weights/unet3d_2in.pth \
  --model unet3d --in-channels 2 --split val \
  --output results/unet3d_2in_val.csv
```

Permeability (optionally compared against an LBM reference CSV):

```bash
python src/permeability.py \
  --input-folder data/input --mask-folder data/sim \
  --checkpoint weights/unet3d_2in.pth \
  --model unet3d --in-channels 2 \
  --delta-p 5e-4 \
  --lbm-csv data/lbm_data.csv \
  --output results/unet3d_2in_permeability.csv
```

Analysis (tables and plots from one or more evaluation CSVs):

```bash
python src/analysis.py results/*.csv --out-dir results
```

`--model` accepts `unet3d` or `attresunet`. `--in-channels` is `1`
(geometry only) or `2` (geometry + EDT).

## Notes

The train/validation split is fixed (`random_state=42`, stratified by
volume size) for reproducibility. Training uses a batch size of 1 because
whole 128³ volumes are processed. Mixed precision is enabled only on CUDA.


