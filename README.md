# RDDM-UNet-CBAM

This repository contains the code for an RDDM covariance denoiser with a compact conditional UNet backbone and optional CBAM attention. The model denoises complex covariance / SCM matrices stored as two channels `[real, imag]` with shape `[N, 2, M, M]`.

## Contents

- `model/rddm_gat.py`: core model definitions, including `UNetTwoHead`, `CBAMAttention2d`, `ResidualDiffusion`, structural projections, and RGAT ablations.
- `train_rddm_rgat.py`: training entry point. Use `--backbone unet --unet-attention cbam` for the CBAM-UNet RDDM version.
- `test.py`: MUSIC-based evaluation for noisy, denoised, and clean covariance matrices.
- `make_rddm_denoised_dataset.py`: offline denoising script that writes a denoised `.npz` dataset.
- `getdata.py`: synthetic FLOM / SCM matrix dataset generator.
- `run_unet_fair_music_experiment.py`: sequential UNet pretrain, fine-tune, and MUSIC evaluation protocol.
- `run_unet_attention_ablation.py`: wrapper for baseline, SE, CBAM, skip-gate, and CBAM+skip-gate UNet variants.

Large datasets, checkpoints, logs, virtual environments, paper files, and experiment outputs are intentionally excluded.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA version if you train on GPU.

## Data Format

Training and evaluation scripts expect `.npz` files with:

- `xins`: noisy input matrices, shape `[N, 2, M, M]`, `float32`
- `x0s`: clean target matrices, shape `[N, 2, M, M]`, `float32`
- `meta`: optional metadata used by MUSIC evaluation

Generate a small SCM example dataset:

```powershell
python getdata.py --M 8 --T 128 --noise-type gaussian --min-sep-deg 3 --snr-min -8 --snr-max -5 --num-samples 1000 --out-dir data/demo_train
```

## Train CBAM-UNet RDDM

```powershell
python train_rddm_rgat.py `
  --train data/train/flom_mat_000.npz `
  --val data/val/flom_mat_000.npz `
  --backbone unet `
  --objective rddm `
  --unet-attention cbam `
  --image-size 8 `
  --steps 400 `
  --epochs 150 `
  --batch-size 1024 `
  --outdir runs/unet_cbam_rddm
```

## Evaluate

```powershell
python test.py `
  --ckpt runs/unet_cbam_rddm/<run>/best_model_loss=<value>.pth `
  --data-dir data/test `
  --sampler ddim `
  --ddim-steps 200 `
  --sample-proj-toeplitz `
  --sample-proj-psd `
  --plot-num 0 `
  --out-json results/unet_cbam_test.json
```

## Offline Denoising

```powershell
python make_rddm_denoised_dataset.py `
  --src data/test/flom_mat_000.npz `
  --dst-dir data/test_denoised `
  --ckpt runs/unet_cbam_rddm/<run>/best_model_loss=<value>.pth `
  --project-output
```

## Notes Before Uploading

The `.gitignore` is configured to keep GitHub clean by excluding generated data, `.npz` arrays, `.pth` checkpoints, logs, virtual environments, and paper/rendering temporary files. If you want to publish trained weights later, use GitHub Releases or Git LFS instead of committing them directly.
