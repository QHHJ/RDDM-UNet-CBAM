# Package Manifest

Package name: `RDDM-CBAM-UNet`

Prepared on: `2026-05-11`

## Included Best Checkpoint

```text
checkpoints/rddm_cbam_unet_best_music.pth
```

Original source:

```text
exp_unet_attention_ablation/attn_lambda1_pretrain_20260501_143335/cbam/attn_lambda1_pretrain_cbam_20260501_145057/runs/unet_ft_sep3_t128/20260501_145100/best_music_sr_sr=0.3055_rmse=1.6785.pth
```

## Main Metrics

Evaluation setting: `lambda=1`, MUSIC backend, 5000 test samples, `M=8`, `T=128`, SNR `[-8 dB, -5 dB]`, minimum angle separation `3 deg`.

| Method | SR@0.5 | SR@1 | RMSE |
|---|---:|---:|---:|
| RDDM-CBAM-UNet + MUSIC | 0.3875 | 0.6730 | 1.0195 |

## Key Config

```text
backbone = unet
objective = rddm
unet_attention = cbam
unet_skip_gate = false
unet_base = 64
steps = 400
w_res = 1
w_noise = 0
sample_lambda = 1
sample_project_toeplitz = true
sample_project_psd = true
```

Full training args are saved in:

```text
configs/rddm_cbam_unet_args.json
```
