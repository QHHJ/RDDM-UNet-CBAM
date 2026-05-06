# RDDM-GAT 消融实验设计

## 固定设置

- 数据：`exp_scm_t128_snr_-8_to_-5_sep3_gaussian_*`
- 条件：M=8，snapshots=128，SNR=[-8,-5] dB，K=2，最小角间隔 3 deg
- 训练默认：20 epochs，batch=1024，lr=1e-4，EMA=0.999
- 测试默认：5000 samples，`sampling_timestep=100`，`lambda_res=1.0`
- 主指标：SR@0.5、SR@1、RMSE
- 汇总文件：每次运行输出 `summary.csv` 和 `summary.json`

## 核心消融

| 编号 | 实验名 | 改动 |
|---|---|---|
| A0 | `A0_noisy_scm_music` | 原始 Noisy SCM + MUSIC 基线 |
| A1 | `A1_full_rddm_rgat` | 完整 RDDM-GAT |
| A2 | `A2_direct_gat_x0` | 去掉 RDDM，RGAT 直接回归 clean SCM |
| A3 | `A3_rddm_predict_x0` | 保留 RDDM 前向加噪，但模型直接预测 x0，不预测残差 |
| A4 | `A4_no_time_film` | 去掉时间步 FiLM 调制 |
| A7 | `A7_no_edge_sign` | 去掉边方向 sign |
| A8 | `A8_no_hermitian_decoder` | 解码器不强制 Hermitian |
| A9 | `A9_no_toeplitz_psd_projection` | 用 A1 checkpoint 测试时关闭 Toeplitz/PSD 后投影 |
| A11 | `A11_unet_backbone` | RGAT backbone 替换为 UNet |
| A12 | `A12_plain_ddim` | RDDM 退化为不带残差桥接项的普通条件 DDIM |

## 超参数消融

采用 one-factor-at-a-time，其他参数固定为 A1：

| 消融项 | 设置 |
|---|---|
| GAT layers | 2 / 4 / 6 / 8 |
| attention heads | 2 / 4 / 8 |
| hidden dim | 128 / 256 / 512 / 768 |
| kmax | 1 / 2 / 4 / 7 |
| dropout | 0 / 0.05 / 0.1 |
| low-rank decoder | 0 / 32 / 64 / 128 |

## 损失函数消融

| 实验名 | 设置 |
|---|---|
| `L_res_only` | `w_res=1, w_noise=0` |
| `L_res_plus_noise` | `w_res=1, w_noise=1` |
| `L_noise_only` | `w_res=0, w_noise=1` |

## 运行命令

先看计划，不启动训练：

```powershell
.\.venv_cuda\Scripts\python.exe .\run_rddm_ablation_suite.py --preset core --dry-run
.\.venv_cuda\Scripts\python.exe .\run_rddm_ablation_suite.py --preset hparam --dry-run
.\.venv_cuda\Scripts\python.exe .\run_rddm_ablation_suite.py --preset loss --dry-run
```

正式运行：

```powershell
.\.venv_cuda\Scripts\python.exe -u .\run_rddm_ablation_suite.py --preset core --amp --fp16-eval
.\.venv_cuda\Scripts\python.exe -u .\run_rddm_ablation_suite.py --preset hparam --amp --fp16-eval
.\.venv_cuda\Scripts\python.exe -u .\run_rddm_ablation_suite.py --preset loss --amp --fp16-eval
```

一次跑完全部：

```powershell
.\.venv_cuda\Scripts\python.exe -u .\run_rddm_ablation_suite.py --preset all --amp --fp16-eval
```

输出目录形如：

```text
exp_rddm_ablation/core_YYYYMMDD_HHMMSS/
exp_rddm_ablation/hparam_YYYYMMDD_HHMMSS/
exp_rddm_ablation/loss_YYYYMMDD_HHMMSS/
```

