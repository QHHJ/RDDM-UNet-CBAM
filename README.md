# RDDM-CBAM-UNet

本包对应论文第三章主线方法：基于残差扩散模型的 CBAM-UNet 协方差矩阵去噪方法。

模型输入为有限快拍和低信噪比条件下估计得到的受噪样本协方差矩阵实值化表示，输出为恢复后的理想协方差矩阵实值表示。随后将恢复结果送入 MUSIC 谱峰搜索，完成 DOA 估计。

## 方法名称

推荐在论文、图表和代码说明中统一写作：

```text
RDDM-CBAM-UNet
```

若需要强调后端估计算法，可写作：

```text
RDDM-CBAM-UNet + MUSIC
```

## 当前打包结果

实验条件：

- 阵列形式：均匀线阵
- 阵元数：M=8
- 快拍数：T=128
- 信源数：K=2
- SNR 范围：[-8 dB, -5 dB]
- DOA 范围：[-60 deg, 60 deg]
- 最小角度间隔：3 deg
- 恢复步长：lambda=1
- 主干网络：UNet
- 注意力模块：CBAM
- 扩散时间步：400
- 后端估计算法：MUSIC

当前用于论文第三章主线的最佳 CBAM-UNet 结果：

| Method | SR@0.5 | SR@1 | RMSE |
|---|---:|---:|---:|
| RDDM-CBAM-UNet + MUSIC | 0.3875 | 0.6730 | 1.0195 |

对比纯 UNet 主干：

| Method | SR@0.5 | SR@1 | RMSE |
|---|---:|---:|---:|
| RDDM-UNet + MUSIC | 0.3875 | 0.6736 | 1.0469 |
| RDDM-CBAM-UNet + MUSIC | 0.3875 | 0.6730 | 1.0195 |

结论口径：CBAM 对严格成功率提升不明显，但能够降低 MUSIC 后端平均角度估计误差，说明通道-空间注意力对协方差矩阵恢复质量有正向作用。

## 目录结构

```text
RDDM-CBAM-UNet/
  model/rddm_gat.py                    # RDDM、UNet、CBAM、结构投影等核心模型定义
  train_rddm_cbam_unet.py              # RDDM-CBAM-UNet 训练入口
  test.py                              # MUSIC 后端评测入口
  getdata.py                           # SCM/FLOM 数据生成脚本
  make_rddm_denoised_dataset.py        # 离线生成恢复后协方差矩阵数据
  run_unet_fair_music_experiment.py    # UNet 公平对比实验封装
  run_unet_attention_ablation.py       # 注意力消融实验封装
  checkpoints/
    rddm_cbam_unet_best_music.pth      # 当前打包的最佳 CBAM-UNet 权重
  configs/
    rddm_cbam_unet_args.json           # 最佳权重对应训练参数
  results/
    comparison.csv                     # 注意力消融对比结果
    unet_summary.csv                   # CBAM-UNet 汇总结果
    unet_best_music_lam1_sr05.json     # SR@0.5 评测结果
    unet_best_music_lam1_sr10.json     # SR@1 评测结果
```

## 安装依赖

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

如果使用 GPU 训练，需要安装与本机 CUDA 版本匹配的 PyTorch。

## 训练 RDDM-CBAM-UNet

```powershell
python train_rddm_cbam_unet.py `
  --train data/train/flom_mat_000.npz `
  --val data/val/flom_mat_000.npz `
  --backbone unet `
  --objective rddm `
  --unet-attention cbam `
  --image-size 8 `
  --steps 400 `
  --epochs 150 `
  --batch-size 1024 `
  --outdir runs/rddm_cbam_unet
```

关键参数是：

```text
--backbone unet
--objective rddm
--unet-attention cbam
```

## 使用已打包权重评测

```powershell
python test.py `
  --ckpt checkpoints/rddm_cbam_unet_best_music.pth `
  --data-dir data/test `
  --sampler ddim `
  --ddim-steps 200 `
  --sample-lambda 1 `
  --sample-proj-toeplitz `
  --sample-proj-psd `
  --plot-num 0 `
  --out-json results/rddm_cbam_unet_music_test.json
```

如果测试数据目录沿用原项目，可将 `data/test` 替换为：

```text
../exp_scm_t128_snr_-8_to_-5_sep3_gaussian_test_5000
```

## 数据格式

训练和评测脚本读取 `.npz` 文件，核心字段为：

- `xins`：受噪输入矩阵，形状为 `[N, 2, M, M]`
- `x0s`：理想监督矩阵，形状为 `[N, 2, M, M]`
- `meta`：DOA、SNR、信源数等评测元信息

其中两个通道分别对应复协方差矩阵的实部和虚部。

## 论文表述建议

可以将本方法描述为：

```text
本文在残差扩散协方差矩阵恢复框架中采用 UNet 作为噪声预测与残差预测主干网络，并在卷积特征块中引入 CBAM 注意力模块。CBAM 通过通道注意力和空间注意力对中间特征进行重标定，使网络在恢复协方差矩阵时更加关注与阵元相关结构和 DOA 子空间估计相关的关键特征区域。
```

实验结论建议写作：

```text
与普通 RDDM-UNet 相比，RDDM-CBAM-UNet 在 SR@0.5 和 SR@1 上基本保持一致，但 RMSE 由 1.0469 降低至 1.0195，表明 CBAM 注意力主要改善平均角度估计误差，而非显著改变严格阈值下的成功率。
```
