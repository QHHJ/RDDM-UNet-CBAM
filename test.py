# -*- coding: utf-8 -*-
"""
测试脚本：RDDM + RGAT 的 DOA 性能评估（仅指标版，无绘图）
- 基于你的 eval_rddm_music_rgat.py 精简/增强：保留三路对比（Noisy/Denoised/Clean），
  统一输出 SR@τ 与 RMSE；支持按 SNR（逐 dB）与按 K 分层统计；可导出 CSV/JSON。
- 兼容 meta 读法（列表/0-d/JSON 字符串）；兼容 best_model/last/state_dict 等权重键位。
- 采样接口与原版保持一致：one-step 与 ddim 少步长逆向（η 仅占位）。

运行示例：
    python test_rddm_rgat_performance.py \
        --ckpt runs_rddm/20251102_003348/best_model_loss=0.009774.pth \
        --data-dir dataset_snap=1024_snr_-10_to_-5_gaussian_test_1000 \
        --eval-num 1000 --batch-size 128 --fp16 \
        --out-csv perf.csv --out-json perf.json

注意：
- 默认在跑 MUSIC 之前对三路矩阵做结构投影（Hermitian + Toeplitz + PSD）。
- SNR 分层默认对 snr_db 四舍五入到整数 dB；可通过 --snr-bin-mode 改成 floor/ceil。
"""

import os, glob, json, argparse
import numpy as np
import torch as th
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast

# 解决 OpenMP 重复库问题（Windows 常见）
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 导入你工程内的模型
import sys
sys.path.insert(0, os.path.dirname(__file__))
from model.rddm_gat import (
    DirectDenoiser,
    GATTwoHead,
    PlainDDIMDiffusion,
    ResidualDiffusion,
    ResidualX0Diffusion,
    UNetTwoHead,
)

# ===================== 数据集 =====================
class FlomMatNPZWithMeta(Dataset):
    def __init__(self, data_dir, split_ratio=(0.90, 0.10), train=False, use_all=False):
        """支持目录或单个 .npz。期望键：xins / x0s / (可选) meta。
        meta 兼容 JSON 字符串、0-d ndarray、object 数组等形式。
        返回：(cond[2,M,M], tgt[2,M,M], mask[1,M,M], meta_dict)
        """
        if os.path.isfile(data_dir) and data_dir.endswith('.npz'):
            files = [data_dir]
        else:
            candidates = []
            for pattern in ('flom_mat_*.npz', 'ideal_mat_*.npz', 'scm_mat_*.npz'):
                candidates.extend(glob.glob(os.path.join(data_dir, pattern)))
            files = []
            for candidate in sorted(set(candidates)):
                with np.load(candidate, allow_pickle=True) as npz:
                    if 'xins' in npz and 'x0s' in npz:
                        files.append(candidate)
        assert files, f"No npz found in {data_dir}"

        self.items = []
        self.M = None
        for f in files:
            with np.load(f, allow_pickle=True) as npz:
                assert 'xins' in npz and 'x0s' in npz, f"{f} missing keys: expect xins/x0s"
                Xc, Xt = npz['xins'], npz['x0s']
                N = Xc.shape[0]
                self.M = Xc.shape[-1] if self.M is None else self.M

                meta_raw = npz.get('meta', None)
                if meta_raw is None:
                    metas = [{} for _ in range(N)]
                else:
                    if isinstance(meta_raw, np.ndarray):
                        if meta_raw.ndim == 0:
                            metas = json.loads(meta_raw.item())
                        elif meta_raw.dtype == object:
                            metas = list(meta_raw)
                        else:
                            metas = json.loads(str(meta_raw))
                    else:
                        metas = json.loads(meta_raw)
                    assert len(metas) == N, f"meta length mismatch in {f}"

                for i in range(N):
                    self.items.append((f, i, metas[i]))

        if not use_all:
            n_all = len(self.items)
            n_train = int(n_all * split_ratio[0])
            self.items = self.items[:n_train] if train else self.items[n_train:]

        Mmask = self.M if self.M is not None else 8
        self.mask = th.ones(1, Mmask, Mmask, dtype=th.float32)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        f, i, meta = self.items[idx]
        with np.load(f) as npz:
            cond = npz['xins'][i].astype(np.float32)
            tgt  = npz['x0s'][i].astype(np.float32)
        return th.from_numpy(cond), th.from_numpy(tgt), self.mask.clone(), meta

# ===================== 结构投影 =====================
def project_struct_numpy(C: np.ndarray) -> np.ndarray:
    """Toeplitz + Hermitian + PSD 投影（复 MxM）。"""
    C = 0.5 * (C + C.conj().T)
    M = C.shape[0]
    Tproj = np.zeros_like(C, dtype=np.complex64)
    for k in range(-(M-1), M):
        d = np.diag(C, k)
        m = d.mean()
        Tproj += np.diag(np.full(M-abs(k), m, dtype=np.complex64), k)
    C = 0.5 * (Tproj + Tproj.conj().T)
    w, V = np.linalg.eigh(C)
    w = np.maximum(w, 0.0)
    return (V * w) @ V.conj().T

# ===================== 简化版 ULA-MUSIC =====================
def steering_vec_ula(M, d_over_lambda, theta_deg):
    m = np.arange(M)
    th_rad = np.deg2rad(theta_deg)
    phase = 2*np.pi*d_over_lambda*np.sin(th_rad)*m
    return np.exp(1j*phase)[:, None]

def build_music_scan(M, d_over_lambda, theta_grid_deg):
    th_rad = np.deg2rad(np.asarray(theta_grid_deg))
    m = np.arange(M)[:, None]
    phase = 2*np.pi*d_over_lambda * np.sin(th_rad)[None, :] * m
    return np.exp(1j * phase)

def music_spectrum(R, K, d_over_lambda, theta_grid_deg, aScan=None):
    """
    与 MATLAB:
        P(θ) = 1 / ||E_n^H a(θ)||_2^2
    等价实现（一次性把所有 θ 的导向向量拼起来做矩阵乘，避免 for-loop）。
    """
    R = 0.5 * (R + R.conj().T)                          # Hermitian 化
    w, V = np.linalg.eigh(R)                            # eigh: 本征值升序
    M = R.shape[0]
    En = V[:, :M-K]                                     # 最小的 M-K 个 -> 噪声子空间

    # 扫描角导向矩阵 aScan: (M, G). It is sample-independent, so callers can
    # precompute it once for large SNR sweeps.
    if aScan is None:
        aScan = build_music_scan(M, d_over_lambda, theta_grid_deg)

    # P(θ) = 1 ./ sum(|En' * a|.^2, 1)
    EnHa = En.conj().T @ aScan                          # (M-K, G)
    denom = np.sum(np.abs(EnHa)**2, axis=0)             # (G,)
    Pmusic = 1.0 / (denom + 1e-12)
    return Pmusic

def run_music(R, K, d_over_lambda, theta_grid_deg, aScan=None):
    """
    - 先按上面的向量化公式算 P(θ)
    - 再按 MATLAB findpeaks 的方式取峰：
        * 按峰高降序
        * NPeaks = K
        * MinPeakDistance = round(1 / scanRes) 个网格点（≈ 1°）
      若不够 K 个峰，则放宽间距补齐，确保返回 K 个角度（便于后续指标与绘图）。
    """
    P = music_spectrum(R, K, d_over_lambda, theta_grid_deg, aScan=aScan)
    th = np.asarray(theta_grid_deg)
    G  = len(th)
    if G < 3:
        idx = np.argsort(P)[-K:][::-1]
        return th[idx], P

    # 网格分辨率与等效最小间隔（≈1°）
    scan_res = float(abs(th[1] - th[0]))
    min_dist_pts = max(1, int(round(1.0 / scan_res)))   # 例如 0.167° -> 6 点 ≈ 1°

    # ---- 找局部极大值（strict left/ right）----
    y = P
    is_peak = np.zeros(G, dtype=bool)
    is_peak[1:-1] = (y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:])
    cand = np.where(is_peak)[0]

    # 若未检测到局部峰，退化为全局排序
    if cand.size == 0:
        idx = np.argsort(y)[-K:][::-1]
        return th[idx], P

    # 候选按峰高降序
    cand = cand[np.argsort(y[cand])[::-1]]

    # ---- NPeaks=K + MinPeakDistance 筛选（MATLAB 风格）----
    picked = []
    for i in cand:
        if all(abs(i - j) >= min_dist_pts for j in picked):
            picked.append(i)
            if len(picked) == K:
                break

    # 如果因最小间隔导致 <K，则放宽间隔补齐（保证下游流程稳定）
    if len(picked) < K:
        for i in cand:
            if i not in picked:
                picked.append(i)
            if len(picked) == K:
                break
    if len(picked) < K:
        for i in np.argsort(y)[::-1]:
            if i not in picked:
                picked.append(int(i))
            if len(picked) == K:
                break

    idx = np.array(picked[:K], dtype=int)
    # 画图时需要返回 (预测角度, 伪谱)
    return th[idx], P

# ===================== 指标 =====================
def match_metrics(pred_deg, gt_deg, thr=0.5):
    """贪心一一匹配：SR@thr 与 RMSE（度）。"""
    pred = list(pred_deg)
    gt = list(gt_deg)
    used = [False]*len(pred)
    hits = 0
    sq = []
    for g in gt:
        j_best, d_best = -1, 1e9
        for j, p in enumerate(pred):
            if used[j]:
                continue
            d = abs(p - g)
            if d < d_best:
                d_best, j_best = d, j
        if j_best >= 0:
            used[j_best] = True
        if d_best <= thr:
            hits += 1
        sq.append(d_best**2)
    sr = hits / max(1, len(gt))
    rmse = (np.mean(sq)**0.5) if sq else np.nan
    return sr, rmse

# ===================== 绘图：三子图 Noisy/Denoised/Clean =====================
def plot_three_spectra(theta_grid, pseu_noisy, pseu_deno, pseu_clean,
                       pred_noisy, pred_deno, pred_clean, gt_deg, 
                       title, out_path, normalize="per", yscale="log"):
    """绘制 Noisy / Denoised / Clean 三路 MUSIC 伪谱对比图
    
    Args:
        pred_noisy, pred_deno, pred_clean: 各方法的预测角度（数组）
        gt_deg: 真实角度（ground truth）
    """
    import matplotlib.pyplot as plt
    curves = [("Noisy", pseu_noisy), ("Denoised", pseu_deno), ("Clean", pseu_clean)]
    preds = [pred_noisy, pred_deno, pred_clean]
    
    # 归一化
    if normalize != "none":
        if normalize == "global":
            gmax = max([c[1].max() for c in curves]) + 1e-12
        for i, (name, y) in enumerate(curves):
            m = (y.max() + 1e-12) if normalize == "per" else gmax
            curves[i] = (name, y / m)
    
    # 绘制三子图
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    for ax, (name, y), pred in zip(axes, curves, preds):
        if yscale == "log":
            ax.semilogy(theta_grid, y)
        else:
            ax.plot(theta_grid, y)
        # 标记真实DOA（灰色虚线）
        for g in gt_deg:
            ax.axvline(g, linestyle="--", linewidth=1.2, color="gray", alpha=0.6)
        # 标记该方法的预测值（红色点线）
        for p in pred:
            ax.axvline(p, linestyle=":", linewidth=1.5, color="red", alpha=0.8)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.25)
    axes[0].set_title(title, fontsize=10)
    axes[-1].set_xlabel("θ (deg)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

# ===================== 采样封装 =====================
@th.no_grad()
def sample_rddm_denoised(netD, cond, sampling_timestep, lambda_res, init_noise_scale,
                         do_project=False, proj_hermitian=True, proj_toeplitz=False, proj_psd=False):
    x0_hat = netD.p_sample_loop(
        cond, steps=1, sum_scale=init_noise_scale,
        proj_hermitian=proj_hermitian, proj_toeplitz=proj_toeplitz, proj_psd=proj_psd,
        lambda_res=lambda_res, start_t=sampling_timestep
    )
    re = x0_hat[:,0,:,:].cpu().numpy()
    im = x0_hat[:,1,:,:].cpu().numpy()
    C = re + 1j*im
    if do_project:
        C = np.stack([project_struct_numpy(c) for c in C], axis=0)
    return C

@th.no_grad()
def sample_rddm_multistep(
    netD,
    cond8: th.Tensor,                 # [B, 2, M, M]
    steps: int = 5,                   # 采样步数 S
    lambda_res: float = 1.0,
    init_noise_scale: float = 1e-4,
    do_project: bool = False,
    proj_hermitian: bool = True,
    proj_toeplitz: bool = False,
    proj_psd: bool = False,
    eta: float = 0.0                  # 与原接口对齐（这里内部用确定性 DDIM；η留作占位）
):
    x0_hat = netD.p_sample_loop(
        cond8, steps=steps, sum_scale=init_noise_scale,
        proj_hermitian=proj_hermitian, proj_toeplitz=proj_toeplitz, proj_psd=proj_psd,
        lambda_res=lambda_res
    )
    re = x0_hat[:,0,:,:].cpu().numpy()
    im = x0_hat[:,1,:,:].cpu().numpy()
    C = re + 1j*im
    if do_project:
        C = np.stack([project_struct_numpy(c) for c in C], axis=0)
    return C, list(range(steps))

# ===================== 工具 =====================
def _nanmean(x):
    x = np.array(x, dtype=np.float64)
    return float(np.nanmean(x)) if x.size else float('nan')

def _snr_bin(v: float, mode: str):
    if np.isnan(v):
        return None
    if mode == 'round':
        return int(np.round(v))
    if mode == 'floor':
        return int(np.floor(v))
    if mode == 'ceil':
        return int(np.ceil(v))
    return int(np.round(v))

# ===================== 主流程 =====================
def main():
    parser = argparse.ArgumentParser('RDDM-RGAT 性能评估（仅指标）')
    parser.add_argument('--ckpt', type=str, required=False,
                        default='runs_rddm/20251107_184117/best_model_loss=0.034635.pth',
                        help='RDDM+RGAT 权重路径（best_model/last/state_dict）')
    parser.add_argument('--data-dir', type=str, required=False,
                        default='dataset_snap=1024_snr_-8_to_-5_gaussian_test_1000',
                        help='测试数据目录或单个 npz')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--project-inputs', action='store_true', default=False,
                        help='在 MUSIC 前对 noisy/clean/denoised 三路做结构投影')
    parser.add_argument('--eval-num', type=int, default=1000, help='评估样本上限')
    parser.add_argument('--use-all-data', action='store_true', default=True,
                        help='使用测试集全部样本')

    # 采样参数
    parser.add_argument('--lambda-res', type=float, default=1.0)
    parser.add_argument('--init-noise-scale', type=float, default=0)
    parser.add_argument('--sampler', type=str, default='ddim', choices=['one','ddim'])
    parser.add_argument('--ddim-steps', type=int, default=200)
    parser.add_argument('--eta', type=float, default=0.0)
    parser.add_argument('--sampling-timestep', type=int, default=None)
    parser.add_argument('--sample-no-hermitian', action='store_true', default=False,
                        help='采样输出不做 Hermitian 投影，用于结构投影消融')
    parser.add_argument('--sample-proj-toeplitz', action='store_true', default=False,
                        help='采样过程中对 denoised 矩阵做 Toeplitz 投影')
    parser.add_argument('--sample-proj-psd', action='store_true', default=False,
                        help='采样过程中对 denoised 矩阵做 PSD 投影')

    # 阈值/分层/导出
    parser.add_argument('--sr-thr', type=float, default=0.5, help='SR 命中阈值（度）')
    parser.add_argument('--snr-bin-mode', type=str, default='round', choices=['round','floor','ceil'])
    parser.add_argument('--out-csv', type=str, default='')
    parser.add_argument('--out-json', type=str, default='')
    
    # 绘图选项
    parser.add_argument('--plot-num', type=int, default=200, help='保存谱图数量（0=不绘图）')
    parser.add_argument('--plot-out', type=str, default='rddm_music_plots', help='谱图输出目录')
    parser.add_argument('--yscale', type=str, default='log', choices=['linear','log'], help='Y轴刻度')
    parser.add_argument('--norm', type=str, default='per', choices=['per','global','none'], help='归一化方式')

    args = parser.parse_args()

    # 读取 meta.json 以获得 M / d_over_lambda / θ 扫描范围
    if os.path.isdir(args.data_dir):
        meta_path = os.path.join(args.data_dir, 'meta.json')
    else:
        meta_path = os.path.join(os.path.dirname(args.data_dir), 'meta.json')

    if not os.path.exists(meta_path):
        print(f"[Warning] 未找到 {meta_path}，使用默认 M=8, d/λ=0.5, θ∈[-60,60]")
        M = 8; d_over_lambda = 0.5; theta_grid = np.linspace(-60, 60, 721)
    else:
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta_all = json.load(f)
        M = int(meta_all['M'])
        d_over_lambda = float(meta_all['d_over_lambda'])
        theta_grid = np.linspace(meta_all['doa_min'], meta_all['doa_max'], 721)
    music_scan = build_music_scan(M, d_over_lambda, theta_grid)

    # DataLoader
    def collate_fn(batch):
        conds, tgts, masks, metas = zip(*batch)
        return th.stack(conds, 0), th.stack(tgts, 0), th.stack(masks, 0), list(metas)

    val_set = FlomMatNPZWithMeta(args.data_dir, train=False, use_all=args.use_all_data)
    total = min(len(val_set), args.eval_num)
    loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, pin_memory=True, collate_fn=collate_fn)
    print(f"[Data] 加载测试样本 {len(val_set)}，评估上限 {args.eval_num} → 实际 {total}")

    # 读取训练配置（用于构建 RGAT 与 RDDM）
    ckpt_dir = os.path.dirname(args.ckpt)
    args_json_path = os.path.join(ckpt_dir, 'args.json')
    train_steps = 400

    model_cfg = dict(
        backbone='rgat',
        objective='rddm',
        direct_target='x0',
        hidden=64,
        heads=4,
        layers=4,
        kmax=2,
        dropout=0.1,
        lowrank=0,
        use_edge_sign=True,
        use_time_film=True,
        force_hermitian_decoder=True,
        unet_base=64,
        unet_attention='none',
        unet_skip_gate=False,
    )
    if os.path.exists(args_json_path):
        print(f"[Config] 从 {args_json_path} 加载训练配置")
        with open(args_json_path, 'r', encoding='utf-8') as f:
            train_args = json.load(f)
        train_steps = train_args.get('steps', train_steps)
        model_cfg['backbone'] = train_args.get('backbone', model_cfg['backbone'])
        model_cfg['objective'] = train_args.get('objective', model_cfg['objective'])
        model_cfg['direct_target'] = train_args.get('direct_target', model_cfg['direct_target'])
        model_cfg['hidden']  = train_args.get('gat_hidden',  model_cfg['hidden'])
        model_cfg['heads']   = train_args.get('gat_heads',   model_cfg['heads'])
        model_cfg['layers']  = train_args.get('gat_layers',  model_cfg['layers'])
        model_cfg['kmax']    = train_args.get('gat_kmax',    model_cfg['kmax'])
        model_cfg['dropout'] = train_args.get('gat_dropout', model_cfg['dropout'])
        model_cfg['lowrank'] = train_args.get('lowrank_decoder', model_cfg['lowrank'])
        model_cfg['use_edge_sign'] = train_args.get('use_edge_sign', model_cfg['use_edge_sign'])
        model_cfg['use_time_film'] = not train_args.get('no_time_film', False)
        model_cfg['force_hermitian_decoder'] = not train_args.get('no_hermitian_decoder', False)
        model_cfg['unet_base'] = train_args.get('unet_base', model_cfg['unet_base'])
        model_cfg['unet_attention'] = train_args.get('unet_attention', train_args.get('unet_attn', model_cfg['unet_attention']))
        model_cfg['unet_skip_gate'] = train_args.get('unet_skip_gate', model_cfg['unet_skip_gate'])
    print(
        f"[Config] model: backbone={model_cfg['backbone']}, objective={model_cfg['objective']}, "
        f"M={M}, hidden={model_cfg['hidden']}, heads={model_cfg['heads']}, layers={model_cfg['layers']}, "
        f"kmax={model_cfg['kmax']}, dropout={model_cfg['dropout']}, lowrank={model_cfg['lowrank']}, "
        f"edge_sign={model_cfg['use_edge_sign']}, time_film={model_cfg['use_time_film']}, "
        f"hermitian_decoder={model_cfg['force_hermitian_decoder']}, "
        f"unet_attention={model_cfg['unet_attention']}, unet_skip_gate={model_cfg['unet_skip_gate']}, "
        f"steps={train_steps}"
    )
    print(f"[Config] 采样: sampler={args.sampler}, ddim_steps={args.ddim_steps}, λ_res={args.lambda_res}, init_noise_scale={args.init_noise_scale}, hermitian={not args.sample_no_hermitian}, proj_toeplitz={args.sample_proj_toeplitz}, proj_psd={args.sample_proj_psd}")

    # 构建模型
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    if model_cfg['backbone'] == 'unet':
        model = UNetTwoHead(
            M=M,
            base=model_cfg['unet_base'],
            t_dim=model_cfg['hidden'],
            dropout=model_cfg['dropout'],
            use_time_film=model_cfg['use_time_film'],
            force_hermitian_decoder=model_cfg['force_hermitian_decoder'],
            attention=model_cfg['unet_attention'],
            use_skip_gate=model_cfg['unet_skip_gate'],
        )
    else:
        model = GATTwoHead(
            M=M,
            hidden=model_cfg['hidden'], heads=model_cfg['heads'], layers=model_cfg['layers'],
            k_max=model_cfg['kmax'], t_dim=model_cfg['hidden'],
            use_edge_sign=model_cfg['use_edge_sign'],
            dropout=model_cfg['dropout'], lowrank_decoder=model_cfg['lowrank'],
            use_time_film=model_cfg['use_time_film'],
            force_hermitian_decoder=model_cfg['force_hermitian_decoder'],
        )
    if model_cfg['objective'] == 'direct':
        netD = DirectDenoiser(
            model=model, image_size=M, timesteps=train_steps,
            loss_type='l1', w_res=1.0, w_noise=0.0,
            target=model_cfg['direct_target'], device=device,
        ).to(device)
    elif model_cfg['objective'] == 'rddm_x0':
        netD = ResidualX0Diffusion(
            model=model, image_size=M, timesteps=train_steps,
            loss_type='l1', w_res=1.0, w_noise=1.0, device=device,
        ).to(device)
    elif model_cfg['objective'] == 'ddim':
        netD = PlainDDIMDiffusion(
            model=model, image_size=M, timesteps=train_steps,
            loss_type='l1', w_res=0.0, w_noise=1.0, device=device,
        ).to(device)
    else:
        netD = ResidualDiffusion(
            model=model, image_size=M, timesteps=train_steps,
            loss_type='l1', w_res=1.0, w_noise=1.0, device=device
        ).to(device)
    print(f"[Model] 构建完成，T={train_steps}, 设备={device}")

    # 加载权重
    ckpt = th.load(args.ckpt, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        print('[Model] 从 \"model\" 键加载权重')
        missing, unexpected = netD.load_state_dict(ckpt['model'], strict=False)
    elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
        print('[Model] 从 \"state_dict\" 键加载权重')
        missing, unexpected = netD.load_state_dict(ckpt['state_dict'], strict=False)
    else:
        print('[Model] 直接加载 state_dict')
        missing, unexpected = netD.load_state_dict(ckpt, strict=False)
    if missing:   print(f"[Model] 警告：缺失键 {missing}")
    if unexpected: print(f"[Model] 警告：多余键 {unexpected}")

    netD.eval()
    print('[Model] 已切换到评估模式')

    # 评测累积器
    sr_noisy, rmse_noisy = [], []
    sr_deno , rmse_deno  = [], []
    sr_clean, rmse_clean = [], []

    # 分层桶
    from collections import defaultdict
    bins_snr = defaultdict(lambda: dict(sr_noisy=[], rmse_noisy=[], sr_deno=[], rmse_deno=[], sr_clean=[], rmse_clean=[]))
    bins_K   = defaultdict(lambda: dict(sr_noisy=[], rmse_noisy=[], sr_deno=[], rmse_deno=[], sr_clean=[], rmse_clean=[]))

    # 明细（可导出）
    details = []

    # 采样设置（打印与原风格对齐）
    sampling_t = args.sampling_timestep if args.sampling_timestep is not None else train_steps - 1
    print(f"[Sampling] 模式={args.sampler}, t={sampling_t}（仅 one-step 打印对齐）, ddim_steps={args.ddim_steps}, eta={args.eta}")

    # 准备绘图
    if args.plot_num > 0:
        os.makedirs(args.plot_out, exist_ok=True)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt  # noqa
        print(f"[Plot] 将保存谱图至: {args.plot_out}")

    n_done = 0
    n_plotted = 0
    device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    
    # 计算要绘制的样本索引（从数据集均匀抽样）
    total_samples = min(len(val_set), args.eval_num)
    if args.plot_num > 0 and total_samples > 0:
        plot_indices = set(np.linspace(0, total_samples - 1, min(args.plot_num, total_samples), dtype=int).tolist())
        print(f"[Plot] 将从 {total_samples} 个样本中均匀抽取 {len(plot_indices)} 个绘制谱图")
    else:
        plot_indices = set()
    
    print('[Eval] 开始评估...')

    for cond8, tgt8, mask, meta in loader:
        B = cond8.size(0)
        cond8 = cond8.to(device)

        with autocast(device_type, enabled=args.fp16):
            if args.sampler == 'one':
                C_deno = sample_rddm_denoised(
                    netD, cond8, sampling_t, args.lambda_res, args.init_noise_scale,
                    do_project=False, proj_hermitian=not args.sample_no_hermitian,
                    proj_toeplitz=args.sample_proj_toeplitz,
                    proj_psd=args.sample_proj_psd
                )
            else:
                C_deno, _ = sample_rddm_multistep(netD, cond8, steps=args.ddim_steps,
                                                  lambda_res=args.lambda_res, init_noise_scale=args.init_noise_scale,
                                                  do_project=False, proj_hermitian=not args.sample_no_hermitian,
                                                  proj_toeplitz=args.sample_proj_toeplitz,
                                                  proj_psd=args.sample_proj_psd, eta=args.eta)
        cond_np = cond8.cpu().numpy()
        tgt_np  = tgt8.cpu().numpy()

        for b in range(B):
            C_noisy = cond_np[b,0,:,:] + 1j*cond_np[b,1,:,:]
            C_clean = tgt_np [b,0,:,:] + 1j*tgt_np [b,1,:,:]
            C_d     = C_deno[b]

            if args.project_inputs:
                C_noisy = project_struct_numpy(C_noisy)
                C_clean = project_struct_numpy(C_clean)
                C_d     = project_struct_numpy(C_d)

            sm = meta[b] if isinstance(meta[b], dict) else {}
            K = int(sm.get('K', 2))
            gt = [float(x) for x in sm.get('thetas_deg', [])]
            alpha = float(sm.get('alpha', np.nan))
            snr_db = float(sm.get('snr_db', np.nan))

            # MUSIC（同时获取伪谱用于绘图）
            pred_noisy, pseu_noisy = run_music(C_noisy, K, d_over_lambda, theta_grid, music_scan)
            pred_deno , pseu_deno  = run_music(C_d,     K, d_over_lambda, theta_grid, music_scan)
            pred_clean, pseu_clean = run_music(C_clean, K, d_over_lambda, theta_grid, music_scan)

            # 指标
            sr, rm = match_metrics(pred_noisy, gt, thr=args.sr_thr)
            sr_noisy.append(sr); rmse_noisy.append(rm)
            s_no, r_no = sr, rm

            sr, rm = match_metrics(pred_deno, gt, thr=args.sr_thr)
            sr_deno.append(sr); rmse_deno.append(rm)
            s_de, r_de = sr, rm

            sr, rm = match_metrics(pred_clean, gt, thr=args.sr_thr)
            sr_clean.append(sr); rmse_clean.append(rm)
            s_cl, r_cl = sr, rm

            # 分层：SNR（逐 dB）、K
            snr_key = _snr_bin(snr_db, args.snr_bin_mode)
            if snr_key is not None:
                bins_snr[snr_key]['sr_noisy'].append(s_no)
                bins_snr[snr_key]['rmse_noisy'].append(r_no)
                bins_snr[snr_key]['sr_deno'].append(s_de)
                bins_snr[snr_key]['rmse_deno'].append(r_de)
                bins_snr[snr_key]['sr_clean'].append(s_cl)
                bins_snr[snr_key]['rmse_clean'].append(r_cl)

            bins_K[K]['sr_noisy'].append(s_no)
            bins_K[K]['rmse_noisy'].append(r_no)
            bins_K[K]['sr_deno'].append(s_de)
            bins_K[K]['rmse_deno'].append(r_de)
            bins_K[K]['sr_clean'].append(s_cl)
            bins_K[K]['rmse_clean'].append(r_cl)

            # 明细（可选导出）
            details.append(dict(
                idx=n_done,
                K=K, snr_db=snr_db, snr_bin=snr_key, alpha=alpha,
                sr_noisy=s_no, rmse_noisy=r_no,
                sr_deno=s_de, rmse_deno=r_de,
                sr_clean=s_cl, rmse_clean=r_cl,
            ))

            # 绘制谱图（仅当当前样本索引在抽样集合中）
            if n_done in plot_indices:
                out_path = os.path.join(args.plot_out, f"spectrum_{n_plotted:04d}.png")
                title = f"Sample#{n_done} | K={K} | α={alpha:.2f}, SNR={snr_db:.1f} dB | RDDM-RGAT"
                plot_three_spectra(theta_grid, pseu_noisy, pseu_deno, pseu_clean,
                                   pred_noisy, pred_deno, pred_clean, gt,
                                   title, out_path, normalize=args.norm, yscale=args.yscale)
                n_plotted += 1
                if n_plotted % 20 == 0:
                    print(f"[Plot] 已保存 {n_plotted}/{len(plot_indices)} 张谱图")

            n_done += 1
            if n_done >= args.eval_num:
                break
        if n_done >= args.eval_num:
            break

    # ========== 汇总打印 ==========
    def _fmt_row(name, a, b, c):
        return f"{name:<12} {a:<12.3f} {b:<12.3f} {c:<12.3f}"

    print('\n' + '='*64)
    print(f"Eval done: {n_done} samples (threshold={args.sr_thr} deg)")
    print('='*64)
    print(f"{'指标':<12} {'Noisy':<12} {'Denoised':<12} {'Clean':<12}")
    print('-'*64)
    print(_fmt_row('SR@τ', _nanmean(sr_noisy), _nanmean(sr_deno), _nanmean(sr_clean)))
    print(_fmt_row('RMSE(°)', _nanmean(rmse_noisy), _nanmean(rmse_deno), _nanmean(rmse_clean)))
    print('='*64)
    if args.plot_num > 0:
        print(f"谱图已保存至: {args.plot_out} (共 {n_plotted} 张，从 {n_done} 个样本中均匀抽样)")
        print('='*64)

    # 分层：按 SNR bin 升序
    if len(bins_snr):
        print('[Group] 按 SNR(dB) 分层：')
        keys = sorted(k for k in bins_snr.keys() if k is not None)
        for k in keys:
            g = bins_snr[k]
            print(f"  SNR={k:>3} dB -> SR@τ: no={_nanmean(g['sr_noisy']):.3f}, de={_nanmean(g['sr_deno']):.3f}, cl={_nanmean(g['sr_clean']):.3f} | "
                  f"RMSE: no={_nanmean(g['rmse_noisy']):.3f}, de={_nanmean(g['rmse_deno']):.3f}, cl={_nanmean(g['rmse_clean']):.3f}")

    # 分层：按 K 升序
    if len(bins_K):
        print('[Group] 按 K 分层：')
        for k in sorted(bins_K.keys()):
            g = bins_K[k]
            print(f"  K={k} -> SR@τ: no={_nanmean(g['sr_noisy']):.3f}, de={_nanmean(g['sr_deno']):.3f}, cl={_nanmean(g['sr_clean']):.3f} | "
                  f"RMSE: no={_nanmean(g['rmse_noisy']):.3f}, de={_nanmean(g['rmse_deno']):.3f}, cl={_nanmean(g['rmse_clean']):.3f}")

    # ========== 可选导出 ==========
    if args.out_csv:
        import csv
        with open(args.out_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['idx','K','snr_db','snr_bin','alpha','sr_noisy','rmse_noisy','sr_deno','rmse_deno','sr_clean','rmse_clean'])
            for d in details:
                w.writerow([d['idx'], d['K'], d['snr_db'], d['snr_bin'], d['alpha'],
                            d['sr_noisy'], d['rmse_noisy'], d['sr_deno'], d['rmse_deno'], d['sr_clean'], d['rmse_clean']])
        print(f"[Save] 明细已导出 CSV: {args.out_csv}")

    if args.out_json:
        with open(args.out_json, 'w', encoding='utf-8') as f:
            json.dump(dict(
                overall=dict(
                    SR_noisy=_nanmean(sr_noisy), SR_deno=_nanmean(sr_deno), SR_clean=_nanmean(sr_clean),
                    RMSE_noisy=_nanmean(rmse_noisy), RMSE_deno=_nanmean(rmse_deno), RMSE_clean=_nanmean(rmse_clean)
                ),
                by_snr={int(k): dict(
                    SR_noisy=_nanmean(v['sr_noisy']), SR_deno=_nanmean(v['sr_deno']), SR_clean=_nanmean(v['sr_clean']),
                    RMSE_noisy=_nanmean(v['rmse_noisy']), RMSE_deno=_nanmean(v['rmse_deno']), RMSE_clean=_nanmean(v['rmse_clean'])
                ) for k, v in sorted(bins_snr.items()) if k is not None},
                by_K={int(k): dict(
                    SR_noisy=_nanmean(v['sr_noisy']), SR_deno=_nanmean(v['sr_deno']), SR_clean=_nanmean(v['sr_clean']),
                    RMSE_noisy=_nanmean(v['rmse_noisy']), RMSE_deno=_nanmean(v['rmse_deno']), RMSE_clean=_nanmean(v['rmse_clean'])
                ) for k, v in sorted(bins_K.items())},
            ), f, ensure_ascii=False, indent=2)
        print(f"[Save] 汇总已导出 JSON: {args.out_json}")

    print('='*64 + '\n')


if __name__ == '__main__':
    main()
