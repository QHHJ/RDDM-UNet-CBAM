#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_dataset_flom_or_scm.py
生成用于 “RDDM+RGAT” 训练/评测的矩阵数据集（8x8，Re/Im 两通道）。

新增：
- --noise-type {alpha, gaussian}
  * alpha   → SαS 噪声 + FLOM（p<2）构矩阵
  * gaussian→ 复高斯白噪声 + 二阶矩（SCM）构矩阵

保持：
- 保存键名：大 NPZ 使用 x0s/xins（与训练脚本一致）
- 默认强制：只写一个大 npz（chunk_size 自动设成 num_samples）
- 支持 SNR 分层（--snr-step-db 控制步长）
"""

import os, json, math
import argparse
import numpy as np

# -----------------------------
# 阵列/信号生成
# -----------------------------
def steering_vec_ula(M: int, d_over_lambda: float, theta_deg: float):
    m = np.arange(M)
    theta = np.deg2rad(theta_deg)
    phase = 2.0 * np.pi * d_over_lambda * np.sin(theta) * m
    return np.exp(1j * phase)  # [M]

def simulate_clean_snapshots(M, T, thetas_deg, d_over_lambda, rng):
    K = len(thetas_deg)
    A = np.stack([steering_vec_ula(M, d_over_lambda, th) for th in thetas_deg], axis=1)  # [M,K]
    S = (rng.standard_normal((K, T)) + 1j * rng.standard_normal((K, T))) / np.sqrt(2.0)  # CN(0,1)
    Y = A @ S  # [M,T]
    return Y, A, S

# -----------------------------
# 矩阵构造：FLOM / SCM
# -----------------------------
def flom_matrix(Y: np.ndarray, p: float, eps: float = 1e-12):
    """
    FLOM：C^(p) = (Y ∘ |Y|^{p/2-1}) (·)^H / T
    Y: [M,T] 复矩阵；0<p<2（实际需 p<alpha）
    """
    M, T = Y.shape
    absY = np.abs(Y) + eps
    w = absY ** (p / 2.0 - 1.0)        # [M,T]
    Yw = Y * w                         # [M,T]
    return (Yw @ Yw.conj().T) / float(T)  # [M,M]

def scm_matrix(Y: np.ndarray):
    """
    SCM（二阶矩）：R = YY^H / T
    """
    M, T = Y.shape
    return (Y @ Y.conj().T) / float(T)

def hermitian_symmetrize(C: np.ndarray):
    return 0.5 * (C + C.conj().T)

def toeplitz_project(C: np.ndarray):
    """沿副对角线平均（可选“扶正”）"""
    M = C.shape[0]
    out = np.zeros_like(C, dtype=C.dtype)
    for k in range(-(M-1), M):
        diag = np.diag(C, k)
        mean_val = diag.mean()
        out += np.diag([mean_val]*(M - abs(k)), k)
    return hermitian_symmetrize(out)

def psd_project(C: np.ndarray, eps: float = 0.0):
    """投影到 PSD（裁剪负特征值）"""
    vals, vecs = np.linalg.eigh(hermitian_symmetrize(C))
    vals = np.maximum(vals, eps)
    return (vecs * vals) @ vecs.conj().T

def normalize_matrix(C: np.ndarray, method: str = "trace", eps: float = 1e-12):
    """
    用观测侧尺度 c 做归一化，返回 C/c 与 c
    method: 'trace' or 'diag'
    """
    M = C.shape[0]
    if method == "trace":
        c = (np.trace(C).real / M) + eps
    elif method == "diag":
        c = (np.mean(np.real(np.diag(C)))) + eps
    else:
        raise ValueError("norm method must be 'trace' or 'diag'")
    return C / c, c

# -----------------------------
# 噪声：SαS（CMS）/ 复高斯
# -----------------------------
def salpha_stable_real(alpha: float, size, rng: np.random.Generator):
    U = rng.uniform(-np.pi/2, np.pi/2, size)
    if abs(alpha - 1.0) < 1e-12:
        X = np.tan(U)
    else:
        W = rng.exponential(1.0, size)
        num = np.sin(alpha * U)
        den = (np.cos(U)) ** (1.0 / alpha)
        frac = num / den
        expo = (np.cos(U - alpha * U) / W) ** ((1.0 - alpha) / alpha)
        X = frac * expo
    return X

def salpha_stable_complex(alpha: float, size, rng: np.random.Generator):
    x = salpha_stable_real(alpha, size, rng)
    y = salpha_stable_real(alpha, size, rng)
    return x + 1j * y

def complex_gaussian(size, rng: np.random.Generator):
    return (rng.standard_normal(size) + 1j * rng.standard_normal(size)) / np.sqrt(2.0)


def sample_thetas(rng: np.random.Generator, K: int, doa_min: float, doa_max: float, min_sep_deg: float):
    if K <= 1 or min_sep_deg <= 0:
        return rng.uniform(doa_min, doa_max, size=K).tolist()
    span = float(doa_max) - float(doa_min)
    if span < float(min_sep_deg) * float(K - 1):
        raise ValueError("DOA range is too small for the requested min separation")
    for _ in range(10000):
        thetas = np.sort(rng.uniform(doa_min, doa_max, size=K))
        if np.min(np.diff(thetas)) >= float(min_sep_deg):
            return thetas.tolist()
    raise RuntimeError("Failed to sample DOAs with the requested min separation")

# -----------------------------
# p-阶矩 SNR 标定（样本级）
# -----------------------------
def scale_noise_to_psnr(Y_clean: np.ndarray, W: np.ndarray, snr_db: float, p_for_snr: float, eps: float = 1e-12):
    """
    将噪声 W 缩放，使 SNR_p = 10 log10(E|signal|^p / E|noise|^p) 达到 snr_db
    """
    sig_p = np.mean(np.abs(Y_clean) ** p_for_snr)
    noi_p0 = np.mean(np.abs(W) ** p_for_snr) + eps
    ratio = 10.0 ** (snr_db / 10.0)
    s = (sig_p / (ratio * noi_p0)) ** (1.0 / p_for_snr)
    return s * W

# -----------------------------
# 生成一个样本（按 noise-type 分支）
# -----------------------------
def make_one_sample(args, rng: np.random.Generator, snr_db_override: float = None):
    # 1) K 与 DOA
    if args.K_fixed > 0:
        K = args.K_fixed
    else:
        K = rng.integers(args.K_min, args.K_max + 1)
    thetas = sample_thetas(rng, K, args.doa_min, args.doa_max, args.min_sep_deg)

    # 2) 干净快拍
    Y_clean, A, S = simulate_clean_snapshots(args.M, args.T, thetas, args.d_over_lambda, rng)

    # 3) 噪声与 SNR 标定
    if snr_db_override is not None:
        snr_db = float(snr_db_override)
    else:
        if args.snr_grid is None:
            snr_db = rng.uniform(args.snr_min, args.snr_max)
        else:
            snr_db = float(rng.choice(args.snr_grid))

    noise_type = args.noise_type.lower()
    if noise_type == "alpha":
        alpha = rng.uniform(args.alpha_min, args.alpha_max)
        W = salpha_stable_complex(alpha, size=Y_clean.size, rng=rng).reshape(Y_clean.shape)
        # FLOM：p 由 --p / --p-rand 决定；SNR 也用 p（或 --p-snr）
        p_use = args.p if not args.p_rand else rng.uniform(args.p_min, min(args.p_max, alpha - 1e-3))
        p_snr = args.p_snr if args.p_snr > 0 else p_use
        W = scale_noise_to_psnr(Y_clean, W, snr_db, p_for_snr=p_snr)
        Y_obs = Y_clean + W
        # 4) 构矩阵：FLOM
        Cx = flom_matrix(Y_clean, p=p_use, eps=1e-12)
        Cy = flom_matrix(Y_obs,   p=p_use, eps=1e-12)
        meta_extra = dict(alpha=float(alpha), p=float(p_use), p_snr=float(p_snr))
    elif noise_type == "gaussian":
        alpha = 2.0
        W = complex_gaussian(size=Y_clean.size, rng=rng).reshape(Y_clean.shape)  # CN(0,1)
        # 高斯：SNR 用 p=2（功率）
        W = scale_noise_to_psnr(Y_clean, W, snr_db, p_for_snr=2.0)
        Y_obs = Y_clean + W
        # 4) 构矩阵：SCM（二阶矩）
        Cx = scm_matrix(Y_clean)
        Cy = scm_matrix(Y_obs)
        meta_extra = dict(alpha=float(alpha), p=float(2.0), p_snr=float(2.0))
    else:
        raise ValueError("noise-type must be 'alpha' or 'gaussian'")

    # 5) 可选投影 & 厄米化
    if args.toeplitz_project:
        Cx = toeplitz_project(Cx)
        Cy = toeplitz_project(Cy)
    if args.hermitian_fix:
        Cx = hermitian_symmetrize(Cx)
        Cy = hermitian_symmetrize(Cy)

    # 6) 归一化（观测侧尺度）
    if args.disable_norm:
        Cy_norm = Cy
        Cx_norm = Cx
        c_scale = 1.0
    else:
        Cy_norm, c_scale = normalize_matrix(Cy, method=args.norm, eps=1e-12)
        Cx_norm = Cx / c_scale

    # 7) 可选 PSD 投影（一般训练集不做；推断再投影）
    if args.psd_project:
        Cx_norm = psd_project(Cx_norm)
        Cy_norm = psd_project(Cy_norm)

    # 8) 打包（Re/Im 两通道）
    def to_reim(C):
        return np.stack([C.real, C.imag], axis=0).astype(np.float32)  # [2,M,M]

    sample = dict(
        xins=to_reim(Cy_norm),
        x0=to_reim(Cx_norm),
        meta=dict(
            M=int(args.M), T=int(args.T), K=int(K), thetas_deg=thetas,
            snr_db=float(snr_db), norm=args.norm, scale_c=float(c_scale),
            noise_type=noise_type, min_sep_deg=float(args.min_sep_deg), **meta_extra
        )
    )
    return sample

# -----------------------------
# SNR 分层
# -----------------------------
def build_snr_bins(args):
    """
    返回 (bins[np.ndarray], counts[np.ndarray])：每个 bin 的 SNR 值与应生成的样本数
    - 若 snr_grid 指定：按 grid 值逐一分层，均分样本。
    - 否则：
        * 若 snr_min == snr_max → 单值层
        * 否则按 [snr_min, snr_max] 以 snr_step_db 步长均匀取（浮点网格）。
    """
    N = args.num_samples
    grid = None
    if args.snr_grid is not None:
        items = [x for x in args.snr_grid.split(",") if x.strip() != ""]
        if len(items) > 0:
            grid = np.array([float(x) for x in items], dtype=np.float32)

    if grid is not None:
        bins = np.array(sorted(grid.tolist()), dtype=np.float32)
    else:
        if abs(args.snr_min - args.snr_max) < 1e-12:
            bins = np.array([float(args.snr_min)], dtype=np.float32)
        else:
            step = float(args.snr_step_db)
            if step <= 0:
                step = 0.5
            n_steps = int(round((args.snr_max - args.snr_min) / step)) + 1
            bins = np.linspace(args.snr_min, args.snr_max, n_steps, dtype=np.float32)

    B = len(bins)
    if B <= 0:
        raise ValueError("SNR 分层失败：未得到有效的 bins")

    base = N // B
    rem  = N % B
    counts = np.array([base + (1 if i < rem else 0) for i in range(B)], dtype=int)
    return bins, counts

# -----------------------------
# CLI
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Generate FLOM/SCM-matrix dataset for diffusion+GNN (with SNR stratification)")
    # 尺寸
    ap.add_argument("--M", type=int, default=8)
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--d-over-lambda", type=float, default=0.5, dest="d_over_lambda")
    # 源与方位
    ap.add_argument("--K-min", type=int, default=2)
    ap.add_argument("--K-max", type=int, default=2)
    ap.add_argument("--K-fixed", type=int, default=0)
    ap.add_argument("--doa-min", type=float, default=-60.0)
    ap.add_argument("--doa-max", type=float, default=60.0)
    ap.add_argument("--min-sep-deg", type=float, default=0.0)
    # 噪声类型
    ap.add_argument("--noise-type", type=str, default="gaussian", choices=["alpha","gaussian"],
                    help="alpha: SαS 噪声 + FLOM；gaussian: 复高斯噪声 + SCM")
    # α 稳定噪声与 p（当 noise-type=alpha 时生效）
    ap.add_argument("--alpha-min", type=float, default=1.2)
    ap.add_argument("--alpha-max", type=float, default=1.8)
    ap.add_argument("--p", type=float, default=1.0)
    ap.add_argument("--p-rand", action="store_true")
    ap.add_argument("--p-min", type=float, default=0.8)
    ap.add_argument("--p-max", type=float, default=1.2)
    # SNR & 分层
    ap.add_argument("--snr-min", type=float, default=-8.0)
    ap.add_argument("--snr-max", type=float, default=-5.0)
    ap.add_argument("--snr-grid", type=str, default=None,
                    help='离散 SNR 列表，如 "-10,-5,0"；设置后覆盖 min/max')
    ap.add_argument("--snr-stratify", dest="snr_stratify", action="store_true", default=True)
    ap.add_argument("--no-snr-stratify", dest="snr_stratify", action="store_false")
    ap.add_argument("--snr-step-db", type=float, default=0.1)
    ap.add_argument("--p-snr", type=float, default=0.0,
                    help="用于 SNR 标定的 p 值；<=0 时与 FLOM 的 p 相同；gaussian 强制用 2.0")
    # 归一化与投影
    ap.add_argument("--disable-norm", action="store_true", default=False)
    ap.add_argument("--norm", type=str, default="trace", choices=["trace", "diag"])
    ap.add_argument("--hermitian-fix", action="store_true", default=True)
    ap.add_argument("--toeplitz-project", action="store_true", default=False)
    ap.add_argument("--psd-project", action="store_true", default=False)
    # 数据量与输出
    ap.add_argument("--num-samples", type=int, default=1000)
    ap.add_argument("--chunk-size", type=int, default=1000, help="将被自动设为 num_samples 以写单文件")
    ap.add_argument("--out-dir", type=str, default="dataset_snap=16_snr_-10_to_-5_gaussian_test_1000")
    ap.add_argument("--seed", type=int, default=2025)
    return ap.parse_args()

# -----------------------------
# 主流程
# -----------------------------
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # 强制单文件，保证与训练脚本兼容
    if args.chunk_size != args.num_samples:
        print(f"[Warn] chunk_size({args.chunk_size}) != num_samples({args.num_samples}) → 为兼容训练脚本，已强制合并为单文件。")
        args.chunk_size = args.num_samples

    # 写 meta
    meta = {
        "M": args.M, "T": args.T, "d_over_lambda": args.d_over_lambda,
        "K_min": args.K_min, "K_max": args.K_max, "K_fixed": args.K_fixed,
        "doa_min": args.doa_min, "doa_max": args.doa_max, "min_sep_deg": args.min_sep_deg,
        "noise_type": args.noise_type,
        "alpha_min": args.alpha_min, "alpha_max": args.alpha_max,
        "p_fixed": args.p, "p_rand": args.p_rand, "p_min": args.p_min, "p_max": args.p_max,
        "snr_min": args.snr_min, "snr_max": args.snr_max, "snr_grid": args.snr_grid,
        "snr_stratify": args.snr_stratify, "snr_step_db": args.snr_step_db,
        "p_snr": args.p_snr, "disable_norm": args.disable_norm, "norm": args.norm,
        "hermitian_fix": args.hermitian_fix, "toeplitz_project": args.toeplitz_project, "psd_project": args.psd_project,
        "num_samples": args.num_samples, "chunk_size": args.chunk_size, "seed": args.seed
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 分层/随机 SNR 采样器
    def sample_iter():
        if args.snr_stratify:
            bins, counts = build_snr_bins(args)
            print(f"[SNR stratify] bins={bins.tolist()}  counts={counts.tolist()}  (sum={counts.sum()})")
            for snr_db, cnt in zip(bins, counts):
                for _ in range(int(cnt)):
                    yield make_one_sample(args, rng, snr_db_override=float(snr_db))
        else:
            for _ in range(args.num_samples):
                yield make_one_sample(args, rng, snr_db_override=None)

    # 只写一个大 npz
    xins_arr = np.zeros((args.num_samples, 2, args.M, args.M), dtype=np.float32)
    x0s_arr  = np.zeros_like(xins_arr)
    metas = []

    gen = sample_iter()
    for i in range(args.num_samples):
        sample = next(gen)
        xins_arr[i] = sample["xins"]
        x0s_arr[i]  = sample["x0"]
        metas.append(sample["meta"])
        if (i+1) % 5000 == 0 or (i+1) == args.num_samples:
            print(f"[Gen] {i+1}/{args.num_samples}")

    out_path = os.path.join(args.out_dir, "flom_mat_000.npz")
    np.savez_compressed(out_path,
                        x0s=x0s_arr,   # 训练脚本期望的大 NPZ 键名
                        xins=xins_arr,
                        meta=json.dumps(metas, ensure_ascii=False))
    print(f"[Save] {out_path}  samples={args.num_samples}")
    print("Done.")

if __name__ == "__main__":
    main()
