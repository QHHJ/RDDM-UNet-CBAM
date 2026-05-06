# -*- coding: utf-8 -*-
# train_rddm_rgat.py
# -------------------------------------------------------------
# RDDM + (R)GAT 训练脚本（增强版，日志 & 保存路径保持原样）
# 新增开关：
#   --identity-assert      恒等任务自检（第1个batch检查 max|xin-x0| 与理想损失）
#   --decoder-zero-init    训练启动时把解码器权重置零/近零（sanity / 调试）
#   --freeze-backbone      冻结骨干，只训解码器（sanity 快速定位）
#   --film-probe           每个 epoch 记录一次 FiLM γ/β 在 t=0 与 t=T-1 的均值差
#   --ema/--ema-decay      维护 EMA 权重（验证与 best 保存用 EMA）
#   --warmup-steps         余弦调度 + warmup
#   --accum-steps          梯度累积（等效放大 batch）
#   --res-diag-weight      残差头“实部对角”加权（默认=1.0 不变更）
#   --res-off-penalty      残差头“实部非对角”L1正则（默认=0.0 不变更）
# 其余代码风格、日志与保存方式保持一致
# -------------------------------------------------------------
import os, json, time, glob, argparse
from datetime import datetime

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
import torch.nn.functional as F

# 兼容 import：优先 model.rddm_gat，失败则退回同目录 rddm_gat.py
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



# ----------------------------- 数据集 -----------------------------
class NPZPairDataset(Dataset):
    """
    支持：
      1) 单个大 npz（包含 xins/x0s/meta）
      2) 目录下多个 flom_mat_*.npz（每个文件一个样本或一批样本）
    """
    def __init__(self, path, train=True, split=0.95, use_all=False):
        self.train = train
        self.split = split
        self.use_all = use_all

        if os.path.isfile(path) and path.endswith(".npz"):
            self.mode = "big_npz"
            with np.load(path, allow_pickle=True) as npz:
                self.xins = npz["xins"].astype(np.float32)  # [N,2,M,M]
                self.x0s  = npz["x0s"].astype(np.float32)   # [N,2,M,M]
                self.meta = npz.get("meta", None)
            self.N = self.xins.shape[0]
            self.M = self.xins.shape[-1]
            n_tr = int(self.N * self.split)
            if self.use_all:
                self.idx = np.arange(self.N)
            else:
                self.idx = np.arange(n_tr) if train else np.arange(n_tr, self.N)
        else:
            self.mode = "list_npz"
            candidates = []
            for pattern in ("flom_mat_*.npz", "ideal_mat_*.npz", "scm_mat_*.npz"):
                candidates.extend(glob.glob(os.path.join(path, pattern)))
            self.files = []
            for candidate in sorted(set(candidates)):
                with np.load(candidate, allow_pickle=True) as npz:
                    if "xins" in npz and "x0s" in npz:
                        self.files.append(candidate)
            assert self.files, f"[Data] {path} 下未找到 flom_mat_*.npz / ideal_mat_*.npz / scm_mat_*.npz"
            self.N = len(self.files)
            # 读取一个文件以确定 M
            with np.load(self.files[0], allow_pickle=True) as npz:
                self.M = npz["xins"].shape[-1]
            n_tr = int(self.N * self.split)
            self.files = self.files if self.use_all else (self.files[:n_tr] if train else self.files[n_tr:])

        # 兼容用：占位 mask（目前不参与损失）
        self.mask = torch.ones(1, self.M, self.M, dtype=torch.float32)

    def __len__(self):
        return len(self.idx) if self.mode == "big_npz" else len(self.files)

    def __getitem__(self, i):
        if self.mode == "big_npz":
            j = self.idx[i]
            xin = torch.from_numpy(self.xins[j])  # [2,M,M]
            x0  = torch.from_numpy(self.x0s[j])   # [2,M,M]
            return xin, x0, self.mask
        else:
            with np.load(self.files[i], allow_pickle=True) as npz:
                xin = torch.from_numpy(npz["xins"].astype(np.float32))
                x0  = torch.from_numpy(npz["x0s"].astype(np.float32))
                if xin.ndim == 3:  # 单样本
                    xin = xin.unsqueeze(0); x0 = x0.unsqueeze(0)
                # 合并 batch 维
                xin = xin[0]  # 这里每个文件一个样本，便于与 big_npz 对齐
                x0  = x0[0]
                return xin, x0, self.mask


def collate_fn(batch):
    xin = torch.stack([b[0] for b in batch], dim=0)  # (B,2,M,M)
    x0  = torch.stack([b[1] for b in batch], dim=0)
    msk = batch[0][2]                                # (1,M,M)
    return xin, x0, msk


def _parse_meta_array(meta_raw, n_items):
    if meta_raw is None:
        return [{} for _ in range(n_items)]
    if isinstance(meta_raw, np.ndarray):
        if meta_raw.ndim == 0:
            return json.loads(meta_raw.item())
        if meta_raw.dtype == object:
            return list(meta_raw)
        return json.loads(str(meta_raw))
    if isinstance(meta_raw, str):
        return json.loads(meta_raw)
    return list(meta_raw)


class MusicValDataset(Dataset):
    """Validation subset with meta for MUSIC-based model selection."""
    def __init__(self, path, max_items=0):
        self.items = []
        self.xins = None
        self.x0s = None
        if os.path.isfile(path) and path.endswith(".npz"):
            with np.load(path, allow_pickle=True) as npz:
                self.xins = npz["xins"].astype(np.float32)
                self.x0s = npz["x0s"].astype(np.float32)
                metas = _parse_meta_array(npz.get("meta", None), self.xins.shape[0])
            n = self.xins.shape[0] if max_items <= 0 else min(max_items, self.xins.shape[0])
            self.items = [(None, i, metas[i]) for i in range(n)]
            self.M = self.xins.shape[-1]
        else:
            candidates = []
            for pattern in ("flom_mat_*.npz", "ideal_mat_*.npz", "scm_mat_*.npz"):
                candidates.extend(glob.glob(os.path.join(path, pattern)))
            files = []
            for candidate in sorted(set(candidates)):
                with np.load(candidate, allow_pickle=True) as npz:
                    if "xins" in npz and "x0s" in npz:
                        files.append(candidate)
            assert files, f"[MusicVal] {path} 下未找到 flom_mat_*.npz / ideal_mat_*.npz / scm_mat_*.npz"
            for f in files:
                with np.load(f, allow_pickle=True) as npz:
                    n = npz["xins"].shape[0]
                    self.M = npz["xins"].shape[-1]
                    metas = _parse_meta_array(npz.get("meta", None), n)
                for i in range(n):
                    if max_items > 0 and len(self.items) >= max_items:
                        break
                    self.items.append((f, i, metas[i]))
                if max_items > 0 and len(self.items) >= max_items:
                    break

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        f, j, meta = self.items[i]
        if f is None:
            xin = self.xins[j]
            x0 = self.x0s[j]
        else:
            with np.load(f, allow_pickle=True) as npz:
                xin = npz["xins"][j].astype(np.float32)
                x0 = npz["x0s"][j].astype(np.float32)
        return torch.from_numpy(xin), torch.from_numpy(x0), meta


def music_collate_fn(batch):
    xin = torch.stack([b[0] for b in batch], dim=0)
    x0 = torch.stack([b[1] for b in batch], dim=0)
    meta = [b[2] for b in batch]
    return xin, x0, meta


def load_music_grid(data_path, fallback_M=8):
    meta_path = os.path.join(data_path, "meta.json") if os.path.isdir(data_path) else os.path.join(os.path.dirname(data_path), "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        M = int(meta.get("M", fallback_M))
        d_over_lambda = float(meta.get("d_over_lambda", 0.5))
        doa_min = float(meta.get("doa_min", -60.0))
        doa_max = float(meta.get("doa_max", 60.0))
    else:
        M, d_over_lambda, doa_min, doa_max = fallback_M, 0.5, -60.0, 60.0
    return M, d_over_lambda, np.linspace(doa_min, doa_max, 721)


def music_spectrum_np(R, K, d_over_lambda, theta_grid_deg):
    R = 0.5 * (R + R.conj().T)
    _, V = np.linalg.eigh(R)
    M = R.shape[0]
    K = max(1, min(int(K), M - 1))
    En = V[:, :M - K]
    th_rad = np.deg2rad(np.asarray(theta_grid_deg))
    m = np.arange(M)[:, None]
    phase = 2 * np.pi * d_over_lambda * np.sin(th_rad)[None, :] * m
    a_scan = np.exp(1j * phase)
    denom = np.sum(np.abs(En.conj().T @ a_scan) ** 2, axis=0)
    return 1.0 / (denom + 1e-12)


def run_music_np(R, K, d_over_lambda, theta_grid_deg):
    P = music_spectrum_np(R, K, d_over_lambda, theta_grid_deg)
    th = np.asarray(theta_grid_deg)
    G = len(th)
    if G < 3:
        idx = np.argsort(P)[-K:][::-1]
        return th[idx]

    scan_res = float(abs(th[1] - th[0]))
    min_dist_pts = max(1, int(round(1.0 / scan_res)))
    is_peak = np.zeros(G, dtype=bool)
    is_peak[1:-1] = (P[1:-1] > P[:-2]) & (P[1:-1] >= P[2:])
    cand = np.where(is_peak)[0]
    if cand.size == 0:
        idx = np.argsort(P)[-K:][::-1]
        return th[idx]

    cand = cand[np.argsort(P[cand])[::-1]]
    picked = []
    for i in cand:
        if all(abs(i - j) >= min_dist_pts for j in picked):
            picked.append(i)
            if len(picked) == K:
                break
    if len(picked) < K:
        for i in cand:
            if i not in picked:
                picked.append(i)
            if len(picked) == K:
                break
    if len(picked) < K:
        for i in np.argsort(P)[::-1]:
            if i not in picked:
                picked.append(int(i))
            if len(picked) == K:
                break
    return th[np.array(picked[:K], dtype=int)]


def match_metrics_np(pred_deg, gt_deg, thr=0.5):
    pred = list(pred_deg)
    gt = list(gt_deg)
    used = [False] * len(pred)
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
        sq.append(d_best ** 2)
    return hits / max(1, len(gt)), (np.mean(sq) ** 0.5 if sq else np.nan)


@torch.no_grad()
def evaluate_music_validation(netD, loader, device, d_over_lambda, theta_grid, sr_thr=0.5,
                              sr_thr2=1.0, sample_steps=1, amp_enabled=False):
    netD.eval()
    sr_deno, sr_deno2, rmse_deno = [], [], []
    sr_noisy, rmse_noisy = [], []
    sr_clean, rmse_clean = [], []
    device_type = "cuda" if device.type == "cuda" else "cpu"
    for xin, x0, metas in loader:
        xin = xin.to(device, non_blocking=True)
        with autocast(device_type=device_type, enabled=amp_enabled):
            deno = netD.p_sample_loop(
                xin, steps=sample_steps, sum_scale=0.0,
                proj_hermitian=True, proj_toeplitz=False, proj_psd=False
            )
        xin_np = xin.cpu().numpy()
        x0_np = x0.numpy()
        deno_np = deno.cpu().numpy()

        for b, meta in enumerate(metas):
            K = int(meta.get("K", 2)) if isinstance(meta, dict) else 2
            gt = [float(x) for x in meta.get("thetas_deg", [])] if isinstance(meta, dict) else []
            if not gt:
                continue
            C_noisy = xin_np[b, 0] + 1j * xin_np[b, 1]
            C_clean = x0_np[b, 0] + 1j * x0_np[b, 1]
            C_deno = deno_np[b, 0] + 1j * deno_np[b, 1]

            p_no = run_music_np(C_noisy, K, d_over_lambda, theta_grid)
            p_cl = run_music_np(C_clean, K, d_over_lambda, theta_grid)
            p_de = run_music_np(C_deno, K, d_over_lambda, theta_grid)

            s, r = match_metrics_np(p_no, gt, thr=sr_thr)
            sr_noisy.append(s); rmse_noisy.append(r)
            s, r = match_metrics_np(p_cl, gt, thr=sr_thr)
            sr_clean.append(s); rmse_clean.append(r)
            s, r = match_metrics_np(p_de, gt, thr=sr_thr)
            sr_deno.append(s); rmse_deno.append(r)
            s2, _ = match_metrics_np(p_de, gt, thr=sr_thr2)
            sr_deno2.append(s2)

    def mean(x):
        return float(np.nanmean(np.asarray(x, dtype=np.float64))) if len(x) else float("nan")

    return {
        "SR_noisy": mean(sr_noisy),
        "SR_deno": mean(sr_deno),
        "SR_deno_thr2": mean(sr_deno2),
        "SR_clean": mean(sr_clean),
        "RMSE_noisy": mean(rmse_noisy),
        "RMSE_deno": mean(rmse_deno),
        "RMSE_clean": mean(rmse_clean),
    }


# ----------------------------- EMA -----------------------------
class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.collected = {}
        # 仅对需要 grad 的参数做 EMA
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, p in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(p.detach(), alpha=(1.0 - self.decay))

    def store(self, model: nn.Module):
        self.collected = {name: p.detach().clone() for name, p in model.named_parameters() if name in self.shadow}

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        for name, p in model.named_parameters():
            if name in self.shadow:
                p.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        for name, p in model.named_parameters():
            if name in self.collected:
                p.data.copy_(self.collected[name])
        self.collected = {}


# ----------------------------- Scheduler -----------------------------
def build_warmup_cosine(optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.1):
    def lr_lambda(step):
        if step < warmup_steps:
            return max(1e-8, float(step + 1) / max(1, warmup_steps))
        # 余弦从 1 -> min_lr_ratio
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


# ----------------------------- 训练/验证 -----------------------------
def run_one_epoch(loader, netD, opt, scaler, device, is_train=True, amp_enabled=False,
                  grad_clip=1.0, accum_steps=1, identity_assert=False, logger=None, global_state=None, ema=None, scheduler=None):
    """
    global_state: dict(step=..., probed=False) 用于统计全局 step（给调度器用）与只在首batch做一次 identity 断言
    ema: ModelEMA 实例，若非 None 则在每次优化步骤后更新
    scheduler: 学习率调度器，若非 None 则在每次优化步骤后更新
    """
    netD.train(is_train)
    tot, tot_res, tot_noi, n = 0.0, 0.0, 0.0, 0
    opt_zeroed = False
    step_in_epoch = 0

    for xin, x0, _ in loader:
        step_in_epoch += 1
        xin = xin.to(device, non_blocking=True)
        x0  = x0.to(device, non_blocking=True)

        # ---- 恒等任务自检（只在首个batch触发一次）----
        if identity_assert and global_state is not None and global_state.get("first_batch_check", True):
            with torch.no_grad():
                d = (xin - x0).abs()
                maxd, meand = d.max().item(), d.mean().item()
                if logger: logger(f"[CHK-DATA] max|xin-x0|={maxd:.3e}, mean={meand:.3e}")
                ideal = (torch.zeros_like(xin) - (xin - x0)).abs().mean().item()
                if logger: logger(f"[CHK-LOSS] ideal residual (Δhat≡0) = {ideal:.3e}")
                # 恒等严格断言（你也可以把阈值放宽到 1e-7～1e-8）
                if maxd > 1e-12 or ideal > 1e-12:
                    raise RuntimeError("[IdentityAssert] 当前数据并非恒等 (xin != x0) 或损失实现不匹配 Δ 目标")
            global_state["first_batch_check"] = False

        # ---- 训练/验证前向 ----
        if is_train:
            if not opt_zeroed:
                opt.zero_grad(set_to_none=True)
                opt_zeroed = True

            with autocast(device_type=device.type, enabled=amp_enabled):
                loss, lres, lnoi = netD(x0, xin)
            loss = loss / max(1, accum_steps)  # for gradient accumulation
            scaler.scale(loss).backward()

            # 累积到达
            if (global_state["step"] + 1) % max(1, accum_steps) == 0:
                if grad_clip and grad_clip > 0:
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(netD.parameters(), max_norm=grad_clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                opt_zeroed = False
                # ★ 每次权重真的更新之后，立刻推进 EMA 和调度器
                if ema is not None:
                    ema.update(netD)
                if scheduler is not None:
                    scheduler.step()  # ★ 每次权重更新后推进调度
        else:
            with torch.no_grad():
                with autocast(device_type=device.type, enabled=amp_enabled):
                    loss, lres, lnoi = netD(x0, xin)

        bs = xin.size(0)
        tot += float(loss) * bs
        tot_res += float(lres) * bs
        tot_noi += float(lnoi) * bs
        n += bs

        # 全局 step 递增
        if global_state is not None:
            global_state["step"] += 1

    return tot / n, tot_res / n, tot_noi / n


# ----------------------------- 工具 -----------------------------
def save_args_json(args, outdir):
    path = os.path.join(outdir, "args.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

def save_ckpt(path, netD, opt, epoch, best_loss=None, extra=None):
    obj = {
        "epoch": epoch,
        "model": netD.state_dict(),
        "opt": opt.state_dict(),
        "best_loss": best_loss
    }
    if extra:
        obj.update(extra)
    torch.save(obj, path)

class Logger:
    """日志记录器：同时输出到控制台和文件"""
    def __init__(self, log_path):
        self.log_path = log_path
        self.file = open(log_path, "w", encoding="utf-8")
    
    def __call__(self, msg):
        print(msg)
        self.file.write(msg + "\n")
        self.file.flush()
    
    def close(self):
        self.file.close()


# ----------------------------- 主函数 -----------------------------
def main():
    ap = argparse.ArgumentParser("RDDM + RGAT 训练脚本",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # 数据
    ap.add_argument("--train", type=str, default="dataset_snap=1024_snr_-8_to_-5_gaussian_train_400000/flom_mat_000.npz")
    ap.add_argument("--val",   type=str, default="dataset_snap=1024_snr_-8_to_-5_gaussian_eval_40000/flom_mat_000.npz")
    ap.add_argument("--train-use-all", dest="train_use_all", action="store_true", default=True,
                    help="使用整个 train 文件；适用于 train/val 已经分开的数据集")
    ap.add_argument("--no-train-use-all", dest="train_use_all", action="store_false",
                    help="保留旧行为：只用 train 文件前 split 比例")
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--num-workers", type=int, default=0)

    # 模型 / 扩散
    ap.add_argument("--image-size", type=int, default=8)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--backbone", type=str, default="rgat", choices=["rgat", "unet"],
                    help="骨干网络：rgat=原方法，unet=RGAT 消融为卷积 UNet")
    ap.add_argument("--objective", type=str, default="rddm",
                    choices=["rddm", "rddm_x0", "direct", "ddim"],
                    help="训练目标：rddm=残差扩散，rddm_x0=扩散但直接预测x0，direct=无扩散直接回归，ddim=普通非残差DDIM")
    ap.add_argument("--direct-target", type=str, default="x0", choices=["x0", "residual"],
                    help="direct 目标：直接预测 clean x0 或 residual")
    ap.add_argument("--loss", type=str, default="l1", choices=["l1", "l2"])
    ap.add_argument("--w-res", type=float, default=1.0)
    ap.add_argument("--w-noise", type=float, default=0)

    # GAT 主干
    ap.add_argument("--gat-hidden", type=int, default=256)
    ap.add_argument("--gat-heads", type=int, default=8)
    ap.add_argument("--gat-layers", type=int, default=6)
    ap.add_argument("--gat-dropout", type=float, default=0.1)
    ap.add_argument("--use-edge-sign", action="store_true", default=True)
    ap.add_argument("--no-edge-sign", dest="use_edge_sign", action="store_false")
    ap.add_argument("--no-time-film", action="store_true",
                    help="关闭 GAT/UNet 层内的时间 FiLM 调制")
    ap.add_argument("--no-hermitian-decoder", action="store_true",
                    help="关闭输出头/解码器内置 Hermitian 约束")
    ap.add_argument("--lowrank-decoder", type=int, default=0)
    ap.add_argument("--gat-kmax", type=int, default=2, help="关系半径 |i-j|<=k_max；建议<=M-1")
    ap.add_argument("--unet-base", type=int, default=64, help="UNet 消融的基础通道数")
    ap.add_argument("--unet-attention", "--unet-attn", dest="unet_attention",
                    type=str, default="none", choices=["none", "se", "cbam"],
                    help="UNet FiLM 卷积块后的轻量注意力")
    ap.add_argument("--unet-skip-gate", action="store_true",
                    help="启用 decoder-conditioned 门控 skip connection")

    # 优化 / 运行
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--outdir", type=str, default="runs_rddm_t1024_snr_-8_to_-5_bestcfg")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--init-ckpt", type=str, default="", help="可选：加载已有模型权重后继续训练/微调")

    # （新增）训练稳定性与自检
    ap.add_argument("--identity-assert", action="store_true", help="恒等任务自检（仅在首个batch断言）")
    ap.add_argument("--decoder-zero-init", action="store_true", help="解码器零/近零初始化（sanity用）")
    ap.add_argument("--freeze-backbone", action="store_true", help="冻结骨干/编码器/时间支路，只训练解码器")
    ap.add_argument("--film-probe", action="store_true", help="每个 epoch 打印一次 FiLM γ/β 的 Δ 均值（t=0 vs t=T-1）")

    # （新增）EMA + warmup/cosine + grad accumulation
    ap.add_argument("--ema", action="store_true", default=True, help="开启 EMA（评估/保存用 EMA 权重）")
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--warmup-steps", type=int, default=2000)
    ap.add_argument("--accum-steps", type=int, default=1)

    # （已有增强）残差损失两个开关
    ap.add_argument("--res-diag-weight", type=float, default=8.0, help="仅对实部对角加权（>1 强化对角）")
    ap.add_argument("--res-off-penalty", type=float, default=0.2, help="实部非对角 L1 正则系数")

    # （新增）DOA/结构友好损失，默认 0 不改变旧行为
    ap.add_argument("--toeplitz-loss-weight", type=float, default=0.0, help="输出矩阵 Toeplitz 一致性损失权重")
    ap.add_argument("--psd-loss-weight", type=float, default=0.0, help="输出矩阵负特征值惩罚权重")
    ap.add_argument("--subspace-loss-weight", type=float, default=0.0, help="MUSIC 噪声子空间投影损失权重")
    ap.add_argument("--music-loss-weight", type=float, default=0.0, help="粗角度网格 log-MUSIC 谱损失权重")
    ap.add_argument("--music-margin-loss-weight", type=float, default=0.0, help="clean MUSIC 峰窗口 vs 假峰的 margin 损失权重")
    ap.add_argument("--subspace-k", type=int, default=2, help="子空间/MUSIC损失使用的信源数 K")
    ap.add_argument("--music-grid-size", type=int, default=121, help="MUSIC谱损失角度网格点数")
    ap.add_argument("--music-d-over-lambda", type=float, default=0.5, help="MUSIC谱损失 d/lambda")
    ap.add_argument("--music-doa-min", type=float, default=-60.0, help="MUSIC谱损失扫描角下界")
    ap.add_argument("--music-doa-max", type=float, default=60.0, help="MUSIC谱损失扫描角上界")
    ap.add_argument("--music-margin", type=float, default=0.5, help="谱峰 margin 损失的 log 谱间隔")
    ap.add_argument("--music-target-window-deg", type=float, default=0.5, help="clean 谱峰目标窗口半宽（度）")
    ap.add_argument("--music-peak-min-dist-deg", type=float, default=1.0, help="clean 谱峰选择的最小间距（度）")

    # （新增）按验证集 MUSIC 指标选模，默认关闭
    ap.add_argument("--music-val-every", type=int, default=0, help="每 N 个 epoch 跑一次 MUSIC 验证；<=0 关闭")
    ap.add_argument("--music-val-num", type=int, default=1000, help="MUSIC 验证最多使用的样本数；<=0 使用全部")
    ap.add_argument("--music-val-batch-size", type=int, default=256, help="MUSIC 验证 batch size")
    ap.add_argument("--music-val-steps", type=int, default=1, help="MUSIC 验证时 p_sample_loop 的采样步数")
    ap.add_argument("--music-sr-thr", type=float, default=0.5, help="MUSIC 验证主 SR 阈值（度）")
    ap.add_argument("--music-sr-thr2", type=float, default=1.0, help="MUSIC 验证辅助 SR 阈值（度）")
    ap.add_argument("--music-val-select", type=str, default="sr", choices=["sr", "rmse"],
                    help="MUSIC 验证 checkpoint 选择准则：sr=SR优先，rmse=RMSE优先")

    args = ap.parse_args()

    # 随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    # 数据
    train_set = NPZPairDataset(args.train, train=True,  split=0.95, use_all=args.train_use_all)
    val_set   = NPZPairDataset(args.val,   train=False, split=0.95, use_all=True)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              drop_last=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True, drop_last=False,
                            collate_fn=collate_fn)

    music_val_loader = None
    music_d_over_lambda = 0.5
    music_theta_grid = None
    if args.music_val_every and args.music_val_every > 0:
        music_val_set = MusicValDataset(args.val, max_items=args.music_val_num)
        _, music_d_over_lambda, music_theta_grid = load_music_grid(args.val, fallback_M=val_set.M)
        music_val_loader = DataLoader(
            music_val_set,
            batch_size=args.music_val_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
            collate_fn=music_collate_fn,
        )

    # 模型
    # 断言 hidden 可整除 heads
    assert args.gat_hidden % args.gat_heads == 0, f"hidden({args.gat_hidden}) 必须能被 heads({args.gat_heads}) 整除"

    use_time_film = not bool(args.no_time_film)
    force_hermitian_decoder = not bool(args.no_hermitian_decoder)
    if args.backbone == "rgat":
        model = GATTwoHead(
            M=train_set.M,
            hidden=args.gat_hidden,
            heads=args.gat_heads,
            layers=args.gat_layers,
            k_max=args.gat_kmax,
            t_dim=args.gat_hidden,
            use_edge_sign=args.use_edge_sign,
            dropout=args.gat_dropout,
            lowrank_decoder=args.lowrank_decoder,
            use_time_film=use_time_film,
            force_hermitian_decoder=force_hermitian_decoder,
        ).to(device)
    else:
        model = UNetTwoHead(
            M=train_set.M,
            base=args.unet_base,
            t_dim=args.gat_hidden,
            dropout=args.gat_dropout,
            use_time_film=use_time_film,
            force_hermitian_decoder=force_hermitian_decoder,
            attention=args.unet_attention,
            use_skip_gate=args.unet_skip_gate,
        ).to(device)

    diffusion_cls = {
        "rddm": ResidualDiffusion,
        "rddm_x0": ResidualX0Diffusion,
        "ddim": PlainDDIMDiffusion,
    }.get(args.objective)
    if args.objective == "direct":
        netD = DirectDenoiser(
            model=model, image_size=args.image_size, timesteps=args.steps,
            loss_type=args.loss, w_res=args.w_res, w_noise=args.w_noise,
            target=args.direct_target, device=device,
        ).to(device)
    else:
        netD = diffusion_cls(
            model=model, image_size=args.image_size, timesteps=args.steps,
            loss_type=args.loss, w_res=args.w_res, w_noise=args.w_noise, device=device
        ).to(device)

    # 把两个损失开关传入模型（默认1.0/0.0不改变旧行为）
    netD.res_diag_weight = float(args.res_diag_weight)
    netD.res_off_penalty = float(args.res_off_penalty)
    netD.toeplitz_loss_weight = float(args.toeplitz_loss_weight)
    netD.psd_loss_weight = float(args.psd_loss_weight)
    netD.subspace_loss_weight = float(args.subspace_loss_weight)
    netD.music_loss_weight = float(args.music_loss_weight)
    netD.music_margin_loss_weight = float(args.music_margin_loss_weight)
    netD.subspace_k = int(args.subspace_k)
    netD.music_grid_size = int(args.music_grid_size)
    netD.music_d_over_lambda = float(args.music_d_over_lambda)
    netD.music_doa_min = float(args.music_doa_min)
    netD.music_doa_max = float(args.music_doa_max)
    netD.music_margin = float(args.music_margin)
    netD.music_target_window_deg = float(args.music_target_window_deg)
    netD.music_peak_min_dist_deg = float(args.music_peak_min_dist_deg)

    if args.init_ckpt:
        ckpt = torch.load(args.init_ckpt, map_location=device, weights_only=False)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        missing, unexpected = netD.load_state_dict(state, strict=False)
        print(f"[Init] 从 checkpoint 加载权重: {args.init_ckpt}")
        if missing:
            print(f"[Init] 缺失键: {missing}")
        if unexpected:
            print(f"[Init] 多余键: {unexpected}")

    # （可选）解码器近零/零初始化 & 冻结骨干
    if args.decoder_zero_init:
        with torch.no_grad():
            for name, m in netD.model.named_modules():
                if isinstance(m, nn.Module) and ("dec_res" in name or "dec_noise" in name):
                    for p_name, p in m.named_parameters(recurse=False):
                        # 兼容 Wr/Wi 或 Ur/Ui
                        if p is not None and p.data is not None:
                            p.zero_()
        print("[Init] 解码器参数已置零（decoder-zero-init）")

    if args.freeze_backbone:
        for n, p in netD.model.named_parameters():
            # 只保留解码器训练
            if (".dec_res." not in n) and (".dec_noise." not in n):
                p.requires_grad_(False)
        print("[Freeze] 骨干参数已冻结，仅训练解码器（freeze-backbone）")

    # 优化器 & 调度 & AMP & EMA
    opt = AdamW(filter(lambda p: p.requires_grad, netD.parameters()),
                lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=args.amp)

    total_updates = (len(train_loader) * args.epochs) // max(1, args.accum_steps)
    scheduler = build_warmup_cosine(opt, warmup_steps=max(0, args.warmup_steps),
                                    total_steps=max(1, total_updates), min_lr_ratio=0.1)
    ema = ModelEMA(netD, decay=args.ema_decay) if args.ema else None

    # 输出目录与配置保存
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.outdir, tag)
    os.makedirs(out_dir, exist_ok=True)
    save_args_json(args, out_dir)
    
    # 初始化日志记录器
    log_path = os.path.join(out_dir, "train_log.txt")
    log = Logger(log_path)
    
    log("=" * 80)
    log("RDDM + RGAT 训练开始")
    log("=" * 80)
    log(f"训练集: {args.train} ({len(train_set)} 样本)")
    log(f"验证集: {args.val} ({len(val_set)} 样本)")
    log(f"输出目录: {out_dir}")
    log(f"模型参数: backbone={args.backbone}, objective={args.objective}, M={train_set.M}, "
        f"hidden={args.gat_hidden}, heads={args.gat_heads}, layers={args.gat_layers}, "
        f"kmax={args.gat_kmax}, unet_base={args.unet_base}, steps={args.steps}, "
        f"unet_attention={args.unet_attention}, unet_skip_gate={args.unet_skip_gate}, "
        f"edge_sign={args.use_edge_sign}, time_film={not args.no_time_film}, "
        f"hermitian_decoder={not args.no_hermitian_decoder}")
    log(f"训练参数: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}, "
        f"amp={args.amp}, grad_clip={args.grad_clip}, accum={args.accum_steps}, warmup={args.warmup_steps}, ema={args.ema}")
    log(f"损失参数: res_diag_w={args.res_diag_weight}, res_off_penalty={args.res_off_penalty}")
    log(f"结构/DOA损失: toeplitz={args.toeplitz_loss_weight}, psd={args.psd_loss_weight}, "
        f"subspace={args.subspace_loss_weight}, music={args.music_loss_weight}, "
        f"margin={args.music_margin_loss_weight}, K={args.subspace_k}, grid={args.music_grid_size}, "
        f"margin_gap={args.music_margin}, target_win={args.music_target_window_deg}")
    if music_val_loader is not None:
        log(f"MUSIC选模: every={args.music_val_every}, samples={len(music_val_loader.dataset)}, "
            f"batch={args.music_val_batch_size}, steps={args.music_val_steps}, "
            f"SR@{args.music_sr_thr} / SR@{args.music_sr_thr2}")
    log("=" * 80)

    best_val, best_ckpt_path = float("inf"), None
    best_music_sr, best_music_rmse, best_music_ckpt_path = -float("inf"), float("inf"), None
    global_state = {"step": 0, "first_batch_check": True}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # ---- 训练 ----
        tr_loss, tr_res, tr_noi = run_one_epoch(
            train_loader, netD, opt, scaler, device,
            is_train=True, amp_enabled=args.amp, grad_clip=args.grad_clip,
            accum_steps=args.accum_steps, identity_assert=args.identity_assert,
            logger=log, global_state=global_state, ema=ema, scheduler=scheduler  # ★ 传入 ema 和 scheduler
        )

        # ---- 验证（EMA 权重）----
        if args.ema:
            ema.store(netD); ema.copy_to(netD)
        va_loss, va_res, va_noi = run_one_epoch(
            val_loader, netD, opt, scaler, device,
            is_train=False, amp_enabled=args.amp, grad_clip=args.grad_clip,
            accum_steps=1, identity_assert=False, logger=None, global_state={"step": 0}
        )
        if args.ema:
            ema.restore(netD)

        dt = time.time() - t0

        # ---- epoch 日志（保持原样式）----
        log(f"[Epoch {epoch:03d}] 训练: {tr_loss:.4f} (残差损失 {tr_res:.4f}, 噪声损失 {tr_noi:.4f}) | "
            f"验证: {va_loss:.4f} (残差损失 {va_res:.4f}, 噪声损失 {va_noi:.4f}) | {dt:.1f}s")

        # ---- film-probe：观察 t→FiLM 是否学出差异 ----
        if args.film_probing if hasattr(args, "film_probing") else args.film_probe:
            with torch.no_grad():
                layer0 = netD.model.gnn[0] if hasattr(netD.model, "gnn") and len(netD.model.gnn) else None
                if (
                    layer0 is not None
                    and getattr(layer0, "time_to_affine_out", None) is not None
                    and hasattr(netD.model, "time_mlp")
                    and hasattr(netD.model, "time_pos")
                ):
                    B = 2
                    t0_ = torch.zeros(B, dtype=torch.long, device=device)
                    t1_ = torch.full((B,), netD.T - 1, dtype=torch.long, device=device)
                    emb0 = netD.model.time_mlp(netD.model.time_pos(t0_))
                    emb1 = netD.model.time_mlp(netD.model.time_pos(t1_))
                    g0, b0 = layer0.time_to_affine_out(emb0).chunk(2, -1)
                    g1, b1 = layer0.time_to_affine_out(emb1).chunk(2, -1)
                    log(f"[FiLM] Δ‖γ‖={(g1-g0).abs().mean().item():.3e}  Δ‖β‖={(b1-b0).abs().mean().item():.3e}")
                else:
                    log("[FiLM] 当前模型没有可探测的 RGAT FiLM 层")

        # ---- MUSIC 验证选模（不反传，只决定额外 best checkpoint）----
        if music_val_loader is not None and (epoch % args.music_val_every == 0 or epoch == args.epochs):
            if args.ema:
                ema.store(netD); ema.copy_to(netD)
            music_metrics = evaluate_music_validation(
                netD, music_val_loader, device, music_d_over_lambda, music_theta_grid,
                sr_thr=args.music_sr_thr, sr_thr2=args.music_sr_thr2,
                sample_steps=args.music_val_steps, amp_enabled=args.amp
            )
            log(f"[MUSIC-Val] SR@{args.music_sr_thr:.2f}: no={music_metrics['SR_noisy']:.3f}, "
                f"de={music_metrics['SR_deno']:.3f}, cl={music_metrics['SR_clean']:.3f} | "
                f"SR@{args.music_sr_thr2:.2f}: de={music_metrics['SR_deno_thr2']:.3f} | "
                f"RMSE: no={music_metrics['RMSE_noisy']:.3f}, de={music_metrics['RMSE_deno']:.3f}, "
                f"cl={music_metrics['RMSE_clean']:.3f}")

            sr = music_metrics["SR_deno"]
            rmse = music_metrics["RMSE_deno"]
            if args.music_val_select == "rmse":
                better_music = (rmse < best_music_rmse - 1e-12) or (
                    abs(rmse - best_music_rmse) <= 1e-12 and sr > best_music_sr
                )
            else:
                better_music = (sr > best_music_sr + 1e-12) or (
                    abs(sr - best_music_sr) <= 1e-12 and rmse < best_music_rmse
                )
            if better_music:
                if best_music_ckpt_path is not None and os.path.exists(best_music_ckpt_path):
                    os.remove(best_music_ckpt_path)
                    log(f"[Delete-MUSIC] 已删除旧模型: {os.path.basename(best_music_ckpt_path)}")
                best_music_sr, best_music_rmse = sr, rmse
                best_music_ckpt_path = os.path.join(
                    out_dir,
                    f"best_music_{args.music_val_select}_sr={sr:.4f}_rmse={rmse:.4f}.pth"
                )
                save_ckpt(
                    best_music_ckpt_path, netD, opt, epoch, best_loss=best_val,
                    extra={"best_music": music_metrics}
                )
                log(f"[Save-MUSIC] {os.path.basename(best_music_ckpt_path)} 已保存 (epoch={epoch})")
            if args.ema:
                ema.restore(netD)

        # ---- 保存 best ----
        if va_loss < best_val:
            # 删除旧的 best 模型
            if best_ckpt_path is not None and os.path.exists(best_ckpt_path):
                os.remove(best_ckpt_path)
                log(f"[Delete] 已删除旧模型: {os.path.basename(best_ckpt_path)}")
            best_val = va_loss
            best_ckpt_path = os.path.join(out_dir, f"best_model_loss={best_val:.6f}.pth")
            # 保存 EMA 权重版本（更稳）
            if args.ema:
                ema.store(netD); ema.copy_to(netD)
                save_ckpt(best_ckpt_path, netD, opt, epoch, best_loss=best_val)
                ema.restore(netD)
            else:
                save_ckpt(best_ckpt_path, netD, opt, epoch, best_loss=best_val)
            log(f"[Save] {os.path.basename(best_ckpt_path)} 已保存 (epoch={epoch})")

        # ---- 保存 last ----
        last_path = os.path.join(out_dir, "last.pth")
        # 同样用 EMA 权重保存 last，便于稳定复现
        if args.ema:
            ema.store(netD); ema.copy_to(netD)
            save_ckpt(last_path, netD, opt, epoch, best_loss=best_val)
            ema.restore(netD)
        else:
            save_ckpt(last_path, netD, opt, epoch, best_loss=best_val)
    
    log("=" * 80)
    log(f"训练完成！最佳验证损失: {best_val:.6f}")
    log(f"最佳模型: {best_ckpt_path}")
    if best_music_ckpt_path is not None:
        log(f"最佳MUSIC模型: {best_music_ckpt_path} (SR={best_music_sr:.4f}, RMSE={best_music_rmse:.4f})")
    log("=" * 80)
    log.close()


if __name__ == "__main__":
    main()
