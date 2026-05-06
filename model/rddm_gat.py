# -*- coding: utf-8 -*-
"""
rddm_gat.py
-----------
RGAT 双头骨干 + 残差去噪扩散 (RDDM) 包装（可被训练脚本 import）
- 不依赖 DGL；行=节点；关系 r=|i-j|；边方向特征 sign(i-j)
- 时间步 t：正弦位置编码 + MLP，经 FiLM 调制通道（gamma/beta 仿射变换）
- 双头输出：预测 (x_in - x0) 与 eps；内部强制 Hermitian 结构
- 采样：DDIM 风格（确定性），可选 Hermitian/Toeplitz/PSD 投影
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------- 基础工具 -----------------------------

def to_complex(x2: torch.Tensor) -> torch.Tensor:
    """(B,2,M,M) → complex (B,M,M)"""
    re = x2[:, 0]
    im = x2[:, 1]
    return torch.complex(re, im)


def hermitian_project(x2: torch.Tensor) -> torch.Tensor:
    """(B,2,M,M) → Hermitian：Re 对称，Im 反对称"""
    re = x2[:, 0]
    im = x2[:, 1]
    re = 0.5 * (re + re.transpose(-1, -2))
    im = 0.5 * (im - im.transpose(-1, -2))
    return torch.stack([re, im], dim=1)


def toeplitz_project(x2: torch.Tensor) -> torch.Tensor:
    """近似 Toeplitz 投影：每条对角线取均值回填"""
    B, _, M, _ = x2.shape
    re = x2[:, 0].clone()
    im = x2[:, 1].clone()
    device = x2.device
    for b in range(B):
        for k in range(-(M - 1), M):
            diag_re = torch.diagonal(re[b], offset=k)
            diag_im = torch.diagonal(im[b], offset=k)
            mean_re = diag_re.mean()
            mean_im = diag_im.mean()
            idx = torch.arange(max(0, k), min(M, M + k), device=device)
            i = idx
            j = idx - k
            re[b, i, j] = mean_re
            im[b, i, j] = mean_im
    return torch.stack([re, im], dim=1)


def psd_project(x2: torch.Tensor) -> torch.Tensor:
    """Z = Re + j Im → 投影到 PSD（裁剪负特征值）"""
    z = to_complex(x2)  # (B,M,M) complex
    recons = []
    for b in range(z.size(0)):
        Z = z[b]
        Z = 0.5 * (Z + Z.conj().T)  # Hermitian
        evals, evecs = torch.linalg.eigh(Z)
        evals = torch.clamp(evals, min=0.0)
        Zp = (evecs * evals) @ evecs.conj().T
        recons.append(Zp)
    Zp = torch.stack(recons, dim=0)
    return torch.stack([Zp.real, Zp.imag], dim=1)


def toeplitz_consistency_loss(x2: torch.Tensor) -> torch.Tensor:
    """Penalize variation along each matrix diagonal."""
    B, _, M, _ = x2.shape
    loss = x2.new_tensor(0.0)
    count = 0
    for c in range(2):
        X = x2[:, c]
        for k in range(-(M - 1), M):
            d = torch.diagonal(X, offset=k, dim1=-2, dim2=-1)
            loss = loss + (d - d.mean(dim=-1, keepdim=True)).abs().mean()
            count += 1
    return loss / max(1, count)


def psd_negative_eig_loss(x2: torch.Tensor) -> torch.Tensor:
    """Penalize negative eigenvalues of the Hermitian matrix."""
    z = to_complex(hermitian_project(x2))
    evals = torch.linalg.eigvalsh(z)
    return F.relu(-evals).mean()


def noise_subspace_projector(x2: torch.Tensor, K: int) -> torch.Tensor:
    """Return the MUSIC noise-subspace projector En En^H."""
    z = to_complex(hermitian_project(x2))
    M = z.size(-1)
    k = max(1, min(int(K), M - 1))
    _, evecs = torch.linalg.eigh(z)
    En = evecs[:, :, : M - k]
    return En @ En.conj().transpose(-1, -2)


def subspace_projector_loss(x_hat: torch.Tensor, x_true: torch.Tensor, K: int) -> torch.Tensor:
    """Match predicted and target MUSIC noise-subspace projectors."""
    P_hat = noise_subspace_projector(x_hat, K)
    P_true = noise_subspace_projector(x_true, K)
    return (P_hat - P_true).abs().pow(2).mean()


def ula_steering_matrix(M: int, grid_size: int, d_over_lambda: float,
                        doa_min: float, doa_max: float, device, dtype) -> torch.Tensor:
    """ULA steering matrix with shape (M,G), complex dtype."""
    theta = torch.linspace(doa_min, doa_max, grid_size, device=device, dtype=dtype)
    m = torch.arange(M, device=device, dtype=dtype).unsqueeze(1)
    phase = 2 * math.pi * d_over_lambda * torch.sin(theta * math.pi / 180.0).unsqueeze(0) * m
    return torch.exp(1j * phase)


def music_spectrum_from_matrix(x2: torch.Tensor, K: int, grid_size: int = 121,
                               d_over_lambda: float = 0.5, doa_min: float = -60.0,
                               doa_max: float = 60.0) -> torch.Tensor:
    """Normalized MUSIC spectrum on an angular grid, shape (B,G)."""
    z = to_complex(hermitian_project(x2))
    B, M, _ = z.shape
    k = max(1, min(int(K), M - 1))
    A = ula_steering_matrix(M, grid_size, d_over_lambda, doa_min, doa_max,
                            z.device, z.real.dtype)
    _, evecs = torch.linalg.eigh(z)
    En = evecs[:, :, : M - k]
    EnHa = En.conj().transpose(-1, -2) @ A.unsqueeze(0)
    denom = EnHa.abs().pow(2).sum(dim=1)
    P = 1.0 / (denom + 1e-8)
    return P / (P.amax(dim=-1, keepdim=True) + 1e-8)


def music_log_spectrum_loss(x_hat: torch.Tensor, x_true: torch.Tensor, K: int,
                            grid_size: int = 121, d_over_lambda: float = 0.5,
                            doa_min: float = -60.0, doa_max: float = 60.0) -> torch.Tensor:
    """Match normalized log-MUSIC spectra on a coarse angular grid."""
    P_hat = music_spectrum_from_matrix(x_hat, K, grid_size, d_over_lambda, doa_min, doa_max)
    with torch.no_grad():
        P_true = music_spectrum_from_matrix(x_true, K, grid_size, d_over_lambda, doa_min, doa_max)
    return F.mse_loss(torch.log(P_hat + 1e-8), torch.log(P_true + 1e-8))


def _separated_peak_target_mask(P_true: torch.Tensor, K: int, doa_min: float, doa_max: float,
                                target_window_deg: float, min_peak_dist_deg: float) -> torch.Tensor:
    """Build a non-differentiable target mask around separated clean MUSIC peaks."""
    B, G = P_true.shape
    if G <= 1:
        return torch.ones_like(P_true, dtype=torch.bool)
    step_deg = abs(float(doa_max) - float(doa_min)) / float(G - 1)
    win_pts = max(0, int(round(float(target_window_deg) / max(step_deg, 1e-12))))
    min_dist_pts = max(1, int(round(float(min_peak_dist_deg) / max(step_deg, 1e-12))))
    mask = torch.zeros_like(P_true, dtype=torch.bool)

    local = torch.zeros_like(P_true, dtype=torch.bool)
    if G >= 3:
        local[:, 1:-1] = (P_true[:, 1:-1] > P_true[:, :-2]) & (P_true[:, 1:-1] >= P_true[:, 2:])

    for b in range(B):
        candidates = torch.where(local[b])[0]
        if candidates.numel() > 0:
            order = torch.argsort(P_true[b, candidates], descending=True)
            candidates = candidates[order]
        else:
            candidates = torch.argsort(P_true[b], descending=True)

        picked = []
        for idx in candidates.detach().cpu().tolist():
            if all(abs(idx - prev) >= min_dist_pts for prev in picked):
                picked.append(idx)
                if len(picked) >= K:
                    break
        if len(picked) < K:
            for idx in torch.argsort(P_true[b], descending=True).detach().cpu().tolist():
                if all(abs(idx - prev) >= min_dist_pts for prev in picked):
                    picked.append(idx)
                    if len(picked) >= K:
                        break

        for idx in picked:
            lo = max(0, idx - win_pts)
            hi = min(G, idx + win_pts + 1)
            mask[b, lo:hi] = True
    return mask


def music_peak_margin_loss(x_hat: torch.Tensor, x_true: torch.Tensor, K: int,
                           grid_size: int = 241, d_over_lambda: float = 0.5,
                           doa_min: float = -60.0, doa_max: float = 60.0,
                           margin: float = 0.5, target_window_deg: float = 0.5,
                           min_peak_dist_deg: float = 1.0) -> torch.Tensor:
    """
    Encourage predicted MUSIC peaks near clean-target peaks to outrank false peaks.
    The target mask is derived from the clean SCM spectrum and is not backpropagated.
    """
    P_hat = music_spectrum_from_matrix(x_hat, K, grid_size, d_over_lambda, doa_min, doa_max)
    with torch.no_grad():
        P_true = music_spectrum_from_matrix(x_true, K, grid_size, d_over_lambda, doa_min, doa_max)
        target_mask = _separated_peak_target_mask(
            P_true, K, doa_min, doa_max, target_window_deg, min_peak_dist_deg
        )
        if not target_mask.any(dim=-1).all():
            target_mask = target_mask | (P_true == P_true.amax(dim=-1, keepdim=True))

    logP = torch.log(P_hat + 1e-8)
    neg_inf = torch.finfo(logP.dtype).min
    target_score = logP.masked_fill(~target_mask, neg_inf).amax(dim=-1)
    false_score = logP.masked_fill(target_mask, neg_inf).amax(dim=-1)
    valid_false = torch.isfinite(false_score)
    if not valid_false.any():
        return logP.new_tensor(0.0)
    return F.softplus(float(margin) + false_score[valid_false] - target_score[valid_false]).mean()

    return F.mse_loss(spectrum(z_hat), spectrum(z_true))


# ----------------------------- 时间编码 -----------------------------

class SinusoidalPosEmb(nn.Module):
    """标准 diffusion 正弦时间步编码"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: (B,) int/float
        return: (B, dim)
        """
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            torch.linspace(math.log(1.0), math.log(10000.0), steps=half, device=device)
        )
        t = t.float().unsqueeze(-1)  # (B,1)
        args = t * freqs.unsqueeze(0)  # (B,half)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1), mode="constant")
        return emb


# ----------------------------- RGAT 核心 -----------------------------

def build_edges(M: int, k_max: int, self_loop: bool = True, device=None):
    """
    构建边表：src, dst, rel=|i-j|, esgn=sign(i-j)
    - src: 出发节点（被 softmax 归一化）
    - 消息从 dst 聚合到 src（与 GAT 约定一致）
    """
    src = []
    dst = []
    rel = []
    esg = []
    for i in range(M):
        if self_loop:
            src.append(i); dst.append(i); rel.append(0); esg.append(0)
        for k in range(1, k_max + 1):
            j1 = i - k
            j2 = i + k
            if j1 >= 0:
                src.append(i); dst.append(j1); rel.append(k); esg.append(+1)
            if j2 < M:
                src.append(i); dst.append(j2); rel.append(k); esg.append(-1)
    src = torch.tensor(src, dtype=torch.long, device=device)
    dst = torch.tensor(dst, dtype=torch.long, device=device)
    rel = torch.tensor(rel, dtype=torch.long, device=device)             # 0..k_max
    esg = torch.tensor(esg, dtype=torch.float32, device=device).unsqueeze(-1)  # (-1,0,+1), shape (E,1)
    return src, dst, rel, esg


class RelEdgeGATLayer(nn.Module):
    """
    关系感知 GAT 层（后置 FiLM 版）
    - 输入:  H (B,N,Cin), t_emb (B,Ct)；Ct 通常等于 hidden
    - 输出:  H' (B,N,Cout)，其中 Cout = heads * head_dim
    - 注意力 logits（已去掉时间项，只保留结构/方向）:
        <W h_i, a_src[r]> + <W h_j, a_dst[r]> + <We e_ij, a_e>
    - FiLM 注入（本改动）:
        Z = out_proj( Σ_j α_ij * V_j )  →  pre_norm →  FiLM(t_emb) → GELU → 残差+LN
    """
    def __init__(self, in_dim: int, out_heads: int, head_dim: int, n_rel: int,
                 use_edge_sign: bool = True, dropout: float = 0.1,
                 use_time_film: bool = True):
        super().__init__()
        self.H = out_heads
        self.d = head_dim
        self.Cout = out_heads * head_dim
        self.n_rel = n_rel
        self.use_edge_sign = use_edge_sign
        self.use_time_film = use_time_film
        self.dropout = nn.Dropout(dropout)

        # 节点投影
        self.W   = nn.Linear(in_dim, self.Cout, bias=False)  # 给 Q/K 共用
        self.W_v = nn.Linear(in_dim, self.Cout, bias=False)  # 给 V

        # 关系模板（按对角距 r 分桶）
        self.a_src = nn.Parameter(torch.randn(n_rel, self.H, self.d) * 0.02)
        self.a_dst = nn.Parameter(torch.randn(n_rel, self.H, self.d) * 0.02)

        # 边方向项
        if use_edge_sign:
            self.W_e = nn.Linear(1, self.Cout, bias=False)
            self.a_e = nn.Parameter(torch.randn(self.H, self.d) * 0.02)

        # -------- 后置 FiLM（新增）--------
        # 用 t_emb 生成 (gamma, beta) ∈ R^{Cout}，对通道做 scale/shift
        if use_time_film:
            self.time_to_affine_out = nn.Sequential(
                nn.SiLU(),
                nn.Linear(in_dim, 2 * self.Cout)  # → [gamma | beta]
            )
            # 置零初始化：初始等价于不调制，训练再慢慢学
            nn.init.zeros_(self.time_to_affine_out[-1].weight)
            nn.init.zeros_(self.time_to_affine_out[-1].bias)
        else:
            self.time_to_affine_out = None

        self.pre_norm = nn.LayerNorm(self.Cout)   # FiLM 前做一次 LN 稳定分布
        self.out_proj = nn.Linear(self.Cout, self.Cout, bias=True)
        self.norm     = nn.LayerNorm(self.Cout)
        self.act      = nn.GELU()

        self._res_proj = None  # lazy residual

    def forward(self, H: torch.Tensor, t_emb: torch.Tensor,
                src: torch.Tensor, dst: torch.Tensor,
                rel: torch.Tensor, esg: torch.Tensor,
                num_nodes: int) -> torch.Tensor:
        """
        H: (B,N,Cin), t_emb: (B,Ct)，Ct 通常等于 hidden（与 GATTwoHead.time_mlp 输出一致）
        src/dst/rel/esg: (E,)
        """
        B, N, Cin = H.shape
        E = src.numel()

        # Q/K/V
        Wh = self.W(H).view(B, N, self.H, self.d)     # (B,N,H,d)
        Wv = self.W_v(H).view(B, N, self.H, self.d)   # (B,N,H,d)

        Wh_src = Wh[:, src, :, :]                     # (B,E,H,d)
        Wh_dst = Wh[:, dst, :, :]                     # (B,E,H,d)

        a_src = self.a_src[rel]                       # (E,H,d)
        a_dst = self.a_dst[rel]                       # (E,H,d)

        # 注意力打分（无时间项）
        score = (Wh_src * a_src).sum(-1) + (Wh_dst * a_dst).sum(-1)  # (B,E,H)
        if self.use_edge_sign:
            We = self.W_e(esg).view(E, self.H, self.d)
            score = score + (We * self.a_e).sum(-1).unsqueeze(0)     # (B,E,H)

        score = F.leaky_relu(score, 0.2)

        # 按目标节点 i 的入边 softmax
        alpha = torch.zeros_like(score)                               # (B,E,H)
        for i in range(num_nodes):
            mask = (src == i)
            if mask.any():
                s = score[:, mask, :]                                 # (B,E_i,H)
                a = torch.softmax(s, dim=1)
                alpha[:, mask, :] = a
        alpha = self.dropout(alpha)

        # 聚合：Σ_j α_ij * V_j
        out = torch.zeros(B, N, self.H, self.d, device=H.device, dtype=H.dtype)
        Wv_dst = Wv[:, dst, :, :]                                     # (B,E,H,d)
        m = alpha.unsqueeze(-1) * Wv_dst                              # (B,E,H,d)
        for i in range(num_nodes):
            mask = (src == i)
            if mask.any():
                out[:, i, :, :] += m[:, mask, :, :].sum(dim=1)

        # 线性 -> FiLM(t_emb) -> 激活 -> 残差 + LN
        out = out.view(B, N, self.Cout)           # (B,N,Cout)
        out = self.out_proj(out)                  # 先通道混合
        out = self.dropout(out)
        out = self.pre_norm(out)                  # FiLM 前归一化（更稳）

        if self.use_time_film and self.time_to_affine_out is not None:
            gamma, beta = self.time_to_affine_out(t_emb).chunk(2, dim=-1)  # (B,Cout),(B,Cout)
            gamma = gamma.unsqueeze(1)               # (B,1,Cout) 广播到节点维
            beta  = beta .unsqueeze(1)
            out = out * (1.0 + gamma) + beta
        out = self.act(out)

        out = self.norm(out + self._residual(H))
        return out

    def _residual(self, H: torch.Tensor) -> torch.Tensor:
        if H.size(-1) != self.Cout:
            if self._res_proj is None:
                self._res_proj = nn.Linear(H.size(-1), self.Cout, bias=False)
            return self._res_proj(H)
        return H


class BilinearMatrixDecoder(nn.Module):
    """节点特征 H:(B,M,C) → (B,2,M,M)，并强制 Hermitian"""
    def __init__(self, in_dim: int, rank_shrink: int = 0, force_hermitian: bool = True):
        super().__init__()
        self.force_hermitian = force_hermitian
        C = in_dim
        scale = 1e-3 / math.sqrt(C)
        if rank_shrink > 0 and rank_shrink < C:
            self.Ur = nn.Parameter(torch.randn(C, rank_shrink) * scale)
            self.Ui = nn.Parameter(torch.randn(C, rank_shrink) * scale)
            self.low_rank = True
        else:
            self.Wr = nn.Parameter(torch.randn(C, C) * scale)
            self.Wi = nn.Parameter(torch.randn(C, C) * scale)
            self.low_rank = False

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        B, M, C = H.shape
        if self.low_rank:
            Wr = self.Ur @ self.Ur.t()
            Wi = self.Ui @ self.Ui.t()
        else:
            Wr = self.Wr
            Wi = self.Wi
        Hr = H @ Wr
        Hi = H @ Wi
        re = Hr @ H.transpose(1, 2)
        im = Hi @ H.transpose(1, 2)
        if self.force_hermitian:
            re = 0.5 * (re + re.transpose(1, 2))
            im = 0.5 * (im - im.transpose(1, 2))
        return torch.stack([re, im], dim=1)


class GATTwoHead(nn.Module):
    """
    RGAT 双头骨干：
    - 节点 = 行；关系 r=|i-j|；方向 sign(i-j)
    - 节点特征：concat [x_t.Re_row, x_t.Im_row, x_in.Re_row, x_in.Im_row] → 4M
    - 时间步：正弦位置编码 + MLP → (B, hidden)
    """
    def __init__(self, M: int, hidden: int = 64, heads: int = 4, layers: int = 4,
                 k_max: int = 2, t_dim: int = 64, use_edge_sign: bool = True,
                 dropout: float = 0.1, lowrank_decoder: int = 0,
                 use_time_film: bool = True, force_hermitian_decoder: bool = True):
        super().__init__()
        assert hidden % heads == 0, f"hidden({hidden}) must be divisible by heads({heads})"
        self.M = M
        self.hidden = hidden
        self.heads = heads
        self.layers = layers
        self.k_max = k_max
        self.n_rel = k_max + 1
        self.use_time_film = use_time_film
        self.force_hermitian_decoder = force_hermitian_decoder

        # 时间编码
        self.time_pos = SinusoidalPosEmb(t_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(t_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

        # 节点编码：4M → hidden
        self.node_enc = nn.Sequential(
            nn.Linear(4 * M, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )

        # 边表缓存（模块 buffer，随 .to(device) 搬运）
        device = torch.device("cpu")
        s, d, r, e = build_edges(M, k_max, True, device=device)
        self.register_buffer("edge_src", s)
        self.register_buffer("edge_dst", d)
        self.register_buffer("edge_rel", r)
        self.register_buffer("edge_esg", e)

        # GAT 层堆叠
        self.gnn = nn.ModuleList()
        in_dim = hidden
        for _ in range(layers):
            self.gnn.append(
                RelEdgeGATLayer(in_dim=in_dim,
                                out_heads=heads,
                                head_dim=hidden // heads,
                                n_rel=self.n_rel,
                                use_edge_sign=use_edge_sign,
                                dropout=dropout,
                                use_time_film=use_time_film)
            )
            in_dim = hidden

        # 双头解码
        self.dec_res = BilinearMatrixDecoder(
            in_dim, rank_shrink=lowrank_decoder, force_hermitian=force_hermitian_decoder
        )
        self.dec_noise = BilinearMatrixDecoder(
            in_dim, rank_shrink=lowrank_decoder, force_hermitian=force_hermitian_decoder
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, x_in: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x_t, x_in: (B,2,M,M); t: (B,)
        return: pred_res, pred_noise: (B,2,M,M)
        """
        B, _, M, _ = x_t.shape
        assert M == self.M, f"M mismatch: got {M}, expected {self.M}"

        xt_re, xt_im = x_t[:, 0], x_t[:, 1]
        xi_re, xi_im = x_in[:, 0], x_in[:, 1]
        node = torch.cat([xt_re, xt_im, xi_re, xi_im], dim=-1)  # (B,M,4M)

        H = self.node_enc(node)                                 # (B,M,hidden)
        t_emb = self.time_mlp(self.time_pos(t))                 # (B,hidden)

        s, d, r, e = self.edge_src, self.edge_dst, self.edge_rel, self.edge_esg
        for layer in self.gnn:
            H = layer(H, t_emb, s, d, r, e, num_nodes=self.M)   # (B,M,hidden)

        pred_res = self.dec_res(H)
        pred_noise = self.dec_noise(H)
        return pred_res, pred_noise


def _valid_group_count(channels: int) -> int:
    for g in (8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


class ConvFiLMBlock2d(nn.Module):
    """Small convolutional block used by the UNet backbone ablation."""
    def __init__(self, in_ch: int, out_ch: int, t_dim: int, dropout: float = 0.0,
                 use_time_film: bool = True, attention: str = "none"):
        super().__init__()
        self.use_time_film = use_time_film
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(_valid_group_count(out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(_valid_group_count(out_ch), out_ch)
        self.dropout = nn.Dropout2d(dropout)
        self.res = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)
        if use_time_film:
            self.time_to_affine = nn.Sequential(nn.SiLU(), nn.Linear(t_dim, 2 * out_ch))
            nn.init.zeros_(self.time_to_affine[-1].weight)
            nn.init.zeros_(self.time_to_affine[-1].bias)
        else:
            self.time_to_affine = None
        self.attn = make_unet_attention(attention, out_ch)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(self.conv1(x))
        if self.use_time_film and self.time_to_affine is not None:
            gamma, beta = self.time_to_affine(t_emb).chunk(2, dim=-1)
            h = h * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        h = F.gelu(h)
        h = self.dropout(h)
        h = self.norm2(self.conv2(h))
        h = F.gelu(h + self.res(x))
        return self.attn(h)


class SEAttention2d(nn.Module):
    """Residual SE channel attention, initialized as identity."""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, channels, 1)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.fc2(F.silu(self.fc1(self.pool(x))))
        return x * (2.0 * torch.sigmoid(scale))


class SpatialAttention2d(nn.Module):
    """CBAM-style spatial attention, initialized as identity."""
    def __init__(self, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx = x.max(dim=1, keepdim=True).values
        logits = self.conv(torch.cat([avg, mx], dim=1))
        return x * (2.0 * torch.sigmoid(logits))


class CBAMAttention2d(nn.Module):
    """Lightweight channel + spatial attention for matrix denoising UNet blocks."""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.channel = SEAttention2d(channels, reduction=reduction)
        self.spatial = SpatialAttention2d(kernel_size=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial(self.channel(x))


def make_unet_attention(attention: str, channels: int) -> nn.Module:
    attention = (attention or "none").lower()
    if attention == "none":
        return nn.Identity()
    if attention == "se":
        return SEAttention2d(channels)
    if attention == "cbam":
        return CBAMAttention2d(channels)
    raise ValueError(f"Unsupported UNet attention: {attention}")


class SkipGate2d(nn.Module):
    """Decoder-conditioned skip gate, initialized as an identity skip path."""
    def __init__(self, channels: int):
        super().__init__()
        self.proj = nn.Conv2d(channels * 2, channels, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, dec: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        gate = torch.tanh(self.proj(torch.cat([dec, skip], dim=1)))
        return skip * (1.0 + gate)


class UNetTwoHead(nn.Module):
    """
    Compact conditional UNet ablation for replacing the RGAT backbone.

    Input channels are concat[x_t, x_in] = 4; output channels are split into
    residual/x0 head and noise head, each shaped (B,2,M,M).
    """
    def __init__(self, M: int, base: int = 64, t_dim: int = 128, dropout: float = 0.05,
                 use_time_film: bool = True, force_hermitian_decoder: bool = True,
                 attention: str = "none", use_skip_gate: bool = False):
        super().__init__()
        self.M = M
        self.base = base
        self.use_time_film = use_time_film
        self.force_hermitian_decoder = force_hermitian_decoder
        self.attention = (attention or "none").lower()
        self.use_skip_gate = use_skip_gate
        self.time_pos = SinusoidalPosEmb(t_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(t_dim, t_dim),
            nn.SiLU(),
            nn.Linear(t_dim, t_dim),
        )

        self.enc1 = ConvFiLMBlock2d(4, base, t_dim, dropout, use_time_film, self.attention)
        self.down = nn.Conv2d(base, base * 2, 3, stride=2, padding=1)
        self.enc2 = ConvFiLMBlock2d(base * 2, base * 2, t_dim, dropout, use_time_film, self.attention)
        self.mid = ConvFiLMBlock2d(base * 2, base * 2, t_dim, dropout, use_time_film, self.attention)
        self.up = nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1)
        self.skip_gate = SkipGate2d(base) if use_skip_gate else nn.Identity()
        self.dec = ConvFiLMBlock2d(base * 2, base, t_dim, dropout, use_time_film, self.attention)
        self.out = nn.Conv2d(base, 4, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def _maybe_project_head(self, x: torch.Tensor) -> torch.Tensor:
        if self.force_hermitian_decoder:
            return hermitian_project(x)
        return x

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, x_in: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        t_emb = self.time_mlp(self.time_pos(t))
        x = torch.cat([x_t, x_in], dim=1)
        skip = self.enc1(x, t_emb)
        h = self.down(skip)
        h = self.enc2(h, t_emb)
        h = self.mid(h, t_emb)
        h = self.up(h)
        if h.shape[-2:] != skip.shape[-2:]:
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        if self.use_skip_gate:
            skip = self.skip_gate(h, skip)
        h = self.dec(torch.cat([h, skip], dim=1), t_emb)
        out = self.out(h)
        pred_res, pred_noise = out[:, :2], out[:, 2:]
        return self._maybe_project_head(pred_res), self._maybe_project_head(pred_noise)


# ----------------------------- 扩散包装 -----------------------------

def cosine_cumprod(T: int, s: float = 0.008, device=None) -> torch.Tensor:
    """
    余弦累计调度（固定版）：
    abar_t = f(t)/f(0), f(t) = cos((t/T + s)/(1+s) * pi / 2)^2
    返回 (T+1,) 张量
    """
    steps = torch.arange(T + 1, device=device, dtype=torch.float32)
    tt = steps / T
    f = torch.cos((tt + s) / (1.0 + s) * math.pi / 2) ** 2
    f = f / f[0]
    return f


class ResidualDiffusion(nn.Module):
    """
    残差去噪扩散：
    x_t = x0 + a_t (x_in - x0) + sqrt(b_t) eps
    训练预测：(x_in - x0) 与 eps；推理 x ← x0_hat
    """
    def __init__(self, model: nn.Module, image_size: int, timesteps: int = 400,
                 loss_type: str = "l1", w_res: float = 1.0, w_noise: float = 1.0,
                 device=None):
        super().__init__()
        self.model = model
        self.M = image_size
        self.T = timesteps
        self.loss_type = loss_type
        self.w_res = w_res
        self.w_noise = w_noise

        abar = cosine_cumprod(timesteps, device=device)      # (T+1,)
        a = 1.0 - abar[:-1]  # 0..T-1
        b = 1.0 - abar[:-1]  # 绑定（与当前实现一致）
        self.register_buffer("abar", abar)
        self.register_buffer("a", a)
        self.register_buffer("b", b)

        # >>> 新增：残差头损失的两个开关（默认值保持旧行为） <<<
        self.res_diag_weight: float = 1.0   # 仅对“实部对角”加权；例如 16
        self.res_off_penalty: float = 0.0   # 对“实部非对角”的 L1 额外惩罚；例如 0.2~0.5
        self.toeplitz_loss_weight: float = 0.0
        self.psd_loss_weight: float = 0.0
        self.subspace_loss_weight: float = 0.0
        self.music_loss_weight: float = 0.0
        self.music_margin_loss_weight: float = 0.0
        self.subspace_k: int = 2
        self.music_grid_size: int = 121
        self.music_d_over_lambda: float = 0.5
        self.music_doa_min: float = -60.0
        self.music_doa_max: float = 60.0
        self.music_margin: float = 0.5
        self.music_target_window_deg: float = 0.5
        self.music_peak_min_dist_deg: float = 1.0

    def q_sample(self, x0: torch.Tensor, x_in: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        a_t = self.a[t].view(-1, 1, 1, 1)
        b_t = self.b[t].view(-1, 1, 1, 1)
        x_t = x0 + a_t * (x_in - x0) + torch.sqrt(torch.clamp(b_t, min=1e-8)) * eps
        return x_t

    def model_predictions(self, x_t: torch.Tensor, t: torch.Tensor, x_in: torch.Tensor, lambda_res: float = 1.0):
        pred_res, pred_noise = self.model(x_t, t, x_in)
        x0_hat = x_in - float(lambda_res) * pred_res
        return pred_res, pred_noise, x0_hat

    @torch.no_grad()
    def p_sample_loop(self, x_in: torch.Tensor, steps: int = None, sum_scale: float = 0.0,
                      proj_hermitian: bool = True, proj_toeplitz: bool = False, proj_psd: bool = False,
                      lambda_res: float = 1.0, start_t: int = None):
        """
        DDIM 风格确定性采样：逐步从 x_in 去噪到 x0
        - steps=1: 单步直接预测 x0
        - steps>1: 多步 DDIM 逐步去噪
        """
        B = x_in.size(0)
        device = x_in.device
        steps = self.T if (steps is None or steps <= 0 or steps > self.T) else steps
        if start_t is None:
            start_t = self.T - 1
        start_t = max(0, min(int(start_t), self.T - 1))
        times = torch.linspace(start_t, 0, steps=steps, device=device).long()

        # 初始化：从 x_in 开始（可选加小噪声）
        x = x_in.clone()
        if sum_scale > 0.0:
            x = x + torch.randn_like(x) * math.sqrt(sum_scale)

        for i, ti in enumerate(times.tolist()):
            t = torch.full((B,), ti, device=device, dtype=torch.long)
            
            # 预测 x0 和 噪声
            _, pred_noise, x0_hat = self.model_predictions(x, t, x_in, lambda_res=lambda_res)
            
            # 结构投影 x0_hat
            if proj_hermitian:
                x0_hat = hermitian_project(x0_hat)
            if proj_toeplitz:
                x0_hat = toeplitz_project(x0_hat)
            if proj_psd:
                x0_hat = psd_project(x0_hat)
            
            if i < len(times) - 1:
                # 不是最后一步，使用 DDIM 更新到下一时间步
                t_next = times[i + 1].item()
                
                # 当前和下一步的系数
                a_t = self.a[ti]
                b_t = self.b[ti]
                a_next = self.a[t_next]
                b_next = self.b[t_next]
                
                # 预测噪声（从当前状态反推）
                if b_t > 1e-8:
                    # x_t = x0 + a_t * (x_in - x0) + sqrt(b_t) * eps
                    # => eps = (x_t - x0 - a_t*(x_in - x0)) / sqrt(b_t)
                    eps_pred = (x - x0_hat - a_t * (x_in - x0_hat)) / torch.sqrt(torch.clamp(b_t, min=1e-8))
                else:
                    eps_pred = pred_noise  # 使用模型预测的噪声
                
                # DDIM 更新：x_{t-1} = x0 + a_{t-1} * (x_in - x0) + sqrt(b_{t-1}) * eps
                x = x0_hat + a_next * (x_in - x0_hat) + torch.sqrt(torch.clamp(b_next, min=1e-8)) * eps_pred
                
                # 可选：每步投影（有助于保持结构约束）
                if proj_hermitian:
                    x = hermitian_project(x)
            else:
                # 最后一步，直接用 x0_hat
                x = x0_hat
        
        return x

    def loss_fn(self, pred_res, pred_noise, true_res, eps, x0_hat=None, x0=None):
        """
        残差头：
          - 基础 L1/L2 与原逻辑一致（对全元素求平均）；
          - 对"实部对角"的误差乘 res_diag_weight（权重化平均）；
          - 对"实部非对角"的误差加 L1 正则：res_off_penalty * |误差|。
        噪声头：与原逻辑一致（L1/L2）。
        """
        # ---- 基础误差 ----
        diff_res = pred_res - true_res  # (B,2,M,M)
        if self.loss_type == "l2":
            elem_res = diff_res ** 2
            l_noi = F.mse_loss(pred_noise, eps)
        else:
            elem_res = diff_res.abs()
            l_noi = F.l1_loss(pred_noise, eps)

        B, _, M, _ = diff_res.shape
        device = diff_res.device
        dtype = elem_res.dtype

        # ---- 实部对角权重化 ----
        eye = torch.eye(M, device=device, dtype=dtype).view(1, 1, M, M)
        W = torch.ones_like(elem_res)
        # 只给"实部对角"乘权
        if hasattr(self, "res_diag_weight") and float(self.res_diag_weight) != 1.0:
            W[:, 0, :, :] = W[:, 0, :, :] + (float(self.res_diag_weight) - 1.0) * eye

        l_res_base = (elem_res * W).mean()

        # ---- 实部非对角 L1 正则（作用在误差，不是预测值）----
        l_res = l_res_base
        if hasattr(self, "res_off_penalty") and float(self.res_off_penalty) > 0.0:
            off_mask = (1.0 - eye)                      # 非对角=1，对角=0
            err_real = (diff_res[:, 0].abs())           # 只惩罚实部
            l_res_off = (err_real * off_mask).mean()
            l_res = l_res + float(self.res_off_penalty) * l_res_off

        # ---- 结构/子空间附加损失（默认权重为 0，保持旧行为）----
        extra = pred_res.new_tensor(0.0)
        if x0_hat is not None:
            if float(self.toeplitz_loss_weight) > 0.0:
                extra = extra + float(self.toeplitz_loss_weight) * toeplitz_consistency_loss(x0_hat)
            if float(self.psd_loss_weight) > 0.0:
                extra = extra + float(self.psd_loss_weight) * psd_negative_eig_loss(x0_hat)
        if x0_hat is not None and x0 is not None:
            if float(self.subspace_loss_weight) > 0.0:
                extra = extra + float(self.subspace_loss_weight) * subspace_projector_loss(
                    x0_hat, x0, self.subspace_k
                )
            if float(self.music_loss_weight) > 0.0:
                extra = extra + float(self.music_loss_weight) * music_log_spectrum_loss(
                    x0_hat, x0, self.subspace_k, self.music_grid_size,
                    self.music_d_over_lambda, self.music_doa_min, self.music_doa_max
                )
            if float(self.music_margin_loss_weight) > 0.0:
                extra = extra + float(self.music_margin_loss_weight) * music_peak_margin_loss(
                    x0_hat, x0, self.subspace_k, self.music_grid_size,
                    self.music_d_over_lambda, self.music_doa_min, self.music_doa_max,
                    self.music_margin, self.music_target_window_deg, self.music_peak_min_dist_deg
                )

        # ---- 总损失 ----
        loss = self.w_res * l_res + self.w_noise * l_noi + extra
        return loss, l_res.item(), l_noi.item()

    # def loss_fn(self, pred_res, pred_noise, true_res, eps):
    #     """
    #     最小改动：
    #     - 对残差头的损失，在“实部对角”处乘以 res_diag_weight
    #     - 可选对预测残差的“实部非对角”施加 L1 惩罚（鼓励 Δ 接近对角阵）
    #     其它保持不变。
    #     """
    #     # 基础 L1/L2
    #     if self.loss_type == "l2":
    #         base_res = (pred_res - true_res).pow(2)
    #         loss_noi = (pred_noise - eps).pow(2).mean()
    #     else:
    #         base_res = (pred_res - true_res).abs()
    #         loss_noi = (pred_noise - eps).abs().mean()

    #     # 对角加权（仅实部通道）
    #     diag_w = float(self.res_diag_weight)
    #     off_p  = float(self.res_off_penalty)

    #     if diag_w != 1.0 or off_p > 0.0:
    #         M = pred_res.shape[-1]
    #         device = pred_res.device
    #         eye = torch.eye(M, device=device).view(1, 1, M, M)   # [1,1,M,M]

    #         w = torch.ones_like(base_res)
    #         # 仅对实部对角加权；虚部对角理论上应接近 0（SCM）
    #         w[:, 0] = w[:, 0] + (diag_w - 1.0) * eye
    #         loss_res = (w * base_res).mean()

    #         # 可选：实部非对角惩罚（L1）
    #         if off_p > 0.0:
    #             off_mask = 1.0 - eye
    #             err_real = (pred_res[:, 0] - true_res[:, 0]).abs()  # 惩罚“实部非对角”的预测误差
    #             loss_res = loss_res + off_p * (err_real * off_mask).mean()

    #     else:
    #         loss_res = base_res.mean()

    #     loss = self.w_res * loss_res + self.w_noise * loss_noi
    #     return loss, loss_res.item(), loss_noi.item()

    def forward(self, x0: torch.Tensor, x_in: torch.Tensor):
        B = x0.size(0)
        device = x0.device
        t = torch.randint(0, self.T, (B,), device=device)
        eps = torch.randn_like(x0)
        x_t = self.q_sample(x0, x_in, t, eps)
        pred_res, pred_noise, x0_hat = self.model_predictions(x_t, t, x_in)
        true_res = x_in - x0
        loss, lres, lnoi = self.loss_fn(pred_res, pred_noise, true_res, eps, x0_hat=x0_hat, x0=x0)
        return loss, lres, lnoi


class ResidualX0Diffusion(ResidualDiffusion):
    """RDDM no-residual-formulation ablation: keep residual q(x_t), predict x0 directly."""
    def model_predictions(self, x_t: torch.Tensor, t: torch.Tensor, x_in: torch.Tensor, lambda_res: float = 1.0):
        pred_x0, pred_noise = self.model(x_t, t, x_in)
        return pred_x0, pred_noise, pred_x0

    def loss_fn(self, pred_x0, pred_noise, x0, eps, x0_hat=None, x0_true=None):
        if self.loss_type == "l2":
            l_x0 = F.mse_loss(pred_x0, x0)
            l_noi = F.mse_loss(pred_noise, eps)
        else:
            l_x0 = F.l1_loss(pred_x0, x0)
            l_noi = F.l1_loss(pred_noise, eps)

        extra = pred_x0.new_tensor(0.0)
        if x0_hat is not None:
            if float(self.toeplitz_loss_weight) > 0.0:
                extra = extra + float(self.toeplitz_loss_weight) * toeplitz_consistency_loss(x0_hat)
            if float(self.psd_loss_weight) > 0.0:
                extra = extra + float(self.psd_loss_weight) * psd_negative_eig_loss(x0_hat)
        if x0_hat is not None and x0_true is not None:
            if float(self.subspace_loss_weight) > 0.0:
                extra = extra + float(self.subspace_loss_weight) * subspace_projector_loss(
                    x0_hat, x0_true, self.subspace_k
                )
            if float(self.music_loss_weight) > 0.0:
                extra = extra + float(self.music_loss_weight) * music_log_spectrum_loss(
                    x0_hat, x0_true, self.subspace_k, self.music_grid_size,
                    self.music_d_over_lambda, self.music_doa_min, self.music_doa_max
                )
            if float(self.music_margin_loss_weight) > 0.0:
                extra = extra + float(self.music_margin_loss_weight) * music_peak_margin_loss(
                    x0_hat, x0_true, self.subspace_k, self.music_grid_size,
                    self.music_d_over_lambda, self.music_doa_min, self.music_doa_max,
                    self.music_margin, self.music_target_window_deg, self.music_peak_min_dist_deg
                )

        loss = self.w_res * l_x0 + self.w_noise * l_noi + extra
        return loss, l_x0.item(), l_noi.item()

    def forward(self, x0: torch.Tensor, x_in: torch.Tensor):
        B = x0.size(0)
        device = x0.device
        t = torch.randint(0, self.T, (B,), device=device)
        eps = torch.randn_like(x0)
        x_t = self.q_sample(x0, x_in, t, eps)
        pred_x0, pred_noise, x0_hat = self.model_predictions(x_t, t, x_in)
        loss, lx0, lnoi = self.loss_fn(pred_x0, pred_noise, x0, eps, x0_hat=x0_hat, x0_true=x0)
        return loss, lx0, lnoi


class DirectDenoiser(nn.Module):
    """No-diffusion ablation: one conditional network maps noisy SCM to clean SCM."""
    def __init__(self, model: nn.Module, image_size: int, timesteps: int = 1,
                 loss_type: str = "l1", w_res: float = 1.0, w_noise: float = 0.0,
                 target: str = "x0", device=None):
        super().__init__()
        self.model = model
        self.M = image_size
        self.T = max(1, int(timesteps))
        self.loss_type = loss_type
        self.w_res = w_res
        self.w_noise = w_noise
        self.target = target

    def model_predictions(self, x_t: torch.Tensor, t: torch.Tensor, x_in: torch.Tensor, lambda_res: float = 1.0):
        pred_main, pred_noise = self.model(x_t, t, x_in)
        if self.target == "residual":
            x0_hat = x_in - float(lambda_res) * pred_main
        else:
            x0_hat = pred_main
        return pred_main, pred_noise, x0_hat

    @torch.no_grad()
    def p_sample_loop(self, x_in: torch.Tensor, steps: int = None, sum_scale: float = 0.0,
                      proj_hermitian: bool = True, proj_toeplitz: bool = False, proj_psd: bool = False,
                      lambda_res: float = 1.0, start_t: int = None):
        t_value = 0 if start_t is None else max(0, min(int(start_t), self.T - 1))
        t = torch.full((x_in.size(0),), t_value, device=x_in.device, dtype=torch.long)
        _, _, x0_hat = self.model_predictions(x_in, t, x_in, lambda_res=lambda_res)
        if proj_hermitian:
            x0_hat = hermitian_project(x0_hat)
        if proj_toeplitz:
            x0_hat = toeplitz_project(x0_hat)
        if proj_psd:
            x0_hat = psd_project(x0_hat)
        return x0_hat

    def forward(self, x0: torch.Tensor, x_in: torch.Tensor):
        B = x0.size(0)
        t = torch.zeros(B, device=x0.device, dtype=torch.long)
        pred_main, pred_noise, x0_hat = self.model_predictions(x_in, t, x_in)
        target = (x_in - x0) if self.target == "residual" else x0
        if self.loss_type == "l2":
            l_main = F.mse_loss(pred_main, target)
            l_noi = F.mse_loss(pred_noise, torch.zeros_like(pred_noise))
        else:
            l_main = F.l1_loss(pred_main, target)
            l_noi = F.l1_loss(pred_noise, torch.zeros_like(pred_noise))
        loss = self.w_res * l_main + self.w_noise * l_noi
        return loss, l_main.item(), l_noi.item()


class PlainDDIMDiffusion(ResidualDiffusion):
    """Classic conditional DDIM ablation: no residual bridge term in q(x_t)."""
    def q_sample(self, x0: torch.Tensor, x_in: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        alpha = self.abar[t].view(-1, 1, 1, 1)
        return torch.sqrt(torch.clamp(alpha, min=1e-8)) * x0 + torch.sqrt(torch.clamp(1.0 - alpha, min=1e-8)) * eps

    def model_predictions(self, x_t: torch.Tensor, t: torch.Tensor, x_in: torch.Tensor, lambda_res: float = 1.0):
        pred_aux, pred_noise = self.model(x_t, t, x_in)
        alpha = self.abar[t].view(-1, 1, 1, 1)
        x0_hat = (x_t - torch.sqrt(torch.clamp(1.0 - alpha, min=1e-8)) * pred_noise) / torch.sqrt(torch.clamp(alpha, min=1e-8))
        return pred_aux, pred_noise, x0_hat

    @torch.no_grad()
    def p_sample_loop(self, x_in: torch.Tensor, steps: int = None, sum_scale: float = 0.0,
                      proj_hermitian: bool = True, proj_toeplitz: bool = False, proj_psd: bool = False,
                      lambda_res: float = 1.0, start_t: int = None):
        B = x_in.size(0)
        device = x_in.device
        steps = self.T if (steps is None or steps <= 0 or steps > self.T) else steps
        if start_t is None:
            start_t = self.T - 1
        start_t = max(0, min(int(start_t), self.T - 1))
        times = torch.linspace(start_t, 0, steps=steps, device=device).long()
        x = torch.randn_like(x_in)
        if sum_scale > 0.0:
            x = x * math.sqrt(sum_scale)

        for i, ti in enumerate(times.tolist()):
            t = torch.full((B,), ti, device=device, dtype=torch.long)
            _, pred_noise, x0_hat = self.model_predictions(x, t, x_in, lambda_res=lambda_res)
            if proj_hermitian:
                x0_hat = hermitian_project(x0_hat)
            if proj_toeplitz:
                x0_hat = toeplitz_project(x0_hat)
            if proj_psd:
                x0_hat = psd_project(x0_hat)
            if i < len(times) - 1:
                t_next = times[i + 1].item()
                alpha_next = self.abar[t_next]
                x = (
                    torch.sqrt(torch.clamp(alpha_next, min=1e-8)) * x0_hat
                    + torch.sqrt(torch.clamp(1.0 - alpha_next, min=1e-8)) * pred_noise
                )
                if proj_hermitian:
                    x = hermitian_project(x)
            else:
                x = x0_hat
        return x

    def forward(self, x0: torch.Tensor, x_in: torch.Tensor):
        B = x0.size(0)
        device = x0.device
        t = torch.randint(0, self.T, (B,), device=device)
        eps = torch.randn_like(x0)
        x_t = self.q_sample(x0, x_in, t, eps)
        _, pred_noise, x0_hat = self.model_predictions(x_t, t, x_in)
        if self.loss_type == "l2":
            l_x0 = F.mse_loss(x0_hat, x0)
            l_noi = F.mse_loss(pred_noise, eps)
        else:
            l_x0 = F.l1_loss(x0_hat, x0)
            l_noi = F.l1_loss(pred_noise, eps)
        loss = self.w_res * l_x0 + self.w_noise * l_noi
        return loss, l_x0.item(), l_noi.item()


__all__ = [
    "GATTwoHead",
    "UNetTwoHead",
    "RelEdgeGATLayer",
    "BilinearMatrixDecoder",
    "ResidualDiffusion",
    "ResidualX0Diffusion",
    "DirectDenoiser",
    "PlainDDIMDiffusion",
    "cosine_cumprod",
    "hermitian_project",
    "toeplitz_project",
    "psd_project",
    "toeplitz_consistency_loss",
    "psd_negative_eig_loss",
    "subspace_projector_loss",
    "music_log_spectrum_loss",
    "music_peak_margin_loss",
]
