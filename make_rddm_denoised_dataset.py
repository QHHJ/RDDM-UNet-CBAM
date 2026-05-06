# -*- coding: utf-8 -*-
"""Create an offline RDDM-denoised SCM dataset for CGNN training/evaluation.

Input dataset:
    xins: noisy SCM, shape [N, 2, M, M]
    x0s:  clean SCM, shape [N, 2, M, M]
    meta: DOA/SNR metadata

Output dataset:
    xins: RDDM-denoised SCM, shape [N, 2, M, M]
    x0s/meta copied from input

The output keeps the same npz format used by train_cgnn_like_doa.py, so CGNN
can be trained directly on the denoised covariance distribution.
"""

import argparse
import glob
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from torch.amp import autocast

from model.rddm_gat import (
    DirectDenoiser,
    GATTwoHead,
    PlainDDIMDiffusion,
    ResidualDiffusion,
    ResidualX0Diffusion,
    UNetTwoHead,
)


def resolve_npz(path: str) -> str:
    if os.path.isdir(path):
        candidates = []
        for pat in ("flom_mat_*.npz", "scm_mat_*.npz", "ideal_mat_*.npz", "*.npz"):
            candidates.extend(glob.glob(os.path.join(path, pat)))
        candidates = sorted(set(candidates))
        for p in candidates:
            try:
                with np.load(p, allow_pickle=True) as z:
                    if "xins" in z.files and "x0s" in z.files:
                        return p
            except Exception:
                continue
        raise FileNotFoundError(f"No npz with xins/x0s found under {path}")
    return path


def load_train_config(ckpt_path: str):
    cfg = dict(
        backbone="rgat",
        objective="rddm",
        direct_target="x0",
        steps=400,
        hidden=256,
        heads=8,
        layers=6,
        kmax=2,
        dropout=0.1,
        lowrank=0,
        use_edge_sign=True,
        use_time_film=True,
        force_hermitian_decoder=True,
        unet_base=64,
        unet_attention="none",
        unet_skip_gate=False,
    )
    args_path = os.path.join(os.path.dirname(ckpt_path), "args.json")
    if os.path.exists(args_path):
        with open(args_path, "r", encoding="utf-8") as f:
            args = json.load(f)
        cfg["backbone"] = str(args.get("backbone", cfg["backbone"]))
        cfg["objective"] = str(args.get("objective", cfg["objective"]))
        cfg["direct_target"] = str(args.get("direct_target", cfg["direct_target"]))
        cfg["steps"] = int(args.get("steps", cfg["steps"]))
        cfg["hidden"] = int(args.get("gat_hidden", cfg["hidden"]))
        cfg["heads"] = int(args.get("gat_heads", cfg["heads"]))
        cfg["layers"] = int(args.get("gat_layers", cfg["layers"]))
        cfg["kmax"] = int(args.get("gat_kmax", cfg["kmax"]))
        cfg["dropout"] = float(args.get("gat_dropout", cfg["dropout"]))
        cfg["lowrank"] = int(args.get("lowrank_decoder", cfg["lowrank"]))
        cfg["use_edge_sign"] = bool(args.get("use_edge_sign", cfg["use_edge_sign"]))
        cfg["use_time_film"] = not bool(args.get("no_time_film", False))
        cfg["force_hermitian_decoder"] = not bool(args.get("no_hermitian_decoder", False))
        cfg["unet_base"] = int(args.get("unet_base", cfg["unet_base"]))
        cfg["unet_attention"] = str(args.get("unet_attention", args.get("unet_attn", cfg["unet_attention"])))
        cfg["unet_skip_gate"] = bool(args.get("unet_skip_gate", cfg["unet_skip_gate"]))
    return cfg


def project_struct_numpy(C: np.ndarray, toeplitz: bool = True, psd: bool = True) -> np.ndarray:
    C = 0.5 * (C + C.conj().T)
    if toeplitz:
        M = C.shape[0]
        Tproj = np.zeros_like(C, dtype=np.complex64)
        for k in range(-(M - 1), M):
            diag = np.diag(C, k=k)
            if diag.size == 0:
                continue
            mean = diag.mean()
            Tproj += np.diag(np.full(M - abs(k), mean, dtype=np.complex64), k)
        C = 0.5 * (Tproj + Tproj.conj().T)
    if psd:
        w, V = np.linalg.eigh(C)
        w = np.clip(w, 0.0, None)
        C = (V * w) @ V.conj().T
    return C.astype(np.complex64)


def complex_to_2ch(C: np.ndarray) -> np.ndarray:
    return np.stack([C.real, C.imag], axis=0).astype(np.float32)


def build_model(ckpt_path: str, m: int, device: torch.device):
    cfg = load_train_config(ckpt_path)
    if cfg["backbone"] == "unet":
        model = UNetTwoHead(
            M=m,
            base=cfg["unet_base"],
            t_dim=cfg["hidden"],
            dropout=cfg["dropout"],
            use_time_film=cfg["use_time_film"],
            force_hermitian_decoder=cfg["force_hermitian_decoder"],
            attention=cfg["unet_attention"],
            use_skip_gate=cfg["unet_skip_gate"],
        )
    else:
        model = GATTwoHead(
            M=m,
            hidden=cfg["hidden"],
            heads=cfg["heads"],
            layers=cfg["layers"],
            k_max=cfg["kmax"],
            t_dim=cfg["hidden"],
            use_edge_sign=cfg["use_edge_sign"],
            dropout=cfg["dropout"],
            lowrank_decoder=cfg["lowrank"],
            use_time_film=cfg["use_time_film"],
            force_hermitian_decoder=cfg["force_hermitian_decoder"],
        )

    if cfg["objective"] == "direct":
        net = DirectDenoiser(
            model=model,
            image_size=m,
            timesteps=cfg["steps"],
            loss_type="l1",
            w_res=1.0,
            w_noise=0.0,
            target=cfg["direct_target"],
            device=device,
        ).to(device)
    elif cfg["objective"] == "rddm_x0":
        net = ResidualX0Diffusion(
            model=model,
            image_size=m,
            timesteps=cfg["steps"],
            loss_type="l1",
            w_res=1.0,
            w_noise=1.0,
            device=device,
        ).to(device)
    elif cfg["objective"] == "ddim":
        net = PlainDDIMDiffusion(
            model=model,
            image_size=m,
            timesteps=cfg["steps"],
            loss_type="l1",
            w_res=0.0,
            w_noise=1.0,
            device=device,
        ).to(device)
    else:
        net = ResidualDiffusion(
            model=model,
            image_size=m,
            timesteps=cfg["steps"],
            loss_type="l1",
            w_res=1.0,
            w_noise=1.0,
            device=device,
        ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing:
        print(f"[Warn] missing keys: {missing}")
    if unexpected:
        print(f"[Warn] unexpected keys: {unexpected}")
    net.eval()
    print(
        "[Model] "
        f"backbone={cfg['backbone']} objective={cfg['objective']} M={m} "
        f"hidden={cfg['hidden']} unet_base={cfg['unet_base']} "
        f"unet_attention={cfg['unet_attention']} unet_skip_gate={cfg['unet_skip_gate']} "
        f"steps={cfg['steps']}"
    )
    return net, cfg


@torch.no_grad()
def denoise_array(args, xins: np.ndarray, net, cfg, device: torch.device) -> np.ndarray:
    n, _, m, _ = xins.shape
    out = np.empty_like(xins, dtype=np.float32)
    start_t = args.sampling_timestep if args.sampling_timestep is not None else cfg["steps"] - 1
    dev_type = "cuda" if device.type == "cuda" else "cpu"
    for s in range(0, n, args.batch_size):
        e = min(n, s + args.batch_size)
        cond = torch.from_numpy(xins[s:e]).to(device)
        with autocast(device_type=dev_type, enabled=args.fp16):
            x0_hat = net.p_sample_loop(
                cond,
                steps=1,
                sum_scale=0.0,
                proj_hermitian=True,
                proj_toeplitz=False,
                proj_psd=False,
                lambda_res=args.lambda_res,
                start_t=start_t,
            )
        batch = x0_hat.detach().float().cpu().numpy()
        if args.project_output:
            projected = np.empty_like(batch, dtype=np.float32)
            for i in range(batch.shape[0]):
                C = batch[i, 0] + 1j * batch[i, 1]
                projected[i] = complex_to_2ch(
                    project_struct_numpy(C, toeplitz=not args.no_toeplitz, psd=not args.no_psd)
                )
            batch = projected
        out[s:e] = batch
        if (e == n) or (e % args.log_every == 0):
            print(f"[Denoise] {e}/{n}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Input npz file or dataset directory")
    ap.add_argument("--dst-dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--lambda-res", type=float, default=1.0)
    ap.add_argument("--sampling-timestep", type=int, default=100)
    ap.add_argument("--project-output", action="store_true")
    ap.add_argument("--no-toeplitz", action="store_true")
    ap.add_argument("--no-psd", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=20000)
    args = ap.parse_args()

    src_npz = resolve_npz(args.src)
    dst_dir = Path(args.dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_npz = dst_dir / Path(src_npz).name

    with np.load(src_npz, allow_pickle=True) as z:
        xins = z["xins"].astype(np.float32)
        x0s = z["x0s"].astype(np.float32)
        extras = {k: z[k] for k in z.files if k not in {"xins", "x0s"}}
    original_n = int(xins.shape[0])

    if args.limit and args.limit > 0:
        xins = xins[: args.limit]
        x0s = x0s[: args.limit]
        for k, v in list(extras.items()):
            if isinstance(v, np.ndarray) and v.shape[:1] == (original_n,):
                extras[k] = v[: args.limit]

    device = torch.device(args.device)
    net, cfg = build_model(args.ckpt, int(xins.shape[-1]), device)
    print(f"[Config] src={src_npz}")
    print(f"[Config] dst={dst_npz}")
    print(f"[Config] samples={len(xins)} M={xins.shape[-1]} steps={cfg['steps']} lambda={args.lambda_res} t={args.sampling_timestep}")
    print(f"[Config] project_output={args.project_output} toeplitz={not args.no_toeplitz} psd={not args.no_psd}")

    denoised = denoise_array(args, xins, net, cfg, device)
    np.savez_compressed(dst_npz, xins=denoised, x0s=x0s, **extras)

    src_meta = os.path.join(os.path.dirname(src_npz), "meta.json")
    if os.path.exists(src_meta):
        shutil.copy2(src_meta, dst_dir / "meta.json")

    manifest = {
        "src": os.path.abspath(src_npz),
        "dst": os.path.abspath(dst_npz),
        "ckpt": os.path.abspath(args.ckpt),
        "samples": int(len(denoised)),
        "lambda_res": float(args.lambda_res),
        "sampling_timestep": args.sampling_timestep,
        "project_output": bool(args.project_output),
        "toeplitz": not args.no_toeplitz,
        "psd": not args.no_psd,
    }
    with open(dst_dir / "denoise_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print("[Done]")


if __name__ == "__main__":
    main()
