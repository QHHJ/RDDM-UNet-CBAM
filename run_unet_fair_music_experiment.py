# -*- coding: utf-8 -*-
"""Run the fair UNet-backbone experiment for the RDDM denoiser.

Protocol:
  1. Pretrain UNet-RDDM on the old T128 Gaussian dataset.
  2. Fine-tune that checkpoint on the T128 sep3 dataset.
  3. Evaluate fine-tuned best-loss and best-MUSIC checkpoints with the same
     MUSIC lambda sweep used by the old RDDM-GAT comparison.

The script is intentionally sequential and writes intermediate summaries so a
long run can be monitored safely.
"""

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_PYTHON = ROOT / ".venv_cuda" / "Scripts" / "python.exe"
TRAIN = ROOT / "train_rddm_rgat.py"
EVAL = ROOT / "test.py"

PRETRAIN_TRAIN = ROOT / "dataset_snap=128_snr_-8_to_-5_gaussian_train_400000" / "flom_mat_000.npz"
PRETRAIN_VAL = ROOT / "dataset_snap=128_snr_-8_to_-5_gaussian_eval_40000" / "flom_mat_000.npz"
FINETUNE_TRAIN = ROOT / "exp_scm_t128_snr_-8_to_-5_sep3_gaussian_train_200000" / "flom_mat_000.npz"
FINETUNE_VAL = ROOT / "exp_scm_t128_snr_-8_to_-5_sep3_gaussian_eval_20000" / "flom_mat_000.npz"
TEST_DATA = ROOT / "exp_scm_t128_snr_-8_to_-5_sep3_gaussian_test_5000" / "flom_mat_000.npz"
REFERENCE_JSON = ROOT / "exp_fixed_eval_after_sampling_fix" / "finetune_compare_project_lam_sweep_sep3_5000.json"


def run(cmd, cwd, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, text=True)
    return proc.returncode


def newest_subdir(path):
    if not path.exists():
        return None
    dirs = [p for p in path.iterdir() if p.is_dir()]
    return max(dirs, key=lambda p: p.stat().st_mtime) if dirs else None


def ckpts_by_kind(run_dir):
    out = {}
    if run_dir is None:
        return out
    matches = sorted(run_dir.glob("best_model_*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    if matches:
        out["unet_best_loss"] = matches[0]
    matches = sorted(run_dir.glob("best_music_*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    if matches:
        out["unet_best_music"] = matches[0]
    last = run_dir / "last.pth"
    if last.exists():
        out.setdefault("unet_last", last)
    return out


def read_overall(path):
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("overall", {})


def train_cmd(py, args, outdir, train_data, val_data, epochs, lr, warmup_steps, init_ckpt="", music_val=False):
    cmd = [
        str(py), "-u", str(TRAIN),
        "--train", str(train_data),
        "--val", str(val_data),
        "--outdir", str(outdir),
        "--backbone", "unet",
        "--objective", "rddm",
        "--image-size", "8",
        "--steps", str(args.steps),
        "--loss", "l1",
        "--w-res", "1.0",
        "--w-noise", "0.0",
        "--gat-hidden", str(args.gat_hidden),
        "--gat-heads", "8",
        "--gat-layers", "6",
        "--gat-kmax", "2",
        "--gat-dropout", "0.1",
        "--lowrank-decoder", "0",
        "--unet-base", str(args.unet_base),
        "--unet-attention", args.unet_attention,
        "--epochs", str(epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(lr),
        "--weight-decay", "0.0",
        "--grad-clip", "1.0",
        "--warmup-steps", str(warmup_steps),
        "--ema-decay", "0.999",
        "--res-diag-weight", "8.0",
        "--res-off-penalty", "0.2",
        "--toeplitz-loss-weight", "0.0",
        "--psd-loss-weight", "0.0",
        "--subspace-loss-weight", "0.0",
        "--music-loss-weight", "0.0",
        "--music-margin-loss-weight", "0.0",
        "--seed", str(args.seed),
        "--device", args.device,
    ]
    if init_ckpt:
        cmd += ["--init-ckpt", str(init_ckpt)]
    if args.unet_skip_gate:
        cmd.append("--unet-skip-gate")
    if music_val:
        cmd += [
            "--music-val-every", str(args.music_val_every),
            "--music-val-num", str(args.music_val_num),
            "--music-val-batch-size", str(args.music_val_batch_size),
            "--music-val-steps", "1",
            "--music-val-select", "sr",
        ]
    else:
        cmd += ["--music-val-every", "0"]
    if args.amp:
        cmd.append("--amp")
    return cmd


def eval_cmd(py, args, ckpt, out_json, lambda_res, sr_thr):
    cmd = [
        str(py), "-u", str(EVAL),
        "--ckpt", str(ckpt),
        "--data-dir", str(args.test),
        "--batch-size", str(args.eval_batch_size),
        "--eval-num", str(args.eval_num),
        "--sampler", "one",
        "--sampling-timestep", str(args.sampling_timestep),
        "--lambda-res", str(lambda_res),
        "--project-inputs",
        "--plot-num", "0",
        "--sr-thr", str(sr_thr),
        "--out-json", str(out_json),
    ]
    if args.fp16_eval:
        cmd.append("--fp16")
    return cmd


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_reference_rows():
    rows = []
    if not REFERENCE_JSON.exists():
        return rows
    with open(REFERENCE_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    keep = ["ft128_plain_loss", "ft128_plain_music"]
    for name in keep:
        item = data.get(name)
        if not item:
            continue
        for lam_name, metrics in item.get("all", {}).items():
            rows.append({
                "model": "old_rgat",
                "ckpt_kind": name,
                "lambda": lam_name.replace("lam", ""),
                "SR05": metrics.get("SR05"),
                "SR10": metrics.get("SR10"),
                "RMSE": metrics.get("RMSE"),
                "ckpt": item.get("ckpt"),
                "source": str(REFERENCE_JSON),
            })
    return rows


def unet_label(args):
    label = "unet" if args.unet_attention == "none" else f"unet_{args.unet_attention}"
    if args.unet_skip_gate:
        label += "_skipgate"
    return label


def evaluate_sweep(py, args, ckpt_kind, ckpt, eval_dir, logs_dir):
    rows = []
    for lam in args.lambdas:
        lam_label = f"{lam:g}".replace(".", "p")
        combined = {
            "model": unet_label(args),
            "ckpt_kind": ckpt_kind,
            "lambda": f"{lam:g}",
            "ckpt": str(ckpt),
        }
        for thr, label in [(0.5, "05"), (1.0, "10")]:
            out_json = eval_dir / f"{ckpt_kind}_lam{lam_label}_sr{label}.json"
            log_path = logs_dir / f"eval_{ckpt_kind}_lam{lam_label}_sr{label}.log"
            code = run(eval_cmd(py, args, ckpt, out_json, lam, thr), ROOT, log_path)
            overall = read_overall(out_json)
            combined[f"eval_returncode_sr{label}"] = code
            combined[f"out_json_sr{label}"] = str(out_json)
            combined[f"SR{label}"] = overall.get("SR_deno")
            if label == "05":
                combined["RMSE"] = overall.get("RMSE_deno")
                combined["SR05_noisy"] = overall.get("SR_noisy")
                combined["RMSE_noisy"] = overall.get("RMSE_noisy")
                combined["SR05_clean"] = overall.get("SR_clean")
                combined["RMSE_clean"] = overall.get("RMSE_clean")
        rows.append(combined)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=ROOT / "exp_unet_fair_music")
    parser.add_argument("--tag", type=str, default="unet_fair")
    parser.add_argument("--pretrain-train", type=Path, default=PRETRAIN_TRAIN)
    parser.add_argument("--pretrain-val", type=Path, default=PRETRAIN_VAL)
    parser.add_argument("--finetune-train", type=Path, default=FINETUNE_TRAIN)
    parser.add_argument("--finetune-val", type=Path, default=FINETUNE_VAL)
    parser.add_argument("--test", type=Path, default=TEST_DATA)
    parser.add_argument("--pretrain-epochs", type=int, default=150)
    parser.add_argument("--finetune-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--eval-num", type=int, default=5000)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--sampling-timestep", type=int, default=100)
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.8, 0.9, 1.0, 1.1])
    parser.add_argument("--gat-hidden", type=int, default=256)
    parser.add_argument("--unet-base", type=int, default=64)
    parser.add_argument("--unet-attention", "--unet-attn", dest="unet_attention",
                        type=str, default="none", choices=["none", "se", "cbam"])
    parser.add_argument("--unet-skip-gate", action="store_true")
    parser.add_argument("--music-val-every", type=int, default=2)
    parser.add_argument("--music-val-num", type=int, default=2000)
    parser.add_argument("--music-val-batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--fp16-eval", action="store_true")
    parser.add_argument("--skip-pretrain", action="store_true")
    parser.add_argument("--pretrain-ckpt", type=Path, default=None)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else Path(sys.executable))
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.out_root / f"{args.tag}_{stamp}"
    logs_dir = run_dir / "logs"
    runs_dir = run_dir / "runs"
    eval_dir = run_dir / "eval"
    for path in (logs_dir, runs_dir, eval_dir):
        path.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_dir": str(run_dir),
        "protocol": "UNet old-T128 pretrain -> sep3 fine-tune -> sep3 MUSIC lambda sweep",
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "reference_json": str(REFERENCE_JSON),
    }
    write_json(run_dir / "manifest.json", manifest)

    status = []
    summary_rows = []
    comparison_rows = load_reference_rows()

    pretrain_ckpt = args.pretrain_ckpt
    pretrain_run_dir = None
    if not args.skip_pretrain:
        print("[Stage] pretrain UNet on old T128 data", flush=True)
        pretrain_out = runs_dir / "unet_pretrain_old_t128"
        code = run(
            train_cmd(
                args.python, args, pretrain_out, args.pretrain_train, args.pretrain_val,
                args.pretrain_epochs, 5e-4, 2000, music_val=False,
            ),
            ROOT,
            logs_dir / "pretrain.log",
        )
        pretrain_run_dir = newest_subdir(pretrain_out)
        pretrain_ckpts = ckpts_by_kind(pretrain_run_dir)
        pretrain_ckpt = pretrain_ckpts.get("unet_best_loss") or pretrain_ckpts.get("unet_last")
        status.append({
            "stage": "pretrain",
            "returncode": code,
            "run_dir": str(pretrain_run_dir or ""),
            "ckpt": str(pretrain_ckpt or ""),
        })
        write_json(run_dir / "status.json", status)
        if code != 0 or pretrain_ckpt is None:
            write_json(run_dir / "summary.json", {"status": status, "comparison_rows": comparison_rows})
            raise SystemExit(f"pretrain failed, returncode={code}, ckpt={pretrain_ckpt}")
    elif pretrain_ckpt is None:
        raise SystemExit("--skip-pretrain requires --pretrain-ckpt")

    print("[Stage] fine-tune UNet on sep3 T128 data", flush=True)
    finetune_out = runs_dir / "unet_ft_sep3_t128"
    code = run(
        train_cmd(
            args.python, args, finetune_out, args.finetune_train, args.finetune_val,
            args.finetune_epochs, 1e-4, 100, init_ckpt=pretrain_ckpt, music_val=True,
        ),
        ROOT,
        logs_dir / "finetune.log",
    )
    finetune_run_dir = newest_subdir(finetune_out)
    finetune_ckpts = ckpts_by_kind(finetune_run_dir)
    status.append({
        "stage": "finetune",
        "returncode": code,
        "run_dir": str(finetune_run_dir or ""),
        "ckpts": {k: str(v) for k, v in finetune_ckpts.items()},
    })
    write_json(run_dir / "status.json", status)
    if code != 0 or not finetune_ckpts:
        write_json(run_dir / "summary.json", {"status": status, "comparison_rows": comparison_rows})
        raise SystemExit(f"finetune failed, returncode={code}, ckpts={finetune_ckpts}")

    fields = [
        "model", "ckpt_kind", "lambda", "SR05", "SR10", "RMSE",
        "SR05_noisy", "RMSE_noisy", "SR05_clean", "RMSE_clean",
        "ckpt", "source", "eval_returncode_sr05", "eval_returncode_sr10",
        "out_json_sr05", "out_json_sr10",
    ]
    write_csv(run_dir / "comparison.csv", comparison_rows, fields)

    print("[Stage] evaluate MUSIC lambda sweep", flush=True)
    for kind in ["unet_best_loss", "unet_best_music"]:
        ckpt = finetune_ckpts.get(kind)
        if ckpt is None:
            continue
        rows = evaluate_sweep(args.python, args, kind, ckpt, eval_dir, logs_dir)
        summary_rows.extend(rows)
        comparison_rows.extend(rows)
        write_csv(run_dir / "unet_summary.csv", summary_rows, fields)
        write_csv(run_dir / "comparison.csv", comparison_rows, fields)
        write_json(run_dir / "summary.json", {
            "status": status,
            "unet_rows": summary_rows,
            "comparison_rows": comparison_rows,
        })

    write_csv(run_dir / "unet_summary.csv", summary_rows, fields)
    write_csv(run_dir / "comparison.csv", comparison_rows, fields)
    write_json(run_dir / "summary.json", {
        "status": status,
        "unet_rows": summary_rows,
        "comparison_rows": comparison_rows,
    })
    print(f"[Done] comparison: {run_dir / 'comparison.csv'}", flush=True)


if __name__ == "__main__":
    main()
