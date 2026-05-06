# -*- coding: utf-8 -*-
"""Run UNet attention variants with the existing fair UNet protocol.

Default variants:
  baseline, se, cbam, skipgate, cbam_skipgate

Any unknown arguments are forwarded to run_unet_fair_music_experiment.py, so
training length, datasets, batch sizes, device, and evaluation settings stay in
one place.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FAIR = ROOT / "run_unet_fair_music_experiment.py"
DEFAULT_PYTHON = ROOT / ".venv_cuda" / "Scripts" / "python.exe"

VARIANTS = {
    "baseline": ("none", False),
    "se": ("se", False),
    "cbam": ("cbam", False),
    "skipgate": ("none", True),
    "cbam_skipgate": ("cbam", True),
}


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Run UNet attention ablations by wrapping run_unet_fair_music_experiment.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out-root", type=Path, default=ROOT / "exp_unet_attention_ablation")
    parser.add_argument("--tag", type=str, default="unet_attention")
    parser.add_argument("--variants", nargs="+", choices=sorted(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else Path(sys.executable))
    parser.add_argument("--continue-on-error", action="store_true")
    args, forwarded = parser.parse_known_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    master_dir = args.out_root / f"{args.tag}_{stamp}"
    logs_dir = master_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    status = []
    write_json(master_dir / "manifest.json", {
        "script": str(FAIR),
        "python": str(args.python),
        "variants": args.variants,
        "forwarded_args": forwarded,
    })

    for name in args.variants:
        attention, skip_gate = VARIANTS[name]
        variant_out = master_dir / name
        cmd = [
            str(args.python), "-u", str(FAIR),
            "--out-root", str(variant_out),
            "--tag", f"{args.tag}_{name}",
            "--python", str(args.python),
            "--unet-attention", attention,
        ]
        if skip_gate:
            cmd.append("--unet-skip-gate")
        cmd.extend(forwarded)

        print(f"[Variant] {name}: attention={attention}, skip_gate={skip_gate}", flush=True)
        log_path = logs_dir / f"{name}.log"
        with open(log_path, "w", encoding="utf-8") as f:
            code = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, text=True).returncode
        item = {
            "variant": name,
            "attention": attention,
            "skip_gate": skip_gate,
            "returncode": code,
            "out_root": str(variant_out),
            "log": str(log_path),
        }
        status.append(item)
        write_json(master_dir / "status.json", status)
        if code != 0 and not args.continue_on_error:
            raise SystemExit(f"variant {name} failed, returncode={code}, log={log_path}")

    print(f"[Done] status: {master_dir / 'status.json'}", flush=True)


if __name__ == "__main__":
    main()
