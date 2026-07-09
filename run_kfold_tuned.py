#!/usr/bin/env python
"""
5-fold CV with Config defaults (regularized warning track), plus optional ablations.

Baseline (full regularized config from config.py):
    python run_kfold_tuned.py --output-dir ./ablation/baseline

Ablations (change ONE thing, separate output dir):
    python run_kfold_tuned.py --output-dir ./ablation/no_consist --lambda-consist 0
    python run_kfold_tuned.py --output-dir ./ablation/late_concat --fusion-mode late_concat
    python run_kfold_tuned.py --output-dir ./ablation/direct_smoke --fusion-mode direct_smoke_temp
    python run_kfold_tuned.py --output-dir ./ablation/pretrain --pretrain

Override any Config field:
    python run_kfold_tuned.py --output-dir ./ablation/low_lr --set lr=0.0005

Optional JSON overrides (ablation dicts only):
    python run_kfold_tuned.py --config my_ablate_overrides.json --output-dir ./ablation/custom
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from config import Config, parse_config_value
from data_pipeline.preprocess import discover_and_configure
from run_kfold import main as run_kfold_main

_ROOT = Path(__file__).resolve().parent


def load_tuned_config(json_path: Path | None = None) -> tuple[Config, dict]:
    """Return Config; optional JSON applies overrides on top of config.py defaults."""
    if json_path is None:
        return Config(), {}
    return Config.from_json(json_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="5-fold CV: config.py defaults + ablation overrides",
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Optional JSON with Config field overrides (default: config.py only)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./trimodal_5fold_tuned_nopretrain",
        help="Root directory for all folds and 5fold_summary.json",
    )
    parser.add_argument("--kfold", type=int, default=5)
    parser.add_argument("--pretrain", action="store_true")
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip folds with existing eval/eval_summary.json; merge into summary.",
    )
    parser.add_argument(
        "--start-fold", type=int, default=None,
        help="Only run folds with index >= N (combine with --resume).",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--fusion-mode", type=str, default=None,
        choices=("layer_bridge", "direct_smoke_temp", "late_concat"),
    )
    parser.add_argument("--lambda-trend", type=float, default=None)
    parser.add_argument("--lambda-consist", type=float, default=None)
    parser.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE",
        help="Extra override on Config (e.g. lambda_warn=1.5). Applied before kfold main.",
    )
    args = parser.parse_args()

    cfg, tuned_cfg = load_tuned_config(args.config)

    for item in args.set:
        if "=" not in item:
            raise ValueError(f"--set expects KEY=VALUE, got: {item!r}")
        key, raw = item.split("=", 1)
        key = key.strip()
        if not hasattr(cfg, key):
            raise AttributeError(f"--set: unknown Config field '{key}'")
        setattr(cfg, key, parse_config_value(raw))

    if args.fusion_mode is not None:
        cfg.fusion_mode = args.fusion_mode
    if args.lambda_trend is not None:
        cfg.lambda_trend = args.lambda_trend
    if args.lambda_consist is not None:
        cfg.lambda_consist = args.lambda_consist
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.device is not None:
        cfg.device = args.device

    discover_and_configure(cfg)
    print("Detected experiments:", cfg.full_process_ids)
    if args.config is not None:
        print("JSON overrides:", args.config.resolve())
        print("Applied from JSON:", {k: tuned_cfg[k] for k in sorted(tuned_cfg)})
    else:
        print("Using config.py defaults (regularized warning track)")
    if args.set:
        print("--set overrides:", args.set)

    argv = [
        "run_kfold.py",
        "--kfold", str(args.kfold),
        "--output-dir", args.output_dir,
    ]
    if args.pretrain:
        argv.append("--pretrain")
    if args.max_folds is not None:
        argv.extend(["--max-folds", str(args.max_folds)])
    if args.resume:
        argv.append("--resume")
    if args.start_fold is not None:
        argv.extend(["--start-fold", str(args.start_fold)])
    if args.epochs is not None:
        argv.extend(["--epochs", str(args.epochs)])
    if args.device is not None:
        argv.extend(["--device", args.device])
    if args.fusion_mode is not None:
        argv.extend(["--fusion-mode", args.fusion_mode])
    if args.lambda_trend is not None:
        argv.extend(["--lambda-trend", str(args.lambda_trend)])
    if args.lambda_consist is not None:
        argv.extend(["--lambda-consist", str(args.lambda_consist)])

    sys.argv = argv
    run_kfold_main(cfg)


if __name__ == "__main__":
    main()
