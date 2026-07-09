#!/usr/bin/env python3
"""
预警轨 — 仅 10 份 full_process，关闭 post-fire 监督。

默认超参来自 ``config.py``（regularized warning track）。
输出目录默认 dual_track_analysis/outputs/warning/kfold_regularized

Usage (from Fire/):
    python -m dual_track_analysis.warning.run_kfold
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from run_kfold_tuned import main as kfold_main


def main() -> None:
    ap = argparse.ArgumentParser(description="Warning-only 5-fold CV (no post-fire in training)")
    ap.add_argument(
        "--config", type=Path, default=None,
        help="Optional JSON overrides on top of config.py defaults",
    )
    ap.add_argument(
        "--output-dir", type=str,
        default="dual_track_analysis/outputs/warning/kfold_regularized",
    )
    ap.add_argument("--pretrain", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args, extra = ap.parse_known_args()

    argv = [
        "run_kfold_tuned.py",
        "--output-dir", args.output_dir,
    ]
    if args.config is not None:
        argv.extend(["--config", str(args.config.resolve())])
    if args.pretrain:
        argv.append("--pretrain")
    if args.resume:
        argv.append("--resume")
    for item in args.set:
        argv.extend(["--set", item])

    sys.argv = argv + extra

    print("═══ 预警轨 Warning-only 5-fold ═══")
    print(f"  config: {args.config or 'config.py (defaults)'}")
    print(f"  output: {args.output_dir}")
    kfold_main()

    from dual_track_analysis.warning.export_results import export_warning_results
    kfold_out = _ROOT / args.output_dir
    export_warning_results(kfold_out, _ROOT / "dual_track_analysis/outputs/warning/summary")


if __name__ == "__main__":
    main()
