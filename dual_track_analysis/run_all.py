#!/usr/bin/env python3
"""
双轨分析入口（默认只跑轻量步骤：registry + 演化 extract + 汇总）。

完整预警 5-fold 较耗时，请显式加 --with-warning-kfold。

Usage (from Fire/):
    python -m dual_track_analysis.run_all
    python -m dual_track_analysis.run_all --with-warning-kfold
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _run(cmd: list[str]) -> None:
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, cwd=str(_ROOT), check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Dual-track analysis pipeline")
    ap.add_argument("--with-warning-kfold", action="store_true", help="运行预警 5-fold（耗时）")
    ap.add_argument("--with-baselines", action="store_true", help="运行预警轨基线 5-fold（耗时）")
    args = ap.parse_args()

    py = sys.executable

    _run([py, "-m", "dual_track_analysis.common.registry"])
    _run([py, "-m", "dual_track_analysis.evolution.run_extract"])
    _run([py, "-m", "dual_track_analysis.analysis.fire_evolution_timescale_analysis"])

    if args.with_warning_kfold:
        _run([py, "-m", "dual_track_analysis.warning.run_kfold"])
        _run([py, "-m", "dual_track_analysis.warning.export_results"])

    if args.with_baselines:
        _run([
            py, "-m", "dual_track_analysis.warning.run_baselines",
            "--resume", "--skip-existing",
        ])

    print("\n完成。见 dual_track_analysis/outputs/")


if __name__ == "__main__":
    main()
