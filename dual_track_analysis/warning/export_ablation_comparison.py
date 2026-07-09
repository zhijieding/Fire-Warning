#!/usr/bin/env python3
"""从已有 ablation 目录重建对比表（不重训）。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dual_track_analysis.warning.ablation_runner import (
    _DEFAULT_OUT,
    _DEFAULT_STUDY,
    export_ablation_comparison,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Rebuild ablation comparison tables")
    ap.add_argument("--study", type=Path, default=_DEFAULT_STUDY)
    ap.add_argument("--ablation-root", type=Path, default=_DEFAULT_OUT)
    args = ap.parse_args()

    export_ablation_comparison(
        args.ablation_root.resolve(),
        study_path=args.study.resolve(),
        write_files=True,
    )


if __name__ == "__main__":
    main()
