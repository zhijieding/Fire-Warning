#!/usr/bin/env python3
"""从已有 baseline / Ours summary 重建对比表（不重训）。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dual_track_analysis.warning.run_baselines import run_baselines, _DEFAULT_BASELINE_ROOT, _DEFAULT_CFG, _DEFAULT_OURS_SUMMARY
from dual_track_analysis.warning.baseline_runner import BASELINE_MODELS


def main() -> None:
    ap = argparse.ArgumentParser(description="Rebuild warning baseline comparison tables")
    ap.add_argument("--baseline-root", type=Path, default=_DEFAULT_BASELINE_ROOT)
    ap.add_argument("--ours-summary", type=Path, default=_DEFAULT_OURS_SUMMARY)
    ap.add_argument("--config", type=Path, default=_DEFAULT_CFG)
    args = ap.parse_args()

    run_baselines(
        models=list(BASELINE_MODELS),
        config_path=args.config,
        baseline_root=args.baseline_root,
        ours_summary_path=args.ours_summary,
        extra_sets=[],
        max_folds=None,
        start_fold=None,
        resume=False,
        skip_existing=True,
        compare_only=True,
        epochs=None,
        device=None,
    )


if __name__ == "__main__":
    main()
