#!/usr/bin/env python3
"""
LBCA 预警轨消融实验（基线参数 = kfold/5fold_summary.json）。

输出：``dual_track_analysis/outputs/warning/ablation/<ablation_id>/``

Usage (from Fire_prediction/):
    python -m dual_track_analysis.warning.run_ablations --dry-run

    # 先跑 reeval 消融（快，共用 kfold checkpoint）
    python -m dual_track_analysis.warning.run_ablations --only thresh_max_f1,thresh_recall85

    # 训练消融（各 5-fold）
    python -m dual_track_analysis.warning.run_ablations --only warn_only,small_model,late_concat

    # 全部 + 断点续跑
    python -m dual_track_analysis.warning.run_ablations --resume --skip-existing

    # 仅重建对比表
    python -m dual_track_analysis.warning.export_ablation_comparison
"""
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
    run_ablation_study,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="LBCA warning-track ablation study")
    ap.add_argument("--study", type=Path, default=_DEFAULT_STUDY)
    ap.add_argument(
        "--config", type=Path, default=None,
        help="Optional JSON overrides on top of config.py defaults",
    )
    ap.add_argument("--ablation-root", type=Path, default=_DEFAULT_OUT)
    ap.add_argument(
        "--only", type=str, default=None,
        help="逗号分隔的 ablation id，如 warn_only,thresh_max_f1",
    )
    ap.add_argument("--max-folds", type=int, default=None)
    ap.add_argument("--start-fold", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()

    only = [x.strip() for x in args.only.split(",") if x.strip()] if args.only else None

    run_ablation_study(
        study_path=args.study.resolve(),
        config_path=args.config.resolve() if args.config else None,
        ablation_root=args.ablation_root.resolve(),
        only=only,
        resume=args.resume,
        skip_existing=args.skip_existing,
        dry_run=args.dry_run,
        max_folds=args.max_folds,
        start_fold=args.start_fold,
        extra_sets=list(args.set),
    )


if __name__ == "__main__":
    main()
