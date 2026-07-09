#!/usr/bin/env python3
"""
预警轨基线 5-fold — 与 ``warning/run_kfold`` 同参数、同划分。

Usage (from Fire_prediction/):
    python -m dual_track_analysis.warning.run_baselines
    python -m dual_track_analysis.warning.run_baselines --models lstm gru
    python -m dual_track_analysis.warning.run_baselines --models lstm --max-folds 1
    python -m dual_track_analysis.warning.run_baselines --skip-existing --resume
    python -m dual_track_analysis.warning.run_baselines --compare-only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dual_track_analysis.warning.baseline_runner import (
    BASELINE_MODELS,
    load_warning_baseline_cfg,
    run_baseline_kfold,
    write_comparison_reports,
)

_DEFAULT_BASELINE_ROOT = _ROOT / "dual_track_analysis/outputs/warning/baseline_regularized"
_DEFAULT_OURS_SUMMARY = _ROOT / "dual_track_analysis/outputs/warning/kfold_regularized/5fold_summary.json"


def _load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_baselines(
    *,
    models: list[str],
    config_path: Path | None,
    baseline_root: Path,
    ours_summary_path: Path,
    extra_sets: list[str],
    max_folds: int | None,
    start_fold: int | None,
    resume: bool,
    skip_existing: bool,
    compare_only: bool,
    epochs: int | None,
    device: str | None,
) -> None:
    baseline_root = baseline_root.resolve()
    baseline_root.mkdir(parents=True, exist_ok=True)

    ours_summary = _load_json(ours_summary_path.resolve())
    if ours_summary is None:
        print(f"[warning-baseline] WARNING: Ours summary not found: {ours_summary_path}")

    all_summaries: dict[str, dict] = {}

    if not compare_only:
        base_cfg, tuned = load_warning_baseline_cfg(
            config_path.resolve() if config_path else None,
            extra_sets,
        )
        if epochs is not None:
            base_cfg.epochs = epochs
        if device is not None:
            base_cfg.device = device

        print("═══ 预警轨 Baseline 5-fold ═══")
        print(f"  config      : {config_path or 'config.py (defaults)'}")
        print(f"  output root : {baseline_root}")
        print(f"  ours summary: {ours_summary_path}")
        print(f"  tuned keys  : {sorted(tuned)}")
        if extra_sets:
            print(f"  --set       : {extra_sets}")

        for mt in models:
            out_dir = baseline_root / mt
            out_dir.mkdir(parents=True, exist_ok=True)
            sum_path = out_dir / "5fold_summary.json"

            if skip_existing and sum_path.is_file():
                print(f"\n[skip] {mt}: {sum_path}")
                all_summaries[mt] = _load_json(sum_path) or {}
                continue

            summary = run_baseline_kfold(
                mt, base_cfg, out_dir,
                max_folds=max_folds,
                start_fold=start_fold,
                resume=resume,
            )
            all_summaries[mt] = summary
    else:
        for mt in models:
            sum_path = baseline_root / mt / "5fold_summary.json"
            if sum_path.is_file():
                all_summaries[mt] = _load_json(sum_path) or {}

    write_comparison_reports(
        baseline_root,
        all_summaries,
        ours_summary,
        ours_summary_path=ours_summary_path.resolve(),
        config_path=config_path.resolve() if config_path else None,
    )
    print(f"\n对比表 → {baseline_root / 'baseline_comparison_wide.csv'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Warning-track baseline 5-fold comparison")
    ap.add_argument(
        "--config", type=Path, default=None,
        help="Optional JSON overrides on top of config.py defaults",
    )
    ap.add_argument("--baseline-root", type=Path, default=_DEFAULT_BASELINE_ROOT)
    ap.add_argument("--ours-summary", type=Path, default=_DEFAULT_OURS_SUMMARY)
    ap.add_argument(
        "--models", nargs="+", default=list(BASELINE_MODELS),
        choices=list(BASELINE_MODELS),
    )
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--max-folds", type=int, default=None)
    ap.add_argument("--start-fold", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument(
        "--compare-only", action="store_true",
        help="Only rebuild comparison CSV/MD from existing baseline summaries",
    )
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()

    run_baselines(
        models=list(args.models),
        config_path=args.config,
        baseline_root=args.baseline_root,
        ours_summary_path=args.ours_summary,
        extra_sets=list(args.set),
        max_folds=args.max_folds,
        start_fold=args.start_fold,
        resume=args.resume,
        skip_existing=args.skip_existing,
        compare_only=args.compare_only,
        epochs=args.epochs,
        device=args.device,
    )


if __name__ == "__main__":
    main()
