#!/usr/bin/env python3
"""
仅重跑 evaluate（不重训），刷新各 fold 的 eval/ 与 5-fold 汇总。

从各 fold 的 run_config.json 恢复数据划分与训练配置；eval 阈值策略默认来自 config.py。

Usage (from Fire/):
    python -m dual_track_analysis.warning.reeval_kfold
    python -m dual_track_analysis.warning.reeval_kfold --fold fold_0_test5_1
    python -m dual_track_analysis.warning.reeval_kfold --set eval_high_recall_min_recall=0.85
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import Config, parse_config_value
from evaluate import evaluate
from run_kfold import main as run_kfold_main


def _fold_dirs(kfold_root: Path, fold_filter: str | None) -> list[Path]:
    dirs = sorted(p for p in kfold_root.glob("fold_*") if p.is_dir())
    if fold_filter:
        dirs = [p for p in dirs if p.name == fold_filter or p.name.endswith(fold_filter)]
    return dirs


def _cfg_for_fold(
    fold_dir: Path,
    overrides: dict[str, object],
) -> Config:
    run_cfg_path = fold_dir / "run_config.json"
    if not run_cfg_path.is_file():
        raise FileNotFoundError(f"缺少 run_config.json: {fold_dir}")

    cfg = Config()
    with open(run_cfg_path, encoding="utf-8") as f:
        run_cfg = json.load(f).get("config") or {}
    cfg.apply_overrides(run_cfg)
    cfg.apply_overrides(overrides)

    cfg.output_dir = str(fold_dir.resolve())
    model_path = fold_dir / "best_model.pt"
    if not model_path.is_file():
        raise FileNotFoundError(f"缺少 best_model.pt: {fold_dir}")
    return cfg


def _reaggregate_kfold_summary(kfold_root: Path, *, n_folds: int) -> None:
    """刷新 5fold_results.csv / 5fold_summary.json（--resume，不重训）。"""
    cfg = Config()
    out_dir = (
        kfold_root.relative_to(_ROOT)
        if kfold_root.is_relative_to(_ROOT)
        else kfold_root
    )
    cfg.output_dir = str(out_dir)

    print("\n═══ Re-aggregate k-fold summary ═══")

    prev_argv = sys.argv
    try:
        sys.argv = [
            "run_kfold.py",
            "--resume",
            "--output-dir", str(out_dir),
            "--kfold", str(n_folds),
        ]
        run_kfold_main(cfg)
    finally:
        sys.argv = prev_argv


def reeval_kfold(
    kfold_root: Path,
    *,
    overrides: dict[str, object],
    fold_filter: str | None = None,
    skip_export: bool = False,
) -> None:
    kfold_root = kfold_root.resolve()
    fold_dirs = _fold_dirs(kfold_root, fold_filter)
    if not fold_dirs:
        raise SystemExit(f"未找到 fold 目录: {kfold_root}")

    print(f"═══ Re-evaluate only ({len(fold_dirs)} folds) ═══")
    print(f"  kfold root : {kfold_root}")
    if overrides:
        print(f"  overrides  : {overrides}")

    for fold_dir in fold_dirs:
        print(f"\n{'─' * 60}")
        print(f"  {fold_dir.name}")
        print(f"{'─' * 60}")
        cfg = _cfg_for_fold(fold_dir, overrides)
        evaluate(cfg)

    if skip_export:
        return

    from dual_track_analysis.warning.export_results import export_warning_results

    summary_dir = _ROOT / "dual_track_analysis/outputs/warning/summary"
    export_warning_results(kfold_root, summary_dir)

    _reaggregate_kfold_summary(kfold_root, n_folds=len(fold_dirs))


def main() -> None:
    ap = argparse.ArgumentParser(description="Re-run evaluate for warning k-fold (no retrain)")
    ap.add_argument(
        "--kfold-root", type=Path,
        default=_ROOT / "dual_track_analysis/outputs/warning/kfold_regularized",
    )
    ap.add_argument(
        "--fold", type=str, default=None,
        help="Only re-eval one fold dir name, e.g. fold_0_test5_1",
    )
    ap.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE",
        help="Extra Config overrides for evaluate",
    )
    ap.add_argument(
        "--skip-export", action="store_true",
        help="Skip summary export after re-eval",
    )
    args = ap.parse_args()

    overrides: dict[str, object] = {}
    for item in args.set:
        if "=" not in item:
            raise SystemExit(f"--set 需要 KEY=VALUE，收到: {item!r}")
        key, raw = item.split("=", 1)
        overrides[key.strip()] = parse_config_value(raw)

    reeval_kfold(
        args.kfold_root.resolve(),
        overrides=overrides,
        fold_filter=args.fold,
        skip_export=args.skip_export,
    )


if __name__ == "__main__":
    main()
