"""LBCA 预警轨消融实验：与 kfold 同协议，输出到 outputs/warning/ablation/."""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from evaluate import WARNING_PRIMARY_METRIC_KEYS
from dual_track_analysis.warning.baseline_runner import (
    EXP_METRICS,
    PRIMARY_METRICS,
    WARNING_TRACK_SETS,
    _fmt,
    _pull_metric,
    load_warning_baseline_cfg,
)

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_STUDY = Path(__file__).resolve().parent / "ablation_study_regularized.json"
_DEFAULT_OUT = _ROOT / "dual_track_analysis/outputs/warning/ablation_regularized"

_REEVAL_COPY_FILES = ("best_model.pt", "run_config.json", "scaler.json", "train_history.csv")


def load_ablation_study(study_path: Path) -> dict:
    with open(study_path, encoding="utf-8") as f:
        study = json.load(f)
    if "ablations" not in study:
        raise SystemExit(f"ablation study 缺少 ablations: {study_path}")
    return study


def _resolve_under_root(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else _ROOT / p


def _load_summary(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _run_kfold_train(
    overrides: dict[str, object],
    *,
    config_path: Path | None,
    output_dir: Path,
    resume: bool,
    max_folds: int | None,
    start_fold: int | None,
    extra_sets: list[str],
) -> None:
    from run_kfold import main as run_kfold_main

    cfg, _ = load_warning_baseline_cfg(config_path, extra_sets)
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise AttributeError(f"unknown Config field '{key}'")
        setattr(cfg, key, value)

    output_dir.mkdir(parents=True, exist_ok=True)
    rel_out = (
        output_dir.relative_to(_ROOT)
        if output_dir.is_relative_to(_ROOT)
        else output_dir
    )
    cfg.output_dir = str(rel_out)

    argv = ["run_kfold.py", "--output-dir", str(rel_out)]
    if resume:
        argv.append("--resume")
    if max_folds is not None:
        argv.extend(["--max-folds", str(max_folds)])
    if start_fold is not None:
        argv.extend(["--start-fold", str(start_fold)])

    prev_argv = sys.argv
    try:
        sys.argv = argv
        run_kfold_main(cfg)
    finally:
        sys.argv = prev_argv


def _copy_fold_for_reeval(src_fold: Path, dst_fold: Path) -> None:
    dst_fold.mkdir(parents=True, exist_ok=True)
    for name in _REEVAL_COPY_FILES:
        src = src_fold / name
        if src.is_file():
            shutil.copy2(src, dst_fold / name)
    if not (dst_fold / "best_model.pt").is_file():
        raise FileNotFoundError(f"reeval 缺少 checkpoint: {dst_fold / 'best_model.pt'}")


def _reeval_ablation(
    source_root: Path,
    output_dir: Path,
    *,
    overrides: dict[str, object],
) -> None:
    from dual_track_analysis.warning.reeval_kfold import (
        _reaggregate_kfold_summary,
        reeval_kfold,
    )

    source_root = source_root.resolve()
    output_dir = output_dir.resolve()
    fold_dirs = sorted(p for p in source_root.glob("fold_*") if p.is_dir())
    if not fold_dirs:
        raise FileNotFoundError(f"source 无 fold 目录: {source_root}")

    for src in fold_dirs:
        dst = output_dir / src.name
        if not (dst / "best_model.pt").is_file():
            print(f"  copy {src.name} → {dst.relative_to(_ROOT)}")
            _copy_fold_for_reeval(src, dst)

    reeval_kfold(output_dir, overrides=overrides, skip_export=True)
    _reaggregate_kfold_summary(output_dir, n_folds=len(fold_dirs))


def _write_ablation_marker(
    output_dir: Path,
    *,
    ablation_id: str,
    mode: str,
    overrides: dict[str, object],
    display_name: str,
    description: str = "",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = _load_summary(output_dir / "5fold_summary.json")
    marker = {
        "ablation_id": ablation_id,
        "display_name": display_name,
        "mode": mode,
        "overrides": overrides,
        "description": description,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "test_primary": {
            m: (summary or {}).get(m, {}).get("mean")
            for m in WARNING_PRIMARY_METRIC_KEYS
        },
    }
    with open(output_dir / "ablation_run.json", "w", encoding="utf-8") as f:
        json.dump(marker, f, indent=2, ensure_ascii=False, default=str)


def run_one_ablation(
    spec: dict,
    *,
    study: dict,
    config_path: Path | None,
    ablation_root: Path,
    source_kfold: Path,
    resume: bool,
    skip_existing: bool,
    max_folds: int | None,
    start_fold: int | None,
    extra_sets: list[str],
) -> Path | None:
    ablation_id = str(spec["id"])
    mode = str(spec.get("mode", "train"))
    display_name = str(spec.get("display_name", ablation_id))
    overrides = dict(spec.get("overrides") or {})
    description = str(spec.get("description", ""))
    output_dir = ablation_root / ablation_id
    marker = output_dir / "ablation_run.json"

    if mode == "reference":
        return _resolve_under_root(study.get("base_kfold_root", source_kfold))

    if skip_existing and marker.is_file() and (output_dir / "5fold_summary.json").is_file():
        print(f"  skip existing: {ablation_id}")
        return output_dir

    print(f"\n{'=' * 60}")
    print(f"  Ablation: {display_name}  [{mode}]")
    print(f"  overrides: {overrides}")
    print(f"  output   : {output_dir.relative_to(_ROOT)}")
    print(f"{'=' * 60}")

    if mode == "train":
        _run_kfold_train(
            overrides,
            config_path=config_path,
            output_dir=output_dir,
            resume=resume,
            max_folds=max_folds,
            start_fold=start_fold,
            extra_sets=extra_sets,
        )
    elif mode == "reeval":
        src = _resolve_under_root(spec.get("source_kfold", study.get("base_kfold_root", source_kfold)))
        _reeval_ablation(
            src,
            output_dir,
            overrides=overrides,
        )
    else:
        raise ValueError(f"unknown ablation mode: {mode!r}")

    _write_ablation_marker(
        output_dir,
        ablation_id=ablation_id,
        mode=mode,
        overrides=overrides,
        display_name=display_name,
        description=description,
    )
    return output_dir


def run_ablation_study(
    *,
    study_path: Path,
    config_path: Path | None,
    ablation_root: Path,
    only: list[str] | None = None,
    resume: bool = False,
    skip_existing: bool = False,
    dry_run: bool = False,
    max_folds: int | None = None,
    start_fold: int | None = None,
    extra_sets: list[str] | None = None,
) -> pd.DataFrame:
    study = load_ablation_study(study_path)
    source_kfold = _resolve_under_root(study.get("base_kfold_root", "dual_track_analysis/outputs/warning/kfold"))
    ablation_root.mkdir(parents=True, exist_ok=True)

    specs: list[dict] = []
    ref = study.get("reference")
    if ref:
        specs.append(ref)
    specs.extend(study["ablations"])

    if only:
        allow = set(only)
        specs = [s for s in specs if s.get("id") in allow or s.get("mode") == "reference"]

    print("═══ LBCA Ablation Study ═══")
    print(f"  study  : {study_path}")
    print(f"  base   : {source_kfold}")
    print(f"  output : {ablation_root}")
    print(f"  runs   : {len([s for s in specs if s.get('mode') != 'reference'])}")

    if dry_run:
        for spec in specs:
            mode = spec.get("mode", "train")
            aid = spec.get("id", "?")
            print(f"  [{mode:8s}] {aid:20s}  {spec.get('overrides', {})}")
        return pd.DataFrame()

    extra = list(extra_sets or [])
    for spec in specs:
        if spec.get("mode") == "reference":
            continue
        run_one_ablation(
            spec,
            study=study,
            config_path=config_path,
            ablation_root=ablation_root,
            source_kfold=source_kfold,
            resume=resume,
            skip_existing=skip_existing,
            max_folds=max_folds,
            start_fold=start_fold,
            extra_sets=extra,
        )

    df = export_ablation_comparison(
        ablation_root,
        study_path=study_path,
        write_files=True,
    )
    return df


def export_ablation_comparison(
    ablation_root: Path,
    *,
    study_path: Path | None = None,
    write_files: bool = True,
) -> pd.DataFrame:
    study = load_ablation_study(study_path or _DEFAULT_STUDY)
    source_kfold = _resolve_under_root(study.get("base_kfold_root", "dual_track_analysis/outputs/warning/kfold"))

    rows: list[dict] = []

    def _append_row(aid: str, display_name: str, summary: dict | None, mode: str) -> None:
        row: dict[str, object] = {
            "ablation_id": aid,
            "display_name": display_name,
            "mode": mode,
        }
        for m in EXP_METRICS:
            mean, std = _pull_metric(summary, m, pooled=False)
            row[f"{m}_mean"] = mean
            row[f"{m}_std"] = std
            if m == "mean_lead_time":
                row[m] = _fmt(mean, std, digits=2)
            else:
                row[m] = _fmt(mean, std, digits=4)
        rows.append(row)

    ref = study.get("reference") or {}
    ref_summary = _load_summary(source_kfold / "5fold_summary.json")
    _append_row(
        str(ref.get("id", "full")),
        str(ref.get("display_name", "Full (Ours)")),
        ref_summary,
        "reference",
    )

    for spec in study["ablations"]:
        aid = str(spec["id"])
        out_dir = ablation_root / aid
        summary = _load_summary(out_dir / "5fold_summary.json")
        _append_row(
            aid,
            str(spec.get("display_name", aid)),
            summary,
            str(spec.get("mode", "train")),
        )

    df = pd.DataFrame(rows)
    if not write_files:
        return df

    ablation_root.mkdir(parents=True, exist_ok=True)
    df.to_csv(ablation_root / "ablation_comparison.csv", index=False)

    wide_rows: list[dict] = []
    for _, r in df.iterrows():
        wide_rows.append({
            "model": r["display_name"],
            "ablation_id": r["ablation_id"],
            "mode": r["mode"],
            **{m: r[m] for m in PRIMARY_METRICS if m in r},
        })
    wide = pd.DataFrame(wide_rows)
    wide.to_csv(ablation_root / "ablation_comparison_wide.csv", index=False)

    md = [
        "# LBCA 预警轨消融 (5-Fold CV, test)",
        "",
        f"基线（Full）来自 `{source_kfold.relative_to(_ROOT) if source_kfold.is_relative_to(_ROOT) else source_kfold}`。",
        "训练消融与 Full 同协议（post_fire=false）；阈值消融为 reeval（同一 checkpoint）。",
        "",
        "## 主指标（experiment-level macro, mean ± std）",
        "",
        "| Model | mode | " + " | ".join(PRIMARY_METRICS) + " |",
        "|---|---|" + "|".join(["---"] * len(PRIMARY_METRICS)) + "|",
    ]
    for _, r in wide.iterrows():
        cells = [str(r["model"]), str(r["mode"])] + [str(r[m]) for m in PRIMARY_METRICS]
        md.append("| " + " | ".join(cells) + " |")
    md += ["", "完整表见 `ablation_comparison.csv`。"]
    (ablation_root / "ablation_comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    meta = {
        "study_path": str(study_path or _DEFAULT_STUDY),
        "base_kfold_root": str(source_kfold),
        "primary_metrics": list(PRIMARY_METRICS),
        "rows": rows,
    }
    with open(ablation_root / "ablation_comparison.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n消融对比表 → {ablation_root / 'ablation_comparison_wide.csv'}")
    return df
