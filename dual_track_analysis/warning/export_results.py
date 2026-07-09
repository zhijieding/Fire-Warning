#!/usr/bin/env python3
"""汇总预警轨 5-fold 主指标（train / val / test）到 dual_track_analysis/outputs/warning/summary/。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluate import WARNING_PRIMARY_METRIC_KEYS, _primary_warning_row

_SPLITS = ("train", "val", "test")


def _load_cached_primary(fold_dir: Path) -> dict[str, dict[str, float]] | None:
    """Read per-split primary metrics written by evaluate (incl. val mean_lead_time)."""
    primary_path = fold_dir / "eval" / "warning_primary_metrics.json"
    if not primary_path.is_file():
        return None
    with open(primary_path, encoding="utf-8") as f:
        data = json.load(f)
    cached = data.get("splits") or {}
    return cached if cached else None


def _merge_lead_from_cache(
    lead_sum: dict[str, float] | None,
    cached: dict[str, dict[str, float]] | None,
    split: str,
) -> dict[str, float] | None:
    """Fill mean_lead_time from warning_primary_metrics when eval_summary lacks it."""
    ml = float("nan")
    if lead_sum is not None:
        ml = float(lead_sum.get("mean_lead_time", float("nan")))
    if not np.isfinite(ml) and cached and split in cached:
        ml = float(cached[split].get("mean_lead_time", float("nan")))
    if not np.isfinite(ml):
        return lead_sum
    merged = dict(lead_sum) if lead_sum else {}
    merged["mean_lead_time"] = ml
    return merged


def _load_fold_primary(fold_dir: Path) -> dict[str, dict[str, float]] | None:
    """Read per-split primary metrics from one fold eval dir."""
    cached = _load_cached_primary(fold_dir)
    summary_path = fold_dir / "eval" / "eval_summary.json"
    if summary_path.is_file():
        with open(summary_path, encoding="utf-8") as f:
            summ = json.load(f)
        exp = summ.get("warning_experiment_level") or {}
        lead = summ.get("lead_time") or {}
        splits: dict[str, dict[str, float]] = {}
        for sp in _SPLITS:
            if sp not in exp:
                continue
            lead_sum = _merge_lead_from_cache(lead.get(sp), cached, sp)
            splits[sp] = _primary_warning_row(exp[sp], lead_sum)
        if splits:
            return splits
    return cached


def collect_split_primary(kfold_root: Path) -> dict[str, dict[str, float]]:
    """Per-split mean of primary metrics across folds."""
    fold_dirs = sorted(p for p in kfold_root.glob("fold_*") if p.is_dir())
    acc: dict[str, dict[str, list[float]]] = {
        sp: {m: [] for m in WARNING_PRIMARY_METRIC_KEYS} for sp in _SPLITS
    }
    for fold_dir in fold_dirs:
        splits = _load_fold_primary(fold_dir)
        if not splits:
            continue
        for sp, row in splits.items():
            if sp not in acc:
                continue
            for m in WARNING_PRIMARY_METRIC_KEYS:
                v = row.get(m, float("nan"))
                if np.isfinite(v):
                    acc[sp][m].append(float(v))

    out: dict[str, dict[str, float]] = {}
    for sp, metrics in acc.items():
        out[sp] = {}
        for m, vals in metrics.items():
            out[sp][m] = float(np.mean(vals)) if vals else float("nan")
    return out


def _fmt(mean: float, std: float) -> str:
    if not np.isfinite(mean):
        return "—"
    if not np.isfinite(std) or std == 0:
        return f"{mean:.4f}"
    return f"{mean:.4f} ± {std:.4f}"


def export_warning_results(kfold_root: Path, out_dir: Path) -> None:
    kfold_root = kfold_root.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_dirs = sorted(kfold_root.glob("fold_*"))
    if not fold_dirs:
        print(f"未找到 fold 目录: {kfold_root}")
        return

    by_fold_rows: list[dict] = []
    for fold_dir in fold_dirs:
        splits = _load_fold_primary(fold_dir)
        if not splits:
            print(f"  [skip] {fold_dir.name}: 无 warning_primary_metrics")
            continue
        for sp in _SPLITS:
            if sp not in splits:
                continue
            by_fold_rows.append({
                "fold": fold_dir.name,
                "split": sp,
                **{m: splits[sp].get(m, np.nan) for m in WARNING_PRIMARY_METRIC_KEYS},
            })

    if not by_fold_rows:
        print("无可用 fold 主指标，请先完成训练与 evaluate")
        return

    by_fold_df = pd.DataFrame(by_fold_rows)
    by_fold_df.to_csv(out_dir / "warning_primary_by_fold.csv", index=False)

    # 各 split 跨 fold 汇总（test 为论文主报告；train/val 供过拟合诊断）
    summary_rows: list[dict] = []
    for sp in _SPLITS:
        sub = by_fold_df[by_fold_df["split"] == sp]
        if sub.empty:
            continue
        for m in WARNING_PRIMARY_METRIC_KEYS:
            vals = sub[m].dropna()
            summary_rows.append({
                "split": sp,
                "metric": m,
                "n_folds": int(len(vals)),
                "mean": float(vals.mean()) if len(vals) else float("nan"),
                "std": float(vals.std()) if len(vals) > 1 else 0.0,
                "formatted": _fmt(
                    float(vals.mean()) if len(vals) else float("nan"),
                    float(vals.std()) if len(vals) > 1 else 0.0,
                ),
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "warning_primary_summary.csv", index=False)

    # 宽表：split × 主指标（mean ± std）
    wide_rows: list[dict] = []
    for sp in _SPLITS:
        row: dict = {"split": sp}
        for m in WARNING_PRIMARY_METRIC_KEYS:
            block = summary_df[(summary_df["split"] == sp) & (summary_df["metric"] == m)]
            if block.empty:
                row[m] = "—"
                continue
            row[m] = block.iloc[0]["formatted"]
        wide_rows.append(row)
    wide_df = pd.DataFrame(wide_rows)
    wide_df.to_csv(out_dir / "warning_primary_wide.csv", index=False)

    _metric_labels = {
        "recall": "Recall",
        "mean_lead_time": "mean_lead_time (s)",
        "f1": "F1",
        "mcc": "MCC",
        "precision": "Precision",
        "pr_auc": "Pr-AUC",
    }
    md_cols = [_metric_labels.get(m, m) for m in WARNING_PRIMARY_METRIC_KEYS]
    try:
        kfold_src = kfold_root.relative_to(_ROOT)
    except ValueError:
        kfold_src = kfold_root
    md = [
        "# 预警主指标（实验级 macro，train / val / test）",
        "",
        f"来源: `{kfold_src}`",
        "",
        "主指标: Recall, mean_lead_time (s), F1, MCC, Precision, Pr-AUC",
        "",
        "## 跨 fold 汇总（mean ± std）",
        "",
        "| split | " + " | ".join(md_cols) + " |",
        "|---|" + "|".join(["---"] * len(md_cols)) + "|",
    ]
    for _, r in wide_df.iterrows():
        cells = " | ".join(str(r[m]) for m in WARNING_PRIMARY_METRIC_KEYS)
        md.append(f"| {r['split']} | {cells} |")
    md += [
        "",
        "## 各 fold 明细",
        "",
        "见 `warning_primary_by_fold.csv`",
        "",
        "## 阈值敏感性",
        "",
        "见 `sensitivity/threshold/threshold_sensitivity_report.md`（test pooled；θ 来自各折 val）",
        "完整三类敏感性见 `sensitivity/README.md`（阈值 / 超参 / history_window）。",
    ]
    (out_dir / "warning_primary_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    from dual_track_analysis.warning.export_sensitivity import export_threshold_sensitivity
    sens_dir = out_dir / "sensitivity"
    export_threshold_sensitivity(kfold_root, sens_dir)

    print(f"已导出 → {out_dir}")
    print(wide_df.to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--kfold-root", type=Path,
        default=_ROOT / "dual_track_analysis/outputs/warning/kfold_regularized",
    )
    ap.add_argument(
        "--out-dir", type=Path,
        default=_ROOT / "dual_track_analysis/outputs/warning/summary",
    )
    args = ap.parse_args()
    export_warning_results(args.kfold_root, args.out_dir)


if __name__ == "__main__":
    main()
