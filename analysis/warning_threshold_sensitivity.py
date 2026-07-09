"""
预警阈值敏感性分析（主指标：Recall / lead time / F1 / MCC / Precision / Pr-AUC）。

- 单折：evaluate 阶段对 train/val/test 扫描阈值（实验级 macro）
- K-fold：汇总 test 集 pooled 曲线 + 标注各折 val 选出的 operating threshold

注意：operating threshold 仅在 val 上选取；test 扫描仅用于敏感性/稳健性报告。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluate import (
    WARNING_PRIMARY_METRIC_KEYS,
    _leadtime_eligible_meta,
    _leadtime_summary,
    _primary_warning_row,
    experiment_leadtime,
    experiment_warning_metrics,
)

THRESHOLD_GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)
PRIMARY_COLS = ("threshold", *WARNING_PRIMARY_METRIC_KEYS)


def _meta_with_threshold(meta_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    if meta_df.empty:
        return meta_df
    m = meta_df.copy()
    m["y_pred"] = (m["y_prob"] >= threshold).astype(int)
    return m


def scan_primary_sensitivity(
    meta_df: pd.DataFrame,
    *,
    grid: Iterable[float] | None = None,
    lead_eligible_only: bool = False,
) -> pd.DataFrame:
    """实验级 macro 主指标 × 阈值网格。"""
    if meta_df.empty:
        return pd.DataFrame(columns=list(PRIMARY_COLS))

    grid_arr = np.asarray(list(grid) if grid is not None else THRESHOLD_GRID, dtype=float)
    lead_src = _leadtime_eligible_meta(meta_df) if lead_eligible_only else meta_df

    rows: list[dict] = []
    for th in grid_arr:
        exp = experiment_warning_metrics(meta_df, float(th))
        if lead_src.empty:
            lead_sum = None
        else:
            lead_df = experiment_leadtime(_meta_with_threshold(lead_src, float(th)), float(th))
            lead_sum = _leadtime_summary(lead_df)
        rows.append({
            "threshold": float(th),
            **_primary_warning_row(exp, lead_sum),
        })
    return pd.DataFrame(rows)


def write_fold_threshold_sensitivity(
    eval_dir: Path,
    *,
    meta_by_split: dict[str, pd.DataFrame],
    operating_threshold: float,
) -> Path:
    """写入单折 train/val/test 敏感性 CSV。"""
    eval_dir = Path(eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)

    parts: list[pd.DataFrame] = []
    for split, meta in meta_by_split.items():
        if meta is None or meta.empty:
            continue
        lead_only = split == "train"
        scan = scan_primary_sensitivity(meta, lead_eligible_only=lead_only)
        if scan.empty:
            continue
        scan = scan.copy()
        scan.insert(0, "split", split)
        if np.isfinite(operating_threshold) and len(scan):
            nearest_idx = int((scan["threshold"] - operating_threshold).abs().argmin())
            scan["is_operating"] = False
            scan.loc[scan.index[nearest_idx], "is_operating"] = True
        else:
            scan["is_operating"] = False
        scan.to_csv(eval_dir / f"threshold_sensitivity_{split}.csv", index=False)
        parts.append(scan)

    if not parts:
        return eval_dir / "threshold_sensitivity_primary.csv"

    combined = pd.concat(parts, ignore_index=True)
    out_path = eval_dir / "threshold_sensitivity_primary.csv"
    combined.to_csv(out_path, index=False)
    return out_path


def _nearest_threshold_row(scan: pd.DataFrame, operating_threshold: float) -> pd.Series | None:
    if scan.empty or not np.isfinite(operating_threshold):
        return None
    idx = (scan["threshold"] - operating_threshold).abs().argmin()
    return scan.iloc[int(idx)]


def robustness_summary(
    scan: pd.DataFrame,
    operating_threshold: float,
    *,
    bands: tuple[float, ...] = (0.05, 0.10),
) -> pd.DataFrame:
    """在 operating 阈值及其邻域内汇总主指标范围（稳健性）。"""
    if scan.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    th = float(operating_threshold)
    op_row = _nearest_threshold_row(scan, th)
    op_dict = op_row.to_dict() if op_row is not None else {}

    for band in bands:
        mask = (scan["threshold"] >= th - band) & (scan["threshold"] <= th + band)
        sub = scan.loc[mask]
        if sub.empty:
            continue
        row: dict = {
            "operating_threshold": th,
            "band": band,
            "n_thresholds": int(len(sub)),
        }
        for m in WARNING_PRIMARY_METRIC_KEYS:
            vals = sub[m].dropna()
            row[f"{m}_at_op"] = op_dict.get(m, np.nan)
            row[f"{m}_min"] = float(vals.min()) if len(vals) else float("nan")
            row[f"{m}_max"] = float(vals.max()) if len(vals) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def _read_fold_operating_threshold(fold_dir: Path) -> float:
    for rel in (
        "eval/warning_primary_metrics.json",
        "eval/eval_summary.json",
    ):
        p = fold_dir / rel
        if not p.is_file():
            continue
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        if "operating_threshold" in data:
            return float(data["operating_threshold"])
    return float("nan")


def _plot_primary_sensitivity(
    scan: pd.DataFrame,
    out_png: Path,
    operating_threshold: float,
    *,
    title: str = "Test threshold sensitivity (primary metrics)",
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [skip] matplotlib 不可用，跳过敏感性图。")
        return

    fig, axes = plt.subplots(2, 1, figsize=(6.2, 6.8), sharex=True)

    th = scan["threshold"].values
    ax = axes[0]
    ax.plot(th, scan["recall"], label="Recall", color="#c0392b")
    ax.plot(th, scan["f1"], label="F1", color="#27ae60")
    ax.plot(th, scan["mcc"], label="MCC", color="#1f4e79")
    ax.plot(th, scan["precision"], label="Precision", color="#8e44ad")
    if np.isfinite(operating_threshold):
        ax.axvline(operating_threshold, color="#888888", linestyle="--", linewidth=1.0,
                   label=rf"val $\theta$={operating_threshold:.2f}")
    ax.set_ylabel("Classification")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="center right", fontsize=8)
    ax.set_title(title)

    ax2 = axes[1]
    ax2.plot(th, scan["mean_lead_time"], label="Mean lead (s)", color="#c0392b")
    if np.isfinite(operating_threshold):
        ax2.axvline(operating_threshold, color="#888888", linestyle="--", linewidth=1.0)
    ax2.set_xlabel("Warning threshold")
    ax2.set_ylabel("Mean lead time (s)")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="center right", fontsize=8)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_kfold_sensitivity(
    kfold_root: Path,
    out_dir: Path,
    *,
    bootstrap_from_eval: bool = True,
) -> None:
    """K-fold 汇总：pooled test 敏感性 + 稳健性表 + 图。"""
    from analysis.plot_pr_threshold_sensitivity import (
        load_merged_predictions,
        load_fold_prediction_files,
        materialize_predictions_from_eval,
        threshold_sensitivity_table,
    )

    kfold_root = Path(kfold_root).resolve()
    out_dir = Path(out_dir).resolve()
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_dir = kfold_root / "outputs" / "predictions" / "Ours"
    if not load_fold_prediction_files(pred_dir) and bootstrap_from_eval:
        print("[sensitivity] 从 eval/test_predictions.csv 生成预测文件 …")
        materialize_predictions_from_eval(kfold_root)

    # 各折 val operating threshold
    fold_dirs = sorted(kfold_root.glob("fold_*"))
    op_rows = []
    for fd in fold_dirs:
        op = _read_fold_operating_threshold(fd)
        op_rows.append({"fold": fd.name, "operating_threshold": op})
    op_df = pd.DataFrame(op_rows)
    op_df.to_csv(out_dir / "operating_thresholds_by_fold.csv", index=False)
    op_mean = float(op_df["operating_threshold"].dropna().mean()) if len(op_df) else float("nan")

    # 优先：各折 test 敏感性（实验级，已含 lead time）
    fold_test_scans: list[pd.DataFrame] = []
    for fd in fold_dirs:
        p = fd / "eval" / "threshold_sensitivity_test.csv"
        if p.is_file():
            df = pd.read_csv(p)
            df.insert(0, "fold", fd.name)
            fold_test_scans.append(df)
    if fold_test_scans:
        pd.concat(fold_test_scans, ignore_index=True).to_csv(
            out_dir / "threshold_sensitivity_test_by_fold.csv", index=False,
        )

    # pooled test（窗口级，与 PR 分析一致）
    try:
        merged = load_merged_predictions(pred_dir)
        pooled = threshold_sensitivity_table(merged)
        primary_pooled = pooled[list(PRIMARY_COLS)].copy()
        primary_pooled.to_csv(out_dir / "threshold_sensitivity_test_pooled.csv", index=False)

        rob = robustness_summary(primary_pooled, op_mean)
        rob.to_csv(out_dir / "threshold_sensitivity_robustness.csv", index=False)

        _plot_primary_sensitivity(
            primary_pooled,
            fig_dir / "threshold_sensitivity_primary.png",
            op_mean,
            title="Test pooled sensitivity (primary metrics; θ from val)",
        )

        # Markdown 报告
        md = [
            "# 阈值敏感性分析（test pooled）",
            "",
            "**阈值选取**：各折在 **val** 上选 operating threshold，test 仅做敏感性扫描。",
            "",
            f"- 各折 val θ：见 `operating_thresholds_by_fold.csv`",
            f"- pooled 图标注均值 θ ≈ **{op_mean:.3f}**（各折 val 选阈的平均，仅作参考线）",
            "",
            "## operating θ 邻域稳健性（test pooled）",
            "",
        ]
        if not rob.empty:
            md.append(
                "| band | Recall (at θ) | Recall [min,max] | F1 (at θ) | F1 [min,max] | "
                "Precision (at θ) | Precision [min,max] |"
            )
            md.append("|---:|---:|---:|---:|---:|---:|---:|")
            for _, r in rob.iterrows():
                md.append(
                    f"| ±{r['band']:.2f} | {r.get('recall_at_op', float('nan')):.3f} | "
                    f"[{r.get('recall_min', float('nan')):.3f}, {r.get('recall_max', float('nan')):.3f}] | "
                    f"{r.get('f1_at_op', float('nan')):.3f} | "
                    f"[{r.get('f1_min', float('nan')):.3f}, {r.get('f1_max', float('nan')):.3f}] | "
                    f"{r.get('precision_at_op', float('nan')):.3f} | "
                    f"[{r.get('precision_min', float('nan')):.3f}, {r.get('precision_max', float('nan')):.3f}] |"
                )
        md += [
            "",
            "## 文件",
            "",
            "- `threshold_sensitivity_test_pooled.csv` — test pooled 主指标 × θ",
            "- `threshold_sensitivity_robustness.csv` — θ 邻域 min/max",
            "- `figures/threshold_sensitivity_primary.png` — 敏感性曲线",
        ]
        (out_dir / "threshold_sensitivity_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
        print(f"[sensitivity] pooled test 扫描 → {out_dir}")
    except FileNotFoundError as e:
        print(f"[sensitivity] 跳过 pooled 分析: {e}")

    # 各折 test 实验级：汇总 operating 点跨 fold mean±std
    if fold_test_scans:
        long_df = pd.concat(fold_test_scans, ignore_index=True)
        pick_rows = []
        for _, fr in op_df.iterrows():
            if not np.isfinite(fr["operating_threshold"]):
                continue
            sub = long_df[long_df["fold"] == fr["fold"]]
            nearest = _nearest_threshold_row(sub, float(fr["operating_threshold"]))
            if nearest is not None:
                pick_rows.append(nearest)
        if pick_rows:
            at_op = pd.DataFrame(pick_rows)
            rows = []
            for m in WARNING_PRIMARY_METRIC_KEYS:
                if m not in at_op.columns:
                    v = pd.Series(dtype=float)
                else:
                    v = at_op[m].dropna()
                rows.append({
                    "metric": m,
                    "mean": float(v.mean()) if len(v) else float("nan"),
                    "std": float(v.std()) if len(v) > 1 else 0.0,
                    "n_folds": int(len(v)),
                })
            pd.DataFrame(rows).to_csv(out_dir / "primary_at_operating_threshold.csv", index=False)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="K-fold warning threshold sensitivity (primary metrics)")
    ap.add_argument("--kfold-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--no-bootstrap", action="store_true")
    args = ap.parse_args()
    root = args.kfold_root.resolve()
    out = args.out_dir.resolve() if args.out_dir else root / "outputs" / "sensitivity"
    run_kfold_sensitivity(root, out, bootstrap_from_eval=not args.no_bootstrap)


if __name__ == "__main__":
    main()
