#!/usr/bin/env python3
"""
Precision-Recall curve and warning-threshold sensitivity (test-set analysis only).

Reads merged K-fold exports under ``<kfold_root>/outputs/predictions/``,
or bootstraps from ``fold_*/eval/test_predictions.csv`` when missing.

Usage (from Fire_prediction/)::

    python -m analysis.plot_pr_threshold_sensitivity \\
        --kfold-root ./trimodal_5fold_tuned_nopretrain

    python -m analysis.plot_pr_threshold_sensitivity \\
        --kfold-root ./trimodal_5fold_tuned_nopretrain --bootstrap-from-eval

Note: threshold sweeps on the test set are for analysis / visualization only;
do not use them to pick the deployment threshold.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
)

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

REQUIRED_COLS = ("fold_id", "exp_id", "time_idx", "y_true", "y_prob")
EPS = 1e-12
DEFAULT_THETA = 0.5
THRESHOLDS = np.arange(0.05, 0.96, 0.01)


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "sans-serif"]
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["axes.titlesize"] = 11
    plt.rcParams["legend.fontsize"] = 9
    plt.rcParams["lines.linewidth"] = 1.6
    return plt


def resolve_output_dirs(kfold_root: Path) -> dict[str, Path]:
    root = Path(kfold_root)
    return {
        "root": root,
        "predictions": root / "outputs" / "predictions",
        "figures": root / "outputs" / "figures",
        "analysis": root / "outputs" / "analysis",
    }


def _fold_index_from_dir(name: str) -> int | None:
    m = re.match(r"fold_(\d+)_", name)
    return int(m.group(1)) if m else None


def materialize_predictions_from_eval(kfold_root: Path, model_subdir: str = "Ours") -> list[Path]:
    from analysis.warn_prediction_io import materialize_predictions_from_eval as _mat

    return _mat(Path(kfold_root), model_subdir)


def load_fold_prediction_files(predictions_dir: Path) -> list[Path]:
    from analysis.warn_prediction_io import load_fold_prediction_files as _load

    paths = _load(predictions_dir)
    if paths:
        return paths
    # legacy flat layout under outputs/predictions/
    parent = Path(predictions_dir)
    if parent.name == "Ours" and parent.parent.is_dir():
        return _load(parent.parent)
    return []


def _validate_prediction_df(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"预测文件缺少列 {missing}。"
            "请在 evaluation 阶段保存 y_true / y_prob（evaluate.save_fold_warn_predictions），"
            "或使用 --bootstrap-from-eval 从 test_predictions.csv 生成。"
        )
    if df["y_prob"].isna().all():
        raise ValueError("y_prob 全为 NaN：需要在 evaluation 阶段保存校准后的 warning probability。")
    return df


def load_merged_predictions(
    predictions_dir: Path,
    *,
    fold_ids: Iterable[int] | None = None,
) -> pd.DataFrame:
    paths = load_fold_prediction_files(predictions_dir)
    if not paths:
        raise FileNotFoundError(
            f"未找到 {predictions_dir}/fold_*_warn_predictions.csv；"
            "请先运行 evaluate / K-fold，或加 --bootstrap-from-eval。"
        )
    frames: list[pd.DataFrame] = []
    for p in paths:
        df = pd.read_csv(p)
        if fold_ids is not None and int(df["fold_id"].iloc[0]) not in set(fold_ids):
            continue
        frames.append(_validate_prediction_df(df))
    if not frames:
        raise FileNotFoundError("没有匹配的 fold 预测文件。")
    merged = pd.concat(frames, axis=0, ignore_index=True)
    return merged.sort_values(["fold_id", "exp_id", "time_idx"]).reset_index(drop=True)


def load_single_fold_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return _validate_prediction_df(df)


def _binary_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    yt = y_true.astype(int)
    yp = y_pred.astype(int)
    tp = int(((yt == 1) & (yp == 1)).sum())
    tn = int(((yt == 0) & (yp == 0)).sum())
    fp = int(((yt == 0) & (yp == 1)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def classification_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    yt = y_true.astype(int)
    c = _binary_counts(yt, y_pred)
    tp, tn, fp, fn = c["tp"], c["tn"], c["fp"], c["fn"]

    prec = float(precision_score(yt, y_pred, zero_division=0))
    rec = float(recall_score(yt, y_pred, zero_division=0))
    f1 = float(f1_score(yt, y_pred, zero_division=0))
    acc = float(accuracy_score(yt, y_pred))

    if len(np.unique(yt)) < 2:
        mcc = float("nan")
    else:
        mcc = float(matthews_corrcoef(yt, y_pred))

    fpr = float(fp / (fp + tn + EPS))
    fnr = float(fn / (fn + tp + EPS))
    spec = float(tn / (tn + fp + EPS))

    return dict(
        precision=prec,
        recall=rec,
        f1=f1,
        mcc=mcc,
        accuracy=acc,
        fpr=fpr,
        fnr=fnr,
        specificity=spec,
    )


def _event_time_valid(v: float) -> bool:
    return bool(np.isfinite(v))


def lead_metrics_at_threshold(
    df: pd.DataFrame,
    threshold: float,
) -> dict[str, float]:
    if "event_time" not in df.columns:
        return {"success_rate": float("nan"), "mean_lead_time": float("nan")}

    successes: list[int] = []
    leads: list[float] = []

    for _, g in df.groupby("exp_id", sort=False):
        et = g["event_time"].dropna()
        if et.empty:
            continue
        event_time = float(et.iloc[0])
        if not _event_time_valid(event_time):
            continue

        g = g.sort_values("time_idx")
        alarms = g[(g["y_prob"] >= threshold) & (g["time_idx"] < event_time)]
        if len(alarms) > 0:
            first_alarm = float(alarms["time_idx"].iloc[0])
            successes.append(1)
            leads.append(event_time - first_alarm)
        else:
            successes.append(0)

    if not successes:
        return {"success_rate": float("nan"), "mean_lead_time": float("nan")}

    sr = float(np.mean(successes))
    mlt = float(np.mean(leads)) if leads else float("nan")
    return {"success_rate": sr, "mean_lead_time": mlt}


def threshold_sensitivity_table(df: pd.DataFrame) -> pd.DataFrame:
    y_true = df["y_true"].values.astype(int)
    y_prob = df["y_prob"].values.astype(float)
    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except ValueError:
        pr_auc = float("nan")
    rows: list[dict] = []
    for th in THRESHOLDS:
        cls = classification_at_threshold(y_true, y_prob, float(th))
        lead = lead_metrics_at_threshold(df, float(th))
        rows.append({"threshold": float(th), **cls, **lead, "pr_auc": pr_auc})
    return pd.DataFrame(rows)


def pr_curve_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = float(average_precision_score(y_true, y_prob))
    pr_auc = float(auc(rec, prec)) if len(rec) > 1 else float("nan")
    return {"ap": ap, "pr_auc": pr_auc, "precision": prec, "recall": rec}


def metrics_at_point(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> tuple[float, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return (
        float(precision_score(y_true, y_pred, zero_division=0)),
        float(recall_score(y_true, y_pred, zero_division=0)),
    )


def plot_pr_curve(
    df: pd.DataFrame,
    out_png: Path,
    out_pdf: Path,
    *,
    theta_default: float = DEFAULT_THETA,
) -> dict[str, float]:
    plt = _setup_matplotlib()

    y_true = df["y_true"].values.astype(int)
    y_prob = df["y_prob"].values.astype(float)
    m = pr_curve_metrics(y_true, y_prob)
    p_op, r_op = metrics_at_point(y_true, y_prob, theta_default)

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.plot(m["recall"], m["precision"], color="#1f4e79", label="Ours (test pooled)")
    ax.scatter(
        [r_op],
        [p_op],
        color="#c0392b",
        s=36,
        zorder=5,
        label=rf"$\theta$={theta_default:.1f} (P={p_op:.3f}, R={r_op:.3f})",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.text(
        0.04,
        0.06,
        f"AP = {m['ap']:.4f}\nPR-AUC = {m['pr_auc']:.4f}",
        transform=ax.transAxes,
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="#cccccc"),
    )
    ax.legend(loc="lower left", framealpha=0.9)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return {"ap": m["ap"], "pr_auc": m["pr_auc"], "precision_at_theta": p_op, "recall_at_theta": r_op}


def _add_theta_vline(ax, theta_default: float) -> None:
    ax.axvline(theta_default, color="#888888", linestyle="--", linewidth=1.0, label=rf"$\theta$={theta_default:.1f}")


def plot_threshold_sensitivity_prf(
    scan: pd.DataFrame,
    out_png: Path,
    out_pdf: Path,
    *,
    theta_default: float = DEFAULT_THETA,
) -> None:
    plt = _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    th = scan["threshold"].values
    ax.plot(th, scan["precision"], label="Precision", color="#1f4e79")
    ax.plot(th, scan["recall"], label="Recall", color="#c0392b")
    ax.plot(th, scan["f1"], label="F1", color="#27ae60")
    _add_theta_vline(ax, theta_default)
    ax.set_xlabel("Warning threshold")
    ax.set_ylabel("Metric value")
    ax.set_xlim(th.min(), th.max())
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="center right", framealpha=0.92)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_threshold_sensitivity_error(
    scan: pd.DataFrame,
    out_png: Path,
    out_pdf: Path,
    *,
    theta_default: float = DEFAULT_THETA,
) -> None:
    plt = _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    th = scan["threshold"].values
    ax.plot(th, scan["fpr"], label="FPR", color="#e67e22")
    ax.plot(th, scan["fnr"], label="FNR", color="#8e44ad")
    _add_theta_vline(ax, theta_default)
    ax.set_xlabel("Warning threshold")
    ax.set_ylabel("Error rate")
    ax.set_xlim(th.min(), th.max())
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="center right", framealpha=0.92)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_threshold_sensitivity_safety(
    scan: pd.DataFrame,
    out_png: Path,
    out_pdf: Path,
    *,
    theta_default: float = DEFAULT_THETA,
) -> None:
    if scan["success_rate"].isna().all() and scan["mean_lead_time"].isna().all():
        print("  [skip] 无有效 event_time，跳过 Success Rate / Mean Lead Time 图。")
        return

    plt = _setup_matplotlib()
    fig, ax1 = plt.subplots(figsize=(5.8, 4.2))
    th = scan["threshold"].values
    ax1.plot(th, scan["success_rate"], color="#1f4e79", label="Success rate")
    ax1.set_xlabel("Warning threshold")
    ax1.set_ylabel("Success rate")
    ax1.set_ylim(-0.02, 1.05)
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(th, scan["mean_lead_time"], color="#c0392b", linestyle="-", label="Mean lead time (s)")
    ax2.set_ylabel("Mean lead time (s)")

    _add_theta_vline(ax1, theta_default)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", framealpha=0.92)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def recommend_thresholds(scan: pd.DataFrame) -> dict[str, dict]:
    out: dict[str, dict] = {}

    if scan.empty:
        return out

    best_f1 = scan.sort_values("f1", ascending=False).iloc[0]
    out["max_f1"] = {
        "threshold": float(best_f1["threshold"]),
        "f1": float(best_f1["f1"]),
        "precision": float(best_f1["precision"]),
        "recall": float(best_f1["recall"]),
    }

    hi = scan[scan["recall"] >= 0.90]
    if len(hi) > 0:
        row = hi.sort_values("precision", ascending=False).iloc[0]
        out["max_precision_recall_ge_0.90"] = {
            "threshold": float(row["threshold"]),
            "precision": float(row["precision"]),
            "recall": float(row["recall"]),
        }

    if "success_rate" in scan.columns and scan["success_rate"].notna().any():
        perfect = scan[np.isclose(scan["success_rate"], 1.0, rtol=0, atol=1e-9)]
        if len(perfect) > 0:
            row = perfect.sort_values("mean_lead_time", ascending=False).iloc[0]
            out["max_lead_time_at_success_1"] = {
                "threshold": float(row["threshold"]),
                "mean_lead_time": float(row["mean_lead_time"]),
                "success_rate": float(row["success_rate"]),
            }

    return out


def format_threshold_summary(rec: dict[str, dict], theta_default: float) -> str:
    lines = [
        "Threshold sensitivity summary (Ours, test-set pooled)",
        "=" * 60,
        "Threshold selection based on test-set sensitivity is reported only for "
        "analysis and should not be used as the final deployment threshold.",
        "",
        f"Default operating point (reference only): theta = {theta_default:.2f}",
        "",
    ]
    if "max_f1" in rec:
        r = rec["max_f1"]
        lines.append(
            f"Max F1 on test sweep: theta={r['threshold']:.3f}  "
            f"F1={r['f1']:.4f}  P={r['precision']:.4f}  R={r['recall']:.4f}"
        )
    if "max_precision_recall_ge_0.90" in rec:
        r = rec["max_precision_recall_ge_0.90"]
        lines.append(
            f"Max Precision with Recall>=0.90: theta={r['threshold']:.3f}  "
            f"P={r['precision']:.4f}  R={r['recall']:.4f}"
        )
    else:
        lines.append("Max Precision with Recall>=0.90: (no threshold satisfies recall>=0.90)")
    if "max_lead_time_at_success_1" in rec:
        r = rec["max_lead_time_at_success_1"]
        lines.append(
            f"Max mean lead time at success_rate=1.0: theta={r['threshold']:.3f}  "
            f"lead={r['mean_lead_time']:.2f}s"
        )
    else:
        lines.append("Max mean lead time at success_rate=1.0: (not available or no perfect success)")
    lines.append("")
    return "\n".join(lines)


def per_fold_metrics(
    predictions_dir: Path,
    *,
    theta_default: float = DEFAULT_THETA,
) -> pd.DataFrame:
    rows: list[dict] = []
    for path in load_fold_prediction_files(predictions_dir):
        df = load_single_fold_predictions(path)
        fold_id = int(df["fold_id"].iloc[0])
        y_true = df["y_true"].values.astype(int)
        y_prob = df["y_prob"].values.astype(float)

        pr = pr_curve_metrics(y_true, y_prob)
        cls = classification_at_threshold(y_true, y_prob, theta_default)
        lead = lead_metrics_at_threshold(df, theta_default)

        rows.append(
            {
                "fold_id": fold_id,
                "pr_auc": pr["pr_auc"],
                "ap": pr["ap"],
                "precision": cls["precision"],
                "recall": cls["recall"],
                "f1": cls["f1"],
                "mcc": cls["mcc"],
                "mean_lead_time": lead["mean_lead_time"],
                "success_rate": lead["success_rate"],
                "n_windows": len(df),
            }
        )
    return pd.DataFrame(rows).sort_values("fold_id").reset_index(drop=True)


def summarize_fold_metrics(fold_df: pd.DataFrame) -> pd.DataFrame:
    if fold_df.empty:
        return pd.DataFrame()
    numeric = [
        "pr_auc",
        "ap",
        "precision",
        "recall",
        "f1",
        "mcc",
        "mean_lead_time",
        "success_rate",
    ]
    rows: list[dict] = []
    for col in numeric:
        if col not in fold_df.columns:
            continue
        vals = fold_df[col].astype(float)
        finite = vals[np.isfinite(vals)]
        if len(finite) == 0:
            rows.append({"metric": col, "mean": np.nan, "std": np.nan, "n_folds": 0})
        else:
            rows.append(
                {
                    "metric": col,
                    "mean": float(finite.mean()),
                    "std": float(finite.std(ddof=1)) if len(finite) > 1 else 0.0,
                    "n_folds": int(len(finite)),
                }
            )
    return pd.DataFrame(rows)


def run_analysis(
    kfold_root: Path,
    *,
    bootstrap_from_eval: bool = False,
    theta_default: float = DEFAULT_THETA,
) -> None:
    dirs = resolve_output_dirs(kfold_root)
    pred_dir = dirs["predictions"] / "Ours"
    if not load_fold_prediction_files(pred_dir):
        pred_dir = dirs["predictions"]
    fig_dir = dirs["figures"]
    ana_dir = dirs["analysis"]

    paths = load_fold_prediction_files(pred_dir)
    if not paths and bootstrap_from_eval:
        print(f"[bootstrap] 从 eval/test_predictions.csv 生成 {pred_dir} …")
        materialize_predictions_from_eval(kfold_root)
        paths = load_fold_prediction_files(pred_dir)

    if not paths:
        raise FileNotFoundError(
            f"未找到预测文件。请先完成 K-fold evaluate，或使用 --bootstrap-from-eval。"
        )

    merged = load_merged_predictions(pred_dir)
    print(f"Loaded {len(merged)} windows from {len(paths)} fold file(s).")

    # A–B: PR curve
    pr_stats = plot_pr_curve(
        merged,
        fig_dir / "pr_curve_ours.png",
        fig_dir / "pr_curve_ours.pdf",
        theta_default=theta_default,
    )
    print(f"Overall PR-AUC = {pr_stats['pr_auc']:.4f}  AP = {pr_stats['ap']:.4f}")

    # C–D: threshold sensitivity
    scan = threshold_sensitivity_table(merged)
    ana_dir.mkdir(parents=True, exist_ok=True)
    scan_path = ana_dir / "threshold_sensitivity_ours.csv"
    scan.to_csv(scan_path, index=False)
    print(f"Wrote {scan_path}")

    # E: figures
    plot_threshold_sensitivity_prf(
        scan,
        fig_dir / "threshold_sensitivity_prf.png",
        fig_dir / "threshold_sensitivity_prf.pdf",
        theta_default=theta_default,
    )
    plot_threshold_sensitivity_error(
        scan,
        fig_dir / "threshold_sensitivity_error.png",
        fig_dir / "threshold_sensitivity_error.pdf",
        theta_default=theta_default,
    )
    plot_threshold_sensitivity_safety(
        scan,
        fig_dir / "threshold_sensitivity_safety.png",
        fig_dir / "threshold_sensitivity_safety.pdf",
        theta_default=theta_default,
    )

    # F: recommended thresholds (analysis only)
    rec = recommend_thresholds(scan)
    summary_text = format_threshold_summary(rec, theta_default)
    print("\n" + summary_text)
    summary_path = ana_dir / "threshold_summary_ours.txt"
    summary_path.write_text(summary_text + "\n", encoding="utf-8")
    print(f"Wrote {summary_path}")

    # G: per-fold metrics
    if len(paths) > 1:
        fold_df = per_fold_metrics(pred_dir, theta_default=theta_default)
        fold_path = ana_dir / "fold_metrics_ours.csv"
        fold_df.to_csv(fold_path, index=False)
        fold_sum = summarize_fold_metrics(fold_df)
        fold_sum_path = ana_dir / "fold_metrics_summary_ours.csv"
        fold_sum.to_csv(fold_sum_path, index=False)
        print(f"Wrote {fold_path} and {fold_sum_path}")
        for _, row in fold_sum.iterrows():
            print(f"  {row['metric']}: {row['mean']:.4f} ± {row['std']:.4f} (n={int(row['n_folds'])})")
    else:
        print("[per-fold] 仅 1 个 fold 文件，跳过 fold_metrics 汇总。")

    print(f"\nDone. Figures → {fig_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PR curve and warning-threshold sensitivity (test analysis only)",
    )
    parser.add_argument(
        "--kfold-root",
        type=Path,
        required=True,
        help="K-fold 输出根目录（含 fold_* 或 outputs/predictions）",
    )
    parser.add_argument(
        "--bootstrap-from-eval",
        action="store_true",
        help="若无 outputs/predictions，从 fold_*/eval/test_predictions.csv 生成",
    )
    parser.add_argument(
        "--theta-default",
        type=float,
        default=DEFAULT_THETA,
        help="图中标注的默认阈值（默认 0.5）",
    )
    args = parser.parse_args()

    run_analysis(
        args.kfold_root.resolve(),
        bootstrap_from_eval=args.bootstrap_from_eval,
        theta_default=args.theta_default,
    )


if __name__ == "__main__":
    main()
