#!/usr/bin/env python3
"""
预警轨三类敏感性分析汇总导出：

1. 阈值敏感性（主模型 kfold_regularized，test pooled）
2. 超参数敏感性（tune/grid_search_results.csv）
3. history_window 敏感性（window_sweep/window_search_results.csv）

Usage (from Fire_prediction/):
    python -m dual_track_analysis.warning.export_sensitivity --all
    python -m dual_track_analysis.warning.export_sensitivity --threshold
    python -m dual_track_analysis.warning.export_sensitivity --hyperparam
    python -m dual_track_analysis.warning.export_sensitivity --window
"""
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

from evaluate import WARNING_PRIMARY_METRIC_KEYS

_DEFAULT_KFOLD = _ROOT / "dual_track_analysis/outputs/warning/kfold_regularized"
_DEFAULT_OUT = _ROOT / "dual_track_analysis/outputs/warning/summary/sensitivity"
_DEFAULT_TUNE = _ROOT / "dual_track_analysis/outputs/warning/tune"
_DEFAULT_WINDOW = _ROOT / "dual_track_analysis/outputs/warning/window_sweep"
_ARCHIVED_WINDOW = _ROOT / "_archived/dual_track_outputs/warning/window_sweep"

_HP_LABELS = {
    "dropout": "dropout",
    "head_dropout": "head_dropout",
    "lambda_warn": "λ_warn",
    "lr": "learning_rate",
    "train_oversample_factor": "oversample",
    "warn_focal_alpha": "focal_α",
    "warn_pos_weight_cap": "pos_weight_cap",
    "weight_decay": "weight_decay",
    "d_model": "d_model",
}

_TEST_METRICS = ("test_f1", "test_recall", "test_precision", "test_mcc", "test_mean_lead_time")
_VAL_METRICS = ("val_f1", "val_recall", "val_precision", "val_mcc")


def _relpath(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _fmt(mean: float, std: float = 0.0) -> str:
    if not np.isfinite(mean):
        return "—"
    if not np.isfinite(std) or std == 0:
        return f"{mean:.4f}"
    return f"{mean:.4f} ± {std:.4f}"


def _setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "sans-serif"]
    plt.rcParams["font.size"] = 10
    return plt


def export_threshold_sensitivity(
    kfold_root: Path,
    out_dir: Path,
    *,
    label: str = "主模型 (kfold_regularized)",
) -> Path:
    """阈值敏感性：复用 analysis.warning_threshold_sensitivity。"""
    from analysis.warning_threshold_sensitivity import run_kfold_sensitivity

    kfold_root = kfold_root.resolve()
    out_dir = out_dir.resolve()
    thresh_dir = out_dir / "threshold"
    run_kfold_sensitivity(kfold_root, thresh_dir, bootstrap_from_eval=True)

    report = thresh_dir / "threshold_sensitivity_report.md"
    if report.is_file():
        text = report.read_text(encoding="utf-8")
        text = text.replace(
            "# 阈值敏感性分析（test pooled）",
            f"# 阈值敏感性分析（{label}，test pooled）",
            1,
        )
        report.write_text(text, encoding="utf-8")

    print(f"[threshold] → {thresh_dir}")
    return thresh_dir


def _load_sweep_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"未找到结果表: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"结果表为空: {path}")
    if "incomplete" in df.columns:
        df = df[df["incomplete"].astype(str).str.lower() != "true"].copy()
    return df


def _resolve_window_sweep_root(sweep_root: Path) -> Path:
    """Prefer active sweep dir; fall back to archived results if needed."""
    sweep_root = sweep_root.resolve()
    if (sweep_root / "window_search_results.csv").is_file():
        return sweep_root
    if (_ARCHIVED_WINDOW / "window_search_results.csv").is_file():
        print(f"[window] using archived sweep root: {_ARCHIVED_WINDOW}")
        return _ARCHIVED_WINDOW.resolve()
    return sweep_root


def _resolve_sweep_run_dir(row: pd.Series, sweep_root: Path) -> Path | None:
    candidates: list[Path] = []
    run_id = row.get("run_id")
    if isinstance(run_id, str) and run_id:
        candidates.append(sweep_root / run_id)
    out_dir = row.get("output_dir")
    if isinstance(out_dir, str) and out_dir:
        candidates.append(_ROOT / out_dir)
        candidates.append(_ARCHIVED_WINDOW / Path(out_dir).name)
    for path in candidates:
        if path.is_dir() and any(path.glob("fold_*")):
            return path
    return None


def _backfill_window_primary_columns(df: pd.DataFrame, sweep_root: Path) -> pd.DataFrame:
    """Fill missing split primary columns (e.g. val_mean_lead_time) from fold eval dirs."""
    from dual_track_analysis.warning.export_results import collect_split_primary

    df = df.copy()
    for idx, row in df.iterrows():
        run_dir = _resolve_sweep_run_dir(row, sweep_root)
        if run_dir is None:
            continue
        metrics = collect_split_primary(run_dir)
        for split in ("train", "val", "test"):
            for metric in WARNING_PRIMARY_METRIC_KEYS:
                col = f"{split}_{metric}"
                if col not in df.columns:
                    continue
                cur = pd.to_numeric(row.get(col), errors="coerce")
                if np.isfinite(cur):
                    continue
                val = metrics.get(split, {}).get(metric, float("nan"))
                if np.isfinite(val):
                    df.at[idx, col] = val
    return df


def _hp_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("hp_")]
    if not cols and "history_window" in df.columns:
        return ["history_window"]
    return cols


def _marginal_table(df: pd.DataFrame, param_col: str, metric: str) -> pd.DataFrame:
    rows: list[dict] = []
    for val, sub in df.groupby(param_col, sort=True):
        vals = sub[metric].dropna()
        rows.append({
            param_col: val,
            "n_runs": int(len(sub)),
            "mean": float(vals.mean()) if len(vals) else float("nan"),
            "std": float(vals.std()) if len(vals) > 1 else 0.0,
            "min": float(vals.min()) if len(vals) else float("nan"),
            "max": float(vals.max()) if len(vals) else float("nan"),
        })
    return pd.DataFrame(rows)


def _plot_marginal_sensitivity(
    df: pd.DataFrame,
    param_cols: list[str],
    *,
    metric: str,
    out_png: Path,
    title: str,
    xlabel_map: dict[str, str] | None = None,
) -> None:
    try:
        plt = _setup_matplotlib()
    except ImportError:
        print("  [skip] matplotlib 不可用，跳过敏感性图。")
        return

    n = len(param_cols)
    if n == 0:
        return

    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows), squeeze=False)
    xmap = xlabel_map or {}

    for idx, col in enumerate(param_cols):
        ax = axes[idx // ncols][idx % ncols]
        sub = df[[col, metric]].dropna()
        if sub.empty:
            ax.set_visible(False)
            continue

        grouped = sub.groupby(col)[metric].agg(["mean", "std", "count"]).reset_index()
        x = grouped[col].values
        y = grouped["mean"].values
        yerr = grouped["std"].fillna(0).values

        ax.errorbar(x, y, yerr=yerr, fmt="o-", color="#1f4e79", capsize=3, markersize=5)
        for _, row in sub.iterrows():
            ax.scatter(row[col], row[metric], color="#c0392b", alpha=0.35, s=18, zorder=1)

        label = xmap.get(col, col.replace("hp_", ""))
        ax.set_xlabel(label)
        ax.set_ylabel(metric.replace("test_", "test ").replace("val_", "val "))
        ax.set_title(label)
        ax.grid(True, alpha=0.25)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(title, fontsize=11, y=1.02)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_window_curves(df: pd.DataFrame, out_png: Path) -> None:
    try:
        plt = _setup_matplotlib()
    except ImportError:
        return

    if "history_window" not in df.columns:
        return

    sub = df.sort_values("history_window")
    x = sub["history_window"].values
    fig, axes = plt.subplots(2, 1, figsize=(6.2, 6.0), sharex=True)

    ax = axes[0]
    for col, color, label in (
        ("test_f1", "#27ae60", "F1"),
        ("test_recall", "#c0392b", "Recall"),
        ("test_precision", "#8e44ad", "Precision"),
        ("test_mcc", "#1f4e79", "MCC"),
    ):
        if col in sub.columns:
            ax.plot(x, sub[col], "o-", color=color, label=label)
    ax.set_ylabel("Test classification")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="center right", fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.set_title("History window sensitivity (test, 5-fold mean)")

    ax2 = axes[1]
    if "test_mean_lead_time" in sub.columns:
        ax2.plot(x, sub["test_mean_lead_time"], "o-", color="#c0392b", label="Mean lead (s)")
    ax2.set_xlabel("history_window (s)")
    ax2.set_ylabel("Mean lead time (s)")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="center right", fontsize=8)

    if 50 in x:
        for a in axes:
            a.axvline(50, color="#888888", linestyle="--", linewidth=1.0, label="baseline")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def export_hyperparam_sensitivity(
    tune_root: Path,
    out_dir: Path,
    *,
    results_name: str = "grid_search_results.csv",
) -> Path | None:
    tune_root = tune_root.resolve()
    out_dir = out_dir.resolve()
    hp_dir = out_dir / "hyperparam"
    fig_dir = hp_dir / "figures"
    hp_dir.mkdir(parents=True, exist_ok=True)

    results_path = tune_root / results_name
    try:
        df = _load_sweep_csv(results_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"[hyperparam] 跳过: {e}")
        return None

    param_cols = _hp_columns(df)
    if not param_cols:
        print("[hyperparam] 跳过: 无 hp_* 列")
        return None

    # 边际汇总表
    marginal_parts: list[pd.DataFrame] = []
    for col in param_cols:
        for metric in _TEST_METRICS:
            if metric not in df.columns:
                continue
            tab = _marginal_table(df, col, metric)
            tab.insert(0, "parameter", col.replace("hp_", ""))
            tab.insert(1, "metric", metric)
            marginal_parts.append(tab)
    if marginal_parts:
        marginal_df = pd.concat(marginal_parts, ignore_index=True)
        marginal_df.to_csv(hp_dir / "hyperparam_marginal_summary.csv", index=False)

    df.to_csv(hp_dir / "hyperparam_runs.csv", index=False)

    xmap = {c: _HP_LABELS.get(c.replace("hp_", ""), c.replace("hp_", "")) for c in param_cols}
    _plot_marginal_sensitivity(
        df, param_cols, metric="test_f1",
        out_png=fig_dir / "hyperparam_sensitivity_test_f1.png",
        title="Hyperparameter sensitivity (test F1)",
        xlabel_map=xmap,
    )
    _plot_marginal_sensitivity(
        df, param_cols, metric="val_f1",
        out_png=fig_dir / "hyperparam_sensitivity_val_f1.png",
        title="Hyperparameter sensitivity (val F1, selection reference)",
        xlabel_map=xmap,
    )

    # Markdown 报告
    n_runs = len(df)
    best = df.sort_values("val_f1", ascending=False).iloc[0]
    md = [
        "# 超参数敏感性分析",
        "",
        f"来源: `{_relpath(results_path, _ROOT)}`",
        f"完整运行数: **{n_runs}**（已排除 incomplete）",
        "",
        "说明：各点为独立网格组合；边际表按参数取值聚合 test 指标均值。",
        "选参应依据 **val F1**，test 仅作敏感性观察。",
        "",
        f"**val F1 最高组合**: `{best.get('run_id', '—')}`",
        "",
    ]
    if "overrides_json" in best.index:
        try:
            ov = json.loads(best["overrides_json"])
            md.append(f"- overrides: `{ov}`")
        except (json.JSONDecodeError, TypeError):
            pass
    md += [
        f"- val F1 = {best.get('val_f1', float('nan')):.4f}",
        f"- test F1 = {best.get('test_f1', float('nan')):.4f}",
        "",
        "## test F1 边际汇总（按单参数分组）",
        "",
        "| parameter | value | n_runs | mean | std | min | max |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    if marginal_parts:
        f1_marg = marginal_df[marginal_df["metric"] == "test_f1"]
        for _, r in f1_marg.iterrows():
            pcol = f"hp_{r['parameter']}" if f"hp_{r['parameter']}" in df.columns else r["parameter"]
            md.append(
                f"| {r['parameter']} | {r[pcol]} | {int(r['n_runs'])} | "
                f"{r['mean']:.4f} | {r['std']:.4f} | {r['min']:.4f} | {r['max']:.4f} |"
            )

    md += [
        "",
        "## 文件",
        "",
        "- `hyperparam_runs.csv` — 全部网格运行",
        "- `hyperparam_marginal_summary.csv` — 单参数边际汇总",
        "- `figures/hyperparam_sensitivity_test_f1.png`",
        "- `figures/hyperparam_sensitivity_val_f1.png`",
    ]
    (hp_dir / "hyperparam_sensitivity_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[hyperparam] {n_runs} runs → {hp_dir}")
    return hp_dir


def export_window_sensitivity(
    sweep_root: Path,
    out_dir: Path,
    *,
    results_name: str = "window_search_results.csv",
) -> Path | None:
    sweep_root = _resolve_window_sweep_root(sweep_root)
    out_dir = out_dir.resolve()
    win_dir = out_dir / "window"
    fig_dir = win_dir / "figures"
    win_dir.mkdir(parents=True, exist_ok=True)

    results_path = sweep_root / results_name
    try:
        df = _load_sweep_csv(results_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"[window] 跳过: {e}")
        (win_dir / "window_sensitivity_report.md").write_text(
            "# History Window 敏感性分析\n\n"
            f"尚无结果。请先运行：\n\n"
            "```bash\n"
            "cd Fire_prediction\n"
            "python -m dual_track_analysis.warning.window_search --resume --skip-existing\n"
            "```\n",
            encoding="utf-8",
        )
        return None

    if "history_window" not in df.columns and "hp_history_window" in df.columns:
        df = df.rename(columns={"hp_history_window": "history_window"})

    df = _backfill_window_primary_columns(df, sweep_root)
    df = df.sort_values("history_window")
    df.to_csv(win_dir / "window_runs.csv", index=False)
    _plot_window_curves(df, fig_dir / "window_sensitivity_primary.png")

    baseline_row = df[df["history_window"] == 50]
    if baseline_row.empty:
        baseline_row = df.loc[df["val_f1"].idxmax()] if "val_f1" in df.columns else df.iloc[0]
    else:
        baseline_row = baseline_row.iloc[0]

    md = [
        "# History Window 敏感性分析",
        "",
        f"来源: `{_relpath(results_path, _ROOT)}`",
        f"窗口数: **{len(df)}**",
        "",
        "各 history_window 独立 5-fold；参考基线 **history_window=50**（若存在）。",
        "",
        "## 结果表（test）",
        "",
        "| history_window | val F1 | test F1 | test Recall | test Precision | test lead (s) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in df.iterrows():
        md.append(
            f"| {int(r['history_window'])} | {r.get('val_f1', float('nan')):.4f} | "
            f"{r.get('test_f1', float('nan')):.4f} | {r.get('test_recall', float('nan')):.4f} | "
            f"{r.get('test_precision', float('nan')):.4f} | "
            f"{r.get('test_mean_lead_time', float('nan')):.1f} |"
        )

    if len(df) >= 2:
        best_hw = int(df.loc[df["val_f1"].idxmax(), "history_window"]) if "val_f1" in df.columns else int(df.iloc[0]["history_window"])
        md += [
            "",
            "## 简要结论",
            "",
            f"- val F1 最优窗口: **{best_hw}**",
        ]
        if int(baseline_row.get("history_window", -1)) == 50:
            b_f1 = float(baseline_row.get("test_f1", float("nan")))
            w_f1 = float(df.loc[df["val_f1"].idxmax(), "test_f1"]) if "val_f1" in df.columns else float("nan")
            if np.isfinite(b_f1) and np.isfinite(w_f1):
                md.append(f"- 相对 baseline (hw=50) test F1 变化: {w_f1 - b_f1:+.4f}")

    md += [
        "",
        "## 文件",
        "",
        "- `window_runs.csv`",
        "- `figures/window_sensitivity_primary.png`",
    ]
    (win_dir / "window_sensitivity_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[window] {len(df)} windows → {win_dir}")
    return win_dir


def export_all_sensitivity(
    *,
    kfold_root: Path,
    out_dir: Path,
    tune_root: Path,
    sweep_root: Path,
    kfold_label: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    export_threshold_sensitivity(kfold_root, out_dir, label=kfold_label)
    export_hyperparam_sensitivity(tune_root, out_dir)
    export_window_sensitivity(sweep_root, out_dir)

    index_md = [
        "# 预警敏感性分析汇总",
        "",
        f"主模型: `{_relpath(kfold_root, _ROOT)}`",
        "",
        "## 三类分析",
        "",
        "1. **阈值敏感性** — `threshold/threshold_sensitivity_report.md`",
        "2. **超参数敏感性** — `hyperparam/hyperparam_sensitivity_report.md`",
        "3. **History window 敏感性** — `window/window_sensitivity_report.md`",
        "",
        "阈值在 val 上选取；test 扫描仅用于稳健性。超参/window 选参看 val F1。",
    ]
    (out_dir / "README.md").write_text("\n".join(index_md) + "\n", encoding="utf-8")
    print(f"\n汇总索引 → {out_dir / 'README.md'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export warning-track sensitivity analyses")
    ap.add_argument("--kfold-root", type=Path, default=_DEFAULT_KFOLD)
    ap.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT)
    ap.add_argument("--tune-root", type=Path, default=_DEFAULT_TUNE)
    ap.add_argument("--window-root", type=Path, default=_DEFAULT_WINDOW)
    ap.add_argument("--kfold-label", type=str, default="主模型 (kfold_regularized)")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--threshold", action="store_true")
    ap.add_argument("--hyperparam", action="store_true")
    ap.add_argument("--window", action="store_true")
    args = ap.parse_args()

    run_all = args.all or not (args.threshold or args.hyperparam or args.window)
    out_dir = args.out_dir.resolve()
    kfold_root = args.kfold_root.resolve()
    tune_root = args.tune_root.resolve()
    sweep_root = args.window_root.resolve()

    if run_all:
        export_all_sensitivity(
            kfold_root=kfold_root,
            out_dir=out_dir,
            tune_root=tune_root,
            sweep_root=sweep_root,
            kfold_label=args.kfold_label,
        )
    else:
        if args.threshold:
            export_threshold_sensitivity(kfold_root, out_dir, label=args.kfold_label)
        if args.hyperparam:
            export_hyperparam_sensitivity(tune_root, out_dir)
        if args.window:
            export_window_sensitivity(sweep_root, out_dir)


if __name__ == "__main__":
    main()
