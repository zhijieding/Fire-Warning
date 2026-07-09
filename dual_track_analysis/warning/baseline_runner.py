"""预警轨基线 5-fold：与 Ours 共享划分、预处理、损失与 val 选阈流程。"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from config import Config, parse_config_value
from evaluate import WARNING_PRIMARY_METRIC_KEYS
from run_kfold_tuned import load_tuned_config

BASELINE_MODELS = (
    "lstm", "bilstm", "gru", "tcn", "informer", "transformer_noxa",
    "patchtst", "itransformer", "timesnet",
)

BASELINE_DISPLAY_NAMES = {
    "trimodal": "Ours (LBCA)",
    "lstm": "LSTM",
    "bilstm": "BiLSTM",
    "gru": "GRU",
    "tcn": "TCN",
    "informer": "Informer",
    "transformer_noxa": "Transformer",
    "patchtst": "PatchTST",
    "itransformer": "iTransformer",
    "timesnet": "TimesNet",
}

WARNING_TRACK_SETS = (
    "post_fire_as_supervised_neg=false",
    "include_post_fire_regression=false",
    "experiment_balanced_sampling=true",
)

EXP_METRICS = [
    "accuracy", "precision", "recall", "f1", "mcc", "auc", "pr_auc",
    "mean_lead_time", "success_rate",
]
POOLED_METRICS = [
    "pooled_accuracy", "pooled_precision", "pooled_recall",
    "pooled_f1", "pooled_mcc", "pooled_auc", "pooled_pr_auc",
]

PRIMARY_METRICS = tuple(WARNING_PRIMARY_METRIC_KEYS)


def load_warning_baseline_cfg(
    config_path: Path | None = None,
    extra_sets: Iterable[str] = (),
) -> tuple[Config, dict]:
    """Load warning Config (defaults from config.py, optional JSON overrides)."""
    cfg, tuned = load_tuned_config(config_path)
    cfg.apply_sets(list(WARNING_TRACK_SETS) + list(extra_sets))
    return cfg, tuned


def run_baseline_kfold(
    model_type: str,
    base_cfg: Config,
    output_dir: Path,
    *,
    max_folds: int | None = None,
    start_fold: int | None = None,
    resume: bool = False,
) -> dict:
    """Train/eval one baseline backbone under warning-track protocol."""
    from run_kfold import (
        _fold_dir_path,
        _fold_is_complete,
        _make_folds,
        _result_from_completed_fold,
        _run_one_fold,
    )
    from data_pipeline.preprocess import _get_event_times, discover_and_configure

    cfg = copy.deepcopy(base_cfg)
    cfg.model_type = model_type
    cfg.output_dir = str(output_dir)
    cfg.make_dirs()

    discover_and_configure(cfg)
    all_ids = list(cfg.full_process_ids)
    n = len(all_ids)
    k = 5
    event_times = _get_event_times(cfg, all_ids)
    folds = _make_folds(all_ids, k, event_times=event_times)

    print(f"\n╔{'═' * 58}╗")
    print(f"║  Baseline: {model_type:<20s}  [5-fold CV on {n} exps]".ljust(59) + "║")
    print(f"╚{'═' * 58}╝")

    results: list[dict] = []
    for fold_idx in range(k):
        if max_folds is not None and fold_idx >= max_folds:
            break
        if start_fold is not None and fold_idx < start_fold:
            continue

        test_ids = folds[fold_idx]
        val_ids = folds[(fold_idx + 1) % k]
        train_ids = [
            eid
            for i, group in enumerate(folds)
            for eid in group
            if i != fold_idx and i != (fold_idx + 1) % k
        ]

        cfg.output_dir = str(output_dir)
        fold_dir = _fold_dir_path(output_dir, fold_idx, test_ids)

        if resume and _fold_is_complete(fold_dir):
            print(f"\n[resume] {model_type} fold {fold_idx}: eval complete — skipping")
            results.append(
                _result_from_completed_fold(fold_idx, test_ids, val_ids, fold_dir)
            )
            continue

        print(f"\n[baseline] {model_type} fold {fold_idx}: test={test_ids} val={val_ids}")
        results.append(
            _run_one_fold(
                fold_idx, test_ids, val_ids, train_ids,
                cfg, pretrained=False,
            )
        )

    if not results:
        raise RuntimeError(f"No fold results for baseline {model_type!r}")

    df = pd.DataFrame(results)
    df.to_csv(output_dir / "5fold_results.csv", index=False)

    summary = {
        m: {
            "mean": float(df[m].dropna().mean()),
            "std": float(df[m].dropna().std()) if len(df[m].dropna()) > 1 else 0.0,
        }
        for m in EXP_METRICS if m in df.columns and df[m].notna().any()
    }
    summary["pooled"] = {
        m.replace("pooled_", ""): {
            "mean": float(df[m].dropna().mean()),
            "std": float(df[m].dropna().std()) if len(df[m].dropna()) > 1 else 0.0,
        }
        for m in POOLED_METRICS if m in df.columns and df[m].notna().any()
    }
    summary["n_folds"] = int(df["fold"].nunique()) if "fold" in df.columns else len(results)
    summary["n_experiments"] = n
    summary["model_type"] = model_type
    summary["display_name"] = BASELINE_DISPLAY_NAMES.get(model_type, model_type)
    summary["training_config"] = {
        "fusion_mode": getattr(cfg, "fusion_mode", "layer_bridge"),
        "lr": cfg.lr,
        "train_oversample_factor": cfg.train_oversample_factor,
        "lambda_pred": cfg.lambda_pred,
        "lambda_warn": cfg.lambda_warn,
        "lambda_trend": cfg.lambda_trend,
        "lambda_consist": cfg.lambda_consist,
        "epochs": cfg.epochs,
        "early_stopping_patience": cfg.early_stopping_patience,
        "eval_threshold_strategy": getattr(cfg, "eval_threshold_strategy", None),
        "eval_high_recall_min_recall": getattr(cfg, "eval_high_recall_min_recall", None),
        "post_fire_as_supervised_neg": cfg.post_fire_as_supervised_neg,
        "include_post_fire_regression": cfg.include_post_fire_regression,
        "output_dir": str(output_dir),
    }
    summary["metric_note"] = (
        "Test metrics: experiment-level macro-averaging across folds (mean ± std). "
        "Operating threshold selected on validation only; no test leakage. "
        "Primary paper metrics: recall, mean_lead_time, f1, mcc, precision, pr_auc."
    )
    summary["fold_assignments"] = [
        {
            "fold": i,
            "test": folds[i],
            "val": folds[(i + 1) % k],
            "train": [
                eid for j, g in enumerate(folds) for eid in g
                if j != i and j != (i + 1) % k
            ],
        }
        for i in range(k)
    ]
    with open(output_dir / "5fold_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    return summary


def _pull_metric(summary: dict | None, metric: str, *, pooled: bool = False) -> tuple[float, float]:
    if summary is None:
        return float("nan"), float("nan")
    node = summary.get("pooled", {}) if pooled else summary
    key = metric.replace("pooled_", "") if pooled else metric
    block = node.get(key)
    if not isinstance(block, dict):
        return float("nan"), float("nan")
    return float(block.get("mean", np.nan)), float(block.get("std", np.nan))


def _fmt(mean: float, std: float, *, digits: int = 4) -> str:
    if not np.isfinite(mean):
        return "—"
    if not np.isfinite(std) or std == 0:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def build_comparison_table(
    baseline_summaries: dict[str, dict],
    ours_summary: dict | None,
) -> pd.DataFrame:
    rows: list[dict] = []

    def _append_row(key: str, summary: dict | None) -> None:
        row: dict = {
            "model_type": key,
            "display_name": BASELINE_DISPLAY_NAMES.get(key, key),
        }
        for m in EXP_METRICS:
            mean, std = _pull_metric(summary, m, pooled=False)
            row[f"{m}_mean"] = mean
            row[f"{m}_std"] = std
            row[m] = _fmt(mean, std, digits=4 if m != "mean_lead_time" else 2)
        for m in POOLED_METRICS:
            mean, std = _pull_metric(summary, m, pooled=True)
            row[f"{m}_mean"] = mean
            row[f"{m}_std"] = std
        rows.append(row)

    _append_row("trimodal", ours_summary)
    for name in BASELINE_MODELS:
        if name in baseline_summaries:
            _append_row(name, baseline_summaries[name])
    return pd.DataFrame(rows)


def build_primary_wide_table(
    baseline_summaries: dict[str, dict],
    ours_summary: dict | None,
) -> pd.DataFrame:
    """Wide table of primary test metrics (experiment-level macro)."""
    rows: list[dict] = []

    def _append(key: str, summary: dict | None) -> None:
        if summary is None:
            return
        row = {"model": BASELINE_DISPLAY_NAMES.get(key, key)}
        for m in PRIMARY_METRICS:
            mean, std = _pull_metric(summary, m, pooled=False)
            if m == "mean_lead_time":
                row[m] = _fmt(mean, std, digits=2)
            else:
                row[m] = _fmt(mean, std, digits=4)
        rows.append(row)

    _append("trimodal", ours_summary)
    for name in BASELINE_MODELS:
        if name in baseline_summaries:
            _append(name, baseline_summaries[name])
    return pd.DataFrame(rows)


def write_comparison_reports(
    output_root: Path,
    baseline_summaries: dict[str, dict],
    ours_summary: dict | None,
    *,
    ours_summary_path: Path | None = None,
    config_path: Path | None = None,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    table = build_comparison_table(baseline_summaries, ours_summary)
    table.to_csv(output_root / "baseline_comparison.csv", index=False)

    wide = build_primary_wide_table(baseline_summaries, ours_summary)
    wide.to_csv(output_root / "baseline_comparison_wide.csv", index=False)

    meta = {
        "ours_summary_path": str(ours_summary_path) if ours_summary_path else None,
        "config_path": str(config_path) if config_path else None,
        "primary_metrics": list(PRIMARY_METRICS),
        "metric_note": (
            "Same 5-fold splits, warning-only training (no post-fire), "
            "val-only threshold selection. Test = experiment-level macro mean ± std."
        ),
        "ours": ours_summary,
        "baselines": baseline_summaries,
    }
    with open(output_root / "baseline_comparison.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)

    md_lines = [
        "# 预警轨基线对比 (5-Fold CV, test)",
        "",
        "与 Ours 共享：10 实验划分、预处理、滑窗、损失、val 温度标定与 val 选阈；"
        "仅时序主干不同；`post_fire_as_supervised_neg=false`。",
        "",
        "## 主指标（experiment-level macro, mean ± std）",
        "",
        "| Model | " + " | ".join(PRIMARY_METRICS) + " |",
        "|---|" + "|".join(["---"] * len(PRIMARY_METRICS)) + "|",
    ]
    for _, r in wide.iterrows():
        cells = [str(r["model"])] + [str(r[m]) for m in PRIMARY_METRICS]
        md_lines.append("| " + " | ".join(cells) + " |")
    md_lines += [
        "",
        "## 完整指标",
        "",
        "见 `baseline_comparison.csv`",
    ]
    (output_root / "baseline_comparison.md").write_text(
        "\n".join(md_lines) + "\n", encoding="utf-8",
    )
